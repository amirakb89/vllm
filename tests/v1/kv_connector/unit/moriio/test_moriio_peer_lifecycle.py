# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A restarted MoRIIO prefill is dropped, and reached again by a new handshake.

The scheduler decides (PeerTracker) and every worker applies the decision from
the same connector metadata; a READ worker also refuses to read KV computed by
another launch of the prefill than the one its sessions reach.
"""

import threading
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOAgentMetadata,
    MoRIIOConnectorMetadata,
    MoRIIOMode,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_peers import (
    PeerTracker,
)

pytestmark = pytest.mark.cpu_test

PEER = "10.0.0.1:6301"
OTHER = "10.0.0.2:6301"


def test_restarted_rank_is_dropped_and_a_late_old_request_does_not_flip_it_back():
    tracker = PeerTracker()
    tracker.note_request(PEER, incarnation="100")
    # Ranks are separate launches: rank 1's own incarnation is not a restart.
    tracker.note_request(PEER, dp_rank=1, incarnation="150")
    assert tracker.take_drops() == {}

    tracker.note_request(PEER, incarnation="200")
    drops = tracker.take_drops()
    assert list(drops) == [PEER]
    assert drops[PEER][1] == "200", "the new launch's sessions must be kept"

    # A request computed by the old launch, scheduled after the new one, must
    # not drop the sessions to the new launch.
    tracker.note_request(PEER, incarnation="100")
    tracker.note_request(PEER, incarnation="")
    assert tracker.take_drops() == {}


def _req(transfer_id: str, incarnation: str = "", dp_rank: int = 0) -> ReqMeta:
    return ReqMeta(
        transfer_id=transfer_id,
        local_block_ids=[[1, 2]],
        remote_block_ids=[[3, 4]],
        remote_host="10.0.0.1",
        remote_port=6301,
        remote_handshake_port=6301,
        remote_notify_port=7000,
        remote_engine_id=PEER,
        tp_size=1,
        remote_dp_size=1,
        remote_dp_rank=dp_rank,
        remote_incarnation=incarnation,
    )


def test_decode_scheduler_drops_a_restarted_prefill_in_the_step_of_its_first_read():
    """The drop rides in the same metadata as the first read from the new
    launch, so no worker reads it over the old launch's sessions."""
    sched = SimpleNamespace(_peers=PeerTracker())
    meta = MoRIIOConnectorMetadata()
    meta.reqs_to_recv["r1"] = _req("t1", "100")
    MoRIIOConnectorScheduler._update_peers(sched, meta)
    assert meta.peers_to_drop == {}

    meta = MoRIIOConnectorMetadata()
    meta.reqs_to_recv["r2"] = _req("t2", "200")
    MoRIIOConnectorScheduler._update_peers(sched, meta)
    assert list(meta.peers_to_drop) == [PEER]


class _Wrapper:
    def __init__(self):
        self.deregistered: list[bytes] = []

    def deregister_remote_engine(self, packed: bytes) -> None:
        self.deregistered.append(packed)

    def async_wait_reqid(self) -> None:
        pass


def _agent(eid: str, incarnation: str) -> MoRIIOAgentMetadata:
    return MoRIIOAgentMetadata(
        engine_id=eid,
        agent_metadata=f"desc-{eid}-{incarnation}".encode(),
        kv_caches_base_addr=[0],
        num_blocks=8,
        block_len=64,
        attn_backend_name="test",
        incarnation=incarnation,
    )


def _worker_with_sessions(sessions: dict[str, str]) -> SimpleNamespace:
    """A producer worker holding sessions to {session key: incarnation}."""
    worker = SimpleNamespace(
        is_producer=True,
        mode=MoRIIOMode.WRITE,
        _handshake_lock=threading.RLock(),
        _remote_agents={eid: {eid} for eid in sessions},
        remote_moriio_metadata={eid: _agent(eid, inc) for eid, inc in sessions.items()},
        layer_name_to_remote_kv_cache_metadata={eid: {} for eid in sessions},
        built_write_session={eid: ["session"] for eid in sessions},
        write_ready_flags={PEER: True, OTHER: True},
        load_ready_flag={PEER: True},
        _eager_handshaked_engines={PEER, OTHER},
        moriio_wrapper=_Wrapper(),
        _consumer_notification_counts={},
        _completed_consumer_notifications=set(),
    )
    worker.invalidate_remote_peer = lambda *args: (
        MoRIIOConnectorWorker.invalidate_remote_peer(worker, *args)
    )
    return worker


def test_worker_drops_every_rank_of_the_peer_except_the_new_launch():
    worker = _worker_with_sessions(
        {
            f"{PEER}_dp0": "100",
            f"{PEER}_dp1": "200",  # already handshaken with the new launch
            f"{OTHER}_dp0": "100",
        }
    )
    meta = MoRIIOConnectorMetadata()
    meta.peers_to_drop = {PEER: ("rank 0 restarted (100 -> 200)", "200")}
    MoRIIOConnectorWorker.start_load_kv(worker, meta)

    assert set(worker.remote_moriio_metadata) == {f"{PEER}_dp1", f"{OTHER}_dp0"}
    assert set(worker.built_write_session) == {f"{PEER}_dp1", f"{OTHER}_dp0"}
    assert worker.moriio_wrapper.deregistered == [f"desc-{PEER}_dp0-100".encode()]
    # The next READ step handshakes the peer again, on every TP worker alike.
    assert worker._eager_handshaked_engines == {OTHER}
    assert PEER not in worker.write_ready_flags


def _read_worker(session_incarnation: str) -> SimpleNamespace:
    worker = SimpleNamespace(
        mode=MoRIIOMode.READ,
        world_size=1,
        tp_rank=0,
        _has_mamba=False,
        _invalid_block_ids=set(),
        layer_name_to_local_kv_cache_metadata={},
        moriio_config=SimpleNamespace(transfer_timeout=30.0),
    )
    worker._remote_tp_rank = lambda size: 0
    worker.get_engine_name_with_dp = lambda name, dp: (
        MoRIIOConnectorWorker.get_engine_name_with_dp(worker, name, dp)
    )
    worker._get_built_session = lambda eid: (
        [],
        _agent(eid, session_incarnation),
    )
    return worker


@pytest.mark.parametrize("session_incarnation", ["200", ""])
def test_read_of_kv_from_another_prefill_launch_is_a_load_error(session_incarnation):
    """Block ids of a request computed by the old launch name other KV in the
    new one: the read must fail as a load error, not load that KV. A peer that
    sends no incarnation is read as before."""
    worker = _read_worker(session_incarnation)
    MoRIIOConnectorWorker._read_blocks(
        worker,
        local_block_ids=[[5, 6]],
        remote_block_ids=[[7, 8]],
        dst_engine_id=PEER,
        request_id="r1",
        transfer_id="t1",
        remote_host="10.0.0.1",
        remote_notify_port=7000,
        remote_tp_size=1,
        remote_incarnation="100",
    )
    expected = {5, 6} if session_incarnation else set()
    assert worker._invalid_block_ids == expected

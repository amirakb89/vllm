# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A READ decode reconnects to a prefill rank that restarted, and does not read
KV that the previous launch computed from the new one."""

import contextlib
import importlib.util
import threading

import msgspec
import pytest
import torch
import zmq

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    make_kv_cache_group,
)
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    HandshakeError,
    MoRIIOAgentMetadata,
    MoRIIOConstants,
    get_port_offset,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    KVConnectorRole,
    MoRIIOConnector,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_peers import (
    PeerTracker,
)
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheTensor

from ..utils import create_request, create_scheduler
from .test_moriio_connector import _find_free_port, create_vllm_config

PEER = "10.0.0.1:6301"


@pytest.mark.cpu_test
def test_tracker_reports_first_launches_and_restarts_not_late_old_requests():
    tracker = PeerTracker()
    tracker.note_request(PEER, dp_rank=0, incarnation="100")
    tracker.note_request(PEER, dp_rank=1, incarnation="150")
    tracker.note_request(PEER, dp_rank=1, incarnation="")  # a peer sending none
    assert tracker.take_changes() == {PEER: {0: "100", 1: "150"}}

    # Launches are compared for identity, not ordered: a restart whose clock
    # reads earlier, or whose id is not a plain number, is still a restart.
    tracker.note_request(PEER, dp_rank=0, incarnation="050_dp0")
    assert tracker.take_changes() == {PEER: {0: "050_dp0"}}

    # Requests the previous launch computed, scheduled after the restart.
    tracker.note_request(PEER, dp_rank=0, incarnation="100")
    tracker.note_request(PEER, dp_rank=0, incarnation="050_dp0")
    assert tracker.take_changes() == {}


LAYERS = ["layer0", "layer1", "layer2"]
NUM_BLOCKS = 16
SPEC = AttentionSpec(block_size=16, num_kv_heads=4, head_size=64, dtype=torch.float16)


class _Done:
    def Succeeded(self):
        return True

    def Failed(self):
        return False


class _Transport:
    """MoRI stand-in: a session is the remote region it reaches, and every
    read and deregistration is recorded."""

    def __init__(self, *args, **kwargs):
        self.lock = threading.Lock()
        self.reads: list[bytes] = []
        self.deregistered: list[bytes] = []

    def set_moriio_engine(self, engine) -> None:
        pass

    def set_backend_type(self, *args, **kwargs) -> None:
        pass

    def async_wait_reqid(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def register_local_tensor(self, tensor: torch.Tensor) -> bytes:
        return b"local"

    def get_agent_metadata(self) -> bytes:
        return b"decode"

    def register_remote_engine(self, packed: bytes) -> bytes:
        return packed

    def deregister_remote_engine(self, packed: bytes) -> None:
        self.deregistered.append(packed)

    def get_unpack_memory_metadata(self, packed: bytes) -> bytes:
        return packed

    def build_session(self, local: bytes, remote: bytes) -> bytes:
        return remote

    def read_remote_data(self, sizes, local_offsets, remote_offsets, session):
        self.reads.append(session)
        return _Done()

    def waiting_for_transfer_complete(self, statuses, fail_fast=True):
        pass


class _PrefillRank:
    """One launch of a prefill DP rank: its real handshake listener, which a
    test can stop so that the next launch takes over the same port."""

    def __init__(
        self, port: int, dp_rank: int, template: MoRIIOAgentMetadata, launch: str
    ):
        self.port = port + get_port_offset(dp_rank, 0, 1)
        ready = threading.Event()
        metadata = msgspec.structs.replace(
            template,
            agent_metadata=f"prefill/{launch}/dp{dp_rank}".encode(),
            incarnation=launch,
        )
        regions = {ln: [f"{launch}/dp{dp_rank}".encode()] for ln in LAYERS}
        self.thread = threading.Thread(
            target=self._serve,
            args=(metadata, ready, self.port, 0, dp_rank, regions),
            daemon=True,
        )
        self.thread.start()
        assert ready.wait(30), "the prefill handshake listener did not start"

    @staticmethod
    def _serve(*args) -> None:
        # The listener closes its socket and returns on a message it does not
        # expect, which is how stop() ends this launch.
        with contextlib.suppress(HandshakeError):
            MoRIIOConnectorWorker._moriio_handshake_listener(*args)

    def stop(self) -> None:
        if self.thread.is_alive():
            path = make_zmq_path("tcp", "127.0.0.1", self.port)
            with zmq_ctx(zmq.DEALER, path) as sock:
                sock.send(b"stop")
                # Closing the socket at once could drop the unsent message.
                self.thread.join(30)
        assert not self.thread.is_alive()


def _kv_cache_config() -> KVCacheConfig:
    page = SPEC.page_size_bytes
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=len(LAYERS) * NUM_BLOCKS * page,
                layers=LAYERS,
                layer_stride=NUM_BLOCKS * page,
                block_stride=page,
            )
        ],
        kv_cache_groups=[make_kv_cache_group(LAYERS, SPEC)],
    )


def _advertised_metadata(port: int) -> MoRIIOAgentMetadata:
    """What a handshake listener advertises; a prefill of the same model and
    layout advertises the same, under its own launch."""
    with zmq_ctx(zmq.DEALER, make_zmq_path("tcp", "127.0.0.1", port)) as sock:
        sock.send(MoRIIOConstants.GET_META_MSG)
        _, payload = sock.recv_multipart()
        sock.recv_multipart()
    return msgspec.msgpack.decode(payload, type=MoRIIOAgentMetadata)


@pytest.mark.skipif(
    importlib.util.find_spec("mori") is None, reason="MoRI is not installed"
)
def test_decode_reconnects_to_a_restarted_prefill_rank(dist_init, monkeypatch):
    """Prefill DP 2. The decode's first request names rank 0, and the eager
    handshake builds sessions to both ranks. Rank 1 restarts before a request
    names it, so its first request comes from the new launch and must be read
    from it. A request the old launch computed must then not be read at all."""
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector."
        "MoRIIOWrapper",
        _Transport,
    )
    prefill_port = _find_free_port()
    config = create_vllm_config(role="kv_consumer", read_mode=True)
    assert config.kv_transfer_config is not None
    config.kv_transfer_config.kv_connector_extra_config.update(
        http_port=_find_free_port(),
        handshake_port=_find_free_port(),
        notify_port=_find_free_port(),
    )
    scheduler = create_scheduler(config, num_blocks=NUM_BLOCKS)
    with set_current_vllm_config(config):
        connector = MoRIIOConnector(config, KVConnectorRole.WORKER, _kv_cache_config())
    shape = (NUM_BLOCKS, 2, SPEC.block_size, SPEC.num_kv_heads, SPEC.head_size)
    connector.register_kv_caches(
        {ln: torch.zeros(shape, dtype=SPEC.dtype) for ln in LAYERS}
    )
    worker = connector.connector_worker
    transport = worker.moriio_wrapper
    template = _advertised_metadata(worker.side_channel_port)

    def read(dp_rank: int, launch: str) -> list[int]:
        """Schedule one request whose KV `launch` of rank `dp_rank` computed,
        and run the decode worker's load; return its local blocks."""
        request = create_request(
            block_size=SPEC.block_size,
            num_tokens=2 * SPEC.block_size,
            do_remote_prefill=True,
        )
        assert request.kv_transfer_params is not None
        request.kv_transfer_params.update(
            transfer_id=f"t{request.request_id}",
            remote_block_ids=[[1, 2]],
            remote_host="127.0.0.1",
            remote_port=prefill_port,
            remote_handshake_port=prefill_port,
            remote_notify_port=_find_free_port(),
            remote_engine_id="prefill",
            tp_size=1,
            remote_dp_size=2,
            remote_dp_rank=dp_rank,
            remote_incarnation=launch,
        )
        scheduler.add_request(request)
        metadata = scheduler.schedule().kv_connector_metadata
        transport.reads.clear()
        worker.start_load_kv(metadata)
        return metadata.reqs_to_recv[request.request_id].local_block_ids[0]

    ranks = [_PrefillRank(prefill_port, dp, template, "100") for dp in (0, 1)]
    try:
        read(0, "100")
        assert set(transport.reads) == {b"100/dp0"}
        assert {
            key.rsplit("_", 1)[1]: meta.incarnation
            for key, meta in worker.remote_moriio_metadata.items()
        } == {"dp0": "100", "dp1": "100"}

        ranks[1].stop()
        ranks[1] = _PrefillRank(prefill_port, 1, template, "200")
        read(1, "200")
        assert set(transport.reads) == {b"200/dp1"}
        # Only the restarted rank is dropped; rank 0 keeps its session.
        assert transport.deregistered == [b"prefill/100/dp1"]

        stale_blocks = read(1, "100")
        assert transport.reads == []
        assert set(stale_blocks) <= connector.get_block_ids_with_load_errors()
        assert {
            key.rsplit("_", 1)[1]: meta.incarnation
            for key, meta in worker.remote_moriio_metadata.items()
        } == {"dp0": "100", "dp1": "200"}
        assert b"prefill/100/dp0" not in transport.deregistered
    finally:
        for rank in ranks:
            rank.stop()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the WRITE-mode consumer timeout + reclaim (ROCm/mori#655).

The MoRIIO WRITE-mode decode consumer passively waits for the producer's
write_done notification. If that notification is lost, the request would hang
in WAITING_FOR_REMOTE_KVS forever. The scheduler arms a per-request watchdog
when the decode request commits to waiting, reaps stale ones each
build_connector_meta tick, and ships the timed-out block ids to the worker so
vLLM frees/recomputes the request. These tests drive the scheduler's reap
logic on a bare scheduler (no GPU / RDMA / engine).
"""

import time

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
)


def _make_scheduler() -> MoRIIOConnectorScheduler:
    """A scheduler with only the watchdog state initialised (no __init__)."""
    s = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    s._write_recv_deadline = {}
    s._write_recv_block_ids = {}
    s._write_load_failed_block_ids = []
    s._write_recv_timeout = 540.0
    return s


def _arm(s, req_id, blocks, *, deadline):
    s._write_recv_deadline[req_id] = deadline
    s._write_recv_block_ids[req_id] = list(blocks)


def test_fresh_recv_not_reaped():
    """A recv whose deadline is in the future must not be timed out."""
    s = _make_scheduler()
    _arm(s, "req-fresh", [1, 2, 3], deadline=time.monotonic() + 100.0)
    s._reap_stale_write_recvs()
    assert s._write_load_failed_block_ids == []
    assert "req-fresh" in s._write_recv_deadline  # still being watched


def test_stale_recv_reaped_and_blocks_queued():
    """A recv past its deadline is failed and its blocks queued for reclaim."""
    s = _make_scheduler()
    _arm(s, "req-stale", [7, 8, 9], deadline=time.monotonic() - 1.0)

    s._reap_stale_write_recvs()

    assert set(s._write_load_failed_block_ids) == {7, 8, 9}
    # Watchdog state cleared for the reaped request.
    assert "req-stale" not in s._write_recv_deadline
    assert "req-stale" not in s._write_recv_block_ids


def test_mixed_fresh_and_stale():
    """Only the stale request is reaped; the fresh one keeps waiting."""
    s = _make_scheduler()
    now = time.monotonic()
    _arm(s, "req-old", [1, 2], deadline=now - 5.0)
    _arm(s, "req-new", [3, 4], deadline=now + 100.0)

    s._reap_stale_write_recvs()

    assert set(s._write_load_failed_block_ids) == {1, 2}
    assert "req-new" in s._write_recv_deadline
    assert "req-old" not in s._write_recv_deadline


def test_clear_watchdog_on_normal_completion():
    """A request that completes normally drops its watchdog state."""
    s = _make_scheduler()
    _arm(s, "req-done", [5, 6], deadline=time.monotonic() + 100.0)

    s.clear_write_recv_watchdog("req-done")

    assert "req-done" not in s._write_recv_deadline
    assert "req-done" not in s._write_recv_block_ids
    # A later reap finds nothing to fail.
    s._reap_stale_write_recvs()
    assert s._write_load_failed_block_ids == []

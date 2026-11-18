# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launches of the peers a MoRIIO READ decode reads from.

The scheduler learns from the requests which launch of each prefill rank
computed their KV, and every worker reconciles its sessions with that from the
same connector metadata. Workers must not decide alone: they handshake in
lockstep, behind a TP collective, so a handshake re-armed on one worker only
would hang the others.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import ReqMeta
from vllm.logger import init_logger

logger = init_logger(__name__)

PeerId = str


def peer_id_of(meta: ReqMeta) -> PeerId:
    """The peer a request reads from: its prefill's handshake address. Session
    keys extend it with the remote DP (and TP) rank."""
    return f"{meta.remote_host}:{meta.remote_handshake_port}"


class PeerTracker:
    """The current launch of each rank of each peer, learned from the requests.

    A launch other than a rank's current one is a restart, unless the rank
    already ran it: requests from the earlier launch can still arrive after
    the restart. Launches are compared for identity only, never ordered: they
    are timestamps from the peer's own clock, and not always plain numbers. So
    a request from an earlier launch the tracker never saw counts as a restart
    once, which costs the workers one needless reconnect.

    A rank's first launch is reported too, not only a restart: the eager
    handshake builds sessions to every rank of a peer on first contact, so a
    session to a rank no request named yet can reach a launch that has since
    restarted.
    """

    def __init__(self) -> None:
        self._current: dict[tuple[PeerId, int], str] = {}
        self._retired: dict[tuple[PeerId, int], set[str]] = {}
        self._changed: dict[PeerId, dict[int, str]] = {}

    def note_request(
        self, peer_id: PeerId, dp_rank: int = 0, incarnation: str = ""
    ) -> None:
        """A request reads from rank `dp_rank` of `peer_id`; `incarnation` is
        the launch it names for that rank ("" from peers that send none)."""
        rank = (peer_id, dp_rank)
        current = self._current.get(rank, "")
        if (
            not incarnation
            or incarnation == current
            or incarnation in self._retired.get(rank, ())
        ):
            return
        if current:
            self._retired.setdefault(rank, set()).add(current)
            logger.warning(
                "MoRIIO: a request names launch %s of %s rank %d, which was "
                "launch %s; reconnecting",
                incarnation,
                peer_id,
                dp_rank,
                current,
            )
        self._current[rank] = incarnation
        self._changed.setdefault(peer_id, {})[dp_rank] = incarnation

    def take_changes(self) -> dict[PeerId, dict[int, str]]:
        """The ranks whose launch changed since the last call:
        peer -> {dp rank: incarnation}."""
        changed, self._changed = self._changed, {}
        return changed

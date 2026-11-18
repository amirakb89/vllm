# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle of the peers a MoRIIO instance opens sessions to.

The scheduler decides when a peer's sessions are dropped, and every worker
applies the decision from the same connector metadata. Workers must not decide
alone: in READ mode the decode workers handshake in lockstep, behind a TP
collective, so a peer dropped on one worker only would hang the others.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

PeerId = str


def is_newer_incarnation(incarnation: str, than: str) -> bool:
    """Whether `incarnation` is a later launch than `than`.

    Incarnations are launch timestamps (`VllmConfig.instance_id`), so they
    order numerically; any other pair of distinct values counts as newer.
    """
    if not incarnation or incarnation == than:
        return False
    if incarnation.isdigit() and than.isdigit():
        return int(incarnation) > int(than)
    return True


class PeerTracker:
    """Decides when to drop the sessions to a peer: when a request names a
    newer incarnation of one of its ranks (it restarted)."""

    def __init__(self) -> None:
        # Latest incarnation seen per peer and remote DP rank: each rank can
        # be a launch of its own.
        self._incarnations: dict[PeerId, dict[int, str]] = {}
        # Peer -> (reason, incarnation whose sessions to keep).
        self._drops: dict[PeerId, tuple[str, str]] = {}

    def note_request(
        self, peer_id: PeerId, dp_rank: int = 0, incarnation: str = ""
    ) -> None:
        """A request reads from rank `dp_rank` of `peer_id`; `incarnation` is
        the launch it names for that rank ("" if unknown)."""
        ranks = self._incarnations.setdefault(peer_id, {})
        known = ranks.get(dp_rank, "")
        if not is_newer_incarnation(incarnation, known):
            return
        if known:
            reason = f"rank {dp_rank} restarted ({known} -> {incarnation})"
            logger.warning("MoRIIO: dropping the sessions to %s: %s", peer_id, reason)
            self._drops[peer_id] = (reason, incarnation)
        ranks[dp_rank] = incarnation

    def take_drops(self) -> dict[PeerId, tuple[str, str]]:
        """Peers to drop this step: the reason, and the incarnation whose
        sessions to keep."""
        drops, self._drops = self._drops, {}
        return drops

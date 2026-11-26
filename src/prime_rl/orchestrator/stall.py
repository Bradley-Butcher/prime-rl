"""An inference stall: the inference server failed a model call because its own upstream stopped answering.

The Tinker bridge does this (its ``GET /stall`` says while): Tinker can hold a run's sampling for many minutes, and a
rollout left waiting runs out its deadline and is scored as a failure, so the policy would be trained against for the
service's pause. The bridge fails the call at once instead, with ``STALL_CODE`` in its error. A rollout that met one says
nothing of the policy, and nor do its group-mates: what finished is the group's fastest rollouts, not a sample of it,
the reason a down-sampled group an overload cut is voided too. So the dispatcher voids the group (``stall``, which
trains nothing and reaches no curriculum, as a shutdown drop) and holds dispatch until the server answers again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import verifiers.v1 as vf

STALL_CODE = "inference_stalled"
"""The code the inference server's error carries, in the message every layer passes on."""

STALL_POLL_S = 10.0
"""Seconds between asking a stalled server whether it answers again."""

STALL_HOLD_S = 60.0
"""How long dispatch holds for a stall reported by a server with no ``GET /stall`` to ask."""


def is_stall(message: str | None) -> bool:
    return message is not None and STALL_CODE in message


def episode_stalled(episode: vf.WireEpisode) -> bool:
    """Whether any model call of the episode, or any error it ended with, was a stall."""
    if any(is_stall(error.message) for error in episode.errors):
        return True
    for trace in episode.traces:
        if any(is_stall(error.message) for error in trace.errors):
            return True
        if any(call.error is not None and is_stall(call.error.message) for call in trace.calls):
            return True
    return False


def exception_stalled(error: BaseException) -> bool:
    return is_stall(str(error))

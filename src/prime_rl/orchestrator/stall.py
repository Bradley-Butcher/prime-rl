"""An inference stall: the inference server failed a model call because its own upstream stopped answering.

The Tinker bridge does this (its ``GET /stall`` says while): Tinker can hold a run's sampling for many minutes, and a
rollout left waiting runs out its deadline and is scored as a failure, so the policy would be trained against for the
service's pause. The bridge fails the call at once instead, with ``STALL_CODE`` in its error. A rollout that met one says
nothing of the policy, and nor do its group-mates: what finished is the group's fastest rollouts, not a sample of it,
the reason a down-sampled group an overload cut is voided too. So the dispatcher voids the group (``stall``, which
trains nothing and reaches no curriculum, as a shutdown drop) and holds dispatch until the server answers again.

The server calls it a stall only after its upstream has been quiet for a while, and a rollout whose own deadline runs
out in that while would be scored a failure for the pause. So a rollout its agent's deadline stopped counts as having
met the stall too when the server says it is stalled, or that its upstream has answered nothing for ``STALL_SUSPECT_S``
while a call waited: with calls under way, a quiet that long is a stall building, whatever the server's threshold.

An eval rollout that met a stall is run again once the server answers, rather than counted as an error: its eval is a
measurement of every one of its Tasks, and the ones a stall catches are the slowest, not a sample of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.base import AGENT_TIMEOUT

if TYPE_CHECKING:
    import verifiers.v1 as vf

STALL_CODE = "inference_stalled"
"""The code the inference server's error carries, in the message every layer passes on."""

STALL_POLL_S = 10.0
"""Seconds between asking a stalled server whether it answers again."""

STALL_HOLD_S = 60.0
"""How long dispatch holds for a stall reported by a server with no ``GET /stall`` to ask."""

STALL_SUSPECT_S = 60.0
"""Seconds of the server's upstream answering nothing, while a call waits on it, past which a rollout its deadline
stopped met the stall building: no reply of 116,000 on the Tinker free tier took more than 160 s, and 99.99% under 40,
so with calls under way the upstream is never that quiet otherwise."""

MAX_EVAL_STALL_RERUNS = 3
"""Times each of an eval group's rollouts may be run again for a stall before the next one is kept as an error."""


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


def timed_out(episode: vf.WireEpisode) -> bool:
    """Whether its agent's own deadline stopped a trace of the episode."""
    return any(trace.is_timeout and trace.stop_condition == AGENT_TIMEOUT for trace in episode.traces)


def stall_building(state: dict | None) -> bool:
    """Whether the server's ``GET /stall`` says it is stalled, or that its upstream has been quiet past
    ``STALL_SUSPECT_S`` while a call waited."""
    if state is None:
        return False
    return bool(state.get("stalled")) or (state.get("quiet_s") or 0.0) >= STALL_SUSPECT_S

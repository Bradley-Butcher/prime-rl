"""Episode-native algorithm hooks."""

from __future__ import annotations

import verifiers.v1 as vf

from prime_rl.configs.algorithm import AlgoConfig, FrozenModelConfig
from prime_rl.orchestrator.clients import InferenceClient, connect_frozen_client
from prime_rl.utils.logger import get_logger

AGENT_TIMEOUT = "agent_timeout"
"""The stop condition verifiers leaves on a trace its agent's own deadline (``timeout.rollout``) stopped."""


def iter_trainable_traces(episodes: list[vf.Episode], drop_truncated: bool = False):
    """Yield clean trainable traces that contain sampled tokens; with ``drop_truncated``, none that a turn or token
    limit cut off (``Trace.is_truncated``)."""
    for episode in episodes:
        for trace in episode.traces:
            if not episode.ok or trace.has_error or not trace.agent.trainable or trace.num_turns == 0:
                continue
            if drop_truncated and trace.is_truncated:
                continue
            if any(any(node.mask) for node in trace.nodes):
                yield episode, trace


def leave_out_unfinished_reply(trace: vf.Trace) -> None:
    """Keep the trace's last committed reply out of training when the engine never finished it (its call has no
    finish reason): its tokens stay in the sequence as context, none of them sampled. verifiers commits no reply that
    an agent's deadline interrupts (the call in flight is refused or cancelled), so this guards a reply committed
    without a finish, such as an aborted generation."""
    call = next((call for call in reversed(trace.calls) if call.node is not None), None)
    if call is None or call.finish_reason is not None:
        return
    node = trace.nodes[call.node]
    node.mask = [False] * len(node.mask)
    node.logprobs = []
    node.sampling_mask = None


def score_timeout_as_failure(trace: vf.Trace) -> None:
    """Score a timed-out trace as a failed attempt: every reward 0, and ``ok``. An error recorded after its deadline
    stays on its record, and is logged."""
    if trace.has_error and trace.last_error is not None:
        get_logger().warning(
            f"Timed-out trace {trace.id} scored 0 though {trace.last_error.type} after its deadline: "
            f"{trace.last_error.message}"
        )
    trace.rewards = {
        name: None if reward is None else reward.model_copy(update={"score": 0.0})
        for name, reward in trace.rewards.items()
    }
    trace.ok = True


class Algorithm:
    """Assign training annotations directly to verifier episodes.

    Override :meth:`score_episode` for rollout-local work and
    :meth:`score_group` for cohort-relative work. The train sink compiles the
    annotated traces into transport samples only after admission.
    """

    def __init__(self, config: AlgoConfig, clients: InferenceClient):
        self.action_loss_type = config.action_loss_type
        self.drop_truncated = config.drop_truncated
        self.timeouts_as_failures = config.timeouts_as_failures
        self.clients = clients
        self.connected: InferenceClient | None = None

    async def setup(self) -> None:
        """Connect resources owned by the algorithm."""

    async def connect(self, reference: FrozenModelConfig) -> InferenceClient:
        """Connect and track the frozen model pool owned by this algorithm."""
        self.connected = await connect_frozen_client(reference)
        return self.connected

    def settle_timeouts(self, episode: vf.Episode) -> None:
        """Settle what each timed-out trace of a train episode is to training, before the sink sees it.

        With ``timeouts_as_failures``, a trace its agent's deadline stopped is a failed attempt scored 0
        (:func:`score_timeout_as_failure`) once the reply the deadline cut off is left out
        (:func:`leave_out_unfinished_reply`), if it still holds a reply to train. Its outcome was settled at the
        deadline, so an error recorded after it, in the task's finalize or scoring, does not fail it. Every other
        timed-out trace is an error, which training leaves out."""
        rescued = False
        for trace in episode.traces:
            if not trace.is_timeout:
                continue
            if self.timeouts_as_failures and trace.stop_condition == AGENT_TIMEOUT:
                leave_out_unfinished_reply(trace)
                if any(any(node.mask) for node in trace.nodes):
                    rescued |= trace.has_error
                    score_timeout_as_failure(trace)
                    continue
            if not trace.has_error:
                trace.errors.append(vf.Error(type="Timeout", message=f"Trace stopped by {trace.stop_condition}"))
                trace.ok = False
                episode.ok = False
        # An error after a deadline failed the episode too. With no error of the env's own, the traces decide.
        if rescued and not episode.errors:
            episode.ok = all(trace.ok for trace in episode.traces)

    async def score_episode(self, episode: vf.Episode) -> None:
        """Assign rollout-local annotations to one finalized episode."""

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        """Assign group-relative annotations to a finalized cohort."""

    async def finalize_episode(self, episode: vf.Episode) -> None:
        """Run rollout-local scoring when the episode has trainable traces."""
        if any(True for _ in iter_trainable_traces([episode])):
            await self.score_episode(episode)

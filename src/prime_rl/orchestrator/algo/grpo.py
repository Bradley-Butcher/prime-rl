from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig, RewardShapingConfig
from prime_rl.orchestrator.algo.base import Algorithm, is_unfinished, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_nodes
from prime_rl.orchestrator.utils import train_work
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def shape_rewards(
    traces: Sequence[vf.Trace], shaping: RewardShapingConfig, *, pass_penalty: bool = True
) -> list[tuple[float, tuple[str, ...]]]:
    """The reward GRPO's baseline and advantages take for each of a group's ``traces`` under ``shaping``, and the
    levers that changed it, of ``"unfinished"``, ``"stuck"``, ``"pass_turns"`` and ``"pass_cost"``.

    An unfinished rollout takes ``unfinished.reward`` when it is set. Every other rollout with a positive reward is a
    pass: its reward is cut for its stuck calls (a trace without the stuck metric is not cut) and then, unless
    ``pass_penalty`` is False, for its turns, or its turns and tokens, relative to the group's other passes. A failure
    keeps its reward."""
    rewards: list[float] = []
    levers: list[tuple[str, ...]] = []
    passes: list[int] = []
    stuck = shaping.stuck
    for index, trace in enumerate(traces):
        if shaping.unfinished.reward is not None and is_unfinished(trace):
            rewards.append(shaping.unfinished.reward)
            levers.append(("unfinished",))
            continue
        reward, changed = trace.reward, ()
        if reward > 0:
            passes.append(index)
            count = trace.metrics.get(stuck.metric)
            if stuck.per_call > 0 and count is not None and (factor := stuck.factor(count)) < 1:
                reward, changed = reward * factor, ("stuck",)
        rewards.append(reward)
        levers.append(changed)
    if pass_penalty and len(passes) >= 2:
        if shaping.pass_turns.weight > 0:
            turns = [traces[index].num_turns for index in passes]
            factors = [shaping.pass_turns.factor(count, min(turns), max(turns)) for count in turns]
            _cut(rewards, levers, passes, factors, "pass_turns")
        if shaping.pass_cost.weight > 0:
            costs = [
                shaping.pass_cost.cost(traces[index].num_turns, generated_tokens(traces[index])) for index in passes
            ]
            _cut(rewards, levers, passes, shaping.pass_cost.factors(costs, len(traces)), "pass_cost")
    return list(zip(rewards, levers, strict=True))


def generated_tokens(trace: vf.Trace) -> int:
    """The tokens the policy generated over the rollout, each model call counted once. ``Trace.num_output_tokens`` sums
    over the trace's branches, so it counts the calls that branches share once for each of them."""
    usage = trace.usage
    return usage.completion_tokens if usage is not None else 0


def _cut(
    rewards: list[float], levers: list[tuple[str, ...]], passes: list[int], factors: list[float], lever: str
) -> None:
    """Multiply each of ``passes``' rewards by its factor, and name ``lever`` where that changed it."""
    for index, factor in zip(passes, factors, strict=True):
        if factor < 1:
            rewards[index] *= factor
            levers[index] += (lever,)


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.shaping = config.shaping
        self.shaping_at: Callable[[int], RewardShapingConfig] | None = None
        """The shaping of a group by the step it was dispatched at, which the orchestrator's live shaping sets;
        None shapes every group by the config's ``shaping``."""
        self.warned_missing_metric = False
        self.length_penalty = config.length_penalty
        self.length_weighted_baseline = config.length_weighted_baseline
        self.loss_aggregation = config.loss_aggregation

    def shaping_for(self, episodes: list[vf.Episode]) -> RewardShapingConfig:
        if self.shaping_at is None:
            return self.shaping
        return self.shaping_at(train_work(episodes[0]).step)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        shaping = self.shaping_for(episodes)
        if shaping.stuck.per_call > 0 and not self.warned_missing_metric:
            if any(trace.metrics.get(shaping.stuck.metric) is None for trace in traces):
                self.warned_missing_metric = True
                get_logger().warning(
                    f"The stuck penalty is on, but a rollout has no {shaping.stuck.metric!r} metric; such rollouts "
                    "keep their reward"
                )
        for trace, advantage in zip(traces, self.group_advantages(traces, shaping), strict=True):
            assign_advantages(trace, advantage)
        nodes = [node for trace in traces for node in trainable_nodes(trace)]
        if self.loss_aggregation == "prompt" and nodes:
            # rl weight 1/T_q per token: each group's weights sum to 1, and the trainer divides the
            # rl loss by the sum of rl weights, i.e. the number of groups.
            weight = 1.0 / sum(sum(node.mask) for node in nodes)
            for node in nodes:
                node.loss_weights = {**(node.loss_weights or {}), "rl": [weight if m else 0.0 for m in node.mask]}

    def group_advantages(
        self, traces: list[vf.Trace], shaping: RewardShapingConfig, *, pass_penalty: bool = True
    ) -> list[float]:
        """Each of a group's ``traces``' advantage: its shaped reward, less any length penalty, minus the baseline.
        ``pass_penalty=False`` leaves the pass penalty out."""
        import torch  # only the trainer-side extras ship torch; an eval process never scores a group

        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        shaped = shape_rewards(traces, shaping, pass_penalty=pass_penalty)
        shaped_rewards = torch.tensor([reward for reward, _ in shaped], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is not None:
            output = torch.tensor([trace.num_output_tokens for trace in traces], dtype=rewards.dtype)
            total = torch.tensor([trace.num_total_tokens for trace in traces], dtype=rewards.dtype)
            turns = torch.tensor([trace.num_turns for trace in traces], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = shaped_rewards - penalty
        baseline = shaped_rewards.mean()
        if self.length_weighted_baseline:
            lengths = torch.tensor(
                [sum(sum(node.mask) for node in trainable_nodes(trace)) for trace in traces], dtype=rewards.dtype
            )
            baseline = (lengths * shaped_rewards).sum() / lengths.sum()
        return (shaped_rewards - baseline).tolist()

    @contextmanager
    def admission_view(self, episodes: list[vf.Episode]) -> Iterator[None]:
        """With the pass penalty's ``gate_on_raw`` (``shaping.pass_turns`` or ``shaping.pass_cost``), the gates judge a
        scored group by its advantages without the penalty, so that the penalty never decides whether a group trains;
        its own advantages are back after."""
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        shaping = self.shaping_for(episodes) if traces else None
        penalty = shaping.pass_penalty if shaping is not None else None
        if penalty is None or not penalty.gate_on_raw:
            yield
            return
        scored = [[node.advantages for node in trace.nodes] for trace in traces]
        for trace, advantage in zip(traces, self.group_advantages(traces, shaping, pass_penalty=False), strict=True):
            assign_advantages(trace, advantage)
        try:
            yield
        finally:
            for trace, advantages in zip(traces, scored, strict=True):
                for node, values in zip(trace.nodes, advantages, strict=True):
                    node.advantages = values

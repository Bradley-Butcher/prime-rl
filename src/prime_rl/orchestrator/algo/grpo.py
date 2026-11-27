from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.trace import min_new_input_tokens

from prime_rl.configs.algorithm import GRPOAlgoConfig, LinearSubagentLengthPenaltyConfig, RewardShapingConfig
from prime_rl.orchestrator.algo.base import Algorithm, is_unfinished, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_nodes
from prime_rl.orchestrator.utils import train_work
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def _length_penalty_tokens(trace: vf.Trace, subagent_token_weight: float) -> tuple[float, float]:
    """Weighted output and new input tokens, counted once on trainable branches."""
    subagents: set[int] = set()
    successors: dict[int, list[int]] = {}
    for node_id, node in enumerate(trace.nodes):
        for link in node.semantic_parents:
            if link.type == "subagent_call":
                subagents.add(node_id)
            # Return, cancel, and message edges cross agent boundaries.
            if link.type in {"subagent_call", "continuation", "compaction_attempt", "compaction"}:
                successors.setdefault(link.node, []).append(node_id)
    pending = list(subagents)
    while pending:
        for child in successors.get(pending.pop(), []):
            if child not in subagents:
                subagents.add(child)
                pending.append(child)

    output_tokens = input_tokens = 0.0
    seen: set[int] = set()
    for branch in trace.branches:
        if not branch.trainable:
            continue
        # Compute increments before weighting: a discounted predecessor still
        # contributes its full history to the next call's prompt.
        for call, increment in min_new_input_tokens(branch.calls):
            if id(call) in seen:
                continue
            seen.add(id(call))
            weight = subagent_token_weight if call.node in subagents else 1.0
            output_tokens += weight * call.usage.completion_tokens
            input_tokens += weight * increment
    return output_tokens, input_tokens


def shape_rewards(
    traces: Sequence[vf.Trace], shaping: RewardShapingConfig, *, pass_turns: bool = True
) -> list[tuple[float, tuple[str, ...]]]:
    """The reward GRPO's baseline and advantages take for each of a group's ``traces`` under ``shaping``, and the
    levers that changed it, of ``"unfinished"``, ``"stuck"`` and ``"pass_turns"``.

    An unfinished rollout takes ``unfinished.reward`` when it is set, and, with ``unfinished.only_with_pass``, the
    group has a finished pass. Every other rollout with a positive reward is a pass: its reward is cut for its stuck
    calls (a trace without the stuck metric is not cut) and then, unless ``pass_turns`` is False, for its turns relative
    to the group's other passes. A failure keeps its reward."""
    rewards: list[float] = []
    levers: list[tuple[str, ...]] = []
    passes: list[int] = []
    stuck = shaping.stuck
    unfinished = shaping.unfinished.reward is not None
    if unfinished and shaping.unfinished.only_with_pass:
        unfinished = any(trace.reward > 0 and not is_unfinished(trace) for trace in traces)
    for index, trace in enumerate(traces):
        if unfinished and is_unfinished(trace):
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
    if pass_turns and shaping.pass_turns.weight > 0 and len(passes) >= 2:
        turns = [traces[index].num_turns for index in passes]
        fewest, most = min(turns), max(turns)
        for index, count in zip(passes, turns, strict=True):
            factor = shaping.pass_turns.factor(count, fewest, most)
            if factor < 1:
                rewards[index] *= factor
                levers[index] += ("pass_turns",)
    return list(zip(rewards, levers, strict=True))


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

    def reconfigure(self, config: GRPOAlgoConfig) -> None:
        self.length_penalty = config.length_penalty

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
            weight = 1.0 / sum(int(np.count_nonzero(node.mask)) for node in nodes)
            for node in nodes:
                rl = np.where(np.asarray(node.mask, dtype=bool), weight, 0.0).astype(np.float32)
                node.loss_weights = {**(node.loss_weights or {}), "rl": rl}

    def group_advantages(
        self, traces: list[vf.Trace], shaping: RewardShapingConfig, *, pass_turns: bool = True
    ) -> list[float]:
        """Each of a group's ``traces``' advantage: its shaped reward, less any length penalty, minus the baseline.
        ``pass_turns=False`` leaves the turn penalty out."""
        import torch  # only the trainer-side extras ship torch; an eval process never scores a group

        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        shaped = shape_rewards(traces, shaping, pass_turns=pass_turns)
        shaped_rewards = torch.tensor([reward for reward, _ in shaped], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is not None:
            subagent_weight = (
                length_penalty.subagent_token_weight
                if isinstance(length_penalty, LinearSubagentLengthPenaltyConfig)
                else 1.0
            )
            token_counts = [_length_penalty_tokens(trace, subagent_weight) for trace in traces]
            output = torch.tensor([output for output, _ in token_counts], dtype=rewards.dtype)
            input = torch.tensor([input for _, input in token_counts], dtype=rewards.dtype)
            turns = torch.tensor(
                [
                    len(
                        {
                            id(node)
                            for branch in trace.branches
                            if branch.trainable
                            for node in branch.nodes
                            if node.sampled
                        }
                    )
                    for trace in traces
                ],
                dtype=rewards.dtype,
            )
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
                [sum(int(np.count_nonzero(node.mask)) for node in trainable_nodes(trace)) for trace in traces],
                dtype=rewards.dtype,
            )
            baseline = (lengths * shaped_rewards).sum() / lengths.sum()
        return (shaped_rewards - baseline).tolist()

    @contextmanager
    def admission_view(self, episodes: list[vf.Episode]) -> Iterator[None]:
        """With ``shaping.pass_turns.gate_on_raw``, the gates judge a scored group by its advantages without the turn
        penalty, so that the penalty never decides whether a group trains; its own advantages are back after."""
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        shaping = self.shaping_for(episodes) if traces else None
        if shaping is None or shaping.pass_turns.weight == 0 or not shaping.pass_turns.gate_on_raw:
            yield
            return
        scored = [[node.advantages for node in trace.nodes] for trace in traces]
        for trace, advantage in zip(traces, self.group_advantages(traces, shaping, pass_turns=False), strict=True):
            assign_advantages(trace, advantage)
        try:
            yield
        finally:
            for trace, advantages in zip(traces, scored, strict=True):
                for node, values in zip(trace.nodes, advantages, strict=True):
                    node.advantages = values

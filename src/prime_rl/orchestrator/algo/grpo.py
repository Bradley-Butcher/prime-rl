from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig, RewardShapingConfig
from prime_rl.orchestrator.algo.base import Algorithm, is_unfinished, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_nodes
from prime_rl.orchestrator.utils import train_work
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def shape_reward(trace: vf.Trace, shaping: RewardShapingConfig) -> tuple[float, str | None]:
    """The reward GRPO's baseline and advantage take for ``trace`` under ``shaping``, and the lever that changed it
    (``"unfinished"`` or ``"stuck"``), if one did. A trace without the stuck metric keeps its reward."""
    if shaping.unfinished.reward is not None and is_unfinished(trace):
        return shaping.unfinished.reward, "unfinished"
    reward, stuck = trace.reward, shaping.stuck
    count = trace.metrics.get(stuck.metric)
    if reward <= 0 or stuck.per_call == 0 or count is None:
        return reward, None
    factor = stuck.factor(count)
    return (reward * factor, "stuck") if factor < 1 else (reward, None)


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
        import torch  # only the trainer-side extras ship torch; an eval process never scores a group

        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        shaping = self.shaping_for(episodes)
        if shaping.stuck.per_call > 0 and not self.warned_missing_metric:
            if any(trace.metrics.get(shaping.stuck.metric) is None for trace in traces):
                self.warned_missing_metric = True
                get_logger().warning(
                    f"The stuck penalty is on, but a rollout has no {shaping.stuck.metric!r} metric; such rollouts "
                    "keep their reward"
                )
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        shaped_rewards = torch.tensor([shape_reward(trace, shaping)[0] for trace in traces], dtype=torch.float32)
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
        advantages = shaped_rewards - baseline
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
        nodes = [node for trace in traces for node in trainable_nodes(trace)]
        if self.loss_aggregation == "prompt" and nodes:
            # rl weight 1/T_q per token: each group's weights sum to 1, and the trainer divides the
            # rl loss by the sum of rl weights, i.e. the number of groups.
            weight = 1.0 / sum(sum(node.mask) for node in nodes)
            for node in nodes:
                node.loss_weights = {**(node.loss_weights or {}), "rl": [weight if m else 0.0 for m in node.mask]}

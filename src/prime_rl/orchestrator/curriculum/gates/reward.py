"""Mean-reward band-pass training-sample admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.gates.base import AdmissionGate

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import RewardRangeGateConfig


class RewardRangeGate(AdmissionGate):
    """Admit groups whose mean raw reward over trainable ok traces falls inside ``[min_mean_reward, max_mean_reward]``.

    Used as a band-pass filter (e.g. ``[0.12, 0.88]`` on 16 rollouts, requiring at least 2 passes and 2 failures) to
    drop extreme ``1/N`` and ``(N-1)/N`` groups. Groups with no trainable ok traces are admitted.
    """

    def __init__(self, config: RewardRangeGateConfig) -> None:
        self.config = config

    def admit(self, group: list[vf.Episode]) -> bool:
        rewards = [
            float(trace.reward)
            for episode in group
            for trace in episode.traces
            if episode.ok and not trace.has_error and trace.agent.trainable and trace.reward is not None
        ]
        if not rewards:
            return True
        mean_reward = sum(rewards) / len(rewards)
        return self.config.min_mean_reward <= mean_reward <= self.config.max_mean_reward

"""PODS rollout down-sampling (arXiv:2504.13818, section 3): a Task's group dispatches n rollouts and trains on the m
of them whose rewards vary the most.

The train sink waits for all n, lets the admission gates, the difficulty sampler and the curriculum records see every
one, then keeps m with ``max_variance_subset`` and computes the advantages over them (``baseline = "kept"``) or over
all n (``baseline = "all"``). The left-out rollouts are not trained. Every group's rollouts are recorded in
``curriculum/downsample.jsonl`` (``downsample_record``).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.orchestrator.selection import reward_variance
from prime_rl.orchestrator.utils import episode_env_name, train_work

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import DownsampleConfig
    from prime_rl.orchestrator.types import DispatchFailure


def max_variance_subset(rewards: Sequence[float], m: int, rng: random.Random) -> list[int]:
    """Indices of the ``m`` of ``rewards`` whose variance is largest, ascending.

    Pass/fail rewards with ``k`` passes keep ``s = min(k, max(m // 2, m - (n - k)))`` passes and ``m - s`` failures.
    Other rewards keep the top ``i`` and the bottom ``m - i`` for the ``i`` that maximises the subset's variance, ties
    between values of ``i`` at random. Rollouts with equal rewards are chosen uniformly at random."""
    n = len(rewards)
    if m >= n:
        return list(range(n))
    order = list(range(n))
    rng.shuffle(order)
    if set(rewards) <= {0.0, 1.0}:
        passed = [index for index in order if rewards[index] == 1.0]
        failed = [index for index in order if rewards[index] == 0.0]
        kept_passes = min(len(passed), max(m // 2, m - len(failed)))
        return sorted(passed[:kept_passes] + failed[: m - kept_passes])
    order.sort(key=lambda index: rewards[index])  # stable, so equal rewards keep their random order
    subsets = [order[n - top :] + order[: m - top] for top in range(m + 1)]
    variances = [reward_variance([rewards[index] for index in subset]) for subset in subsets]
    best = max(variances)
    tied = [subset for subset, variance in zip(subsets, variances) if math.isclose(variance, best, abs_tol=1e-12)]
    return sorted(rng.choice(tied))


def downsample_record(
    group_id: str,
    group: list[vf.Episode],
    failures: list[DispatchFailure],
    kept: list[vf.Trace],
    admitted: bool,
    config: DownsampleConfig,
) -> dict[str, Any]:
    """A down-sampled group's line in ``curriculum/downsample.jsonl``: every rollout in the order it finished, with
    its reward and turns, whether it was kept, and the advantage it trains with (None for a left-out rollout and for
    every rollout of a group the gates rejected). ``failed`` counts the requests that returned no rollout."""
    kept_ids = {trace.id for trace in kept}
    rollouts = [
        {
            "trace": trace.id,
            "finish": finish,
            "reward": trace.reward,
            "turns": trace.num_turns,
            "kept": trace.id in kept_ids,
            "advantage": trained_advantage(trace) if admitted and trace.id in kept_ids else None,
        }
        for finish, episode in enumerate(group, start=1)
        for trace in episode.traces
    ]
    first = group[0]
    return {
        "step": train_work(first).step,
        "env": episode_env_name(first),
        "key": first.task.key,
        "group": group_id,
        "rule": config.rule,
        "baseline": config.baseline,
        "admitted": admitted,
        "failed": len(failures),
        "rollouts": rollouts,
    }


def trained_advantage(trace: vf.Trace) -> float | None:
    """A trace's advantage on its first trainable token: the whole trace's for a group-relative algorithm."""
    return next((advantage for node in trace.nodes for advantage in node.advantages or []), None)

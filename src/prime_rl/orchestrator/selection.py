"""Batch selection: each step trains the most informative admitted groups of an oversampled pool.

With ``batch_selection`` on, the train sink pools every admitted group as a ``Candidate`` while its traces wait in
the queue. The pool ranks its groups by score, highest first, ties going to the oldest: the lowest dispatch policy
version (frozen-sourced groups, which never go stale, after live ones), then the first admitted. A batch takes the
ranked groups' traces up to the batch size, splitting the last group as the sink always has, and leaves the rest
pooled for the next batch. A group that takes the pool past its cap displaces the lowest-ranked groups, possibly
itself; the sink's staleness sweep drops pooled groups past ``max_off_policy_steps`` as it does any queued trace.

Every group's exit from the pool is a line in ``curriculum/selection.jsonl``: ``trained`` at a step, or dropped as
``stale``, ``displaced`` or ``discarded`` (the run stopped training), with how many batches passed it over.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Fate = Literal["trained", "stale", "displaced", "discarded"]


def reward_variance(rewards: Sequence[float]) -> float:
    """The rewards' population variance: ``p(1-p)`` for pass/fail rewards with pass rate ``p``."""
    if not rewards:
        return 0.0
    mean = sum(rewards) / len(rewards)
    return sum((reward - mean) ** 2 for reward in rewards) / len(rewards)


SCORES: dict[str, Callable[[Sequence[float]], float]] = {"variance": reward_variance}


@dataclass
class Candidate:
    """An admitted group in the pool and its traces still queued."""

    group_id: str
    env_name: str
    task_key: str
    step: int
    """The step it was dispatched at."""
    version: int | None
    """The policy version it was dispatched at; None for a frozen-sourced group."""
    score: float
    trace_ids: list[str]
    order: int = 0
    """Its place in admission order, which the pool assigns."""
    passes: int = 0
    """Batches that passed it over."""

    def rank(self) -> tuple[float, float, int]:
        return (-self.score, math.inf if self.version is None else self.version, self.order)

    def record(self, *, step: int, fate: Fate, rollouts: int) -> dict[str, Any]:
        """Its ``selection.jsonl`` line as ``rollouts`` of its traces leave the pool at ``step``."""
        return {
            "step": step,
            "env": self.env_name,
            "key": self.task_key,
            "group": self.group_id,
            "dispatch_step": self.step,
            "score": round(self.score, 4),
            "passed": self.passes,
            "rollouts": rollouts,
            "fate": fate,
        }


Exit = tuple[Candidate, list[str]]
"""A group that left the pool, with the traces that left with it."""


@dataclass
class Selection:
    """One batch's pick from the pool, and the pool's turnover since the previous batch."""

    trace_ids: list[str]
    selected: list[tuple[Candidate, int]]
    """The groups the batch takes traces from, with how many it takes."""
    passed: list[Candidate]
    carried: int
    """Groups left in the pool."""
    admitted: int
    """Groups admitted since the previous batch."""
    dropped: Counter[str] = field(default_factory=Counter)
    """Groups dropped since the previous batch, by fate."""

    def metrics(self) -> dict[str, float]:
        selected = [candidate.score for candidate, _ in self.selected]
        metrics = {
            "selection/pool": float(len(self.selected) + len(self.passed)),
            "selection/candidates": float(self.admitted),
            "selection/selected": float(len(self.selected)),
            "selection/carried": float(self.carried),
            "selection/stale": float(self.dropped["stale"]),
            "selection/displaced": float(self.dropped["displaced"]),
            "selection/score/selected": sum(selected) / len(selected) if selected else 0.0,
        }
        if self.passed:
            metrics["selection/score/passed"] = sum(candidate.score for candidate in self.passed) / len(self.passed)
        return metrics

    def records(self, step: int, fate: Fate = "trained") -> list[dict[str, Any]]:
        return [candidate.record(step=step, fate=fate, rollouts=count) for candidate, count in self.selected]


class CandidatePool:
    """Admitted groups whose traces wait in the sink's queue, ranked for selection."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        """Queued traces the pool holds; an admitted group past it displaces the lowest-ranked groups."""
        self.candidates: dict[str, Candidate] = {}
        self.group_of: dict[str, str] = {}
        """Each queued trace's group."""
        self.next_order = 0
        self.admitted = 0
        self.dropped: Counter[str] = Counter()

    @property
    def size(self) -> int:
        """Queued traces."""
        return len(self.group_of)

    def ranked(self) -> list[Candidate]:
        return sorted(self.candidates.values(), key=Candidate.rank)

    def add(self, candidate: Candidate) -> list[Exit]:
        """Pool an admitted group. Returns the lowest-ranked groups, possibly the new one, that the pool holds its
        cap without."""
        candidate.order = self.next_order
        self.next_order += 1
        self.admitted += 1
        self.candidates[candidate.group_id] = candidate
        self.group_of.update(dict.fromkeys(candidate.trace_ids, candidate.group_id))
        displaced: list[Exit] = []
        ranked = self.ranked()
        while self.size - len(ranked[-1].trace_ids) >= self.cap:
            displaced += self.drop(list(ranked.pop().trace_ids), "displaced")
        return displaced

    def drop(self, trace_ids: Iterable[str], fate: Fate) -> list[Exit]:
        """Forget traces that left the queue other than in a batch. Returns the groups left with none."""
        exits = self._remove(trace_ids)
        self.dropped[fate] += len(exits)
        return exits

    def select(self, batch_size: int) -> Selection:
        """Take the ranked groups' traces up to ``batch_size``. The groups passed over stay pooled, as does the
        rest of a group the batch splits."""
        trace_ids: list[str] = []
        selected: list[tuple[Candidate, int]] = []
        passed: list[Candidate] = []
        for candidate in self.ranked():
            taken = candidate.trace_ids[: batch_size - len(trace_ids)]
            if taken:
                trace_ids += taken
                selected.append((candidate, len(taken)))
            else:
                candidate.passes += 1
                passed.append(candidate)
        self._remove(trace_ids)
        selection = Selection(
            trace_ids, selected, passed, carried=len(self.candidates), admitted=self.admitted, dropped=self.dropped
        )
        self.admitted, self.dropped = 0, Counter()
        return selection

    def _remove(self, trace_ids: Iterable[str]) -> list[Exit]:
        removed: dict[str, list[str]] = defaultdict(list)
        emptied: list[Candidate] = []
        for trace_id in trace_ids:
            group_id = self.group_of.pop(trace_id, None)
            if group_id is None:
                continue
            candidate = self.candidates[group_id]
            candidate.trace_ids.remove(trace_id)
            removed[group_id].append(trace_id)
            if not candidate.trace_ids:
                del self.candidates[group_id]
                emptied.append(candidate)
        return [(candidate, removed[candidate.group_id]) for candidate in emptied]

"""Per-step curriculum plans, posted over the orchestrator's control API or replayed from a recorded run.

A plan chooses the Tasks dispatched while the orchestrator collects one step, or every step without a
plan of its own (``"default"``). A step's plan is fixed when its first Task is dispatched, so the step
dispatches under one decision; a step without a plan samples exactly as configured. Three append-only
JSONL files in one directory keep the run repeatable:

- ``plans.jsonl``: every plan posted or deleted, with its time, author and note.
- ``dispatch.jsonl``: every Task dispatched, with its step and what chose it, and a summary line per step.
  Another run replays the schedule from it.
- ``groups.jsonl``: every finalized group's outcome, which the stats are built from, with the reward shaping it
  trained under (``prime_rl.orchestrator.curriculum.shaping``, whose changes ``shaping.jsonl`` records).

A resume reads them back, keeping the dispatch and group records up to its step.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field

from prime_rl.orchestrator.curriculum.samplers import DifficultyPoolSampler
from prime_rl.orchestrator.utils import episode_env_name, train_work
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.curriculum.shaping import Shaping
    from prime_rl.orchestrator.train_source import TrainSource

Weight = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Share = Annotated[float, Field(ge=0, le=1)]
PlanKey = Annotated[int, Field(ge=1)] | Literal["default"]
"""A plan's step, or ``"default"`` for every step without a plan of its own."""
RequestedStep = PlanKey | Literal["next"]
"""A posted plan's step: ``"next"`` is the first step whose dispatch hasn't started, fixed as it is stored."""
Origin = Literal["step", "default", "replay"]
"""What fixed a step's plan: its own plan, the default plan, or the replayed record."""


class PlanTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env: str
    key: str


class Plan(BaseModel):
    """The Tasks one step trains on. What it leaves out falls back to the configured sampling."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[PlanTask] | None = None
    """Tasks dispatched in order, one per group, from the step's first group; after them the step samples."""

    sources: dict[str, Share] | None = None
    """Shares of the step's groups by env. A named env gets exactly its share; the rest of 1 goes to the
    envs it leaves out by their configured ratios, so nothing if the named shares sum to 1. Relative to the
    config, never to another step's plan."""

    weights: dict[str, Weight] | None = None
    """Multipliers on a Task's weight within its env, keyed ``task:<key>``, ``label:<label>`` or
    ``pool:<pool>``. A Task's weight is its difficulty pool's (1 without a pool sampler) times every
    multiplier it matches, so 0 excludes it."""


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: RequestedStep
    plan: Plan
    author: str | None = None
    note: str | None = None
    """Why the plan was posted, kept with it in ``plans.jsonl``."""


@dataclass
class StoredPlan:
    id: int
    """The plan's line in ``plans.jsonl``, which dispatch records name it by."""
    plan: Plan
    posted_at: str
    author: str | None = None
    note: str | None = None


class PlanError(ValueError):
    """A plan the run cannot apply, such as one naming an unknown env, Task, label or pool."""


class PlanConflictError(Exception):
    """A plan for a step whose dispatch has started or passed, or for a run that replays a record."""

    def __init__(self, message: str, *, current_dispatch_step: int, next_open_step: int | None) -> None:
        super().__init__(message)
        self.current_dispatch_step = current_dispatch_step
        self.next_open_step = next_open_step
        """The step a plan could be posted for instead; None while the run replays a record."""


class PlanNotFoundError(Exception):
    pass


SHARE_TOLERANCE = 1e-9


def source_shares(names: list[str], ratios: list[float], shares: dict[str, float]) -> list[float]:
    """Each env's share of a step: a named env's as given, and the rest of 1 split among the others by their
    configured ratios."""
    rest = 1.0 - sum(shares.values())
    unnamed = sum(ratio for name, ratio in zip(names, ratios) if name not in shares)
    scale = rest / unnamed if rest > SHARE_TOLERANCE and unnamed else 0.0
    return [shares[name] if name in shares else ratio * scale for name, ratio in zip(names, ratios)]


def has_label(task: vf.Task, label: str) -> bool:
    """Whether the Task data's ``labels`` hold ``label``, or, for ``<field>=<value>``, its field has that value."""
    name, sep, value = label.partition("=")
    if sep:
        found = getattr(task.data, name, None)
        return found is not None and str(found) == value
    return label in (getattr(task.data, "labels", None) or ())


def group_outcome(group: list[vf.Episode]) -> dict[str, Any]:
    """A finalized group's line in ``groups.jsonl``. Its reward is the mean over the rollouts a difficulty
    pool scores; the counts are over its rollouts, an episode lost before it had a trace counted as errored."""
    first = group[0]
    traces = [trace for episode in group for trace in episode.traces if trace.agent.trainable]
    lost = sum(not episode.ok and not episode.traces for episode in group)
    rewards = [
        trace.reward
        for episode in group
        for trace in episode.traces
        if episode.ok and not trace.has_error and trace.agent.trainable and trace.num_turns > 0
    ]
    return {
        "step": train_work(first).step,
        "env": episode_env_name(first),
        "key": first.task.key,
        "reward": sum(rewards) / len(rewards) if rewards else None,
        "mixed": len(set(rewards)) > 1,
        "rollouts": len(traces) + lost,
        "turns": sum(trace.num_turns for trace in traces),
        "truncated": sum(trace.is_truncated for trace in traces),
        "timed_out": sum(trace.is_timeout for trace in traces),
        "errored": sum(trace.has_error for trace in traces) + lost,
    }


LABEL_STATS = ("groups", "mixed_share", "mean_reward")
"""What ``/v1/stats`` gives per label, of an env's stats."""


@dataclass
class EnvTally:
    """Finalized groups of one env."""

    groups: int = 0
    mixed: int = 0
    scored: int = 0
    reward: float = 0.0
    rollouts: int = 0
    turns: int = 0
    truncated: int = 0
    timed_out: int = 0
    errored: int = 0

    def add(self, outcome: dict[str, Any]) -> None:
        self.groups += 1
        self.mixed += outcome["mixed"]
        if outcome["reward"] is not None:
            self.scored += 1
            self.reward += outcome["reward"]
        for name in ("rollouts", "turns", "truncated", "timed_out", "errored"):
            setattr(self, name, getattr(self, name) + outcome[name])

    def summary(self) -> dict[str, Any]:
        return {
            "groups": self.groups,
            "mixed_share": _ratio(self.mixed, self.groups),
            "mean_reward": _ratio(self.reward, self.scored),
            "mean_turns": _ratio(self.turns, self.rollouts),
            "truncated_share": _ratio(self.truncated, self.rollouts),
            "timeout_share": _ratio(self.timed_out, self.rollouts),
            "error_share": _ratio(self.errored, self.rollouts),
        }


@dataclass
class Tally:
    """Finalized groups by env, by the labels of their Tasks, and by Task (most recently finalized last)."""

    envs: dict[str, EnvTally] = field(default_factory=dict)
    labels: dict[str, EnvTally] = field(default_factory=dict)
    tasks: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def add(self, outcome: dict[str, Any], labels: Iterable[str] = ()) -> None:
        self.envs.setdefault(outcome["env"], EnvTally()).add(outcome)
        for label in labels:
            self.labels.setdefault(label, EnvTally()).add(outcome)
        task_id = (outcome["env"], outcome["key"])
        visits = self.tasks.pop(task_id, {"visits": 0})["visits"]
        self.tasks[task_id] = {"reward": outcome["reward"], "visits": visits + 1}


@dataclass
class StepPlan:
    """One step's plan as its dispatch started, and what it has dispatched since."""

    step: int
    origin: Origin | None = None
    plan_id: int | None = None
    plan: Plan | None = None
    env_weights: list[float] = field(default_factory=list)
    multipliers: dict[str, list[float]] = field(default_factory=dict)
    """Per env a ``task:``/``label:`` selector matches: each Task's product of the multipliers it matches."""
    pool_multipliers: dict[str, float] = field(default_factory=dict)
    weighted_envs: set[str] = field(default_factory=set)
    """Envs whose Tasks the plan's weights touch; the others keep their sampler."""
    cursor: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    """Dispatched Tasks by what chose them: the plan's ``tasks``, its ``weights``, or the env's ``sampler``."""
    warned: set[str] = field(default_factory=set)

    @property
    def tasks_left(self) -> int:
        return len(self.plan.tasks or []) - self.cursor if self.plan is not None else 0

    def summary(self) -> dict[str, Any]:
        return {
            "type": "step",
            "step": self.step,
            "plan": self.plan_id,
            "origin": self.origin,
            "counts": dict(self.counts),
            "unused": self.tasks_left,
        }


class Plans:
    """Plans keyed by step or ``"default"``, applied to the train source's draws. The control API's thread
    posts and reads them while the orchestrator's loop draws Tasks and reports groups."""

    def __init__(
        self, source: TrainSource, directory: Path, *, resume_step: int | None, replay_from: Path | None = None
    ) -> None:
        """Read the records in ``directory`` back on a resume, the dispatch and group records up to
        ``resume_step``; a fresh start sets earlier plans aside and clears the records. ``replay_from``, a
        recorded run's ``dispatch.jsonl``, fixes each step it lists to the Tasks it dispatched."""
        self.source = source
        self.tasks = {
            name: {task.key: task for task in tasks} for name, tasks in source.tasks.items() if tasks is not None
        }
        self.labels = {
            (name, key): list(labels)
            for name, keyed in self.tasks.items()
            for key, task in keyed.items()
            if (labels := getattr(task.data, "labels", None))
        }
        """Each Task's labels, for those whose data has them."""
        self.pools = {
            name: {*curriculum.sampler.pools, "unseen"}
            for name, curriculum in source.curricula.items()
            if isinstance(curriculum.sampler, DifficultyPoolSampler)
        }
        self.replay_from = replay_from
        self.replay = self._load_replay(replay_from) if replay_from is not None else None
        self.plans: dict[int | str, StoredPlan] = {}
        self.events = 0
        self.dispatch_step = resume_step or 0
        """The latest step whose dispatch started; a plan for it or an earlier step comes too late."""
        self.active: StepPlan | None = None
        self.used: dict[int, Origin] = {}
        self.totals = Tally()
        self.by_step: dict[int, Tally] = {}
        self.shaping: Shaping | None = None
        """The run's live reward shaping, whose setting each group's outcome records."""
        self.lock = threading.Lock()

        directory.mkdir(parents=True, exist_ok=True)
        self.plans_path = directory / "plans.jsonl"
        self.dispatch_path = directory / "dispatch.jsonl"
        self.groups_path = directory / "groups.jsonl"
        self._read_plans(fresh=resume_step is None)
        for record in _keep_records(self.dispatch_path, self.dispatch_step):
            if record["type"] == "task" and record["origin"] is not None:
                self.used[record["step"]] = record["origin"]
        for outcome in _keep_records(self.groups_path, self.dispatch_step):
            self._tally(outcome)

    # ── the API's side ─────────────────────────────────────────────────────

    def set(
        self, requested: RequestedStep, plan: Plan, *, author: str | None = None, note: str | None = None
    ) -> dict[str, Any]:
        """Store a plan for ``requested``, ``"next"`` taking the first step whose dispatch hasn't started."""
        self.validate(plan)
        with self.lock:
            key = self.dispatch_step + 1 if requested == "next" else requested
            self._check_open(key)
            previous = self.plans.get(key)
            event = self._log(
                action="set",
                step=key,
                requested=requested,
                plan=plan.model_dump(mode="json", exclude_none=True),
                shares=self.shares(plan),
                author=author,
                note=note,
                replaces=previous.id if previous is not None else None,
            )
            self.plans[key] = StoredPlan(event["id"], plan, event["time"], author, note)
            return self._describe(key, self.plans[key])

    def get(self, key: PlanKey) -> dict[str, Any]:
        with self.lock:
            return self._describe(key, self._find(key))

    def delete(self, key: PlanKey, *, author: str | None = None, note: str | None = None) -> dict[str, Any]:
        with self.lock:
            stored = self._find(key)
            self._check_open(key)
            self._log(action="delete", step=key, deletes=stored.id, author=author, note=note)
            del self.plans[key]
            return self._describe(key, stored)

    def all(self) -> list[dict[str, Any]]:
        with self.lock:
            keys = sorted(self.plans, key=lambda key: (key != "default", key if key != "default" else 0))
            return [self._describe(key, self.plans[key]) for key in keys]

    def shares(self, plan: Plan) -> dict[str, float]:
        """The share of a step's sampled groups each env gets under ``plan``, from the configured ratios, its
        ``sources``, and its ``weights`` where they exclude an env's every Task."""
        weights = self._resolve(0, "step", None, plan).env_weights
        if not any(weights):
            weights = self.source.weights
        total = sum(weights)
        return {name: round(weight / total, 4) for name, weight in zip(self.source.env_names, weights)}

    def validate(self, plan: Plan) -> None:
        """Raise ``PlanError`` unless every env, Task, label and pool the plan names is the run's."""
        if not plan.tasks and plan.sources is None and not plan.weights:
            raise PlanError("A plan sets tasks, sources or weights")
        names = self.source.env_names
        for item in plan.tasks or []:
            if item.env not in names:
                raise PlanError(f"No env {item.env!r}; the envs are {', '.join(names)}")
            if item.env not in self.tasks:
                raise PlanError(f"Env {item.env!r} streams its Tasks, so a plan cannot list them")
            if item.key not in self.tasks[item.env]:
                raise PlanError(f"Env {item.env!r} has no Task {item.key!r}")
        unknown = sorted(set(plan.sources or {}) - set(names))
        if unknown:
            raise PlanError(f"No env {', '.join(map(repr, unknown))}; the envs are {', '.join(names)}")
        if plan.sources is not None:
            total = sum(plan.sources.values())
            if total > 1 + SHARE_TOLERANCE:
                raise PlanError(f"The sources' shares sum to {total:g}, more than 1")
            if set(plan.sources) == set(names) and total < 1 - SHARE_TOLERANCE:
                raise PlanError(f"The sources name every env, so their shares sum to 1, not {total:g}")
        for selector in plan.weights or {}:
            kind, _, value = selector.partition(":")
            if kind == "task":
                found = any(value in keyed for keyed in self.tasks.values())
            elif kind == "label":
                found = any(has_label(task, value) for keyed in self.tasks.values() for task in keyed.values())
            elif kind == "pool":
                found = any(value in pools for pools in self.pools.values())
            else:
                raise PlanError(f"Weight {selector!r} is not task:<key>, label:<label> or pool:<pool>")
            if not found:
                raise PlanError(f"Weight {selector!r} matches no Task of an env with a finite taskset")
        if not plan.tasks and not any(self._resolve(0, "step", None, plan).env_weights):
            raise PlanError("The plan leaves every env weight 0")

    def status(self) -> dict[str, Any]:
        with self.lock:
            active = self.active
            used = sorted(self.used.items())
            return {
                "dispatch_step": self.dispatch_step,
                "dispatching": None
                if active is None
                else {
                    "step": active.step,
                    "plan": active.plan_id,
                    "origin": active.origin,
                    "groups": active.counts.total(),
                    "tasks_left": active.tasks_left,
                },
                "pending_plans": sorted(key for key in self.plans if key != "default" and key > self.dispatch_step),
                "default_plan": "default" in self.plans,
                "replay_from": str(self.replay_from) if self.replay_from is not None else None,
                "plan_steps": {
                    origin: _ranges(step for step, used_origin in used if used_origin == origin)
                    for origin in ("step", "default", "replay")
                },
            }

    def stats(
        self,
        *,
        step: int | None = None,
        env: str | None = None,
        labels: list[str] | None = None,
        limit: int = 100,
        top_labels: int = 20,
    ) -> dict[str, Any]:
        """Finalized groups by env, by label and by Task, most recently finalized first: the run's, or those
        dispatched at ``step``. ``labels`` names the labels to aggregate, across envs, and keeps the Tasks
        carrying all of them; without it the ``top_labels`` labels with the most groups are aggregated."""
        with self.lock:
            tally = self.totals if step is None else self.by_step.get(step, Tally())
            wanted = set(labels or ())
            tasks = []
            for (env_name, key), task in reversed(tally.tasks.items()):
                task_labels = self.labels.get((env_name, key), [])
                if (env is None or env_name == env) and wanted <= set(task_labels):
                    entry = {"key": key, "env": env_name, "reward": _round(task["reward"]), "visits": task["visits"]}
                    tasks.append(entry | ({"labels": task_labels} if task_labels else {}))
            if labels:
                chosen = [(label, tally.labels.get(label, EnvTally())) for label in labels]
            else:
                chosen = sorted(tally.labels.items(), key=lambda item: (-item[1].groups, item[0]))[:top_labels]
            return {
                "step": step,
                "envs": {
                    name: env_tally.summary()
                    for name, env_tally in sorted(tally.envs.items())
                    if env is None or name == env
                },
                "labels": {
                    label: {name: value for name, value in label_tally.summary().items() if name in LABEL_STATS}
                    for label, label_tally in chosen
                },
                "tasks": tasks[:limit],
                "tasks_total": len(tasks),
            }

    # ── the orchestrator's side ────────────────────────────────────────────

    def draw(self, step: int) -> tuple[str, vf.Task]:
        """The env and Task of the next group dispatched at ``step``, which starts its dispatch if it is
        the step's first."""
        with self.lock:
            if self.active is None or self.active.step != step:
                self._begin(step)
            active = self.active
            env_name, task, via = self._draw(active)
            active.counts[via] += 1
            record = {"type": "task", "step": step, "env": env_name, "key": task.key, "via": via}
            _append(self.dispatch_path, record | {"plan": active.plan_id, "origin": active.origin})
            if via == "tasks" and active.tasks_left == 0:
                get_logger().info(f"Step {step} dispatched its {active.cursor} planned Tasks; it samples from here")
            return env_name, task

    def observe(self, group: list[vf.Episode]) -> None:
        """Count a finalized group in the stats of the run and of the step it was dispatched at."""
        outcome = group_outcome(group)
        if self.shaping is not None:
            outcome |= self.shaping.record(group)
        with self.lock:
            self._tally(outcome)
            _append(self.groups_path, outcome)

    def close(self) -> None:
        """Summarize the step dispatching now, as the next step's dispatch would."""
        with self.lock:
            if self.active is not None:
                self._summarize()
                self.active = None

    # ── internals (under the lock) ─────────────────────────────────────────

    def _begin(self, step: int) -> None:
        if self.active is not None:
            self._summarize()
        if self.replay is not None:
            tasks = self.replay.get(step)
            self.active = StepPlan(step) if tasks is None else self._resolve(step, "replay", None, Plan(tasks=tasks))
        elif step in self.plans or "default" in self.plans:
            origin: Origin = "step" if step in self.plans else "default"
            stored = self.plans[step if origin == "step" else "default"]
            self.active = self._resolve(step, origin, stored.id, stored.plan)
            get_logger().info(f"Step {step} dispatches under plan {stored.id} ({origin})")
        else:
            self.active = StepPlan(step)
        if self.active.origin is not None:
            self.used[step] = self.active.origin
        self.dispatch_step = max(self.dispatch_step, step)

    def _summarize(self) -> None:
        active = self.active
        _append(self.dispatch_path, active.summary())
        if active.tasks_left:
            get_logger().warning(f"Step {active.step} left {active.tasks_left} planned Tasks undispatched")

    def _resolve(self, step: int, origin: Origin, plan_id: int | None, plan: Plan) -> StepPlan:
        """Fix the plan's env weights and per-Task multipliers for a step."""
        resolved = StepPlan(step, origin, plan_id, plan)
        selectors = [(*selector.split(":", 1), weight) for selector, weight in (plan.weights or {}).items()]
        resolved.pool_multipliers = {value: weight for kind, value, weight in selectors if kind == "pool"}
        static = [(kind, value, weight) for kind, value, weight in selectors if kind in ("task", "label")]
        for name in self.tasks:
            factors, matched = [], False
            for task in self.source.tasks[name]:
                factor = 1.0
                for kind, value, weight in static:
                    if (task.key == value) if kind == "task" else has_label(task, value):
                        factor *= weight
                        matched = True
                factors.append(factor)
            if matched:
                resolved.multipliers[name] = factors
                resolved.weighted_envs.add(name)
            if set(resolved.pool_multipliers) & self.pools.get(name, set()):
                resolved.weighted_envs.add(name)

        names = self.source.env_names
        if plan.sources is None:
            weights = list(self.source.weights)
        else:
            weights = source_shares(names, self.source.weights, plan.sources)
        # An env whose every Task the multipliers exclude is out of the step.
        resolved.env_weights = [
            0.0 if name in resolved.multipliers and not any(resolved.multipliers[name]) else weight
            for name, weight in zip(names, weights)
        ]
        return resolved

    def _draw(self, active: StepPlan) -> tuple[str, vf.Task, str]:
        source = self.source
        plan = active.plan
        if plan is None:
            return *source.draw(), "sampler"
        if plan.tasks and active.cursor < len(plan.tasks):
            item = plan.tasks[active.cursor]
            active.cursor += 1
            return item.env, self.tasks[item.env][item.key], "tasks"
        if not any(active.env_weights):
            return *source.draw(), "sampler"
        env_name = source.rng.choices(source.env_names, weights=active.env_weights, k=1)[0]
        task = self._weighted_task(active, env_name) if env_name in active.weighted_envs else None
        if task is None:
            return env_name, next(source.curricula[env_name].sampler), "sampler"
        return env_name, task, "weights"

    def _weighted_task(self, active: StepPlan, env_name: str) -> vf.Task | None:
        tasks = self.source.tasks[env_name]
        sampler = self.source.curricula[env_name].sampler
        weights = list(active.multipliers.get(env_name, [1.0] * len(tasks)))
        if isinstance(sampler, DifficultyPoolSampler):
            for index, task in enumerate(tasks):
                pool = sampler.task_pool(task.key)
                base = 1.0 if pool is None else sampler.pools[pool].weight
                weights[index] *= base * active.pool_multipliers.get(pool or "unseen", 1.0)
        if not any(weights):
            if env_name not in active.warned:
                active.warned.add(env_name)
                get_logger().warning(
                    f"Step {active.step}'s plan leaves env {env_name} no Task with weight; it samples as configured"
                )
            return None
        return self.source.rng.choices(tasks, weights=weights, k=1)[0]

    def _tally(self, outcome: dict[str, Any]) -> None:
        labels = self.labels.get((outcome["env"], outcome["key"]), ())
        self.totals.add(outcome, labels)
        self.by_step.setdefault(outcome["step"], Tally()).add(outcome, labels)

    def _check_open(self, key: PlanKey) -> None:
        if self.replay is not None:
            raise PlanConflictError(
                f"The run replays {self.replay_from} and takes no plans.",
                current_dispatch_step=self.dispatch_step,
                next_open_step=None,
            )
        if key != "default" and key <= self.dispatch_step:
            raise PlanConflictError(
                f"Step {key}'s dispatch has started (the run is at step {self.dispatch_step}); "
                f"post for step {self.dispatch_step + 1} or 'next'.",
                current_dispatch_step=self.dispatch_step,
                next_open_step=self.dispatch_step + 1,
            )

    def _find(self, key: PlanKey) -> StoredPlan:
        stored = self.plans.get(key)
        if stored is None:
            raise PlanNotFoundError(f"No plan for {'the default' if key == 'default' else f'step {key}'}")
        return stored

    def _describe(self, key: PlanKey, stored: StoredPlan) -> dict[str, Any]:
        if key == "default":
            state = "default"
        elif key > self.dispatch_step:
            state = "pending"
        else:
            state = "used" if self.used.get(key) == "step" else "passed"
        described = {
            "step": key,
            "id": stored.id,
            "plan": stored.plan.model_dump(mode="json", exclude_none=True),
            "shares": self.shares(stored.plan),
            "state": state,
            "posted_at": stored.posted_at,
            "author": stored.author,
            "note": stored.note,
        }
        return {name: value for name, value in described.items() if value is not None}

    def _log(self, **event: Any) -> dict[str, Any]:
        """Append an event to the plan history and return it with its id and time."""
        self.events += 1
        now = datetime.now(UTC).isoformat(timespec="seconds")
        event = {"id": self.events, "time": now} | {name: value for name, value in event.items() if value is not None}
        _append(self.plans_path, event)
        return event

    def _read_plans(self, *, fresh: bool) -> None:
        """Rebuild the plans from their history since its last reset. A fresh start resets it."""
        events = _read_jsonl(self.plans_path)
        self.events = max((event["id"] for event in events), default=0)
        if fresh:
            if events:
                self._log(action="reset")
            return
        resets = [index for index, event in enumerate(events) if event["action"] == "reset"]
        for event in events[resets[-1] + 1 if resets else 0 :]:
            if event["action"] == "set":
                plan = Plan.model_validate(event["plan"])
                stored = StoredPlan(event["id"], plan, event["time"], event.get("author"), event.get("note"))
                self.plans[event["step"]] = stored
            elif event["action"] == "delete":
                self.plans.pop(event["step"], None)
        for key, stored in list(self.plans.items()):
            try:
                self.validate(stored.plan)
            except PlanError as error:
                get_logger().warning(f"Setting plan {stored.id} for {key} aside: {error}")
                del self.plans[key]

    def _load_replay(self, path: Path) -> dict[int, list[PlanTask]]:
        """Each step's dispatched Tasks in a dispatch record, in order."""
        if not path.exists():
            raise PlanError(f"No dispatch record to replay at {path}")
        schedule: dict[int, list[PlanTask]] = {}
        for record in _read_jsonl(path):
            if record["type"] == "task":
                schedule.setdefault(record["step"], []).append(PlanTask(env=record["env"], key=record["key"]))
        if not schedule:
            raise PlanError(f"{path} records no dispatched Tasks")
        self.validate(Plan(tasks=[task for tasks in schedule.values() for task in tasks]))
        get_logger().info(f"Replaying the Tasks of {len(schedule)} steps from {path}; plans are refused")
        return schedule


def _ratio(count: float, total: int) -> float | None:
    return round(count / total, 4) if total else None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _ranges(steps: Iterable[int]) -> list[list[int]]:
    """Sorted steps as inclusive ``[first, last]`` runs."""
    ranges: list[list[int]] = []
    for step in steps:
        if ranges and step == ranges[-1][1] + 1:
            ranges[-1][1] = step
        else:
            ranges.append([step, step])
    return ranges


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(record) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """A JSONL file's records; a line a crash cut short is dropped."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _keep_records(path: Path, last_step: int) -> list[dict[str, Any]]:
    """Rewrite a record file to its records up to ``last_step`` and return them."""
    kept = [record for record in _read_jsonl(path) if record["step"] <= last_step]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(record) + "\n" for record in kept))
    os.replace(tmp, path)
    return kept

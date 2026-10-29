"""Live reward shaping: GRPO's ``shaping`` changed during a run, from a coming dispatch step on.

``POST /v1/shaping`` names a step (or ``"next"``) and the fields of ``stuck`` and ``unfinished`` to change. A change
is sticky: every group dispatched at that step or later trains under it, until a later change. A group is shaped by
the step it was dispatched at, not when it is scored, so the groups in flight when a change is posted keep the
setting they were dispatched under; a change for a step whose dispatch has started is refused, as a plan is.

Each change is recorded in ``shaping.jsonl`` beside the plans, and each group's setting in ``groups.jsonl``. A
resume reads the changes back; ``curriculum_plans.replay_from`` replays the changes recorded beside the dispatch
record it names, and refuses new ones.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from prime_rl.configs.algorithm import RewardShapingConfig
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm, shape_reward
from prime_rl.orchestrator.curriculum.plans import PlanConflictError, PlanError, _append, _read_jsonl
from prime_rl.orchestrator.utils import episode_env_name, train_work
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.curriculum.plans import Plans

ShapingStep = Annotated[int, Field(ge=1)] | Literal["next"]
"""The first step a change applies to: ``"next"`` is the first step whose dispatch hasn't started, fixed as the change
is stored."""


class StuckChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str | None = None
    per_call: float | None = None
    free: int | None = None
    cap: float | None = None


class UnfinishedChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: float | None = None
    """``null`` keeps the env's scores again."""


class ShapingRequest(BaseModel):
    """The fields of the config's ``shaping`` to change from ``step`` on; what it leaves out keeps its setting."""

    model_config = ConfigDict(extra="forbid")

    step: ShapingStep
    stuck: StuckChange | None = None
    unfinished: UnfinishedChange | None = None
    author: str | None = None
    note: str | None = None
    """Why the change was posted, kept with it in ``shaping.jsonl``."""

    def change(self) -> dict[str, dict[str, Any]]:
        """The fields the request sets, by lever."""
        fields = self.model_dump(include={"stuck", "unfinished"}, exclude_unset=True)
        return {lever: values for lever, values in fields.items() if values}


@dataclass
class StoredChange:
    id: int
    step: int
    change: dict[str, dict[str, Any]]
    requested: ShapingStep | None = None
    posted_at: str | None = None
    author: str | None = None
    note: str | None = None


class Shaping:
    """Each GRPO train env's shaping by dispatch step: its config's, then every change for that step or an earlier
    one, in step order and then as posted. Changes are refused for steps the plans have started dispatching, under
    the plans' lock, so a step dispatches under one setting."""

    def __init__(
        self,
        plans: Plans,
        algorithms: dict[str, GRPOAlgorithm],
        directory: Path,
        *,
        resume_step: int | None,
        replay_from: Path | None = None,
    ) -> None:
        """Shape each of ``algorithms``' groups by its dispatch step, and record each group's setting with the plans'
        group outcomes. ``replay_from`` is a recorded run's ``dispatch.jsonl``, whose directory's ``shaping.jsonl``
        holds the changes to replay."""
        self.plans = plans
        self.defaults = {name: algorithm.shaping for name, algorithm in algorithms.items()}
        self.changes: list[StoredChange] = []
        self.path = directory / "shaping.jsonl"
        self.replay_from = replay_from.parent / "shaping.jsonl" if replay_from is not None else None
        events = _read_jsonl(self.path)
        self.events = max((event["id"] for event in events), default=0)
        if self.replay_from is not None:
            self.changes = _changes(_read_jsonl(self.replay_from))
            if resume_step is None:
                if events:
                    self._log(action="reset")
                for stored in self.changes:
                    self._log(action="set", step=stored.step, change=stored.change, replayed_from=str(self.replay_from))
            get_logger().info(
                f"Replaying {len(self.changes)} shaping changes from {self.replay_from}; changes are refused"
            )
        elif resume_step is None:
            if events:
                self._log(action="reset")
        else:
            self.changes = _changes(events)
        for default in self.defaults.values():  # Each step's setting, so that a bad record fails here, not mid-run.
            for index in range(len(self.changes)):
                _resolve(default, self.changes[: index + 1])
        for name, algorithm in algorithms.items():
            algorithm.shaping_at = functools.partial(self.at, name)
        plans.shaping = self

    # ── the API's side ─────────────────────────────────────────────────────

    def set(self, request: ShapingRequest) -> dict[str, Any]:
        """Store a change from ``request.step``, ``"next"`` taking the first step whose dispatch hasn't started."""
        change = request.change()
        if not change:
            raise PlanError("A shaping change sets a field of 'stuck' or 'unfinished'")
        if not self.defaults:
            raise PlanError("No train env runs GRPO, so there is no reward to shape")
        with self.plans.lock:
            step = self.plans.dispatch_step + 1 if request.step == "next" else request.step
            self._check_open(step)
            stored = StoredChange(0, step, change, request.step, author=request.author, note=request.note)
            changes = sorted([*self.changes, stored], key=lambda item: item.step)
            for name in self.defaults:
                _resolve(self.defaults[name], (item for item in changes if item.step <= step))
            event = self._log(
                action="set", step=step, requested=request.step, change=change, author=request.author, note=request.note
            )
            stored.id, stored.posted_at = event["id"], event["time"]
            self.changes = changes
            return self._describe(stored) | {"shaping": self._settings(step)}

    def status(self) -> dict[str, Any]:
        """The setting of the step dispatching now, and the changes for steps still to come."""
        with self.plans.lock:
            dispatch_step = self.plans.dispatch_step
            return {
                "active": self._settings(dispatch_step),
                "pending": [self._describe(stored) for stored in self.changes if stored.step > dispatch_step],
            }

    def describe(self) -> dict[str, Any]:
        with self.plans.lock:
            dispatch_step = self.plans.dispatch_step
            return {
                "dispatch_step": dispatch_step,
                "active": self._settings(dispatch_step),
                "pending": [self._describe(stored) for stored in self.changes if stored.step > dispatch_step],
                "changes": [self._describe(stored) for stored in self.changes],
                "replay_from": str(self.replay_from) if self.replay_from is not None else None,
            }

    # ── the orchestrator's side ────────────────────────────────────────────

    def at(self, env: str, step: int) -> RewardShapingConfig:
        """The shaping of ``env``'s groups dispatched at ``step``."""
        with self.plans.lock:
            return _resolve(self.defaults[env], (stored for stored in self.changes if stored.step <= step))

    def record(self, group: list[vf.Episode]) -> dict[str, Any]:
        """A finalized group's shaping for its ``groups.jsonl`` line: the setting it trained under, and how many of
        its rollouts each lever changed. Nothing for an env that doesn't run GRPO."""
        env = episode_env_name(group[0])
        if env not in self.defaults:
            return {}
        shaping = self.at(env, train_work(group[0]).step)
        levers = [
            shape_reward(trace, shaping)[1]
            for episode in group
            for trace in episode.traces
            if episode.ok and not trace.has_error and trace.agent.trainable and trace.num_turns > 0
        ]
        return {
            "shaping": shaping.model_dump(mode="json"),
            "stuck_penalized": levers.count("stuck"),
            "unfinished_scored": levers.count("unfinished"),
        }

    # ── internals ──────────────────────────────────────────────────────────

    def _check_open(self, step: int) -> None:
        if self.replay_from is not None:
            raise PlanConflictError(
                f"The run replays the shaping of {self.replay_from} and takes no changes.",
                current_dispatch_step=self.plans.dispatch_step,
                next_open_step=None,
            )
        if step <= self.plans.dispatch_step:
            raise PlanConflictError(
                f"Step {step}'s dispatch has started (the run is at step {self.plans.dispatch_step}); "
                f"post for step {self.plans.dispatch_step + 1} or 'next'.",
                current_dispatch_step=self.plans.dispatch_step,
                next_open_step=self.plans.dispatch_step + 1,
            )

    def _settings(self, step: int) -> list[dict[str, Any]]:
        """Each distinct setting at ``step`` with the envs that have it."""
        settings: dict[str, dict[str, Any]] = {}
        for name, default in self.defaults.items():
            shaping = _resolve(default, (stored for stored in self.changes if stored.step <= step))
            entry = settings.setdefault(shaping.model_dump_json(), {"envs": [], **shaping.model_dump(mode="json")})
            entry["envs"].append(name)
        return list(settings.values())

    def _describe(self, stored: StoredChange) -> dict[str, Any]:
        described = {
            "id": stored.id,
            "step": stored.step,
            "requested": stored.requested,
            "change": stored.change,
            "state": "pending" if stored.step > self.plans.dispatch_step else "applied",
            "posted_at": stored.posted_at,
            "author": stored.author,
            "note": stored.note,
        }
        return {name: value for name, value in described.items() if value is not None}

    def _log(self, **event: Any) -> dict[str, Any]:
        self.events += 1
        now = datetime.now(UTC).isoformat(timespec="seconds")
        event = {"id": self.events, "time": now} | {name: value for name, value in event.items() if value is not None}
        _append(self.path, event)
        return event


def _resolve(default: RewardShapingConfig, changes: Iterable[StoredChange]) -> RewardShapingConfig:
    """``default`` with each change's fields set in turn; ``PlanError`` if the result is not a valid shaping."""
    fields = default.model_dump()
    for stored in changes:
        for lever, values in stored.change.items():
            fields[lever] = fields[lever] | values
    try:
        return RewardShapingConfig.model_validate(fields)
    except ValidationError as error:
        raise PlanError(f"Not a valid shaping: {error}") from error


def _changes(events: list[dict[str, Any]]) -> list[StoredChange]:
    """The changes a ``shaping.jsonl`` holds since its last reset, in step order and then as posted."""
    resets = [index for index, event in enumerate(events) if event["action"] == "reset"]
    changes = [
        StoredChange(
            event["id"],
            event["step"],
            event["change"],
            event.get("requested"),
            event["time"],
            event.get("author"),
            event.get("note"),
        )
        for event in events[resets[-1] + 1 if resets else 0 :]
        if event["action"] == "set"
    ]
    return sorted(changes, key=lambda stored: stored.step)

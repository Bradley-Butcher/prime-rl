import json
from collections import Counter
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from fastapi.testclient import TestClient
from pydantic import ValidationError

from prime_rl.configs.orchestrator import CurriculumConfig, DifficultyPoolConfig, DifficultyPoolSamplerConfig
from prime_rl.control.app import create_app
from prime_rl.control.controller import Controller
from prime_rl.control.plans import plan_router
from prime_rl.orchestrator.curriculum.plans import Plan, PlanConflictError, PlanError, Plans, PlanTask
from prime_rl.orchestrator.train_source import TrainSource

POOLS = {
    "hard": DifficultyPoolConfig(threshold=0.25, weight=1.0),
    "easy": DifficultyPoolConfig(threshold=1.0, weight=1.0),
}
LABELS = {0: ["fast"], 2: ["hard", "hops:1"]}


def make_source() -> TrainSource:
    """Env ``a`` cycles four Tasks with ``kind`` and ``labels``; env ``b`` samples four by difficulty pool."""
    a = [vf.Task(vf.WireTaskData(idx=i, kind="read" if i % 2 else "write", labels=LABELS.get(i, []))) for i in range(4)]
    b = [vf.Task(vf.WireTaskData(idx=i, prompt="b")) for i in range(4)]
    pool = CurriculumConfig(sampler=DifficultyPoolSamplerConfig(pools=POOLS))
    envs = [
        SimpleNamespace(name="a", tasks=iter(a), num_tasks=4, config=SimpleNamespace(ratio=1.0, curriculum=None)),
        SimpleNamespace(name="b", tasks=iter(b), num_tasks=4, config=SimpleNamespace(ratio=3.0, curriculum=pool)),
    ]
    return TrainSource(envs)


def planned_source(directory, resume_step: int | None = None, replay_from=None) -> TrainSource:
    source = make_source()
    source.plans = Plans(source, directory, resume_step=resume_step, replay_from=replay_from)
    return source


def draws(source: TrainSource, step: int, n: int) -> list[tuple[str, str]]:
    requests = [source.next_task(step=step) for _ in range(n)]
    return [(request.env_name, request.task.key) for request in requests]


def make_group(task: vf.Task, *, env: str, step: int, rewards: list[float]) -> list[vf.Episode]:
    group = []
    for reward in rewards:
        trace = vf.Trace(
            task=vf.TraceTask(type="Task", data=task.data, key=task.key, hash=task.hash),
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            nodes=[vf.MessageNode(message=vf.AssistantMessage(content="answer"), sampled=True)],
            rewards={"reward": vf.Reward(score=reward)},
            ok=True,
        )
        episode = vf.Episode(
            env=vf.EnvInfo(id=env, name=env),
            task=trace.task,
            group=vf.GroupInfo(id="group"),
            traces=[trace],
            ok=True,
            run=vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=step)),
        )
        group.append(episode)
    return group


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def dispatched(directory, kind: str = "task") -> list[dict]:
    return [record for record in read_jsonl(directory / "dispatch.jsonl") if record["type"] == kind]


def test_no_plan_samples_as_configured(tmp_path):
    plain, planned = make_source(), planned_source(tmp_path)
    for source in (plain, planned):
        task = source.tasks["b"][1]
        source.on_result(make_group(task, env="b", step=1, rewards=[1.0]))

    assert [draws(planned, step, 20) for step in (1, 2, 3)] == [draws(plain, step, 20) for step in (1, 2, 3)]
    assert planned.state_dict() == plain.state_dict()
    assert {(record["via"], record["plan"]) for record in dispatched(tmp_path)} == {("sampler", None)}
    assert [record["step"] for record in dispatched(tmp_path, "step")] == [1, 2]


def test_task_list_dispatches_in_order_then_samples(tmp_path):
    plain, planned = make_source(), planned_source(tmp_path)
    a, b = planned.tasks["a"], planned.tasks["b"]
    listed = [("b", b[2].key), ("a", a[0].key), ("a", a[0].key)]
    planned.plans.set(1, Plan(tasks=[PlanTask(env=env, key=key) for env, key in listed]))

    assert draws(planned, 1, 6) == listed + draws(plain, 1, 3)
    draws(planned, 2, 1)
    assert [record["via"] for record in dispatched(tmp_path)] == ["tasks"] * 3 + ["sampler"] * 4
    assert dispatched(tmp_path, "step") == [
        {"type": "step", "step": 1, "plan": 1, "origin": "step", "counts": {"tasks": 3, "sampler": 3}, "unused": 0}
    ]


def test_named_sources_get_their_share_and_the_rest_keep_their_ratios(tmp_path):
    names, ratios = ["A", "B", "C"], [0.5, 0.3, 0.2]
    envs = [
        SimpleNamespace(
            name=name,
            tasks=iter([vf.Task(vf.WireTaskData(idx=0, prompt=name))]),
            num_tasks=1,
            config=SimpleNamespace(ratio=ratio, curriculum=None),
        )
        for name, ratio in zip(names, ratios)
    ]
    source = TrainSource(envs)
    source.plans = Plans(source, tmp_path, resume_step=None)

    assert source.plans.set(2, Plan(sources={"A": 0.2}))["shares"] == {"A": 0.2, "B": 0.48, "C": 0.32}
    # Each step's plan is relative to the config, not to an earlier step's plan.
    assert source.plans.set(3, Plan(sources={"B": 0.1}))["shares"] == {"A": 0.6429, "B": 0.1, "C": 0.2571}
    assert source.plans.set(4, Plan(sources={"A": 0.6, "B": 0.4}))["shares"] == {"A": 0.6, "B": 0.4, "C": 0.0}
    counts = Counter(env for env, _ in draws(source, 2, 5000))
    assert abs(counts["A"] / 5000 - 0.2) < 0.03 and abs(counts["B"] / 5000 - 0.48) < 0.03


def test_sources_apply_to_their_step_only(tmp_path):
    source = planned_source(tmp_path)
    source.plans.set(2, Plan(sources={"a": 1.0}))
    source.plans.set(4, Plan(sources={"a": 0.0}))

    envs = {step: {env for env, _ in draws(source, step, 40)} for step in (1, 2, 3, 4)}

    assert envs == {1: {"a", "b"}, 2: {"a"}, 3: {"a", "b"}, 4: {"b"}}


def test_weights_multiply_task_weights_and_zero_excludes(tmp_path):
    source = planned_source(tmp_path)
    a, b = source.tasks["a"], source.tasks["b"]
    source.on_result(make_group(b[0], env="b", step=1, rewards=[1.0]))
    source.on_result(make_group(b[1], env="b", step=1, rewards=[0.0]))
    only_a, only_b = {"a": 1.0}, {"b": 1.0}
    plans = {
        1: Plan(sources=only_a, weights={"label:kind=write": 0.0}),
        2: Plan(sources=only_a, weights={f"task:{a[1].key}": 1000.0}),
        3: Plan(sources=only_a, weights={"label:fast": 0.0}),
        4: Plan(sources=only_b, weights={"pool:easy": 0.0}),
        5: Plan(sources=only_b, weights={"pool:unseen": 0.0}),
        6: Plan(weights={"label:kind=write": 0.0, "label:kind=read": 0.0}),
    }
    for step, plan in plans.items():
        source.plans.set(step, plan)

    counts = {step: Counter(key for _, key in draws(source, step, 200)) for step in plans}

    assert set(counts[1]) == {a[1].key, a[3].key}
    assert counts[2][a[1].key] > 180
    assert set(counts[3]) == {a[1].key, a[2].key, a[3].key}
    assert set(counts[4]) == {b[1].key, b[2].key, b[3].key}
    assert set(counts[5]) == {b[0].key, b[1].key}
    assert set(counts[6]) == {task.key for task in b}


def test_validation_rejects_unknown_names_and_passed_steps(tmp_path):
    source = planned_source(tmp_path)
    plans = source.plans
    invalid = [
        Plan(),
        Plan(tasks=[PlanTask(env="c", key="x")]),
        Plan(tasks=[PlanTask(env="a", key="missing")]),
        Plan(sources={"c": 1.0}),
        Plan(sources={"others": 1.0}),
        Plan(sources={"a": 0.7, "b": 0.6}),
        Plan(sources={"a": 0.2, "b": 0.3}),
        Plan(sources={"a": 1.0}, weights={"label:kind=write": 0.0, "label:kind=read": 0.0}),
        Plan(weights={"task:missing": 2.0}),
        Plan(weights={"label:slow": 2.0}),
        Plan(weights={"pool:medium": 2.0}),
        Plan(weights={"rung:R2": 2.0}),
    ]
    for plan in invalid:
        with pytest.raises(PlanError):
            plans.set(5, plan)
    with pytest.raises(ValidationError):
        Plan(weights={"label:fast": -1.0})
    with pytest.raises(ValidationError):
        Plan(sources={"a": 1.5})

    plan = Plan(sources={"a": 1.0})
    plans.set(3, plan)
    draws(source, 3, 1)
    for step in (2, 3):
        with pytest.raises(PlanConflictError):
            plans.set(step, plan)
    with pytest.raises(PlanConflictError):
        plans.delete(3)
    plans.set(4, plan)
    plans.set("default", plan)
    assert [(entry["step"], entry["state"]) for entry in plans.all()] == [
        ("default", "default"),
        (3, "used"),
        (4, "pending"),
    ]


def test_plans_records_and_stats_resume_up_to_their_step(tmp_path):
    source = planned_source(tmp_path)
    task = source.tasks["a"][0]
    source.plans.set("default", Plan(sources={"a": 1.0}))
    source.plans.set(2, Plan(tasks=[PlanTask(env="a", key=task.key)]))
    source.plans.set(9, Plan(weights={"label:fast": 0.0}))
    for step in (1, 2, 3):
        draws(source, step, 2)
        source.on_result(make_group(task, env="a", step=step, rewards=[0.0, 1.0]))
    source.plans.close()
    assert [record["step"] for record in dispatched(tmp_path, "step")] == [1, 2, 3]

    resumed = planned_source(tmp_path, resume_step=2).plans

    assert [(entry["step"], entry["state"]) for entry in resumed.all()] == [
        ("default", "default"),
        (2, "used"),
        (9, "pending"),
    ]
    assert [record["step"] for record in dispatched(tmp_path, "step")] == [1, 2]
    assert resumed.status()["plan_steps"] == {"step": [[2, 2]], "default": [[1, 1]], "replay": []}
    assert resumed.stats()["tasks"] == [{"key": task.key, "env": "a", "reward": 0.5, "visits": 2, "labels": ["fast"]}]
    assert resumed.stats(step=3)["envs"] == {}
    with pytest.raises(PlanConflictError):
        resumed.set(2, Plan(sources={"a": 1.0}))

    fresh = planned_source(tmp_path).plans
    assert fresh.all() == []
    assert fresh.stats()["envs"] == {}
    assert read_jsonl(tmp_path / "dispatch.jsonl") == []
    assert [event["action"] for event in read_jsonl(tmp_path / "plans.jsonl")] == ["set"] * 3 + ["reset"]


def test_plan_routes(tmp_path):
    source = planned_source(tmp_path)
    app = create_app(Controller(requires_step=True), [plan_router(source.plans, lambda: {"trainer_step": 0})])
    client = TestClient(app)
    task = source.tasks["a"][0]
    plan = {"tasks": [{"env": "a", "key": task.key}]}

    response = client.post("/v1/plans", json={"step": 2, "plan": plan, "author": "claude", "note": "retry a0"})
    assert response.status_code == 200
    stored = response.json()
    assert stored.pop("posted_at")
    assert stored == {
        "step": 2,
        "id": 1,
        "plan": plan,
        "shares": {"a": 0.25, "b": 0.75},
        "state": "pending",
        "author": "claude",
        "note": "retry a0",
    }
    assert client.post("/v1/plans", json={"step": "default", "plan": {"sources": {"b": 1.0}}}).status_code == 200
    for invalid in ({"tasks": [{"env": "a", "key": "missing"}]}, {"weights": {"label:fast": -1}}, {"extra": 1}):
        assert client.post("/v1/plans", json={"step": 3, "plan": invalid}).status_code == 422
    assert [entry["step"] for entry in client.get("/v1/plans").json()] == ["default", 2]
    assert client.get("/v1/plans/2").json()["plan"] == plan
    assert client.get("/v1/plans/7").status_code == 404
    assert client.get("/v1/plans/next").status_code == 422

    draws(source, 1, 1)
    draws(source, 2, 2)
    assert client.delete("/v1/plans/2").status_code == 409
    assert client.delete("/v1/plans/default", params={"author": "claude"}).json()["step"] == "default"
    status = client.get("/v1/status").json()
    assert status["dispatch_step"] == 2
    assert status["dispatching"] == {"step": 2, "plan": 1, "origin": "step", "groups": 2, "tasks_left": 0}
    assert status["plan_steps"] == {"step": [[2, 2]], "default": [[1, 1]], "replay": []}
    assert status["trainer_step"] == 0

    source.on_result(make_group(task, env="a", step=2, rewards=[0.0, 1.0]))
    stats = client.get("/v1/stats", params={"step": 2}).json()
    assert stats["envs"]["a"] == {
        "groups": 1,
        "mixed_share": 1.0,
        "mean_reward": 0.5,
        "mean_turns": 1.0,
        "truncated_share": 0.0,
        "timeout_share": 0.0,
        "error_share": 0.0,
    }
    assert stats["tasks"] == [{"key": task.key, "env": "a", "reward": 0.5, "visits": 1, "labels": ["fast"]}]
    assert client.get("/v1/stats", params={"limit": 0, "env": "a"}).json()["tasks_total"] == 1
    assert client.post("/v1/actions", json={"type": "pause", "step": 5}).status_code == 202


def test_posting_for_a_started_step_conflicts_with_the_next_open_step(tmp_path):
    source = planned_source(tmp_path)
    client = TestClient(create_app(Controller(requires_step=True), [plan_router(source.plans)]))
    draws(source, 3, 1)

    response = client.post("/v1/plans", json={"step": 3, "plan": {"sources": {"a": 1.0}}})

    assert response.status_code == 409
    assert response.json()["current_dispatch_step"] == 3
    assert response.json()["next_open_step"] == 4


def test_next_takes_the_first_step_whose_dispatch_has_not_started(tmp_path):
    source = planned_source(tmp_path)
    plan = Plan(sources={"a": 1.0})

    assert source.plans.set("next", plan)["step"] == 1
    draws(source, 1, 1)
    draws(source, 2, 1)
    assert source.plans.set("next", plan)["step"] == 3

    assert planned_source(tmp_path / "resumed", resume_step=7).plans.set("next", plan)["step"] == 8


def test_plan_history_keeps_requested_and_resolved_steps_replacements_and_deletions(tmp_path):
    source = planned_source(tmp_path)
    source.plans.set("next", Plan(sources={"a": 1.0}), author="claude", note="only a")
    source.plans.set(1, Plan(sources={"b": 1.0}))
    source.plans.delete(1, note="changed my mind")

    events = [
        {name: event.get(name) for name in ("id", "action", "requested", "step", "replaces", "deletes")}
        for event in read_jsonl(tmp_path / "plans.jsonl")
    ]

    assert events == [
        {"id": 1, "action": "set", "requested": "next", "step": 1, "replaces": None, "deletes": None},
        {"id": 2, "action": "set", "requested": 1, "step": 1, "replaces": 1, "deletes": None},
        {"id": 3, "action": "delete", "requested": None, "step": 1, "replaces": None, "deletes": 2},
    ]
    assert read_jsonl(tmp_path / "plans.jsonl")[0]["note"] == "only a"


def test_replay_dispatches_the_recorded_tasks_per_step(tmp_path):
    recorded = planned_source(tmp_path / "recorded")
    recorded.plans.set(2, Plan(sources={"a": 1.0}, weights={"label:kind=write": 0.0}))
    recorded.plans.set(3, Plan(tasks=[PlanTask(env="b", key=recorded.tasks["b"][3].key)]))
    counts = {1: 5, 2: 7, 3: 4}
    schedule = {step: draws(recorded, step, n) for step, n in counts.items()}
    recorded.plans.close()

    replay = planned_source(tmp_path / "replay", replay_from=tmp_path / "recorded" / "dispatch.jsonl")

    assert {step: draws(replay, step, n) for step, n in counts.items()} == schedule
    draws(replay, 4, 2)
    assert [record["origin"] for record in dispatched(tmp_path / "replay")] == ["replay"] * 16 + [None] * 2
    with pytest.raises(PlanConflictError):
        replay.plans.set("next", Plan(sources={"a": 1.0}))


def test_stats_give_each_task_its_labels(tmp_path):
    source = planned_source(tmp_path)
    a = source.tasks["a"]
    for task in (a[1], a[2]):
        source.on_result(make_group(task, env="a", step=1, rewards=[1.0]))

    assert source.plans.stats()["tasks"] == [
        {"key": a[2].key, "env": "a", "reward": 1.0, "visits": 1, "labels": ["hard", "hops:1"]},
        {"key": a[1].key, "env": "a", "reward": 1.0, "visits": 1},
    ]


def test_stats_aggregate_groups_by_label(tmp_path):
    source = planned_source(tmp_path)
    client = TestClient(create_app(Controller(requires_step=True), [plan_router(source.plans)]))
    a = source.tasks["a"]
    source.on_result(make_group(a[2], env="a", step=1, rewards=[0.0, 1.0]))
    source.on_result(make_group(a[2], env="a", step=2, rewards=[1.0, 1.0]))
    source.on_result(make_group(a[0], env="a", step=2, rewards=[0.0, 0.0]))

    top = client.get("/v1/stats").json()["labels"]
    chosen = client.get("/v1/stats", params=[("label", "hops:1"), ("label", "slow")]).json()

    assert top == {
        "hard": {"groups": 2, "mixed_share": 0.5, "mean_reward": 0.75},
        "hops:1": {"groups": 2, "mixed_share": 0.5, "mean_reward": 0.75},
        "fast": {"groups": 1, "mixed_share": 0.0, "mean_reward": 0.0},
    }
    assert chosen["labels"] == {
        "hops:1": {"groups": 2, "mixed_share": 0.5, "mean_reward": 0.75},
        "slow": {"groups": 0, "mixed_share": None, "mean_reward": None},
    }
    assert chosen["tasks"] == []
    assert [task["key"] for task in client.get("/v1/stats", params={"label": "hard"}).json()["tasks"]] == [a[2].key]
    assert client.get("/v1/stats", params={"step": 2, "top_labels": 1}).json()["labels"] == {
        "fast": {"groups": 1, "mixed_share": 0.0, "mean_reward": 0.0}
    }

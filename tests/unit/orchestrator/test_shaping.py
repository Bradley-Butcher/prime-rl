import asyncio
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from fastapi.testclient import TestClient

from prime_rl.configs.algorithm import GRPOAlgoConfig, RewardShapingConfig, StuckPenaltyConfig
from prime_rl.configs.orchestrator import AdvRangeGateConfig, OrchestratorConfig
from prime_rl.control.app import create_app
from prime_rl.control.controller import Controller
from prime_rl.control.plans import plan_router
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.curriculum.gates import AdvRangeGate
from prime_rl.orchestrator.curriculum.plans import PlanConflictError, Plans
from prime_rl.orchestrator.curriculum.shaping import Shaping, ShapingRequest
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Progress

STUCK_ON = {"stuck": {"per_call": 0.05}}
DEFAULT = RewardShapingConfig().model_dump(mode="json")


def rollout(reward: float, *, stuck: int = 0, stop: str | None = None, step: int = 1, group: str = "g") -> vf.Episode:
    """One train rollout of env ``a``, dispatched at ``step``, with ``stuck`` stuck calls."""
    task = vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None), key="t0")
    nodes = [
        vf.MessageNode(message=vf.UserMessage(content="q"), token_ids=[1], mask=[False]),
        vf.MessageNode(
            parent=0,
            message=vf.AssistantMessage(content="a"),
            token_ids=[2, 3],
            mask=[True, True],
            logprobs=[-0.1, -0.1],
            sampled=True,
        ),
    ]
    trace = vf.Trace(
        task=task,
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        calls=[vf.ModelCall(node=1, usage=vf.Usage(prompt_tokens=1, completion_tokens=2), finish_reason="stop")],
        rewards={"passed": vf.Reward(score=reward)},
        metrics={"stuck_calls": float(stuck)},
        stop_condition=stop,
        is_timeout=stop == "agent_timeout",
        is_completed=True,
        ok=True,
    )
    return vf.Episode(
        env=vf.EnvInfo(id="a", name="a"),
        task=task,
        group=vf.GroupInfo(id=group),
        traces=[trace],
        ok=True,
        run=vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=step, policy=vf.PolicySpan(start=0, end=0))),
    )


def advantages(algorithm: GRPOAlgorithm, group: list[vf.Episode]) -> list[float]:
    """Score ``group`` as the sink does, its timeouts settled first, and read back each rollout's advantage."""
    for episode in group:
        algorithm.settle_timeouts(episode)
    asyncio.run(algorithm.score_group(group))
    return [episode.traces[0].nodes[1].advantages[0] for episode in group]


def mixed_group(step: int = 1, group: str = "g") -> list[vf.Episode]:
    """Two passes, one with 4 stuck calls; a rollout the turn cap cut off that Ocean passed; one that timed out."""
    return [
        rollout(1.0, step=step, group=group),
        rollout(1.0, stuck=4, step=step, group=group),
        rollout(1.0, stop="max_turns", step=step, group=group),
        rollout(1.0, stop="agent_timeout", step=step, group=group),
    ]


UNSHAPED = [0.25, 0.25, 0.25, -0.75]  # rewards 1, 1, 1 and the timeout's 0
SHAPED = [1.025, 0.925, -0.975, -0.975]  # rewards 1, 0.9, -1, -1


def live(directory, *, replay_from=None, resume_step=None):
    """A GRPO env ``a`` with timeouts trained as failures, its plans and its live shaping."""
    tasks = [vf.Task(vf.WireTaskData(idx=i, prompt="a")) for i in range(2)]
    env = SimpleNamespace(name="a", tasks=iter(tasks), num_tasks=2, config=SimpleNamespace(ratio=1.0, curriculum=None))
    source = TrainSource([env])
    source.plans = Plans(source, directory, resume_step=resume_step, replay_from=replay_from)
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(timeouts_as_failures=True), None)
    shaping = Shaping(source.plans, {"a": algorithm}, directory, resume_step=resume_step, replay_from=replay_from)
    return source, algorithm, shaping


def post(shaping: Shaping, step, **change) -> dict:
    return shaping.set(ShapingRequest(step=step, **change))


def test_by_default_a_group_trains_on_the_env_s_scores():
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(timeouts_as_failures=True), None)

    assert advantages(algorithm, mixed_group()) == pytest.approx(UNSHAPED)


def test_the_stuck_penalty_cuts_passes_up_to_its_cap_and_leaves_failures_at_0():
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(shaping={"stuck": {"per_call": 0.05}}), None)
    group = [rollout(1.0), rollout(1.0, stuck=4), rollout(1.0, stuck=20), rollout(0.0, stuck=9)]

    # Rewards 1, 1 - 0.05 * 2, 1 - min(0.3, 0.05 * 18), and 0: their mean is 0.65.
    assert advantages(algorithm, group) == pytest.approx([0.35, 0.25, 0.05, -0.65])
    assert StuckPenaltyConfig(per_call=0.05).factor(2) == 1.0


@pytest.mark.parametrize("shaping", [{}, STUCK_ON, {"unfinished": {"reward": -1.0}}])
def test_the_zero_advantage_gate_sees_the_shaped_rewards(shaping):
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(shaping=shaping), None)
    gate = AdvRangeGate(AdvRangeGateConfig())
    sink = TrainSink(
        OrchestratorConfig(batch_size=100),
        tokenizer=None,
        train_envs=SimpleNamespace(
            get=lambda name: SimpleNamespace(
                config=SimpleNamespace(group_size=2, sampling=SimpleNamespace(temperature=1.0)),
                requires_sampling_masks=False,
                algorithm=algorithm,
            )
        ),
        progress=Progress(),
        batch_size=100,
        on_result=gate.admit,
    )
    groups = {
        "passes with and without stuck calls": [rollout(1.0, group="p"), rollout(1.0, stuck=6, group="p")],
        "failures, one cut off": [rollout(0.0, group="f"), rollout(0.0, stop="max_turns", group="f")],
    }
    for group in groups.values():
        for episode in group:
            asyncio.run(sink.add(episode))
    trained = {name for name, group in groups.items() if group[0].traces[0].id in sink.pending_batch}

    assert (
        trained
        == {
            "{}": set(),
            json.dumps(STUCK_ON): {"passes with and without stuck calls"},
            json.dumps({"unfinished": {"reward": -1.0}}): {"failures, one cut off"},
        }[json.dumps(shaping)]
    )


def test_a_change_applies_from_its_step_on_and_never_to_groups_already_dispatched(tmp_path):
    source, algorithm, shaping = live(tmp_path)
    source.next_task(step=1)
    in_flight = mixed_group(step=1, group="in-flight")

    change = post(shaping, "next", stuck={"per_call": 0.05}, unfinished={"reward": -1.0}, author="me", note="loops")
    source.next_task(step=2)
    source.next_task(step=3)

    assert change["step"] == 2 and change["state"] == "pending"
    assert advantages(algorithm, in_flight) == pytest.approx(UNSHAPED)
    assert advantages(algorithm, mixed_group(step=2, group="at-2")) == pytest.approx(SHAPED)
    assert advantages(algorithm, mixed_group(step=7, group="at-7")) == pytest.approx(SHAPED)

    post(shaping, 4, unfinished={"reward": None})  # unfinished rollouts keep the env's score again from step 4
    at_4 = advantages(algorithm, mixed_group(step=4, group="at-4"))
    assert at_4 == pytest.approx([0.275, 0.175, 0.275, -0.725])  # rewards 1, 0.9, 1 and the timeout's 0

    source.on_result(in_flight)
    source.on_result(mixed_group(step=2, group="at-2"))
    recorded = [json.loads(line) for line in (tmp_path / "groups.jsonl").read_text().splitlines()]
    shaped = {"stuck": dict(DEFAULT["stuck"], per_call=0.05), "unfinished": {"reward": -1.0}}
    assert [(line["shaping"], line["stuck_penalized"], line["unfinished_scored"]) for line in recorded] == [
        (DEFAULT, 0, 0),
        (shaped, 1, 2),
    ]


def test_the_api_refuses_a_started_step_and_bad_changes_and_shows_the_setting(tmp_path):
    source, _, shaping = live(tmp_path)
    client = TestClient(create_app(Controller(requires_step=True), [plan_router(source.plans, shaping=shaping)]))
    source.next_task(step=1)

    started = client.post("/v1/shaping", json={"step": 1, **STUCK_ON})
    assert started.status_code == 409
    assert (started.json()["current_dispatch_step"], started.json()["next_open_step"]) == (1, 2)
    for bad in (
        {"stuck": {"per_call": -0.1}},
        {"stuck": {"per_call": None}},
        {"stuck": {"cap": 2}},
        {"stuck": {"bogus": 1}},
        {"stuck": {}},
        {},
    ):
        assert client.post("/v1/shaping", json={"step": 3, **bad}).status_code == 422, bad
    assert client.post("/v1/shaping", json={"step": "default", **STUCK_ON}).status_code == 422

    posted = client.post("/v1/shaping", json={"step": "next", "author": "me", "note": "loops", **STUCK_ON})
    assert posted.status_code == 200
    pending = {"id": 1, "step": 2, "requested": "next", "change": STUCK_ON, "state": "pending", "author": "me"}
    assert posted.json().items() >= pending.items()
    assert posted.json()["shaping"] == [{"envs": ["a"], **DEFAULT, "stuck": dict(DEFAULT["stuck"], per_call=0.05)}]
    status = client.get("/v1/status").json()["shaping"]
    assert status["active"] == [{"envs": ["a"], **DEFAULT}]
    assert [change["step"] for change in status["pending"]] == [2]
    assert client.get("/v1/shaping").json()["dispatch_step"] == 1


def test_a_resume_reads_the_changes_back_and_a_replay_repeats_them(tmp_path):
    recorded, _, shaping = live(tmp_path / "recorded")
    recorded.next_task(step=1)
    post(shaping, "next", **STUCK_ON)
    recorded.next_task(step=2)
    recorded.next_task(step=3)
    post(shaping, 5, unfinished={"reward": -1.0})
    schedule = [shaping.at("a", step) for step in range(1, 8)]

    _, _, resumed = live(tmp_path / "recorded", resume_step=3)
    replay, algorithm, replayed = live(tmp_path / "replay", replay_from=tmp_path / "recorded" / "dispatch.jsonl")
    replay.next_task(step=1)

    assert [resumed.at("a", step) for step in range(1, 8)] == schedule
    assert [replayed.at("a", step) for step in range(1, 8)] == schedule
    assert advantages(algorithm, mixed_group(step=6)) == pytest.approx(SHAPED)
    with pytest.raises(PlanConflictError):
        post(replayed, "next", stuck={"per_call": 0.1})
    # The replaying run records what it replays, so it can be replayed in turn.
    _, _, again = live(tmp_path / "again", replay_from=tmp_path / "replay" / "dispatch.jsonl")
    assert [again.at("a", step) for step in range(1, 8)] == schedule

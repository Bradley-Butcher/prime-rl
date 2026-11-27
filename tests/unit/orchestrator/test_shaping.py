import asyncio
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from fastapi.testclient import TestClient

from prime_rl.configs.algorithm import (
    GRPOAlgoConfig,
    PassTurnsPenaltyConfig,
    RewardShapingConfig,
    StuckPenaltyConfig,
)
from prime_rl.configs.orchestrator import AdvRangeGateConfig, CurriculumConfig, DownsampleConfig, OrchestratorConfig
from prime_rl.control.app import create_app
from prime_rl.control.controller import Controller
from prime_rl.control.plans import plan_router
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm, shape_rewards
from prime_rl.orchestrator.curriculum.gates import AdvRangeGate
from prime_rl.orchestrator.curriculum.plans import PlanConflictError, Plans
from prime_rl.orchestrator.curriculum.shaping import Shaping, ShapingRequest
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Progress

STUCK_ON = {"stuck": {"per_call": 0.05}}
DEFAULT = RewardShapingConfig().model_dump(mode="json")


def rollout(
    reward: float, *, stuck: int = 0, stop: str | None = None, step: int = 1, group: str = "g", turns: int = 1
) -> vf.Episode:
    """One train rollout of env ``a``, dispatched at ``step``, with ``stuck`` stuck calls and ``turns`` replies, each
    after a user message."""
    task = vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None), key="t0")
    nodes, calls = [], []
    for _ in range(turns):
        parent = len(nodes) - 1 if nodes else None
        nodes.append(vf.MessageNode(parent=parent, message=vf.UserMessage(content="q"), token_ids=[1], mask=[False]))
        calls.append(
            vf.ModelCall(node=len(nodes), usage=vf.Usage(prompt_tokens=1, completion_tokens=2), finish_reason="stop")
        )
        nodes.append(
            vf.MessageNode(
                parent=len(nodes) - 1,
                message=vf.AssistantMessage(content="a"),
                token_ids=[2, 3],
                mask=[True, True],
                logprobs=[-0.1, -0.1],
                sampled=True,
            )
        )
    trace = vf.Trace(
        task=task,
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        calls=calls,
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
    """Two passes, one with 4 stuck calls; a failed rollout the turn cap cut off; one that timed out."""
    return [
        rollout(1.0, step=step, group=group),
        rollout(1.0, stuck=4, step=step, group=group),
        rollout(0.0, stop="max_turns", step=step, group=group),
        rollout(1.0, stop="agent_timeout", step=step, group=group),
    ]


UNSHAPED = [0.5, 0.5, -0.5, -0.5]  # rewards 1, 1, 0 and the timeout's 0
SHAPED = [1.025, 0.925, -0.975, -0.975]  # rewards 1, 0.9, -1, -1


def live(directory, *, replay_from=None, resume_step=None):
    """A GRPO env ``a`` with timeouts trained as failures, its plans and its live shaping."""
    tasks = [vf.Task(vf.WireTaskData(idx=i, id=f"a-{i}", prompt="a")) for i in range(2)]
    env = SimpleNamespace(
        name="a", tasks=iter(tasks), num_tasks=2, config=SimpleNamespace(ratio=1.0, group_size=1, curriculum=None)
    )
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


def gated(algorithm: GRPOAlgorithm, groups: dict[str, list[vf.Episode]]) -> tuple[TrainSink, set[str]]:
    """Feed ``groups``, all of one size, to a train sink with the zero-advantage gate; the sink and the groups that
    train."""
    gate = AdvRangeGate(AdvRangeGateConfig())
    size = len(next(iter(groups.values())))
    sink = TrainSink(
        OrchestratorConfig(batch_size=100),
        tokenizer=None,
        train_envs=SimpleNamespace(
            get=lambda name: SimpleNamespace(
                config=SimpleNamespace(group_size=size, rollouts_per_task=size, downsample=None),
                sampling_at=lambda step: vf.SamplingConfig(temperature=1.0),
                requires_sampling_masks=False,
                requires_sampler_logprobs=False,
                algorithm=algorithm,
            )
        ),
        progress=Progress(),
        batch_size=100,
        on_result=gate.admit,
    )
    for group in groups.values():
        for episode in group:
            asyncio.run(sink.add(episode))
    return sink, {name for name, group in groups.items() if group[0].traces[0].id in sink.pending_batch}


@pytest.mark.parametrize("shaping", [{}, STUCK_ON, {"unfinished": {"reward": -1.0}}])
def test_the_zero_advantage_gate_sees_the_shaped_rewards(shaping):
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(shaping=shaping), None)
    groups = {
        "passes with and without stuck calls": [rollout(1.0, group="p"), rollout(1.0, stuck=6, group="p")],
        "failures, one cut off": [rollout(0.0, group="f"), rollout(0.0, stop="max_turns", group="f")],
    }
    _, trained = gated(algorithm, groups)

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
    at_2 = mixed_group(step=2, group="at-2")
    assert advantages(algorithm, at_2) == pytest.approx(SHAPED)
    assert advantages(algorithm, mixed_group(step=7, group="at-7")) == pytest.approx(SHAPED)

    post(shaping, 4, unfinished={"reward": None})  # unfinished rollouts keep the env's score again from step 4
    at_4 = advantages(algorithm, mixed_group(step=4, group="at-4"))
    assert at_4 == pytest.approx([0.525, 0.425, -0.475, -0.475])  # rewards 1, 0.9, 0 and the timeout's 0

    source.on_result(in_flight)
    source.on_result(at_2)
    recorded = [json.loads(line) for line in (tmp_path / "groups.jsonl").read_text().splitlines()]
    shaped = {
        **DEFAULT,
        "stuck": dict(DEFAULT["stuck"], per_call=0.05),
        "unfinished": dict(DEFAULT["unfinished"], reward=-1.0),
    }
    assert [(line["shaping"], line["stuck_penalized"], line["unfinished_scored"]) for line in recorded] == [
        (DEFAULT, 0, 0),
        (shaped, 1, 2),
    ]


def test_the_api_refuses_a_started_step_and_bad_changes_and_shows_the_setting(tmp_path):
    source, _, shaping = live(tmp_path)
    client = TestClient(
        create_app(Controller(requires_step=True), routers=[plan_router(source.plans, shaping=shaping)])
    )
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


TURNS_ON = {"pass_turns": {"weight": 0.2}}


def shaped(shaping: dict, group: list[vf.Episode]) -> list[tuple[float, tuple[str, ...]]]:
    """``group``'s shaped rewards and the levers that changed each, its timeouts settled first as the sink does."""
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(timeouts_as_failures=True, shaping=shaping), None)
    for episode in group:
        algorithm.settle_timeouts(episode)
    return shape_rewards([episode.traces[0] for episode in group], algorithm.shaping)


def turns_group(step: int = 1, group: str = "g") -> list[vf.Episode]:
    """Passes in 2, 4 and 6 turns, and a failure in 3."""
    return [rollout(1.0 if turns % 2 == 0 else 0.0, turns=turns, step=step, group=group) for turns in (2, 4, 6, 3)]


UNPENALIZED = [0.25, 0.25, 0.25, -0.75]  # rewards 1, 1, 1 and 0
PENALIZED = [0.325, 0.225, 0.125, -0.675]  # rewards 1, 0.9, 0.8 and 0


def trained_advantage(sample) -> float:
    (advantage,) = {advantage for advantage, trains in zip(sample.advantages, sample.mask) if trains}
    return advantage


def test_the_turn_penalty_cuts_each_pass_by_its_turns_among_its_group_s_passes_and_never_a_failure():
    group = [rollout(1.0, turns=4), rollout(1.0, turns=6), rollout(1.0, turns=10), rollout(0.0, turns=2)]
    group.append(rollout(0.0, turns=30))

    # The passes took 4 to 10 turns: the quickest keeps 1, the slowest 1 - 0.2, the one between 1 - 0.2 * 2 / 6. The
    # failures, one quicker and one slower than every pass, stay 0 and set no bound.
    assert shaped(TURNS_ON, group) == [
        (1.0, ()),
        (pytest.approx(1 - 0.2 / 3), ("pass_turns",)),
        (pytest.approx(0.8), ("pass_turns",)),
        (0.0, ()),
        (0.0, ()),
    ]
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(shaping=TURNS_ON), None)
    mean = (1 + (1 - 0.2 / 3) + 0.8) / 5
    assert advantages(algorithm, group) == pytest.approx([1 - mean, 1 - 0.2 / 3 - mean, 0.8 - mean, -mean, -mean])
    assert PassTurnsPenaltyConfig(weight=0.2).factor(7, 4, 10) == pytest.approx(0.9)
    assert PassTurnsPenaltyConfig(weight=0.2).factor(7, 7, 7) == 1.0
    assert RewardShapingConfig().pass_turns.weight == 0.0  # off by default


@pytest.mark.parametrize(
    "shaping, rewards, turns",
    [
        (TURNS_ON, [1.0, 0.0, 0.0], [3, 1, 9]),
        (TURNS_ON, [1.0, 1.0, 0.0], [5, 5, 2]),
        (TURNS_ON, [0.0, 0.0], [5, 9]),
        ({}, [1.0, 1.0], [2, 8]),
    ],
    ids=["one pass", "passes of equal turns", "no pass", "off"],
)
def test_one_pass_or_passes_of_equal_turns_are_unchanged(shaping, rewards, turns):
    group = [rollout(reward, turns=count) for reward, count in zip(rewards, turns)]

    assert shaped(shaping, group) == [(reward, ()) for reward in rewards]


@pytest.mark.parametrize("unfinished", [-1.0, None])
def test_the_turn_penalty_multiplies_the_stuck_one_and_leaves_unfinished_failures_to_their_lever(unfinished):
    shaping = {"stuck": {"per_call": 0.05}, "unfinished": {"reward": unfinished}, "pass_turns": {"weight": 0.3}}
    group = [
        rollout(1.0, turns=2),
        rollout(1.0, turns=4, stuck=4),
        rollout(1.0, turns=3, stuck=4),
        rollout(1.0, turns=10, stop="max_turns"),  # the turn cap cut it off, but the env verified a pass
        rollout(0.0, turns=10, stop="max_turns"),  # unfinished failure
        rollout(1.0, turns=1, stop="agent_timeout"),  # a failure scored 0
        rollout(0.0, turns=1),
    ]

    stuck_and_turns = ("stuck", "pass_turns")
    # The cut-off pass always keeps its pass (slowest of passes spanning 2 to 10 turns).
    # Only unfinished failures (failed max_turns and agent_timeout) take unfinished.reward.
    unfinished_entry = (-1.0, ("unfinished",)) if unfinished is not None else (0.0, ())
    assert shaped(shaping, group) == [
        (1.0, ()),
        (pytest.approx(0.9 * (1 - 0.3 * 2 / 8)), stuck_and_turns),
        (pytest.approx(0.9 * (1 - 0.3 / 8)), stuck_and_turns),
        (pytest.approx(0.7), ("pass_turns",)),
        unfinished_entry,
        unfinished_entry,
        (0.0, ()),
    ]


@pytest.mark.parametrize("gate_on_raw", [True, False])
def test_by_default_the_gate_judges_a_group_without_the_turn_penalty(gate_on_raw):
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(shaping={"pass_turns": {"weight": 0.2, "gate_on_raw": gate_on_raw}}), None)
    groups = {
        "passes that differ in turns": [rollout(1.0, turns=turns, group="p") for turns in (2, 3, 6)],
        "two passes and a failure": [rollout(1.0, turns=turns, group="m") for turns in (2, 6)],
    }
    groups["two passes and a failure"].append(rollout(0.0, turns=4, group="m"))

    sink, trained = gated(algorithm, groups)

    assert trained == ({"two passes and a failure"} if gate_on_raw else set(groups))
    # Either way a group is scored with the penalty, which is what a trained group trains with: rewards 1, 0.95 and
    # 0.8, and 1, 0.8 and 0.
    scored = {"passes that differ in turns": [1 / 12, 1 / 30, -7 / 60], "two passes and a failure": [0.4, 0.2, -0.6]}
    for name, group in groups.items():
        assert [episode.traces[0].nodes[1].advantages[0] for episode in group] == pytest.approx(scored[name], abs=1e-6)
        for episode, advantage in zip(group, scored[name]):
            for sample in sink.pending_batch.get(episode.traces[0].id, []):
                assert trained_advantage(sample) == pytest.approx(advantage, abs=1e-6)


def test_free_turns_mark_nothing_down_within_them_and_the_slowest_pass_still_loses_the_weight():
    turns = PassTurnsPenaltyConfig(weight=0.5, free=2)
    assert [turns.factor(count, 10, 14) for count in (10, 11, 12, 13, 14)] == pytest.approx([1, 1, 1, 0.75, 0.5])
    assert [turns.factor(count, 10, 12) for count in (10, 11, 12)] == [1.0, 1.0, 1.0]  # a spread within free
    assert PassTurnsPenaltyConfig(weight=0.5).factor(11, 10, 11) == 0.5  # without free, one turn is the whole weight


def test_with_the_gate_on_the_penalty_an_all_pass_group_trains_on_turns_past_free_alone():
    shaping = {"pass_turns": {"weight": 0.5, "gate_on_raw": False, "free": 2}}
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(shaping=shaping), None)
    groups = {
        "passes a turn or two apart": [rollout(1.0, turns=turns, group="n") for turns in (10, 11, 12)],
        "passes past free": [rollout(1.0, turns=turns, group="f") for turns in (10, 12, 14)],
    }

    _, trained = gated(algorithm, groups)

    assert trained == {"passes past free"}
    # Rewards 1, 1 and 0.5: the quickest two within free of each other.
    advantages = [episode.traces[0].nodes[1].advantages[0] for episode in groups["passes past free"]]
    assert advantages == pytest.approx([1 / 6, 1 / 6, -1 / 3], abs=1e-6)


def downsampled(directory, baseline: str) -> tuple[TrainSink, TrainSource]:
    """A train sink over env ``a`` rolling out 12 rollouts a Task and training 8, with the zero-advantage gate, the
    curriculum records and live shaping, the turn penalty at 0.5."""
    tasks = [vf.Task(vf.WireTaskData(idx=i, id=f"a-{i}", prompt="a")) for i in range(2)]
    curriculum = CurriculumConfig(gates={"zero_advantage": AdvRangeGateConfig()})
    env = SimpleNamespace(
        name="a",
        tasks=iter(tasks),
        num_tasks=2,
        config=SimpleNamespace(ratio=1.0, group_size=8, rollouts_per_task=12, curriculum=curriculum),
    )
    source = TrainSource([env])
    source.plans = Plans(source, directory, resume_step=None)
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(loss_aggregation="prompt", shaping={"pass_turns": {"weight": 0.5}}), None)
    Shaping(source.plans, {"a": algorithm}, directory, resume_step=None)
    config = SimpleNamespace(
        group_size=8,
        rollouts_per_task=12,
        downsample=DownsampleConfig(rollouts=12, baseline=baseline),
    )
    sink = TrainSink(
        OrchestratorConfig(batch_size=8),
        tokenizer=None,
        train_envs=SimpleNamespace(
            get=lambda name: SimpleNamespace(
                config=config,
                sampling_at=lambda step: vf.SamplingConfig(temperature=1.0),
                requires_sampling_masks=False,
                requires_sampler_logprobs=False,
                algorithm=algorithm,
            )
        ),
        progress=Progress(),
        batch_size=100,
        on_result=source.on_result,
        on_downsample=source.plans.record_downsample,
    )
    return sink, source


@pytest.mark.parametrize("baseline", ["all", "kept"])
def test_the_turn_penalty_ranges_over_the_rollouts_the_baseline_is_taken_over(tmp_path, baseline):
    sink, _ = downsampled(tmp_path, baseline)
    group = [rollout(1.0, turns=turns, group="mixed") for turns in (2, 4, 6, 8, 10, 12)]
    group += [rollout(0.0, turns=turns, group="mixed") for turns in (1, 3, 5, 7, 9, 11)]
    passes = [rollout(1.0, turns=turns, group="passes") for turns in range(2, 14)]

    for episode in group + passes:
        asyncio.run(sink.add(episode))

    traces = {episode.traces[0].id: episode.traces[0] for episode in group}
    kept = [traces[trace_id] for trace_id in sink.pending_batch]
    # Shaped advantages [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0, ...] have max variance at top-3 fastest passes + 5 failures.
    assert len(kept) == 8 and sum(trace.reward for trace in kept) == 3
    assert sorted(trace.num_turns for trace in kept if trace.reward) == [2, 4, 6]
    if baseline == "all":
        # Each pass against all six passes' 2 to 12 turns, minus the mean of all twelve, (1 + 0.9 + ... + 0.5) / 12.
        rewards = {trace.id: 1 - 0.5 * (trace.num_turns - 2) / 10 if trace.reward else 0.0 for trace in kept}
        mean = 4.5 / 12
    else:
        # Against the kept passes' own fewest and most turns, minus the kept eight's mean.
        turns = [trace.num_turns for trace in kept if trace.reward]
        fewest, most = min(turns), max(turns)
        rewards = {
            trace.id: 1 - 0.5 * (trace.num_turns - fewest) / (most - fewest) if trace.reward else 0.0 for trace in kept
        }
        mean = sum(rewards.values()) / 8
    for trace_id, samples in sink.pending_batch.items():
        assert [trained_advantage(sample) for sample in samples] == pytest.approx(
            [rewards[trace_id] - mean] * len(samples)
        )

    # groups.jsonl counts the passes the penalty changed over all twelve. The twelve passes differ only in turns: the
    # gate judges them without the penalty, so they do not train.
    mixed, uniform = [json.loads(line) for line in (tmp_path / "groups.jsonl").read_text().splitlines()]
    assert (mixed["pass_turns_penalized"], mixed["shaping"]["pass_turns"]) == (
        5,
        {"weight": 0.5, "free": 0, "scale": None, "gate_on_raw": True},
    )
    assert mixed["kept"] == {"rollouts": 8, "reward": 0.375, "successes": 3}
    assert (uniform["pass_turns_penalized"], uniform["kept"]) == (11, None)
    records = [json.loads(line) for line in (tmp_path / "downsample.jsonl").read_text().splitlines()]
    assert [record["admitted"] for record in records] == [True, False]
    assert {entry["trace"]: entry["advantage"] for entry in records[0]["rollouts"] if entry["kept"]} == pytest.approx(
        {trace_id: rewards[trace_id] - mean for trace_id in sink.pending_batch}
    )


def test_the_turn_penalty_is_set_live_from_a_step_on_as_the_other_levers_are(tmp_path):
    source, algorithm, shaping = live(tmp_path)
    client = TestClient(
        create_app(Controller(requires_step=True), routers=[plan_router(source.plans, shaping=shaping)])
    )
    source.next_task(step=1)
    in_flight = turns_group(step=1, group="in-flight")

    started = client.post("/v1/shaping", json={"step": 1, **TURNS_ON})
    assert started.status_code == 409 and started.json()["next_open_step"] == 2
    for bad in ({"weight": 1}, {"weight": -0.1}, {"weight": None}, {"gate_on_raw": None}, {"bogus": 1}, {}):
        assert client.post("/v1/shaping", json={"step": 3, "pass_turns": bad}).status_code == 422, bad
    posted = client.post("/v1/shaping", json={"step": "next", "author": "me", "note": "slow passes", **TURNS_ON})
    assert posted.status_code == 200
    assert (posted.json()["step"], posted.json()["change"], posted.json()["state"]) == (2, TURNS_ON, "pending")
    on = {"weight": 0.2, "free": 0, "scale": None, "gate_on_raw": True}
    assert posted.json()["shaping"] == [{"envs": ["a"], **DEFAULT, "pass_turns": on}]
    source.next_task(step=2)
    source.next_task(step=3)
    off = client.post("/v1/shaping", json={"step": 4, "pass_turns": {"weight": 0}})
    assert off.status_code == 200

    # Groups in flight keep the setting they were dispatched under; the change holds from its step to the next.
    assert advantages(algorithm, in_flight) == pytest.approx(UNPENALIZED)
    assert advantages(algorithm, turns_group(step=2)) == pytest.approx(PENALIZED)
    assert advantages(algorithm, turns_group(step=3)) == pytest.approx(PENALIZED)
    assert advantages(algorithm, turns_group(step=4)) == pytest.approx(UNPENALIZED)
    assert advantages(algorithm, turns_group(step=9)) == pytest.approx(UNPENALIZED)
    assert client.get("/v1/status").json()["shaping"]["active"] == [{"envs": ["a"], **DEFAULT, "pass_turns": on}]

    source.on_result(in_flight)
    source.on_result(turns_group(step=2, group="at-2"))
    recorded = [json.loads(line) for line in (tmp_path / "groups.jsonl").read_text().splitlines()]
    assert [(line["shaping"]["pass_turns"], line["pass_turns_penalized"]) for line in recorded] == [
        (DEFAULT["pass_turns"], 0),
        (on, 2),
    ]
    events = [json.loads(line) for line in (tmp_path / "shaping.jsonl").read_text().splitlines()]
    assert [(event["step"], event["change"]) for event in events] == [
        (2, TURNS_ON),
        (4, {"pass_turns": {"weight": 0.0}}),
    ]


def test_a_resume_and_a_replay_read_the_turn_penalty_back(tmp_path):
    recorded, _, shaping = live(tmp_path / "recorded")
    recorded.next_task(step=1)
    post(shaping, "next", **TURNS_ON)
    recorded.next_task(step=2)
    recorded.next_task(step=3)
    post(shaping, 5, pass_turns={"weight": 0.0})
    schedule = [shaping.at("a", step) for step in range(1, 8)]

    _, resumed_algorithm, resumed = live(tmp_path / "recorded", resume_step=3)
    _, replayed_algorithm, replayed = live(tmp_path / "replay", replay_from=tmp_path / "recorded" / "dispatch.jsonl")

    assert [setting.pass_turns.weight for setting in schedule] == [0.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
    assert [resumed.at("a", step) for step in range(1, 8)] == schedule
    assert [replayed.at("a", step) for step in range(1, 8)] == schedule
    assert advantages(resumed_algorithm, turns_group(step=4)) == pytest.approx(PENALIZED)
    assert advantages(resumed_algorithm, turns_group(step=5)) == pytest.approx(UNPENALIZED)
    assert advantages(replayed_algorithm, turns_group(step=2)) == pytest.approx(PENALIZED)


def test_an_absolute_turn_scale_cuts_a_turn_by_the_same_share_whatever_the_group_s_spread():
    relative = PassTurnsPenaltyConfig(weight=0.5, free=2)
    absolute = PassTurnsPenaltyConfig(weight=0.5, free=2, scale=8)
    # Passes at 7, 9 and 10 turns: by the group's spread, the one turn past 9 takes the whole weight; by turns, 1/8.
    assert [relative.factor(count, 7, 10) for count in (7, 9, 10)] == pytest.approx([1, 1, 0.5])
    assert [absolute.factor(count, 7, 10) for count in (7, 9, 10)] == pytest.approx([1, 1, 1 - 0.5 / 8])
    # A pass far past the quickest still loses the whole weight, and no more.
    assert absolute.factor(30, 7, 30) == pytest.approx(0.5)
    assert absolute.factor(13, 7, 30) == pytest.approx(1 - 0.5 * 4 / 8)
    # One pass, or passes within free of the quickest, are unchanged.
    group = [rollout(1.0, turns=7), rollout(1.0, turns=9), rollout(1.0, turns=10), rollout(0.0, turns=3)]
    assert shaped({"pass_turns": {"weight": 0.5, "free": 2, "scale": 8}}, group) == [
        (1.0, ()),
        (1.0, ()),
        (pytest.approx(1 - 0.5 / 8), ("pass_turns",)),
        (0.0, ()),
    ]


def test_unfinished_only_with_a_pass_leaves_a_group_without_one_to_the_env_s_scores():
    on = {"unfinished": {"reward": -0.25, "only_with_pass": True}}
    off = {"unfinished": {"reward": -0.25}}
    no_pass = [rollout(0.0), rollout(0.0), rollout(0.0, turns=5, stop="max_turns"), rollout(1.0, stop="agent_timeout")]
    # Without the switch, the finished failures beat the rollouts still looking: a give-up trains as better.
    assert [reward for reward, _ in shaped(off, no_pass)] == [0.0, 0.0, -0.25, -0.25]
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(timeouts_as_failures=True, shaping=off), None)
    assert advantages(algorithm, no_pass)[0] > 0
    # With it, a group without a finished pass keeps the env's scores: uniform, so no advantage for stopping.
    no_pass = [rollout(0.0), rollout(0.0), rollout(0.0, turns=5, stop="max_turns"), rollout(1.0, stop="agent_timeout")]
    assert shaped(on, no_pass) == [(0.0, ()), (0.0, ()), (0.0, ()), (0.0, ())]
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(timeouts_as_failures=True, shaping=on), None)
    assert advantages(algorithm, no_pass) == pytest.approx([0.0, 0.0, 0.0, 0.0])
    # A cut-off pass is not a finished one, so it keeps its pass where no finished rollout passed.
    cut_pass = [rollout(0.0), rollout(1.0, turns=9, stop="max_turns")]
    assert shaped(on, cut_pass) == [(0.0, ()), (1.0, ())]
    # A group with a finished pass penalizes unfinished failures, while a verified cut-off pass keeps its pass.
    with_pass = [
        rollout(1.0),
        rollout(0.0),
        rollout(0.0, turns=9, stop="max_turns"),
        rollout(1.0, turns=9, stop="max_turns"),
    ]
    assert shaped(on, with_pass) == shaped(off, with_pass) == [(1.0, ()), (0.0, ()), (-0.25, ("unfinished",)), (1.0, ())]


def test_downsample_uses_shaped_advantages_to_keep_turn_extremes_on_all_pass_groups(tmp_path):
    """When gate_on_raw=False admits an all-pass group shaped by pass_turns, PODS picks the fastest and slowest passes
    (guaranteed mixed advantages) rather than a random subset of raw 1.0 rewards."""
    tasks = [vf.Task(vf.WireTaskData(idx=0, id="a-0", prompt="a"))]
    curriculum = CurriculumConfig(gates={"zero_advantage": AdvRangeGateConfig()})
    env = SimpleNamespace(
        name="a",
        tasks=iter(tasks),
        num_tasks=1,
        config=SimpleNamespace(ratio=1.0, group_size=8, rollouts_per_task=12, curriculum=curriculum),
    )
    source = TrainSource([env])
    source.plans = Plans(source, tmp_path, resume_step=None)
    algorithm = GRPOAlgorithm(
        GRPOAlgoConfig(shaping={"pass_turns": {"weight": 0.5, "gate_on_raw": False}}),
        None,
    )
    config = SimpleNamespace(group_size=8, rollouts_per_task=12, downsample=DownsampleConfig(rollouts=12, baseline="kept"))
    sink = TrainSink(
        OrchestratorConfig(batch_size=100),
        tokenizer=None,
        train_envs=SimpleNamespace(
            get=lambda name: SimpleNamespace(
                config=config,
                sampling_at=lambda step: vf.SamplingConfig(temperature=1.0),
                requires_sampling_masks=False,
                requires_sampler_logprobs=False,
                algorithm=algorithm,
            )
        ),
        progress=Progress(),
        batch_size=100,
        on_result=source.on_result,
    )
    group = [rollout(1.0, turns=t, group="all-pass") for t in range(2, 14)]
    for ep in group:
        asyncio.run(sink.add(ep))
    traces = {ep.traces[0].id: ep.traces[0] for ep in group}
    kept_turns = sorted(traces[tid].num_turns for tid in sink.pending_batch)
    assert kept_turns == [2, 3, 4, 5, 10, 11, 12, 13]
    advs = [trained_advantage(samples[0]) for samples in sink.pending_batch.values()]
    assert any(a > 0 for a in advs) and any(a < 0 for a in advs)


def test_the_turn_scale_and_unfinished_only_with_a_pass_are_set_live_and_recorded(tmp_path):
    source, algorithm, shaping = live(tmp_path)
    client = TestClient(
        create_app(Controller(requires_step=True), routers=[plan_router(source.plans, shaping=shaping)])
    )
    source.next_task(step=1)
    for bad in ({"pass_turns": {"scale": 0}}, {"unfinished": {"only_with_pass": None}}, {"unfinished": {"bogus": 1}}):
        assert client.post("/v1/shaping", json={"step": 3, **bad}).status_code == 422, bad
    change = {"pass_turns": {"weight": 0.5, "scale": 8}, "unfinished": {"reward": -0.25, "only_with_pass": True}}
    posted = client.post("/v1/shaping", json={"step": "next", "author": "me", "note": "audit", **change})
    assert posted.status_code == 200
    [setting] = posted.json()["shaping"]
    assert setting["pass_turns"]["scale"] == 8 and setting["unfinished"] == {"reward": -0.25, "only_with_pass": True}
    back = client.post("/v1/shaping", json={"step": 5, "pass_turns": {"scale": None}})
    assert back.status_code == 200 and back.json()["shaping"][0]["pass_turns"]["scale"] is None
    source.next_task(step=2)
    group = [rollout(0.0, step=2, group="at-2"), rollout(1.0, step=2, group="at-2", stop="agent_timeout")]
    for episode in group:
        algorithm.settle_timeouts(episode)
    source.on_result(group)
    [line] = [json.loads(line) for line in (tmp_path / "groups.jsonl").read_text().splitlines()]
    assert line["shaping"]["pass_turns"]["scale"] == 8 and line["shaping"]["unfinished"]["only_with_pass"] is True
    assert line["unfinished_scored"] == 0

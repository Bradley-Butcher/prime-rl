import asyncio
import itertools
import json
import random
from collections import Counter
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import ModelCall
from verifiers.v1.types import AssistantMessage, Usage, UserMessage

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import (
    AdvRangeGateConfig,
    CurriculumConfig,
    DifficultyPoolSamplerConfig,
    DownsampleConfig,
    OrchestratorConfig,
)
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.curriculum.plans import Plans
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.downsample import max_variance_subset
from prime_rl.orchestrator.selection import reward_variance
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import GroupCancellation, Policy, Progress, TaskRequest
from prime_rl.utils.logger import setup_logger

setup_logger("warning")

N, M = 12, 8


def kept_passes(k: int) -> int:
    return min(k, max(4, 8 - (12 - k)))


@pytest.mark.parametrize("k", range(N + 1))
def test_a_pass_fail_group_keeps_the_most_balanced_eight(k):
    rewards = [1.0] * k + [0.0] * (N - k)
    random.Random(k).shuffle(rewards)

    kept = max_variance_subset(rewards, M, random.Random(0))

    assert len(kept) == M == len(set(kept))
    assert sum(rewards[index] for index in kept) == kept_passes(k)
    # No other eight of the twelve varies more.
    best = max(reward_variance(subset) for subset in itertools.combinations(rewards, M))
    assert reward_variance([rewards[index] for index in kept]) == pytest.approx(best)


def test_the_kept_rollouts_are_uniform_within_each_outcome_and_seeded():
    rewards = [1.0] * 6 + [0.0] * 6  # keeps 4 of the 6 passes and 4 of the 6 failures
    draws = 6000
    rng = random.Random(7)
    counts = Counter(index for _ in range(draws) for index in max_variance_subset(rewards, M, rng))

    for index in range(N):
        assert counts[index] / draws == pytest.approx(4 / 6, abs=0.03), index
    assert max_variance_subset(rewards, M, random.Random(3)) == max_variance_subset(rewards, M, random.Random(3))
    assert len({tuple(max_variance_subset(rewards, M, random.Random(seed))) for seed in range(20)}) > 1


def test_other_rewards_keep_the_top_and_bottom_that_vary_most():
    rng = random.Random(0)
    for _ in range(50):
        rewards = [round(rng.random(), 2) for _ in range(N)]
        kept = max_variance_subset(rewards, M, random.Random(1))
        kept_rewards = sorted(rewards[index] for index in kept)
        best = max(reward_variance(subset) for subset in itertools.combinations(rewards, M))
        assert reward_variance(kept_rewards) == pytest.approx(best)
        # A top i and a bottom M - i of the sorted rewards.
        ordered = sorted(rewards)
        assert any(kept_rewards == sorted(ordered[: M - top] + ordered[N - top :]) for top in range(M + 1))

    # Keeping three, one top and two bottom vary as much as two top and one bottom: the tie goes either way.
    rewards = [0.0] * 4 + [1.0] * 4 + [0.5]
    picks = {tuple(sorted(rewards[i] for i in max_variance_subset(rewards, 3, random.Random(s)))) for s in range(40)}
    assert picks == {(0.0, 0.0, 1.0), (0.0, 1.0, 1.0)}
    assert max_variance_subset(rewards[:8], 8, random.Random(0)) == list(range(8))


def rollout(task: vf.Task, reward: float, group: str) -> vf.Episode:
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=task.data, key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig(), trainable=True),
        nodes=[
            MessageNode(message=UserMessage(content="prompt"), token_ids=[1], mask=[False]),
            MessageNode(
                parent=0,
                message=AssistantMessage(content="answer"),
                sampled=True,
                token_ids=[2, 3],
                mask=[True, True],
                logprobs=[-0.1, -0.2],
            ),
        ],
        ok=True,
        is_completed=True,
        rewards={"correct": vf.Reward(score=reward)},
        calls=[ModelCall(node=1, usage=Usage(prompt_tokens=1, completion_tokens=2), finish_reason="stop")],
    )
    return vf.Episode(
        env=vf.EnvInfo(id="env", name="env"),
        task=trace.task,
        group=vf.GroupInfo(id=group),
        traces=[trace],
        ok=True,
        run=vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=1, policy=vf.PolicySpan(start=0, end=0))),
    )


def downsampling_sink(tmp_path, baseline: str):
    """A train sink over one down-sampled source (12 rollouts, 8 trained) with the run's curriculum: difficulty
    pools, the zero-advantage gate and the curriculum records."""
    tasks = [vf.Task(vf.WireTaskData(idx=i, prompt=f"task {i}")) for i in range(3)]
    curriculum = CurriculumConfig(sampler=DifficultyPoolSamplerConfig(), gates={"zero_advantage": AdvRangeGateConfig()})
    source = TrainSource(
        [
            SimpleNamespace(
                name="env", tasks=iter(tasks), num_tasks=3, config=SimpleNamespace(ratio=1.0, curriculum=curriculum)
            )
        ]
    )
    source.plans = Plans(source, tmp_path, resume_step=None)
    config = SimpleNamespace(
        group_size=M,
        rollouts_per_task=N,
        downsample=DownsampleConfig(rollouts=N, baseline=baseline),
        sampling=SimpleNamespace(temperature=1.0),
    )
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(loss_aggregation="prompt"), None)
    env = SimpleNamespace(config=config, requires_sampling_masks=False, algorithm=algorithm)
    sink = TrainSink(
        OrchestratorConfig(batch_size=M),
        tokenizer=None,
        train_envs=SimpleNamespace(get=lambda name: env),
        progress=Progress(),
        batch_size=100,
        on_result=source.on_result,
        on_downsample=source.plans.record_downsample,
    )
    return sink, source, tasks, env


def trained_advantage(sample) -> float:
    (advantage,) = {a for a, trains in zip(sample.advantages, sample.mask) if trains}
    return advantage


@pytest.mark.parametrize("baseline, passed, failed", [("kept", 5 / 8, -3 / 8), ("all", 0.75, -0.25)])
def test_a_group_waits_for_all_twelve_and_trains_eight_by_max_variance(tmp_path, baseline, passed, failed):
    sink, source, tasks, env = downsampling_sink(tmp_path, baseline)
    rewards = [1.0] * 3 + [0.0] * 9  # 3 passes: all 3 kept with 5 failures
    group = [rollout(tasks[0], reward, "mixed") for reward in rewards]

    for episode in group[:-1]:
        asyncio.run(sink.add(episode))
    assert not sink.pending_batch  # an eleventh rollout does not finalize the group
    asyncio.run(sink.add(group[-1]))

    kept = {trace_id for trace_id in sink.pending_batch}
    assert len(kept) == M
    kept_rewards = [episode.traces[0].reward for episode in group if episode.traces[0].id in kept]
    assert sum(kept_rewards) == 3
    # PODS: reward minus the kept eight's mean (3/8); Reinforce-Ada: minus all twelve's (3/12).
    for trace_id, samples in sink.pending_batch.items():
        reward = next(e.traces[0].reward for e in group if e.traces[0].id == trace_id)
        assert all(trained_advantage(sample) == pytest.approx(passed if reward else failed) for sample in samples)
        # Prompt-mean weights: the kept eight's two tokens each share the group's weight of 1.
        assert all(weight == pytest.approx(1 / 16) for sample in samples for weight in sample.rl_weights if weight)

    # The difficulty sampler and the records see all twelve, with the kept eight beside them.
    assert source.curricula["env"].sampler.task_rewards[tasks[0].key] == pytest.approx(3 / 12)
    (line,) = [json.loads(line) for line in (tmp_path / "groups.jsonl").read_text().splitlines()]
    assert (line["rollouts"], line["reward"], line["mixed"]) == (12, pytest.approx(0.25), True)
    assert line["kept"] == {"rollouts": 8, "reward": pytest.approx(3 / 8), "successes": 3}
    assert source.plans.stats()["envs"]["env"]["mean_reward"] == 0.25

    # downsample.jsonl lists all twelve in the order they finished, with what each trains with.
    (record,) = [json.loads(line) for line in (tmp_path / "downsample.jsonl").read_text().splitlines()]
    assert (record["admitted"], record["baseline"], record["failed"]) == (True, baseline, 0)
    assert [entry["trace"] for entry in record["rollouts"]] == [episode.traces[0].id for episode in group]
    assert [entry["finish"] for entry in record["rollouts"]] == list(range(1, N + 1))
    for entry in record["rollouts"]:
        assert entry["kept"] == (entry["trace"] in kept) and entry["turns"] == 1
        expected = (passed if entry["reward"] else failed) if entry["kept"] else None
        assert entry["advantage"] == (pytest.approx(expected) if expected is not None else None)

    marks = sink.take_kept_marks()
    assert len(marks) == N and set(marks) >= kept and sum(marks.values()) == M
    window = sink.take_rollout_window(force=True)
    metrics = window.metrics.to_dict()
    assert metrics["rollout/traces/queued"] == 8 and metrics["rollout/traces/left_out"] == 4
    assert metrics["rollout/traces/discarded"] == 0 and window.discarded == 0
    assert metrics["downsample/reward/all"] == pytest.approx(0.25)
    assert metrics["downsample/reward/kept"] == pytest.approx(3 / 8)


def test_the_gate_looks_at_all_twelve(tmp_path):
    sink, source, tasks, _ = downsampling_sink(tmp_path, "kept")
    for name, task, rewards in [("uniform", tasks[0], [0.0] * 12), ("one pass", tasks[1], [0.0] * 11 + [1.0])]:
        for reward in rewards:
            asyncio.run(sink.add(rollout(task, reward, name)))

    trained = {sink.episode_by_trace[trace_id].group.id for trace_id in sink.pending_batch}
    assert trained == {"one pass"} and len(sink.pending_batch) == M  # 1 pass and 7 failures
    lines = {line["key"]: line for line in map(json.loads, (tmp_path / "groups.jsonl").read_text().splitlines())}
    assert lines[tasks[0].key]["kept"] is None and lines[tasks[0].key]["rollouts"] == 12
    assert lines[tasks[1].key]["kept"]["successes"] == 1
    records = {line["key"]: line for line in map(json.loads, (tmp_path / "downsample.jsonl").read_text().splitlines())}
    assert records[tasks[0].key]["admitted"] is False
    assert not any(entry["kept"] or entry["advantage"] is not None for entry in records[tasks[0].key]["rollouts"])
    assert sum(entry["kept"] for entry in records[tasks[1].key]["rollouts"]) == M
    assert sink.take_kept_marks().keys() >= set(sink.pending_batch)


@pytest.mark.parametrize("downsampled", [True, False])
def test_an_overload_cut_voids_a_downsampled_group_and_leaves_the_rest_as_before(tmp_path, downsampled):
    sink, source, tasks, env = downsampling_sink(tmp_path, "kept")
    if not downsampled:
        env.config.downsample, env.config.rollouts_per_task = None, M
    dispatched = env.config.rollouts_per_task
    # Six rollouts come back, mixed, before the concurrency controller cuts the rest of the group.
    for reward in [1.0, 0.0] * 3:
        asyncio.run(sink.add(rollout(tasks[0], reward, "cut")))
    cut = GroupCancellation(
        kind="train", env_name="env", group_id="cut", step=1, count=dispatched - 6, reason="overload"
    )
    asyncio.run(sink.cancel(cut))

    window = sink.take_rollout_window(force=True)
    groups = (tmp_path / "groups.jsonl").read_text().splitlines() if (tmp_path / "groups.jsonl").exists() else []
    if downsampled:
        # Its six fastest rollouts are not the twelve its eight are to be chosen from: none trains, the curriculum
        # does not hear of it, and downsample.jsonl says why.
        assert not sink.pending_batch and not groups and not source.curricula["env"].sampler.task_rewards
        assert window.cancelled == N
        (record,) = [json.loads(line) for line in (tmp_path / "downsample.jsonl").read_text().splitlines()]
        assert (record["cancelled"], record["admitted"], len(record["rollouts"])) == ("overload", False, 6)
        assert not any(entry["kept"] for entry in record["rollouts"])
    else:
        assert len(sink.pending_batch) == 6 and len(groups) == 1  # without down-sampling, as before


def train(**group) -> dict:
    env = {"taskset": {"id": "harbor"}, "agent": {"harness": {"id": "null"}, "runtime": {"type": "subprocess"}}}
    return {"group_size": M, "source": [{"name": "a", "env": env}], **group}


def test_the_config_dispatches_the_downsampled_rollouts_and_trains_group_size():
    config = OrchestratorConfig(batch_size=256, train=train(downsample={"rollouts": N}))
    (source,) = config.train.source
    assert (source.group_size, source.rollouts_per_task, source.downsample.baseline) == (M, N, "kept")
    assert OrchestratorConfig(batch_size=256, train=train()).train.source[0].rollouts_per_task == M

    for bad, match in [
        ({"downsample": {"rollouts": 6}}, "fewer than the group_size"),
        ({"downsample": {"rollouts": N}, "algo": {"type": "rae"}}, "advantages from a group's rewards alone"),
        ({"downsample": {"rollouts": N, "baseline": "median"}}, "baseline"),
    ]:
        with pytest.raises(ValueError, match=match):
            OrchestratorConfig(batch_size=256, train=train(**bad))
    with pytest.raises(ValueError, match="most rollouts a train source dispatches per Task"):
        OrchestratorConfig(batch_size=256, train=train(downsample={"rollouts": N}), concurrency={"max_inflight": 10})


def test_ingesting_while_shipping_is_its_own_switch_and_batch_selection_implies_it():
    assert OrchestratorConfig().ingest_while_shipping is False
    assert OrchestratorConfig(ingest_while_shipping=True).batch_selection is None
    assert OrchestratorConfig(batch_selection={}).ingest_while_shipping is True


def test_a_downsampled_source_dispatches_its_rollouts_per_task():
    task = vf.Task(vf.WireTaskData(idx=0, prompt="task"))
    config = SimpleNamespace(group_size=M, rollouts_per_task=N)
    dispatcher = Dispatcher.__new__(Dispatcher)
    dispatcher.progress, dispatcher.policy = Progress(), Policy()
    dispatcher.train_source = SimpleNamespace(next_task=lambda step: TaskRequest(env_name="env", task=task, step=step))

    group = dispatcher.next_fresh_group("train", SimpleNamespace(get=lambda name: SimpleNamespace(config=config)))

    assert group.episodes_to_schedule == group.target_episodes == N

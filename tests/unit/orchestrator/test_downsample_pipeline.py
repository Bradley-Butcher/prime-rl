"""Down-sampling along the orchestrator's own path: the dispatcher dispatches 12 rollouts per Task to a fake env with
deterministic rewards, the orchestrator ingests what comes back, and the train sink waits for all 12, gates them,
keeps 8 and cuts each admitted group's TrainBatch."""

import asyncio
import json
from collections import Counter
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import ModelCall
from verifiers.v1.types import AssistantMessage, Usage, UserMessage

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import AdvRangeGateConfig, CurriculumConfig, DownsampleConfig, OrchestratorConfig
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.curriculum.plans import Plans
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Policy, Progress
from prime_rl.utils.logger import setup_logger

setup_logger("warning")

REWARDS = {
    0: [0.0] * 9 + [1.0] * 3,  # 3 passes, dispatched and finishing last
    1: [1.0] * 9 + [0.0] * 3,  # 9 passes
    2: [0.0] * 12,  # no signal: the gate rejects it
    3: [1.0, 0.0] * 6,  # 6 passes
}
"""Each Task's rewards, by the order its rollouts are dispatched in."""

EXPECTED = {
    "kept": {0: (3, 5 / 8, -3 / 8), 1: (5, 3 / 8, -5 / 8), 3: (4, 0.5, -0.5)},
    "all": {0: (3, 0.75, -0.25), 1: (5, 0.25, -0.75), 3: (4, 0.5, -0.5)},
}
"""Per baseline and Task: passes kept, and the advantage a kept pass and a kept failure train with."""


class FakeEnv:
    """A train source whose rollouts score the rewards ``REWARDS`` lists, in dispatch order; Task 0's passes come
    back last."""

    name = "env"
    uses_live_policy = True
    requires_sampling_masks = False

    def __init__(self, baseline: str):
        tasks = [vf.Task(vf.WireTaskData(idx=idx, prompt=f"task {idx}")) for idx in REWARDS]
        self.tasks, self.num_tasks = iter(tasks), len(tasks)
        self.by_idx = {task.data.idx: task for task in tasks}
        self.config = SimpleNamespace(
            ratio=1.0,
            curriculum=CurriculumConfig(gates={"zero_advantage": AdvRangeGateConfig()}),  # cycles the Tasks in order
            group_size=8,
            rollouts_per_task=12,
            downsample=DownsampleConfig(rollouts=12, baseline=baseline),
            sampling=SimpleNamespace(temperature=1.0),
        )
        self.algorithm = GRPOAlgorithm(GRPOAlgoConfig(loss_aggregation="prompt"), None)
        self.clients = SimpleNamespace(train_client=None, model_name="policy", finish_sessions=self.finish_sessions)
        self.dispatched = Counter()
        self.failed = Counter()

    async def finish_sessions(self, session_ids) -> None:
        pass

    async def run(self, *, client, model_name, cache_salt, task_data, on_delta) -> vf.Episode:
        task = self.by_idx[task_data["idx"]]
        index = self.dispatched[task.key] % 12
        self.dispatched[task.key] += 1
        reward = REWARDS[task.data.idx][index]
        if task.data.idx == 0 and reward == 1.0:  # Task 0's passes come back after its 9 failures
            while self.failed[task.key] < 9:
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.02)
        else:
            await asyncio.sleep(0.002 * index)
            self.failed[task.key] += reward == 0.0
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
        return vf.Episode(env=vf.EnvInfo(id="env", name="env"), task=trace.task, traces=[trace], ok=True)


class Envs(list):
    def get(self, name: str) -> FakeEnv:
        return self[0]


async def run_pipeline(tmp_path, baseline: str) -> tuple[list, dict[int, dict], list[str]]:
    """The TrainBatches cut, each Task's ``downsample.jsonl`` record once every Task's group finalized, and the
    rollouts in the order the orchestrator took them in."""
    env = FakeEnv(baseline)
    envs = Envs([env])
    progress, policy = Progress(), Policy(version=0, model_name="policy")
    source = TrainSource(envs)
    source.plans = Plans(source, tmp_path, resume_step=None)
    dispatcher = Dispatcher(
        train_envs=envs,
        eval_envs=None,
        train_source=source,
        eval_source=None,
        policy_clients=None,
        policy=policy,
        progress=progress,
        initial_max_inflight=24,
        max_inflight_ceiling=24,
        dispatch_per_minute=None,
        max_off_policy_steps=8,
        run_id="run",
        run_name=None,
    )
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.config = OrchestratorConfig(batch_size=8, ingest_while_shipping=True)
    orchestrator.progress, orchestrator.train_source, orchestrator.eval_sink = progress, source, None
    orchestrator.heart, orchestrator.draining, orchestrator.stopped = None, False, asyncio.Event()
    orchestrator.train_sink = TrainSink(
        orchestrator.config,
        tokenizer=None,
        train_envs=envs,
        progress=progress,
        batch_size=8,
        on_result=source.on_result,
        on_downsample=source.plans.record_downsample,
    )
    batches = []

    async def finalize_train_batch(batch) -> None:
        batches.append(batch)

    orchestrator.finalize_train_batch = finalize_train_batch

    def records() -> list[dict]:
        path = tmp_path / "downsample.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    arrivals = []
    running = asyncio.create_task(dispatcher.start())
    try:
        while len(records()) < len(REWARDS):
            item = await asyncio.wait_for(dispatcher.out_q.get(), timeout=30)
            arrivals += [trace.id for trace in item.traces]
            await orchestrator.ingest(item)
    finally:
        await dispatcher.stop()
        await asyncio.gather(running, return_exceptions=True)
    tasks = {task.key: idx for idx, task in env.by_idx.items()}
    return batches, {tasks[record["key"]]: record for record in records()[: len(REWARDS)]}, arrivals


@pytest.mark.parametrize("baseline", ["kept", "all"])
def test_each_admitted_group_ships_exactly_its_kept_eight(tmp_path, baseline):
    batches, by_task, arrivals = asyncio.run(run_pipeline(tmp_path, baseline))

    assert {task: record["admitted"] for task, record in by_task.items()} == {0: True, 1: True, 2: False, 3: True}
    assert not any(rollout["kept"] or rollout["advantage"] is not None for rollout in by_task[2]["rollouts"])
    assert len(batches) == 3  # one TrainBatch of 8 per admitted group

    for batch in batches:
        (group,) = {episode.group.id for episode in batch.cohort.episodes}
        (task,) = [task for task, record in by_task.items() if record["group"] == group]
        record = by_task[task]
        passes, pass_advantage, fail_advantage = EXPECTED[baseline][task]
        rollouts = record["rollouts"]
        # Every rollout, numbered in the order the orchestrator took it in.
        finished = [rollout["trace"] for rollout in sorted(rollouts, key=lambda rollout: rollout["finish"])]
        assert sorted(rollout["finish"] for rollout in rollouts) == list(range(1, 13))
        assert finished == [trace_id for trace_id in arrivals if trace_id in set(finished)]
        assert sorted(rollout["reward"] for rollout in rollouts) == sorted(REWARDS[task])
        assert all(rollout["turns"] == 1 for rollout in rollouts)
        kept = {rollout["trace"]: rollout for rollout in rollouts if rollout["kept"]}
        left_out = {rollout["trace"] for rollout in rollouts if not rollout["kept"]}
        assert len(kept) == 8 and sum(rollout["reward"] for rollout in kept.values()) == passes
        assert all(rollout["advantage"] is None for rollout in rollouts if not rollout["kept"])
        if task == 0:  # its 3 passes finish last, and still train: the group waited for all 12
            assert sorted(r["finish"] for r in kept.values() if r["reward"] == 1.0) == [10, 11, 12]

        # The batch holds exactly the kept eight, with the advantages the record names, and nothing of the rest.
        shipped = {sample.trace_id for sample in batch.samples}
        assert shipped == set(kept) and not shipped & left_out
        assert {episode.traces[0].id for episode in batch.cohort.episodes} == set(kept)
        weight = 0.0
        for sample in batch.samples:
            expected = pass_advantage if kept[sample.trace_id]["reward"] == 1.0 else fail_advantage
            trained = [(a, w) for a, w, m in zip(sample.advantages, sample.rl_weights, sample.mask) if m]
            assert all(a == pytest.approx(expected) for a, _ in trained)
            assert kept[sample.trace_id]["advantage"] == pytest.approx(expected)
            weight += sum(w for _, w in trained)
        assert weight == pytest.approx(1.0)  # prompt-mean: the group's kept tokens share its weight of 1

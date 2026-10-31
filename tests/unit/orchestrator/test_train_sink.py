import asyncio
import gc
import weakref
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import ModelCall
from verifiers.v1.types import AssistantMessage, Tool, ToolMessage, Usage, UserMessage

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import BatchSelectionConfig, OrchestratorConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation, Progress
from prime_rl.utils.logger import setup_logger

setup_logger("warning")


def trace(*, advantage=1.0, ok=True, trainable=True, sampled=True, reward=1.0, stop=None, unfinished_reply=False):
    tools = [Tool(name="search", description="schema" * 100, parameters={"type": "object"})]
    nodes = [
        MessageNode(message=UserMessage(content="prompt"), tools=tools, token_ids=[1], mask=[False]),
        MessageNode(
            parent=0,
            message=ToolMessage(content="result", tool_call_id="call1", name="search"),
            token_ids=[2],
            mask=[False],
        ),
        MessageNode(
            parent=1,
            message=AssistantMessage(content="answer"),
            sampled=sampled,
            token_ids=[3, 4],
            mask=[sampled, sampled],
            is_content=[True, True],
            logprobs=[-0.1, -0.2],
            advantages=[advantage, advantage],
            reference_logprobs=[-0.3, -0.4],
            routed_experts=np.ones((2, 2, 2), dtype=np.uint8),
        ),
    ]
    calls = [ModelCall(node=2, usage=Usage(prompt_tokens=2, completion_tokens=2), finish_reason="stop")]
    if unfinished_reply:
        # A reply committed with no finish reason, as an aborted generation is.
        nodes.append(
            MessageNode(
                parent=2,
                message=AssistantMessage(content="ans"),
                sampled=True,
                token_ids=[5, 6],
                mask=[True, True],
                logprobs=[-0.5, -0.6],
            )
        )
        calls.append(ModelCall(node=3, finish_reason=None))
    return vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig(), trainable=trainable),
        tools=tools,
        nodes=nodes,
        ok=ok,
        is_completed=True,
        stop_condition=stop,
        is_timeout=stop is not None and stop.endswith("_timeout"),
        rewards={"correct": vf.Reward(score=reward)},
        calls=calls,
        errors=[] if ok else [vf.Error(type="ProviderError", message="failed")],
    )


def episode(traces, *, group="g", version=0, ok=True):
    return vf.Episode(
        env=vf.EnvInfo(id="test", name="test"),
        task=traces[0].task,
        group=vf.GroupInfo(id=group),
        traces=traces,
        ok=ok,
        run=vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=1, policy=vf.PolicySpan(start=version, end=version))),
    )


def sink(*, constant=True, admitted=True, batch_size=100, group_size=1, algorithm=None, selection=None, exits=None):
    config = OrchestratorConfig(
        batch_size=batch_size,
        constant_trainer_batch_size=constant,
        max_off_policy_steps=1,
        batch_selection=selection,
    )
    env = SimpleNamespace(
        config=SimpleNamespace(group_size=group_size, sampling=SimpleNamespace(temperature=1.0)),
        requires_sampling_masks=False,
        algorithm=algorithm or Algorithm(GRPOAlgoConfig(), None),
    )
    return TrainSink(
        config,
        tokenizer=None,
        train_envs=SimpleNamespace(get=lambda name: env),
        progress=Progress(),
        batch_size=batch_size,
        on_result=lambda group: admitted,
        on_pool_exit=exits.extend if exits is not None else None,
    )


@pytest.mark.parametrize("constant,admitted", [(True, True), (False, True), (True, False)])
def test_zero_output_finalization_releases_objects_without_reporting(constant, admitted):
    async def run():
        s = sink(constant=constant, admitted=admitted, batch_size=4)
        refs = []
        for i in range(40):
            t = trace(advantage=0)
            ep = episode([t], group=str(i))
            refs.extend([weakref.ref(t), weakref.ref(ep)])
            before = msgspec.msgpack.encode(trace_to_samples(t, env_name="test"))
            batch = await s.add(ep)
            assert batch is None or not batch.samples
            assert msgspec.msgpack.encode(trace_to_samples(t, env_name="test")) == before
            assert s.take_rollout_window() is None
            assert s.rollout_window.attempts == i + 1
            del t, ep, batch
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert not s.pending_groups and not s.pending_group_failures and not s.episode_by_trace
        assert not s.pending_batch
        window = s.take_rollout_window(force=True)
        assert window.attempts == 40
        assert window.traces == 40
        assert s.take_rollout_window(force=True) is None
        assert s.progress.step == 1
        if constant:
            assert window.discarded == 40
        else:
            assert window.metrics.to_dict()["rollout/queued/pruned"] == 40

    asyncio.run(run())


def test_queued_trace_does_not_own_discarded_siblings():
    async def run():
        kept = trace()
        discarded = [trace(advantage=0), trace(ok=False), trace(trainable=False), trace(sampled=False)]
        ep = episode([kept, *discarded])
        refs = [weakref.ref(t) for t in discarded] + [weakref.ref(ep)]
        s = sink()
        before = msgspec.msgpack.encode(trace_to_samples(kept, env_name="test"))
        assert await s.add(ep) is None
        assert all(t.nodes[2].token_ids == [3, 4] for t in discarded)
        del discarded, ep
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert s.episode_by_trace[kept.id].traces == [kept]
        assert msgspec.msgpack.encode(trace_to_samples(kept, env_name="test")) == before
        window = s.take_rollout_window(force=True)
        assert window.traces == 5 and window.attempts == 1
        assert window.metrics.to_dict()["rollout/traces/discarded"] == 4

    asyncio.run(run())


def test_stale_queue_releases_trace_without_recounting_arrivals():
    async def run():
        s = sink(batch_size=4)
        old = trace()
        ref = weakref.ref(old)
        await s.add(episode([old]))
        assert s.take_rollout_window(force=True).attempts == 1
        del old
        s.progress.step = 4
        assert await s.take_batch() is None
        gc.collect()
        assert ref() is None
        window = s.take_rollout_window(force=True)
        assert window.attempts == 0 and window.traces == 0
        assert window.metrics.to_dict()["off_policy/dropped"] == 1
        assert s.take_rollout_window(force=True) is None

    asyncio.run(run())


def test_deferred_pruning_and_multiple_batches_preserve_episode_boundaries():
    async def run():
        s = sink(constant=False, batch_size=2)
        zero, kept, sibling, another = trace(advantage=0), trace(), trace(), trace()
        ep = episode([zero, kept, sibling, another])
        ep_id = ep.id
        batch = await s.add(ep)
        assert len(batch.samples) == 1 and batch.samples[0].trace_id == kept.id
        assert len(batch.cohort) == 1 and batch.cohort.episodes[0].id == ep_id
        assert sibling.id in s.pending_batch
        assert zero.nodes[2].token_ids == [3, 4]
        batch = await s.take_batch()
        assert len(batch.samples) == 2
        assert len(batch.cohort) == 1 and batch.cohort.num_traces == 2
        assert batch.cohort.episodes[0].id == ep_id
        assert not s.pending_batch and not s.episode_by_trace

    asyncio.run(run())


def test_failures_and_cancellations_accumulate_until_reporting():
    async def run():
        s = sink(batch_size=4)
        for i in range(8):
            failure = DispatchFailure(
                kind="train",
                env_name="test",
                group_id=str(i),
                step=1,
                policy_version=0,
                task_type="Task",
                task_key="key",
                task_hash="hash",
                error=vf.Error(type="TransportError", message="large" * 1000),
            )
            assert await s.fail(failure) is None
            assert s.take_rollout_window() is None
            assert not s.pending_group_failures
        for i in range(4):
            await s.cancel(GroupCancellation("train", "test", f"cancel{i}", 1, 1, "stale"))
        assert s.take_rollout_window() is None
        window = s.take_rollout_window(force=True)
        assert window.attempts == window.discarded == 12
        assert window.stale == 4
        assert window.errored == 8
        assert window.metrics.to_dict()["train/agg/all/dispatch_failure/mean"] == 1
        assert not s.pending_group_cancellations

    asyncio.run(run())


def test_partial_window_and_shutdown_release():
    async def run():
        s = sink(batch_size=4)
        t = trace()
        ref = weakref.ref(t)
        await s.add(episode([t]))
        assert s.take_rollout_window() is None
        del t
        s.discard_queued()
        gc.collect()
        assert ref() is None
        window = s.take_rollout_window(force=True)
        assert window.attempts == 1
        assert window.metrics.to_dict()["rollout/queued/cancelled"] == 1
        assert s.take_rollout_window(force=True) is None

    asyncio.run(run())


async def arrive(s, algorithm, t, *, ok=True):
    """A completed train episode as the dispatcher hands it to the sink: its timeouts settled first."""
    ep = episode([t], ok=ok)
    algorithm.settle_timeouts(ep)
    return await s.add(ep)


def trained_advantages(sample):
    return [advantage for advantage, trains in zip(sample.advantages, sample.mask) if trains]


@pytest.mark.parametrize("finalize_failed", [False, True])
@pytest.mark.parametrize("timeouts_as_failures", [True, False])
def test_an_agent_timeout_trains_in_its_group_as_a_failure_only_when_asked(timeouts_as_failures, finalize_failed):
    async def run():
        algorithm = GRPOAlgorithm(GRPOAlgoConfig(timeouts_as_failures=timeouts_as_failures), None)
        s = sink(group_size=8, batch_size=8 if timeouts_as_failures else 7, algorithm=algorithm)
        # The env passed the timed-out rollout, or its finalize raised after the deadline and it was never scored.
        timed_out = trace(reward=1.0, stop="agent_timeout", ok=not finalize_failed)
        if finalize_failed:
            timed_out.rewards = {}
            timed_out.errors = [vf.Error(type="TaskError", message="task finalize: POST /finish: HTTP 500")]
        passed = [trace(reward=1.0) for _ in range(3)]
        failed = [trace(reward=0.0) for _ in range(4)]
        for t in [timed_out, *passed, *failed]:
            batch = await arrive(s, algorithm, t, ok=t.ok)

        trained = {sample.trace_id: sample for sample in batch.samples}
        if timeouts_as_failures:
            assert len(trained) == 8
            assert timed_out.reward == 0.0
            assert trained_advantages(trained[timed_out.id]) == pytest.approx([-3 / 8] * 2)
            assert all(trained_advantages(trained[t.id]) == pytest.approx([5 / 8] * 2) for t in passed)
        else:
            assert len(trained) == 7 and timed_out.id not in trained and timed_out.has_error
            assert all(trained_advantages(trained[t.id]) == pytest.approx([4 / 7] * 2) for t in passed)
        metrics = batch.cohort.effective.metrics.to_wandb(prefix="train/agg", subset="effective")
        assert metrics["train/agg/effective/agent/timeouts_as_failures"] == float(timeouts_as_failures)

    asyncio.run(run())


def test_a_timed_out_rollouts_unfinished_last_reply_is_not_trained():
    async def run():
        algorithm = GRPOAlgorithm(GRPOAlgoConfig(timeouts_as_failures=True), None)
        s = sink(group_size=2, batch_size=2, algorithm=algorithm)
        timed_out = trace(reward=0.0, stop="agent_timeout", unfinished_reply=True)
        await arrive(s, algorithm, timed_out)
        batch = await arrive(s, algorithm, trace(reward=1.0))

        [sample] = [sample for sample in batch.samples if sample.trace_id == timed_out.id]
        assert sample.token_ids == [1, 2, 3, 4, 5, 6]
        assert sample.mask == [False, False, True, True, False, False]
        assert trained_advantages(sample) == pytest.approx([-0.5, -0.5])

    asyncio.run(run())


def test_with_timeouts_as_failures_errors_and_timeouts_without_a_reply_are_still_left_out():
    async def run():
        algorithm = GRPOAlgorithm(GRPOAlgoConfig(timeouts_as_failures=True), None)
        s = sink(group_size=5, algorithm=algorithm)
        passed, failed = trace(reward=1.0), trace(reward=0.0)
        task_error = trace(ok=False, stop="task_error")
        never_replied = trace(stop="agent_timeout", sampled=False)
        setup_timeout = trace(stop="setup_timeout", sampled=False)
        for t in (passed, failed, task_error, never_replied, setup_timeout):
            await arrive(s, algorithm, t, ok=t.ok)

        assert set(s.pending_batch) == {passed.id, failed.id}
        assert all(t.has_error for t in (task_error, never_replied, setup_timeout))

    asyncio.run(run())


def group(name, rewards, *, version=0):
    return episode([trace(reward=reward) for reward in rewards], group=name, version=version)


def groups_of(batch):
    return {episode.group.id for episode in batch.cohort.episodes}


def test_batch_selection_trains_the_most_informative_groups_and_carries_the_rest():
    async def run():
        exits = []
        s = sink(batch_size=8, selection=BatchSelectionConfig(oversample=1.5), exits=exits)
        # While a batch ships, groups keep pooling up to the cap (12 rollouts) and none is cut.
        s.shipping = True
        for name, rewards in [("a", [1, 1, 1, 0]), ("b", [1, 0, 1, 0]), ("c", [1, 0, 0, 0]), ("d", [1, 1, 0, 0])]:
            assert await s.add(group(name, rewards)) is None
        # c scores as a does (3/16), and the newer of the two goes when d takes the pool past its cap.
        assert [(record["group"], record["fate"]) for record in exits] == [("c", "displaced")]
        assert len(s.pending_batch) == 12 and "c" not in {ep.group.id for ep in s.episode_by_trace.values()}

        s.shipping = False
        batch = await s.take_batch()
        assert groups_of(batch) == {"b", "d"} and len(batch.samples) == 8
        assert batch.selection.metrics() == {
            "selection/pool": 3.0,
            "selection/candidates": 4.0,
            "selection/selected": 2.0,
            "selection/carried": 1.0,
            "selection/stale": 0.0,
            "selection/displaced": 1.0,
            "selection/score/selected": pytest.approx(0.25),
            "selection/score/passed": pytest.approx(3 / 16),
        }
        assert [(r["group"], r["fate"], r["rollouts"]) for r in batch.selection.records(1)] == [
            ("b", "trained", 4),
            ("d", "trained", 4),
        ]

        # The group passed over trains in the next batch, ahead of an equally scored newer one.
        batch = await s.add(group("e", [0, 0, 0, 1]))
        assert groups_of(batch) == {"a", "e"}
        assert [(r["group"], r["passed"]) for r in batch.selection.records(2)] == [("a", 1), ("e", 0)]
        assert not s.pending_batch and not s.episode_by_trace and s.pool.size == 0

    asyncio.run(run())


def test_batch_selection_drops_a_pooled_group_once_it_is_too_stale_to_train():
    async def run():
        exits = []
        s = sink(batch_size=8, selection=BatchSelectionConfig(oversample=2), exits=exits)
        s.progress.step = 3  # trains v2; max_off_policy_steps = 1 keeps v1 and later
        s.shipping = True
        await s.add(group("old", [1, 0, 0, 0], version=1))
        await s.add(group("b", [1, 0, 1, 0], version=2))
        await s.add(group("c", [0, 1, 0, 1], version=2))
        s.shipping = False
        assert groups_of(await s.take_batch()) == {"b", "c"}

        s.progress.step = 4
        assert await s.take_batch() is None
        assert [(r["group"], r["fate"], r["step"], r["passed"], r["rollouts"], r["score"]) for r in exits] == [
            ("old", "stale", 4, 1, 4, 0.1875)
        ]
        assert not s.pending_batch and not s.episode_by_trace and s.pool.size == 0

        await s.add(group("d", [1, 0, 0, 1], version=3))
        batch = await s.add(group("e", [1, 0, 0, 1], version=3))
        assert groups_of(batch) == {"d", "e"} and batch.selection.metrics()["selection/stale"] == 1.0

        # A run that stops training discards what is still pooled.
        await s.add(group("f", [1, 0, 0, 1], version=3))
        s.discard_queued()
        assert [(r["group"], r["fate"]) for r in exits[1:]] == [("f", "discarded")]
        assert s.pool.size == 0

    asyncio.run(run())

"""An inference stall: the server failed a model call as ``inference_stalled`` because its upstream stopped answering
(the Tinker bridge). The rollout's train group is void, trains nothing and reaches no curriculum, and dispatch holds
until the server answers again."""

import asyncio
from collections import Counter
from types import SimpleNamespace

import httpx
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import ModelCall
from verifiers.v1.types import AssistantMessage, Usage, UserMessage

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import AdvRangeGateConfig, CurriculumConfig, OrchestratorConfig
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.stall import STALL_CODE, episode_stalled, exception_stalled
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation, Policy, Progress
from prime_rl.utils.logger import setup_logger
from tests.unit.orchestrator.test_train_sink import episode, sink, trace

setup_logger("warning")

STALLED = f"Error code: 503 - {{'error': {{'message': '{STALL_CODE}: Tinker has answered nothing for 181 s', 'code': '{STALL_CODE}'}}}}"
"""What the bridge's 503 reads as by the time it reaches a trace, through the train client and the harness."""


def stalled_call() -> ModelCall:
    return ModelCall(error=vf.Error(type="ProviderError", message=STALLED, status_code=503))


def test_a_stall_is_found_wherever_the_rollout_recorded_it():
    called = trace(ok=False)
    called.calls.append(stalled_call())
    assert episode_stalled(episode([called]))

    harness = trace(ok=False)
    harness.errors = [vf.Error(type="InterceptionError", message=f"model connection failed: {STALLED}")]
    assert episode_stalled(episode([harness]))

    ended = episode([trace()], ok=False)
    ended.errors = [vf.Error(type="EnvError", message=STALLED)]
    assert episode_stalled(ended)

    assert not episode_stalled(episode([trace(ok=False)]))  # an ordinary provider error
    assert exception_stalled(RuntimeError(STALLED)) and not exception_stalled(RuntimeError("upstream 503"))


def test_a_stall_drop_voids_the_group_and_closes_it():
    async def run():
        closed, seen = [], []
        s = sink(batch_size=1, closed=closed, group_size=3)
        s.on_result = lambda group: seen.append(group) or True
        # Two rollouts of the group finished before the stall; the third met it.
        assert await s.add(episode([trace()], group="stalled")) is None
        assert await s.add(episode([trace(reward=0.0, advantage=-1.0)], group="stalled")) is None
        assert await s.cancel(GroupCancellation("train", "test", "stalled", 1, 1, "stall")) is None
        assert not seen and not s.pending_batch and closed == ["stalled"]

    asyncio.run(run())


class FakeEnv:
    """Two Tasks of four rollouts each; Task 1's first rollout met a stall, the rest are ordinary."""

    name = "env"
    uses_live_policy = True
    requires_sampling_masks = False
    requires_sampler_logprobs = False

    def __init__(self) -> None:
        tasks = [vf.Task(vf.WireTaskData(idx=idx, prompt=f"task {idx}")) for idx in (0, 1)]
        self.tasks, self.num_tasks = iter(tasks), len(tasks)
        self.by_idx = {task.data.idx: task for task in tasks}
        self.config = SimpleNamespace(
            ratio=1.0,
            curriculum=CurriculumConfig(gates={"zero_advantage": AdvRangeGateConfig()}),
            group_size=4,
            rollouts_per_task=4,
            downsample=None,
        )
        self.algorithm = GRPOAlgorithm(GRPOAlgoConfig(), None)
        self.clients = SimpleNamespace(train_client=None, model_name="policy", finish_sessions=self.finish_sessions)
        self.dispatched = Counter()

    def sampling_at(self, step: int) -> vf.SamplingConfig:
        return vf.SamplingConfig(temperature=1.0)

    async def finish_sessions(self, session_ids) -> None:
        pass

    async def run(self, *, client, model_name, sampling=None, cache_salt, task_data, on_delta) -> vf.Episode:
        task = self.by_idx[task_data["idx"]]
        index = self.dispatched[task.key]
        self.dispatched[task.key] += 1
        stalled = task.data.idx == 1 and index == 0
        await asyncio.sleep(0.001 if stalled else 0.005 + 0.001 * index)
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
            ok=not stalled,
            is_completed=True,
            rewards={"correct": vf.Reward(score=float(index % 2))},
            calls=[stalled_call()]
            if stalled
            else [ModelCall(node=1, usage=Usage(prompt_tokens=1, completion_tokens=2), finish_reason="stop")],
        )
        return vf.Episode(env=vf.EnvInfo(id="env", name="env"), task=trace.task, traces=[trace], ok=not stalled)


class Envs(list):
    def get(self, name: str) -> FakeEnv:
        return self[0]


class Server:
    """The policy's server as the dispatcher asks it: stalled for the first ``polls`` questions, or with no route."""

    def __init__(self, polls: int, route: bool = True) -> None:
        self.polls, self.route, self.asked = polls, route, 0

    async def inference_stalled(self) -> bool | None:
        self.asked += 1
        if not self.route:
            return None
        return self.asked <= self.polls


async def drive(server: Server, *, hold_s: float = 60.0):
    env = FakeEnv()
    envs = Envs([env])
    progress, policy = Progress(), Policy(version=0, model_name="policy")
    source = TrainSource(envs)
    reported = []
    on_result = source.on_result
    source.on_result = lambda group, **kwargs: reported.append(group[0].task.data.idx) or on_result(group, **kwargs)
    dispatcher = Dispatcher(
        train_envs=envs,
        eval_envs=None,
        train_source=source,
        eval_source=None,
        policy_clients=server,
        policy=policy,
        progress=progress,
        initial_max_inflight=8,
        max_inflight_ceiling=8,
        dispatch_per_minute=None,
        run_id="run",
        run_name=None,
    )
    dispatcher.stall_poll_s, dispatcher.stall_hold_s = 0.01, hold_s
    scheduled_while_held = []
    schedule = dispatcher.schedule_group_episode

    async def schedule_checked(group_id, group):
        scheduled_while_held.append(not dispatcher.inference_ok.is_set())
        return await schedule(group_id, group)

    dispatcher.schedule_group_episode = schedule_checked
    train_sink = TrainSink(
        OrchestratorConfig(batch_size=4),
        tokenizer=None,
        train_envs=envs,
        progress=progress,
        batch_size=4,
        on_result=source.on_result,
    )
    cancellations, batches = [], []
    running = asyncio.create_task(dispatcher.start())
    try:
        while not (cancellations and batches):
            item = await asyncio.wait_for(dispatcher.out_q.get(), timeout=10)
            if isinstance(item, GroupCancellation):
                cancellations.append(item)
                batch = await train_sink.cancel(item)
            else:
                assert not isinstance(item, DispatchFailure)
                batch = await train_sink.add(item)
            if batch is not None:
                batches.append(batch)
        # Dispatch comes back once the server says it answers, and not before.
        await asyncio.wait_for(dispatcher.inference_ok.wait(), timeout=10)
        held = len(scheduled_while_held)
        while len(scheduled_while_held) == held:
            await asyncio.sleep(0.005)
    finally:
        dispatcher.disable_train_scheduling()
        await dispatcher.stop()
        await asyncio.gather(running, return_exceptions=True)
    return env, dispatcher, cancellations, batches, reported, scheduled_while_held


def test_a_rollout_that_met_a_stall_voids_its_group_and_holds_dispatch_until_the_server_answers():
    server = Server(polls=3)
    env, dispatcher, cancellations, batches, reported, scheduled_while_held = asyncio.run(drive(server))

    (cancelled,) = cancellations
    assert cancelled.reason == "stall" and cancelled.count == 4  # the stalled rollout and the three it took with it
    assert 1 not in reported  # the void group reached no curriculum
    (batch,) = batches
    assert {episode.task.data.idx for episode in batch.cohort.episodes} == {0}
    assert server.asked == 4 and dispatcher.stalls == 1 and dispatcher.inference_ok.is_set()
    assert not any(scheduled_while_held)


def test_a_server_with_no_way_to_say_is_held_for_a_while_only():
    server = Server(polls=0, route=False)
    _, dispatcher, cancellations, batches, _, scheduled_while_held = asyncio.run(drive(server, hold_s=0.05))
    assert [c.reason for c in cancellations] == ["stall"] and batches and dispatcher.inference_ok.is_set()
    assert server.asked >= 2 and not any(scheduled_while_held)


def test_the_client_reads_the_server_s_stall_route():
    from prime_rl.orchestrator.clients import InferenceClient

    def answering(status: int, body: dict | None = None):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/stall"
            return httpx.Response(status, json=body or {})

        return handler

    async def ask(handler) -> bool | None:
        client = InferenceClient.__new__(InferenceClient)
        client._stall_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://bridge")
        try:
            return await client.inference_stalled()
        finally:
            await client._stall_client.aclose()

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert asyncio.run(ask(answering(200, {"stalled": True}))) is True
    assert asyncio.run(ask(answering(200, {"stalled": False}))) is False
    assert asyncio.run(ask(answering(404))) is None  # vLLM: no such route
    assert asyncio.run(ask(answering(500))) is True
    assert asyncio.run(ask(unreachable)) is True

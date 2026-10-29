import asyncio
import json
import random
from pathlib import Path
from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.dashboard import dj
from prime_rl.dashboard.rollout_stats import rollout, stream_fold
from prime_rl.monitors.file.monitor import FileMonitor
from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.annotations import stamp_arrival, stamp_batch
from prime_rl.orchestrator.curriculum.plans import Plan, Plans, PlanTask
from prime_rl.orchestrator.curriculum.shaping import Shaping, ShapingRequest
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.utils.config import dump_resolved_config
from prime_rl.utils.pathing import get_curriculum_dir

SOURCES = {"butcher-rowe-hand": 0.4, "cradle-templated": 0.3, "conversations": 0.2, "stop-or-ask-blocked": 0.1}
TASKS = [
    (["interface:rest", "domain:chat", "hops:1"], "read"),
    (["interface:rpc", "domain:analytics"], "read"),
    (["interface:rest", "domain:crm"], "write"),
    (["interface:rpc", "domain:issues", "hops:1"], "write"),
    (["interface:rest", "domain:mail"], "read"),
    (["interface:graphql", "domain:analytics"], "read"),
]
GROUP_SIZE, GROUPS_PER_STEP, STEPS, EVAL_EVERY = 4, 8, 10, 5
WEIGHTS = {"label:interface:rpc": 1.5}
EVALS = {"held-out": 0.3, "held-out-vendors": 0.2}
"""Eval sources and their base pass rates."""
QUERIES = ["vendor master list", "ledger close calendar", "payment terms by region", "warehouse transfer log"]


def rollout_episode(env: str, task: vf.Task, *, step: int, reward: float, rng: random.Random, group: str) -> vf.Episode:
    """One rollout as the trace stream records it: tool calls (some repeated, some silent), Ocean's verdict and the
    final answer."""
    repeats, silent = rng.randint(0, 3), rng.random() < 0.4
    nodes = [vf.MessageNode(message=vf.UserMessage(content=f"{env} task {task.key}"))]
    for index in range(3 + repeats):
        query = "open invoices" if index < repeats + 1 else QUERIES[index - repeats - 1]
        call = vf.ToolCall(
            id=f"c{index}", name="search", arguments=json.dumps({"query": query, "intent": f"look {index}"})
        )
        content = None if silent else "Looking it up."
        nodes.append(
            vf.MessageNode(
                parent=len(nodes) - 1, message=vf.AssistantMessage(content=content, tool_calls=[call]), sampled=True
            )
        )
        nodes.append(vf.MessageNode(parent=len(nodes) - 1, message=vf.ToolMessage(tool_call_id=call.id, content="[]")))
    answer = "Done: three invoices are open." if reward or rng.random() < 0.5 else "I couldn't find the invoices."
    nodes.append(vf.MessageNode(parent=len(nodes) - 1, message=vf.AssistantMessage(content=answer), sampled=True))
    stop = "max_turns" if not reward and rng.random() < 0.15 else "end_task"
    failure = (
        ""
        if reward
        else rng.choice(["The Run never read the invoice list.", "The judge found false: the total is wrong."])
    )
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=task.data, key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        rewards={"passed": vf.Reward(score=reward)},
        metrics={"stuck_calls": float(repeats)},
        info={"ocean": {"passed": bool(reward), "failure": failure}, "turns": [{"answer": answer}]},
        stop_condition=stop,
        is_completed=True,
        ok=True,
    )
    return vf.Episode(
        env=vf.EnvInfo(id=env, name=env),
        task=trace.task,
        group=vf.GroupInfo(id=group),
        traces=[trace],
        ok=True,
        run=vf.TrainRunInfo(
            id="run", work=vf.TrainWorkInfo(step=step, policy=vf.PolicySpan(start=max(0, step - 1), end=step))
        ),
    )


def eval_episode(step: int, index: int, reward: float, source: str = "held-out") -> vf.Episode:
    task = vf.Task(vf.WireTaskData(idx=index, prompt=source))
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=task.data, key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[vf.MessageNode(message=vf.AssistantMessage(content="answer"), sampled=True)],
        rewards={"passed": vf.Reward(score=reward)},
        stop_condition="end_task",
        is_completed=True,
        ok=True,
    )
    return vf.Episode(
        env=vf.EnvInfo(id=source, name=source),
        task=trace.task,
        traces=[trace],
        ok=True,
        run=vf.TrainRunInfo(id="run", work=vf.EvalWorkInfo(step=step)),
    )


def build_run(run_dir: Path) -> Path:
    """A ten-step RL run dir written by the orchestrator's own writers: the resolved config, the curriculum records
    (plans by claude and brad, a deleted plan, cued Tasks, a stuck-call penalty from step 5 and a change queued for
    step 11), the file monitor's metrics, trace stream and annotations. Step 10 is still dispatching."""
    rng = random.Random(0)
    config = OrchestratorConfig(
        batch_size=GROUP_SIZE * GROUPS_PER_STEP,
        max_steps=30,
        train={
            "group_size": GROUP_SIZE,
            "source": [
                {"name": name, "ratio": ratio, "env": {"taskset": {"id": "reverse-text"}}}
                for name, ratio in SOURCES.items()
            ],
        },
        eval={
            "interval": EVAL_EVERY,
            "source": [{"name": name, "env": {"taskset": {"id": "reverse-text"}}} for name in EVALS],
        },
    )
    resolved = run_dir / "configs" / "attempt_1" / "resolved"
    resolved.mkdir(parents=True)
    (resolved / "orchestrator.json").write_text(json.dumps(dump_resolved_config(config)))

    envs = []
    for name, ratio in SOURCES.items():
        tasks = [
            vf.Task(vf.WireTaskData(idx=i, prompt=f"{name} {i}", kind=kind, labels=labels))
            for i, (labels, kind) in enumerate(TASKS)
        ]
        envs.append(
            SimpleNamespace(
                name=name, tasks=iter(tasks), num_tasks=len(tasks), config=SimpleNamespace(ratio=ratio, curriculum=None)
            )
        )
    source = TrainSource(envs)
    directory = get_curriculum_dir(run_dir)
    plans = source.plans = Plans(source, directory, resume_step=None)
    shaping = Shaping(
        plans, {name: GRPOAlgorithm(GRPOAlgoConfig(), None) for name in SOURCES}, directory, resume_step=None
    )
    monitor = FileMonitor(FileMonitorConfig())
    asyncio.run(monitor.init(run_dir, producer="orchestrator"))

    shifted = Plan(sources={"cradle-templated": 0.2}, weights=WEIGHTS)
    cued = Plan(
        sources={"cradle-templated": 0.2},
        weights=WEIGHTS | {f"task:{source.tasks['butcher-rowe-hand'][1].key}": 0.5},
        tasks=[PlanTask(env="butcher-rowe-hand", key=source.tasks["butcher-rowe-hand"][3].key)],
    )
    for step in range(1, STEPS + 1):
        if step in (4, 5, 6, 7, 8):
            plans.set(step, shifted, author="claude", note=f"Cut cradle-templated to 20% for step {step}.")
        if step == 6:
            plans.delete(6, author="brad", note="Mismatch KL is above 0.01: run step 6 on the config.")
        if step in (9, 10):
            plans.set(step, cued, author="claude", note="Cue a butcher-rowe-hand rpc Task to test the rpc failures.")
        if step == 5:
            shaping.set(
                ShapingRequest(
                    step=5, stuck={"per_call": 0.05}, author="claude", note="Turn on the stuck-call penalty."
                )
            )
        groups = []
        for index in range(GROUPS_PER_STEP):
            request = source.next_task(step=step)
            rewards = [float(rng.random() < 0.45) for _ in range(GROUP_SIZE)]
            groups.append(
                [
                    rollout_episode(
                        request.env_name, request.task, step=step, reward=reward, rng=rng, group=f"{step}-{index}"
                    )
                    for reward in rewards
                ]
            )
        finished = groups if step < STEPS else groups[: GROUPS_PER_STEP // 2]
        for group in finished:
            source.on_result(group)
            stamp_arrival(group, "train", step)
            asyncio.run(monitor.log_episodes(group, step, "train", "all"))
        if step < STEPS:
            asyncio.run(monitor.log_annotations([update for group in finished for update in stamp_batch(group, step)]))
            row = {
                "entropy/all/mean": 0.44 - 0.01 * step,
                "mismatch_kl/all/mean": 0.003 + 0.0002 * step,
                "optim/grad_norm": 0.2 + 0.01 * step,
            }
            asyncio.run(monitor.log_metrics(row, step))
        if step % EVAL_EVERY == 0 or step == 1:
            for name, base in EVALS.items():
                evals = [
                    eval_episode(step, index, float(rng.random() < base + 0.02 * step), name) for index in range(10)
                ]
                asyncio.run(monitor.log_episodes(evals, step, "eval", "all"))
                pass_rate = sum(e.traces[0].reward for e in evals) / len(evals)
                asyncio.run(monitor.log_metrics({f"eval/{name}/all/default/avg@1": pass_rate}, step))
    plans.set(
        11, Plan(sources={"cradle-templated": 0.2}, weights=WEIGHTS), author="brad", note="Keep the mix; drop the cue."
    )
    shaping.set(ShapingRequest(step=11, stuck={"per_call": 0.1}, author="brad", note="Double the stuck-call penalty."))
    for stream, index in monitor._streams.values():
        stream.flush()
        index.flush()
    monitor.file.flush()
    return run_dir


def loaded_view(run_dir: Path) -> dict:
    fold = stream_fold(get_trace_stream(run_dir))
    while fold.advance():
        pass
    config = json.loads((run_dir / "configs" / "attempt_1" / "resolved" / "orchestrator.json").read_text())
    return dj.view(run_dir, config)


def test_the_view_reads_a_runs_plans_dispatch_groups_and_shaping(tmp_path):
    run_dir = build_run(tmp_path / "demo")
    view = loaded_view(run_dir)
    steps = {row["step"]: row for row in view["steps"]}

    assert view["dispatching"] == STEPS and steps[STEPS]["state"] == "dispatching"
    assert steps[4]["plan"]["author"] == "claude" and steps[4]["shares"]["cradle-templated"] == 0.2
    assert steps[6]["plan"] is None and steps[6]["deleted"] and steps[6]["origin"] is None
    assert steps[2]["shares"]["butcher-rowe-hand"] == SOURCES["butcher-rowe-hand"]
    assert sum(steps[3]["dispatched"].values()) == GROUPS_PER_STEP
    assert steps[STEPS]["via"]["tasks"] == 1 and steps[STEPS]["finalized"] == GROUPS_PER_STEP // 2
    assert view["next"]["step"] == STEPS + 1 and view["next"]["plan"]["author"] == "brad"
    assert view["shaping"]["active"]["stuck"]["per_call"] == 0.05 and view["shaping"]["since"]["stuck"]["step"] == 5
    assert [change["step"] for change in view["shaping"]["pending"]] == [STEPS + 1]
    assert steps[4]["shaping"]["stuck"]["per_call"] == 0.0 and steps[5]["shaping"]["stuck"]["per_call"] == 0.05


def test_the_view_pools_groups_and_rollouts_as_the_records_say(tmp_path):
    run_dir = build_run(tmp_path / "demo")
    view = loaded_view(run_dir)
    groups = [json.loads(line) for line in (get_curriculum_dir(run_dir) / "groups.jsonl").read_text().splitlines()]
    window = range(view["window"][0], view["window"][1] + 1)
    hand = [g for g in groups if g["env"] == "butcher-rowe-hand" and g["step"] in window]
    row = next(source for source in view["sources"] if source["name"] == "butcher-rowe-hand")

    assert row["groups"] == len(hand) and row["signal"] == sum(g["mixed"] for g in hand) / len(hand)
    assert row["cued"] == 1 and row["named_now"] is False and row["share_now"] < SOURCES["butcher-rowe-hand"] + 0.1
    health = {metric["key"]: metric for metric in view["health"]}
    assert health["near_dup"]["series"] and health["heldout_pass1"]["series"][0][0] == 1
    assert view["eval"]["primary"] == "held-out" and [s["name"] for s in view["eval"]["sources"]] == list(EVALS)
    vendors = view["eval"]["sources"][1]["rows"]
    assert [row["step"] for row in vendors] == [1, 5, 10] and vendors[0]["rollouts"] == 10
    assert health["stuck_calls"]["series"] and health["mismatch_kl"]["status"] == "ok"
    weight = view["weights"][0]
    assert weight["selector"] == "label:interface:rpc" and weight["since"] == 7  # step 6's plan was deleted
    assert 0 < weight["by_source"]["cradle-templated"] < 1 and weight["share_now"] is not None


def test_a_rollout_counts_as_gather_counts_it():
    task = vf.Task(vf.WireTaskData(idx=0, prompt="p", kind="read", labels=["interface:rpc"]))
    episode = rollout_episode("a", task, step=3, reward=0.0, rng=random.Random(4), group="g")
    record = json.loads(json.dumps(episode.to_record()))
    item = rollout(record)

    calls = [
        c
        for node in record["traces"][0]["nodes"]
        if node.get("sampled")
        for c in node["message"].get("tool_calls") or []
    ]
    repeats = sum(1 for c in calls if json.loads(c["arguments"])["query"] == "open invoices") - 1
    assert item.kind == "train" and item.step == 3 and item.policy == 2
    assert item.counts["calls"] == len(calls) and item.counts["near_dup"] == repeats
    assert item.mode in ("didnt_find", "declared_success", "gave_up", "unfinished")
    assert "kind=read" in item.labels and "interface:rpc" in item.labels

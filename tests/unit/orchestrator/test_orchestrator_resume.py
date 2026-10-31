from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl.configs.orchestrator import (
    AdvRangeGateConfig,
    CheckpointConfig,
    CurriculumConfig,
    DifficultyPoolSamplerConfig,
)
from prime_rl.orchestrator.ckpt import setup_ckpt_manager
from prime_rl.orchestrator.curriculum.plans import Plan, Plans
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Progress


def source(directory, resume_step: int | None = None) -> TrainSource:
    """Two sources drawn by difficulty pool behind the zero-advantage gate, with the curriculum records, as the
    orchestrator sets them up: the checkpoint loads before the plans read their records back."""
    curriculum = CurriculumConfig(sampler=DifficultyPoolSamplerConfig(), gates={"zero": AdvRangeGateConfig()})
    envs = []
    for name in ("a", "b"):
        tasks = [vf.Task(vf.WireTaskData(idx=idx, prompt=f"{name} {idx}")) for idx in range(6)]
        config = SimpleNamespace(ratio=1.0, curriculum=curriculum)
        envs.append(SimpleNamespace(name=name, tasks=iter(tasks), num_tasks=6, config=config))
    return TrainSource(envs)


def group(task: vf.Task, env: str, rewards: list[float]) -> list[vf.Episode]:
    episodes = []
    for reward in rewards:
        trace = vf.Trace(
            task=vf.TraceTask(type="Task", data=task.data, key=task.key, hash=task.hash),
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            nodes=[vf.MessageNode(message=vf.AssistantMessage(content="answer"), sampled=True)],
            rewards={"reward": vf.Reward(score=reward)},
            ok=True,
        )
        episodes.append(
            vf.Episode(
                env=vf.EnvInfo(id=env, name=env),
                task=trace.task,
                group=vf.GroupInfo(id=f"{env}-{task.key}"),
                traces=[trace],
                ok=True,
                run=vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=2)),
            )
        )
    return episodes


def test_the_orchestrator_resumes_its_steps_curriculum_and_plans(tmp_path):
    curriculum = tmp_path / "curriculum"
    running = source(tmp_path)
    running.plans = Plans(running, curriculum, resume_step=None)
    running.plans.set("default", Plan(sources={"a": 0.25}), author="dj")
    for step in (1, 2, 3):
        for _ in range(4):
            running.next_task(step=step)
    for idx, rewards in enumerate([[1.0] * 12, [0.0] * 12, [1.0, 0.0] * 6]):
        running.on_result(group(running.tasks["a"][idx], "a", rewards))
    progress = Progress(step=4, total_tokens=1234, total_samples=96, total_problems=12, rollout_window=3)
    manager = setup_ckpt_manager(tmp_path, CheckpointConfig(interval=1))
    manager.save(progress, running, step=3)
    running.plans.close()

    resumed, resumed_progress = source(tmp_path), Progress()
    manager.load(resumed_progress, resumed, step=3)
    resumed.plans = Plans(resumed, curriculum, resume_step=3)

    assert resumed_progress == progress
    sampler = resumed.curricula["a"].sampler
    assert sampler.task_rewards == running.curricula["a"].sampler.task_rewards
    assert sampler.task_rewards == {
        running.tasks["a"][0].key: 1.0,
        running.tasks["a"][1].key: 0.0,
        running.tasks["a"][2].key: 0.5,
    }
    # The default plan is back, and step 4 is the first whose dispatch hasn't started.
    assert resumed.plans.get("default")["plan"] == {"sources": {"a": 0.25}}
    assert resumed.plans.status()["dispatch_step"] == 3
    assert resumed.plans.stats()["envs"]["a"]["groups"] == 3
    # The samplers and the source mix pick up where they stopped, under the default plan.
    assert [resumed.next_task(step=4).task.key for _ in range(8)] == [
        running.next_task(step=4).task.key for _ in range(8)
    ]

"""Resuming a LoRA run from a trainable-only checkpoint, on two gloo ranks under FSDP2, each run in fresh processes."""

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from prime_rl.configs.trainer import CheckpointConfig, LinearSchedulerConfig, LoRAConfig

WORLD = 2
STEPS, SAVED = 6, 3
LR, NEW_LR = 1e-3, 4e-3
WARMUP = LinearSchedulerConfig(warmup_steps=2, decay_steps=0)
FROZEN = ("embed.weight", "head.weight", "layers.0.proj.weight", "layers.1.proj.weight")


class Block(nn.Module):
    def __init__(self) -> None:
        from prime_rl.trainer.models.layers.lora import MultiLoRALinear

        super().__init__()
        self.proj = MultiLoRALinear(nn.Linear(16, 16, dtype=torch.bfloat16), rank=8, n_adapters=1, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x)


class Tiny(nn.Module):
    """A frozen bf16 base, an fp32 LoRA in each block, and a persistent buffer, as a LoRA run holds them."""

    def __init__(self, blocks: int) -> None:
        super().__init__()
        torch.manual_seed(0)  # the same base on every load
        self.embed = nn.Linear(8, 16, dtype=torch.bfloat16)
        self.layers = nn.ModuleList(Block() for _ in range(blocks))
        self.head = nn.Linear(16, 4, dtype=torch.bfloat16)
        self.register_buffer("scale", torch.ones(4))
        for module in (self.embed, self.head):
            module.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h).float() * self.scale


def run(rank: int, store: str, plan: dict) -> None:
    """One run of ``plan`` on this rank: build, optionally resume, train some steps, optionally checkpoint, and let
    rank 0 save the adapter, Adam's state and each step's lr and update."""
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(WORLD), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(WORLD))
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        from prime_rl.trainer.ckpt import CheckpointManager, Progress
        from prime_rl.trainer.lora import setup_lora_state
        from prime_rl.trainer.models.layers.lora import set_lora_num_tokens
        from prime_rl.trainer.scheduler import restart_scheduler, setup_scheduler

        setup_lora_state(LoRAConfig(rank=8, alpha=16.0), torch.device("cpu"))
        model = Tiny(plan.get("blocks", 2))
        mesh = init_device_mesh("cpu", (WORLD,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        lora = {name: param for name, param in model.named_parameters() if param.requires_grad}
        optimizer = torch.optim.AdamW(lora.values(), lr=LR, weight_decay=0.0, foreach=True)
        scheduler = setup_scheduler(optimizer, WARMUP, STEPS, LR)

        manager = CheckpointManager(
            Path(plan["dir"]), CheckpointConfig(trainable_only=plan.get("trainable_only", True))
        )
        progress = Progress()
        if "resume" in plan:
            with torch.no_grad():  # a fresh adapter, which the checkpoint must replace
                for param in lora.values():
                    nn.init.normal_(param, std=0.3)
            restart = plan.get("resume_lr") == "config"
            manager.load(plan["resume"], model, [optimizer], None if restart else scheduler, progress)
            if restart:
                lr = plan["lr"]
                scheduler = restart_scheduler(optimizer, plan["scheduler"], STEPS, lr, progress.step)
            progress.step += 1

        lrs, updates = [], []
        for step in range(progress.step, plan["stop"] + 1):
            generator = torch.Generator().manual_seed(1000 * step + rank)
            x = torch.randn(4, 8, generator=generator).bfloat16()
            target = torch.randn(4, 4, generator=generator)
            set_lora_num_tokens(torch.tensor([4], dtype=torch.int32))
            loss = ((model(x) - target) ** 2).mean()
            loss.backward()
            before = {name: param.detach().full_tensor() for name, param in lora.items()}
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            updates.append({name: param.detach().full_tensor() - before[name] for name, param in lora.items()})
            progress.step = step
        if "save" in plan:
            manager.save(plan["save"], model, [optimizer], scheduler, progress)

        state = {
            "params": {name: param.detach().full_tensor() for name, param in lora.items()},
            "moments": {
                name: {
                    key: value.full_tensor() if key != "step" else value for key, value in optimizer.state[p].items()
                }
                for name, p in lora.items()
            },
            "lrs": lrs,
            "updates": updates,
        }
        if rank == 0:
            torch.save(state, plan["out"])
    finally:
        dist.destroy_process_group()


def spawn(tmp_path: Path, name: str, **plan) -> dict:
    out = tmp_path / f"{name}.pt"
    mp.spawn(run, args=(str(tmp_path / f"{name}.store"), {**plan, "out": str(out)}), nprocs=WORLD)
    return torch.load(out, weights_only=False)


def test_a_trainable_only_resume_matches_an_uninterrupted_run(tmp_path):
    uninterrupted = spawn(tmp_path, "uninterrupted", dir=str(tmp_path / "unused"), stop=STEPS)
    spawn(tmp_path, "first", dir=str(tmp_path / "run"), stop=SAVED, save=SAVED)
    spawn(tmp_path, "full", dir=str(tmp_path / "full"), stop=SAVED, save=SAVED, trainable_only=False)
    resumed = spawn(tmp_path, "resumed", dir=str(tmp_path / "run"), resume=SAVED, stop=STEPS)

    for name, param in uninterrupted["params"].items():
        assert param.dtype == torch.float32
        assert torch.equal(resumed["params"][name], param), name
        for key, value in uninterrupted["moments"][name].items():
            assert torch.equal(resumed["moments"][name][key], value), (name, key)
    assert resumed["lrs"] == uninterrupted["lrs"][SAVED:]

    # The checkpoint holds the adapter (and the buffer), with its optimizer state, and no frozen parameter, one file
    # per tensor.
    checkpoint = tmp_path / "run" / "checkpoints" / f"step_{SAVED}" / "trainer"
    model_keys = {
        key.removeprefix("app.model.")
        for key in FileSystemReader(checkpoint).read_metadata().state_dict_metadata
        if key.startswith("app.model.")
    }
    assert model_keys == set(uninterrupted["params"]) | {"scale"}
    assert not model_keys & set(FROZEN)
    files = list(checkpoint.glob("*.distcp"))
    full = tmp_path / "full" / "checkpoints" / f"step_{SAVED}" / "trainer"
    assert len(files) > WORLD and len(list(full.glob("*.distcp"))) == WORLD
    size = sum(path.stat().st_size for path in checkpoint.iterdir())
    assert size < sum(path.stat().st_size for path in full.iterdir())


def test_a_resume_with_the_config_lr_trains_at_it_from_its_first_step(tmp_path):
    spawn(tmp_path, "first", dir=str(tmp_path / "run"), stop=SAVED, save=SAVED)
    kept = spawn(tmp_path, "kept", dir=str(tmp_path / "run"), resume=SAVED, stop=SAVED + 1)
    # The new schedule even has another shape (warmup, a constant phase and a decay: three schedulers in sequence).
    decaying = LinearSchedulerConfig(warmup_steps=2, decay_steps=2)
    changed = spawn(
        tmp_path,
        "changed",
        dir=str(tmp_path / "run"),
        resume=SAVED,
        stop=SAVED + 1,
        resume_lr="config",
        lr=NEW_LR,
        scheduler=decaying,
    )

    assert kept["lrs"] == [pytest.approx(LR)] and changed["lrs"] == [pytest.approx(NEW_LR)]
    for name, update in kept["updates"][0].items():
        # Adam's moments came back the same, so only the step size differs (an update is measured as the difference
        # of two fp32 weights, so to their resolution).
        torch.testing.assert_close(changed["updates"][0][name], update * (NEW_LR / LR), rtol=1e-4, atol=1e-7)
        for key, value in kept["moments"][name].items():
            assert torch.equal(changed["moments"][name][key], value), (name, key)


def test_a_checkpoint_that_misses_a_trainable_parameter_is_refused(tmp_path):
    spawn(tmp_path, "first", dir=str(tmp_path / "run"), stop=SAVED, save=SAVED)
    with pytest.raises(Exception, match=r"2 missing \(layers\.2\.proj\.lora_A\.0, layers\.2\.proj\.lora_B\.0\)"):
        spawn(tmp_path, "grown", dir=str(tmp_path / "run"), resume=SAVED, stop=STEPS, blocks=3)

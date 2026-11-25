"""Check that a LoRA adapter of Inkling 2 Mini is served as it was trained, on one GPU.

The trainer (this process) loads a checkpoint with the base in BF16 and the adapter in FP32, trains a few steps, and
writes two adapters through the filesystem weight broadcast's own code. prime-rl's inference server, started from the
Python environment whose vLLM has the model (``rl-worker/inkling_vllm_env.sh`` builds it), serves the base, loads the
first adapter and hot-swaps it for the second through ``/load_lora_adapter``. Each of the three samples completions,
and the trainer computes the sampled tokens' logprobs under each of its three states, replaying the server's routed
experts as RL does. The diagonal of that table is trainer/server mismatch. Off the diagonal, the server samples from
one adapter and the trainer computes another: the difference an adapter that did not load, did not swap or reached the
wrong weights would make.

Without ``--model`` it writes a small looped checkpoint with random weights, which takes a few minutes. The kernels
vLLM runs need a hidden size of 2304, so that checkpoint is narrow in its experts and short, not small in width.

    python tools/check_inkling2_lora_serving.py --server-python /envs/inkling-vllm/bin/python
    python tools/check_inkling2_lora_serving.py --server-python ... --model /models/Inkling-2-Mini-bf16

Trainer and server share the GPU unless ``--server-gpu`` names a second one. The real weights are 53 GB in each, so
one H200 takes both with ``--gpu-memory-utilization 0.45`` or so.

A run passes when the diagonal is below ``--tolerance`` and everything else is well above it. Compare the diagonal's
adapters with its base: an adapter adds no mismatch of its own.
"""

import argparse
import binascii
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import httpx
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.configs.shared import FileSystemWeightBroadcastConfig
from prime_rl.configs.trainer import LoRAConfig, ModelConfig
from prime_rl.trainer.lora import get_lora_state
from prime_rl.trainer.model import forward, setup_model
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopForCausalLM, Inkling2LoopTextConfig
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.transports.weights.filesystem import FileSystemWeightSender

TARGETS = ["q_proj", "k_proj", "v_proj", "rel_proj", "gate_proj", "sink_proj", "o_proj", "experts"]
STATES = ("base", "adapter 1", "adapter 2")

SMALL_TEXT_CONFIG = dict(
    vocab_size=1024,
    unpadded_vocab_size=1000,
    hidden_size=2304,
    num_hidden_layers=8,
    # Two layers per block, a sliding and a full one: a prelude block, a core of two blocks run three times, a coda.
    n_layers_per_attn_res_block=2,
    local_layer_ids=[0, 2, 4, 6],
    loop_config=dict(core_start=2, core_end=6, num_passes=3, gate_rank=16, is_varloop=False),
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=128,
    swa_num_attention_heads=8,
    swa_num_key_value_heads=2,
    swa_head_dim=128,
    sliding_window_size=128,
    rel_extent=1024,
    d_rel=16,
    log_scaling_n_floor=64,
    n_routed_experts=16,
    num_experts_per_tok=8,
    intermediate_size=256,
    # Spelled out: vLLM's defaults are no caps, the trainer's are the release's.
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
    route_scale=8.0,
    model_max_length=4096,
    eos_token_id=1,
)

PROMPTS = (
    "Write a Python function that merges two sorted lists.",
    "Explain how a hash table resolves collisions.",
    "List three causes of the French Revolution and describe one of them in detail.",
    "What is the derivative of x * sin(x), and why?",
    "SELECT name FROM users WHERE",
    "The quick brown fox jumps over the lazy dog. Continue the story:",
)


def write_small_checkpoint(path: Path) -> None:
    """A looped checkpoint in the released names and dtypes, whose every path moves the output. The weights' scales
    are those of SkyRL's synthetic Inkling 2 checkpoint (Apache-2.0, ``synthetic_checkpoint.py`` at faae9039): with
    larger ones, BF16 rounding alone moves the logprobs of a looped random model by as much as an adapter."""
    # With the trainer's conversion of an earlier one, which it keeps inside.
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    scale = (256 / SMALL_TEXT_CONFIG["hidden_size"]) ** 0.5
    config = Inkling2LoopTextConfig(**SMALL_TEXT_CONFIG)
    model = Inkling2LoopForCausalLM(config)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn(param.shape, generator=generator)
            if name.endswith(("layernorm.weight", "norm.weight", "q_scale", "rel_scale")):
                param.copy_(1.0 + 0.2 * noise)
            elif name.endswith("query") or name.endswith("message_queries"):
                param.copy_(0.5 * scale * noise)
            elif name.endswith(("conv.weight", "static_weight", "rel_table")):
                param.copy_(0.3 * noise)
            elif name.endswith("embed_tokens.weight"):
                param.copy_(noise)
            elif "message_gates" in name:
                param.copy_(noise / param.shape[-1] ** 0.5)
            elif name.endswith(("o_proj.weight", "experts.down_proj")):
                param.copy_(0.02 * noise)
            else:
                param.copy_(0.02 * scale * noise)
        for name, buffer in model.named_buffers():
            if name.endswith("selection_bias"):
                buffer.copy_(0.1 * torch.randn(buffer.shape, generator=generator))
    state_dict = {name: weight.detach().clone() for name, weight in model.state_dict().items()}
    fp32 = {name for name in state_dict if model.keep_in_fp32_for_weight_transfer(name)}
    state_dict = {name: weight if name in fp32 else weight.to(torch.bfloat16) for name, weight in state_dict.items()}
    checkpoint = {name: weight.contiguous() for name, weight in model.convert_to_hf(state_dict).items()}
    save_file(checkpoint, path / "model.safetensors", metadata={"format": "pt"})
    hf_config = {
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "dtype": "bfloat16",
        "eos_token_id": config.eos_token_id,
        "text_config": {"model_type": "inkling2_loop_text", "torch_dtype": "bfloat16", **SMALL_TEXT_CONFIG},
    }
    (path / "config.json").write_text(json.dumps(hf_config, indent=2))


class Server:
    """prime-rl's inference server, as an RL run starts it by hand for this model."""

    def __init__(self, args: argparse.Namespace, model: Path, work: Path, has_tokenizer: bool) -> None:
        self.model = str(model)
        # Rollouts go through the router that fronts the engine, admin calls to the engine (`backend_port`).
        self.url, self.admin_url = f"http://127.0.0.1:{args.port}", f"http://127.0.0.1:{args.port + 100}"
        config = [
            f"[server]\nport = {args.port}\nhost = '127.0.0.1'",
            "[weight_broadcast]\ntype = 'filesystem'",
            "[vllm]",
            f"model = '{model}'",
            "enable_lora = true",
            # The gate's and the up projection's ranks are served as one adapter.
            f"max_lora_rank = {2 * args.rank}",
            f"max_model_len = {args.max_model_len}",
            f"gpu_memory_utilization = {args.gpu_memory_utilization}",
            "enable_return_routed_experts = true",
            "max_num_seqs = 8",
            "tool_call_parser = 'None'",
            "reasoning_parser = 'None'",
        ]
        if not has_tokenizer:
            config += ["skip_tokenizer_init = true", "tokenizer_mode = 'hf'"]
        config += [f"{line}" for line in args.server_option]
        config += ["[vllm.limit_mm_per_prompt]\nimage = 0"]
        (work / "server.toml").write_text("\n".join(config) + "\n")
        env = {name: value for name, value in os.environ.items() if name not in ("PYTHONPATH", "VIRTUAL_ENV")}
        # The trainer's process group is not the server's.
        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(name, None)
        env["PATH"] = f"{Path(args.server_python).parent}:{env['PATH']}"
        if args.server_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.server_gpu)
        self.log_path = work / "server.log"
        self.process = subprocess.Popen(
            [args.server_python, "-m", "prime_rl.entrypoints.inference", "@", str(work / "server.toml")],
            stdout=self.log_path.open("w"),
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    def wait(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"The server exited with {self.process.returncode}: see {self.log_path}")
            try:
                urls = (self.admin_url, self.url)
                if all(httpx.get(f"{url}/health", timeout=5).status_code == 200 for url in urls):
                    return
            except httpx.TransportError:
                pass
            time.sleep(2)
        raise TimeoutError(f"The server is not up after {timeout:.0f} s: see {self.log_path}")

    def models(self) -> list[str]:
        return [model["id"] for model in httpx.get(f"{self.admin_url}/v1/models", timeout=30).json()["data"]]

    def load_adapter(self, path: Path, step: int) -> float:
        start = time.monotonic()
        response = httpx.post(
            f"{self.admin_url}/load_lora_adapter",
            json={"lora_name": self.model, "lora_path": str(path)},
            params={"weight_version": step},
            timeout=300,
        )
        if response.status_code != 200:
            raise RuntimeError(f"/load_lora_adapter returned {response.status_code}: {response.text[:500]}")
        return time.monotonic() - start

    def sample(self, name: str, version: int, prompt: list[int], max_tokens: int, seed: int) -> "Sample":
        response = httpx.post(
            f"{self.url}/inference/v1/generate",
            json={
                "model": name,
                "token_ids": prompt,
                "sampling_params": {
                    "temperature": 1.0,
                    "max_tokens": max_tokens,
                    "logprobs": 1,
                    "seed": seed,
                    "ignore_eos": True,
                },
                # The orchestrator salts the prefix cache per weight version.
                "cache_salt": f"v{version}",
            },
            timeout=600,
        )
        if response.status_code != 200:
            raise RuntimeError(f"/inference/v1/generate returned {response.status_code}: {response.text[:500]}")
        choice = response.json()["choices"][0]
        return Sample(prompt, choice["token_ids"], decode(choice["completion_logprobs"]), choice["routed_experts"])

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)


def decode(array: dict) -> np.ndarray:
    return np.frombuffer(binascii.a2b_base64(array["data"]), dtype=array["dtype"]).reshape(array["shape"])


class Sample:
    def __init__(self, prompt: list[int], completion: list[int], logprobs: np.ndarray, routed_experts: dict) -> None:
        self.prompt, self.completion = prompt, completion
        self.logprobs = torch.from_numpy(logprobs.astype(np.float32))
        rows = decode(routed_experts)
        assert routed_experts["start"] == 0 and len(rows) == len(self.token_ids) - 1, rows.shape
        # The last token is never forwarded: any row does.
        self.routed_experts = torch.from_numpy(np.concatenate([rows, rows[-1:]]).astype(np.int64))

    @property
    def token_ids(self) -> list[int]:
        return [*self.prompt, *self.completion]


class Trainer:
    def __init__(self, args: argparse.Namespace, model: Path, work: Path) -> None:
        self.lora = LoRAConfig(rank=args.rank, alpha=args.alpha, target_modules=TARGETS, dtype="float32")
        config = ModelConfig(
            name=str(model),
            optimization_dtype="bfloat16",
            attn="flash_attention_2",
            lora=self.lora,
            compile=None,
            ac=None,
        )
        for name, value in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0), ("MASTER_PORT", args.port + 50)):
            os.environ.setdefault(name, str(value))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        setup_torch_distributed(timeout=timedelta(minutes=10))
        resolve_ep(config)
        self.model = setup_model(config, get_parallel_dims(config))
        torch.manual_seed(0)
        get_lora_state().reset_adapter_parameters()
        self.adapter = [param for param in self.model.parameters() if param.requires_grad]
        self.sender = FileSystemWeightSender(work, FileSystemWeightBroadcastConfig(), self.lora)
        dtypes = {str(param.dtype) for param in self.adapter}
        base = {str(param.dtype) for param in self.model.parameters() if not param.requires_grad}
        print(f"trainer: {sum(p.numel() for p in self.adapter):,} adapter parameters in {dtypes}, base in {base}")

    def forward(self, documents: list[list[int]], routed_experts: list[Tensor] | None = None) -> Tensor:
        """The logprob of every token of the packed documents given the ones before it (a document's first has none)."""
        row = torch.tensor([token for document in documents for token in document], device="cuda").unsqueeze(0)
        seq_lens = torch.tensor([len(document) for document in documents], device="cuda")
        position_ids = torch.cat([torch.arange(length, device="cuda") for length in seq_lens.tolist()]).unsqueeze(0)
        routed = None
        if routed_experts is not None:
            routed = torch.cat(routed_experts).unsqueeze(0).to("cuda").int()
        out = forward(
            self.model,
            row,
            position_ids,
            labels=row.roll(-1, dims=1),
            temperature=torch.ones_like(row, dtype=torch.float32),
            seq_lens=seq_lens,
            routed_experts=None if routed is None else routed.long(),
        )
        return out["logprobs"][0].roll(1)

    def train(self, documents: list[list[int]], steps: int, learning_rate: float) -> list[float]:
        """Fit the adapter to the documents. Returns the loss of each step."""
        optimizer = torch.optim.AdamW(self.adapter, lr=learning_rate, weight_decay=0.0)
        mask = torch.ones(sum(len(document) for document in documents), dtype=torch.bool, device="cuda")
        mask[torch.tensor([0, *np.cumsum([len(document) for document in documents])[:-1]], device="cuda")] = False
        losses = []
        for _ in range(steps):
            loss = -self.forward(documents)[mask].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        return losses

    def export(self, step: int) -> Path:
        step_dir = self.sender.step_dir(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        self.sender._broadcast(self.model, step, step_dir)
        return step_dir

    def snapshot(self) -> list[Tensor]:
        # On the host: three copies of the real model's adapter are 6 GB.
        return [local(param).detach().cpu().clone() for param in self.adapter]

    @torch.no_grad()
    def restore(self, snapshot: list[Tensor]) -> None:
        for param, value in zip(self.adapter, snapshot, strict=True):
            local(param).copy_(value)


def local(param: Tensor) -> Tensor:
    """The parameter's only shard."""
    return param.to_local() if isinstance(param, DTensor) else param


def moved_entries(before: list[Tensor], after: list[Tensor]) -> float:
    """The share of the adapter's entries that training changed."""
    changed = sum(int((old != new).sum()) for old, new in zip(before, after, strict=True))
    return changed / sum(param.numel() for param in before)


@torch.no_grad()
def differences(trainer: Trainer, samples: list[Sample], replay: bool) -> tuple[float, float]:
    """Mean and largest |trainer logprob - server logprob| of the sampled tokens."""
    routed = [sample.routed_experts for sample in samples] if replay else None
    logprobs = trainer.forward([sample.token_ids for sample in samples], routed).float().cpu()
    ends = np.cumsum([len(sample.token_ids) for sample in samples])
    sampled = torch.cat([logprobs[end - len(sample.completion) : end] for end, sample in zip(ends, samples)])
    delta = (sampled - torch.cat([sample.logprobs for sample in samples])).abs()
    return delta.mean().item(), delta.max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-python", required=True, help="Python of the environment that serves the model")
    parser.add_argument("--server-gpu", type=int, help="GPU of the server (default: the trainer's)")
    parser.add_argument("--model", type=Path, help="BF16 checkpoint (default: a small random one)")
    parser.add_argument("--work", type=Path, default=Path("outputs/check_inkling2_lora"))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=4, help="training steps before each of the two adapters")
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--server-option", action="append", default=[], help="a further line of the [vllm] table")
    parser.add_argument("--server-timeout", type=float, default=900)
    parser.add_argument("--port", type=int, default=18437)
    parser.add_argument("--tolerance", type=float, default=0.03, help="largest mean |delta logprob| on the diagonal")
    args = parser.parse_args()

    # `timeout` and a cancelled job send SIGTERM: leave through `finally`, so that the server does not outlive the check.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("terminated"))
    args.work.mkdir(parents=True, exist_ok=True)
    work = args.work.resolve()
    model = args.model
    if model is None:
        model = work / "model"
        write_small_checkpoint(model)
    model = model.resolve()
    has_tokenizer = (model / "tokenizer.json").exists()
    if has_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)
        prompts = [tokenizer(prompt)["input_ids"] for prompt in PROMPTS]
    else:
        generator = np.random.default_rng(0)
        prompts = [generator.integers(2, 1000, size=int(size)).tolist() for size in (5, 17, 33, 64, 130, 200)]

    server = Server(args, model, work, has_tokenizer)
    try:
        trainer = Trainer(args, model, work)
        snapshots = [trainer.snapshot()]
        adapters = []
        for step in (1, 2):
            documents = prompts[:3] if step == 1 else prompts[3:]
            losses = trainer.train(documents, args.steps, args.learning_rate)
            snapshots.append(trainer.snapshot())
            adapters.append(trainer.export(step))
            size = (adapters[-1] / "adapter_model.safetensors").stat().st_size / 2**20
            moved = moved_entries(snapshots[-2], snapshots[-1])
            print(f"adapter {step}: losses {[round(loss, 3) for loss in losses]}, {moved:.1%} moved, {size:.0f} MiB")
        torch.cuda.empty_cache()

        print("waiting for the server", flush=True)
        server.wait(args.server_timeout)
        base_name = f"{server.model}-base"
        assert server.models() == [base_name], server.models()

        def sample(name: str, version: int) -> list[Sample]:
            return [server.sample(name, version, prompt, args.max_tokens, seed) for seed, prompt in enumerate(prompts)]

        samples = {"base": sample(base_name, 0)}
        seconds = [server.load_adapter(adapters[0], step=1)]
        samples["adapter 1"] = sample(server.model, 1)
        # The hot swap: another directory under the same adapter name.
        seconds.append(server.load_adapter(adapters[1], step=2))
        samples["adapter 2"] = sample(server.model, 2)
        assert sorted(server.models()) == sorted([base_name, server.model]), server.models()
        # The base still serves without a delta, next to the adapter.
        again = server.sample(base_name, 0, prompts[0], args.max_tokens, seed=0)
        repeated = (again.logprobs - samples["base"][0].logprobs).abs().max().item()
        assert again.completion == samples["base"][0].completion, "the base samples other tokens under the adapter"
        print(f"adapter loads took {seconds[0]:.1f} s and {seconds[1]:.1f} s; base again: max |delta| {repeated:.4f}")

        torch.save({state: [vars(drawn) for drawn in samples[state]] for state in STATES}, work / "samples.pt")

        table: dict[bool, dict[str, dict[str, tuple[float, float]]]] = {}
        for replay in (True, False):
            table[replay] = {served: {} for served in STATES}
            for trained, snapshot in zip(STATES, snapshots):
                trainer.restore(snapshot)
                for served in STATES:
                    table[replay][served][trained] = differences(trainer, samples[served], replay)
            print(f"\nmean (max) |delta logprob|, {'replaying' if replay else 'without'} the server's routing")
            print("served \\ trained".ljust(18) + "".join(state.rjust(20) for state in STATES))
            for served in STATES:
                cells = (f"{mean:.4f} ({largest:.2f})" for mean, largest in table[replay][served].values())
                print(served.ljust(18) + "".join(cell.rjust(20) for cell in cells))
        failures = []
        for served in STATES:
            on = table[True][served][served][0]
            if on > args.tolerance:
                failures.append(f"{served}: mean |delta logprob| {on:.4f} above {args.tolerance}")
            for trained in STATES:
                off = table[True][served][trained][0]
                if trained != served and off < max(5 * on, 2 * args.tolerance):
                    failures.append(f"{served} against {trained}: {off:.4f} does not tell the two apart")
        print("\n" + ("FAILED\n" + "\n".join(failures) if failures else "PASSED"), flush=True)
    finally:
        server.stop()
        if dist.is_initialized():
            dist.destroy_process_group()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

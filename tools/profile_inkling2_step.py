"""Profile one forward and backward of Inkling 2 Mini on one GPU, split into attention, experts, convolutions, FP32
chains and loss.

The model runs as the trainer runs it (BF16 compute, activation checkpointing, compiled layers, fused LM head), without
FSDP and the optimizer. The base is frozen unless ``--train`` is given: the backward then computes the gradients of the
hidden states only, what a LoRA step pays for the base. With a checkpoint directory (release or BF16 export) it loads
the real weights; without one it builds random ones, which ``--num-layers`` and ``--num-experts`` shrink to fit.

    python tools/profile_inkling2_step.py /models/Inkling-2-Mini --tokens 131072 --documents 8192
    python tools/profile_inkling2_step.py --num-layers 12 --num-experts 16 --tokens 16384
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.profiler import ProfilerActivity, profile, record_function

from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig
from prime_rl.trainer.model import apply_ac, apply_compile
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopConfig, Inkling2LoopForCausalLM, Inkling2LoopTextConfig
from prime_rl.trainer.models.inkling2_loop.attention import Inkling2Attention
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head

# First match wins. Compiled layers fuse FP32 work across modules, so the names of their kernels say only that.
KERNEL_CATEGORIES = (
    ("loss", ("range:loss",)),
    ("attention", ("inkling2_tiled_attention", "inkling2_rel_attention")),
    ("convolution", ("inkling2_static_conv", "inkling2_dynamic_conv")),
    ("expert activation", ("inkling2_swiglu",)),
    ("expert GEMMs", ("_grouped_mm",)),
    ("other GEMMs", ("aten::mm", "aten::bmm", "aten::addmm", "aten::baddbmm", "aten::matmul", "aten::linear")),
    ("routing, dispatch", ("argsort", "aten::sort", "aten::topk", "aten::index", "aten::gather", "aten::scatter")),
    ("compiled FP32 chains", ("kernel:triton_",)),
)


def build_model(args: argparse.Namespace) -> Inkling2LoopForCausalLM:
    if args.model is None:
        config = Inkling2LoopTextConfig(
            num_hidden_layers=args.num_layers, n_routed_experts=args.num_experts, unpadded_vocab_size=200058
        )
    else:
        config = Inkling2LoopConfig.from_dict(json.loads((args.model / "config.json").read_text()))
        config.text_config.num_hidden_layers = min(args.num_layers, config.text_config.num_hidden_layers)
    with torch.device("meta"):
        model = Inkling2LoopForCausalLM(config).to(torch.bfloat16)
    model.to_empty(device="cuda")
    if args.model is None:
        generator = torch.Generator("cuda").manual_seed(0)
        for name, param in model.named_parameters():
            mean = 1.0 if name.endswith(("norm.weight", "q_scale", "rel_scale")) else 0.0
            torch.nn.init.normal_(param, mean=mean, std=0.02, generator=generator)
        for buffer in model.buffers():
            buffer.zero_()
    else:
        state_dict = {}
        for shard in sorted(args.model.glob("*.safetensors")):
            if shard.name != "mtp.safetensors":
                state_dict.update(load_file(shard))
        state_dict = model.convert_to_prime(state_dict)
        model.load_state_dict({name: state_dict[name] for name in model.state_dict()})
        model.init_buffers_post_meta()
    for name, buffer in model.named_buffers():
        if name.endswith("selection_bias"):
            buffer.data = buffer.data.float()
    return model


def category(event_path: str) -> str:
    return next((name for name, marks in KERNEL_CATEGORIES if any(mark in event_path for mark in marks)), "other")


def kernel_seconds(prof) -> tuple[dict[str, float], dict[str, float]]:
    """GPU seconds by category and by kernel name. A kernel belongs to the operators and ranges that enclose its
    launch on the CPU."""
    by_category, by_kernel = defaultdict(float), defaultdict(float)
    for event in prof.events():
        kernels = [kernel for kernel in event.kernels if kernel.duration > 0]
        if not kernels:
            continue
        path, parent = [], event
        while parent is not None:
            path.append(f"range:{parent.name}" if parent.name == "loss" else parent.name)
            parent = parent.cpu_parent
        for kernel in kernels:
            by_category[category(" ".join([*path, f"kernel:{kernel.name}"]))] += kernel.duration / 1e6
            by_kernel[f"{kernel.name[:120]} < {event.name[:40]}"] += kernel.duration / 1e6
    return by_category, by_kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, nargs="?", help="checkpoint directory; random weights without one")
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-experts", type=int, default=128, help="random weights only")
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--documents", type=int, default=8192, help="tokens per document")
    parser.add_argument("--ac", choices=["full", "selective", "none"], default="full")
    parser.add_argument("--compile", choices=["on", "off"], default="on")
    parser.add_argument("--kernel", choices=["auto", "dense", "triton"], default="auto")
    parser.add_argument("--lm-head-chunk", type=int, default=8192, help="tokens per LM-head chunk")
    parser.add_argument("--loss", choices=["sft", "rl"], default="sft")
    parser.add_argument("--train", action="store_true", help="also compute the gradients of every weight")
    parser.add_argument("--steps", type=int, default=3, help="timed steps after the two that compile")
    parser.add_argument("--top", type=int, default=25, help="kernels listed")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    model = build_model(args)
    model.requires_grad_(args.train)
    inject_prime_lm_head(model, chunk_size=args.lm_head_chunk)
    for module in model.modules():
        if isinstance(module, Inkling2Attention):
            module.kernel = args.kernel
    if args.ac != "none":
        apply_ac(model, ActivationCheckpointConfig(mode=args.ac))
    if args.compile == "on":
        apply_compile(model, CompileConfig())

    generator = torch.Generator("cuda").manual_seed(1)
    vocab = model.config.get_text_config().num_logits
    input_ids = torch.randint(0, vocab, (1, args.tokens), device="cuda", generator=generator)
    labels = torch.randint(0, vocab, (1, args.tokens), device="cuda", generator=generator)
    full, rest = divmod(args.tokens, args.documents)
    seq_lens = torch.tensor([args.documents] * full + [rest] * (rest > 0), device="cuda")
    temperature = torch.ones(1, args.tokens, device="cuda") if args.loss == "rl" else None
    embed = model.get_input_embeddings()
    embed_norm = model.model.embed_norm
    head = model.lm_head
    inner_head = head.forward

    def ranged_head(*head_args, **head_kwargs):
        with record_function("loss"):
            return inner_head(*head_args, **head_kwargs)

    head.forward = ranged_head

    def step() -> tuple[float, float]:
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        embeddings = embed(input_ids)
        embeddings = embed_norm(embeddings, embeddings.dtype)
        if not args.train:
            embeddings = embeddings.detach().requires_grad_()
        output = model(inputs_embeds=embeddings, labels=labels, temperature=temperature, seq_lens=seq_lens)
        loss = output["loss"] if args.loss == "sft" else -output["logprobs"].sum()
        torch.cuda.synchronize()
        middle = time.perf_counter()
        (loss / args.tokens).backward()
        torch.cuda.synchronize()
        return middle - start, time.perf_counter() - middle

    for _ in range(2):
        step()
    torch.cuda.reset_peak_memory_stats()
    forward, backward = min((step() for _ in range(args.steps)), key=sum)
    seconds = forward + backward
    print(
        f"{torch.cuda.get_device_name()}: {args.tokens} tokens in documents of {args.documents}, {args.num_layers} layers, "
        f"AC {args.ac}, compile {args.compile}, kernel {args.kernel}, {args.loss} loss, "
        f"{'all weights' if args.train else 'base frozen'}\n"
        f"forward {forward:.3f} s, backward {backward:.3f} s, {args.tokens / seconds:,.0f} tokens/s, "
        f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
        f"of which model {sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30:.2f} GiB"
    )

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True) as prof:
        step()
    by_category, by_kernel = kernel_seconds(prof)
    total = sum(by_category.values())
    print(f"GPU time of one profiled step: {total:.3f} s")
    for name, value in sorted(by_category.items(), key=lambda item: -item[1]):
        print(f"{name:>22} {value:8.3f} s {100 * value / total:5.1f} %")
    for name, value in sorted(by_kernel.items(), key=lambda item: -item[1])[: args.top]:
        print(f"{value:8.3f} s  {name}")


if __name__ == "__main__":
    main()

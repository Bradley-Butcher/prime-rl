"""Time and peak memory of the Inkling 2 attention kernels: the dense FP32 reference against the Triton tiles.

One document per row, real head sizes, random BF16 operands; forward and backward run through autograd from the
relative states and the table, so the reference's materialized bias counts. Needs one GPU and no weights.

    python tools/bench_inkling2_kernels.py                    # 8K/32K/64K tokens, 24/12/6 heads, window and full
    python tools/bench_inkling2_kernels.py --tokens 131072 --heads 24 --kernels triton
    python tools/bench_inkling2_kernels.py --sweep            # launch shapes of the Triton kernels on this GPU
"""

import argparse
import gc
import itertools
import time

import torch
import triton

from prime_rl.trainer.models.inkling2_loop.attention import rel_attention, tiled_rel_attention
from prime_rl.trainer.models.kernels import inkling2_attention as kernels

HEAD_DIM, D_REL, EXTENT, WINDOW = 128, 16, 1024, 1024


def operands(num_tokens: int, num_heads: int) -> list[torch.Tensor]:
    generator = torch.Generator("cuda").manual_seed(0)
    shapes = ((num_heads, HEAD_DIM), (num_heads // 3, HEAD_DIM), (num_heads // 3, HEAD_DIM), (num_heads, D_REL))
    tensors = [torch.randn(num_tokens, *shape, device="cuda", generator=generator) for shape in shapes]
    tensors.append(0.1 * torch.randn(D_REL, EXTENT, device="cuda", generator=generator))
    return [tensor.bfloat16().requires_grad_() for tensor in tensors]


def first_keys(num_tokens: int, layer: str) -> torch.Tensor:
    token = torch.arange(num_tokens, device="cuda")
    return torch.clamp(token - (WINDOW - 1), min=0) if layer == "window" else torch.zeros_like(token)


def attention(kernel: str, inputs: list[torch.Tensor], first_key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v, rel_states, rel_table = inputs
    if kernel == "triton":
        return tiled_rel_attention(q, k, v, rel_states, rel_table, first_key, HEAD_DIM**-0.5)
    return rel_attention(q, k, v, torch.matmul(rel_states, rel_table), first_key, HEAD_DIM**-0.5)


def elapsed(start: float) -> float:
    torch.cuda.synchronize()
    return time.perf_counter() - start


def measure(kernel: str, layer: str, num_tokens: int, num_heads: int, repeats: int) -> dict | None:
    """Best forward and backward seconds and the peak bytes above the operands. ``None`` when out of memory."""
    inputs = operands(num_tokens, num_heads)
    first_key = first_keys(num_tokens, layer)
    best = [float("inf"), float("inf")]
    try:
        # The first call compiles the kernels.
        for repeat in range(repeats + 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            start = time.perf_counter()
            out, lse = attention(kernel, inputs, first_key)
            forward = elapsed(start)
            grads = (torch.ones_like(out), torch.ones_like(lse))
            start = time.perf_counter()
            torch.autograd.grad((out, lse), inputs, grads)
            backward = elapsed(start)
            peak = torch.cuda.max_memory_allocated() - before
            del out, lse, grads
            if repeat > 0 or repeats == 0:
                best = [min(best[0], forward), min(best[1], backward)]
    except torch.OutOfMemoryError:
        return None
    finally:
        torch.cuda.empty_cache()
    return dict(forward=best[0], backward=best[1], peak=peak)


def pairs(layer: str, num_tokens: int, num_heads: int) -> int:
    """Query-key pairs of the layer, summed over the heads."""
    window = min(WINDOW, num_tokens) if layer == "window" else num_tokens
    return num_heads * (window * (window + 1) // 2 + (num_tokens - window) * window)


def compare(args: argparse.Namespace) -> None:
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, triton {triton.__version__}")
    print(
        f"{'layer':>6} {'tokens':>7} {'heads':>5} {'kernel':>7} {'fwd ms':>9} {'bwd ms':>9} {'peak GB':>8} {'TFLOP/s':>8}"
    )
    for layer, num_tokens, num_heads in itertools.product(args.layers, args.tokens, args.heads):
        for kernel in args.kernels:
            repeats = args.repeats if kernel == "triton" or num_tokens <= 8192 else 0
            result = measure(kernel, layer, num_tokens, num_heads, repeats)
            row = f"{layer:>6} {num_tokens:>7} {num_heads:>5} {kernel:>7}"
            if result is None:
                print(f"{row} {'out of memory':>28}", flush=True)
                continue
            # Two products forward and seven backward, each of head_dim multiply-adds per pair.
            flops = 2 * 9 * HEAD_DIM * pairs(layer, num_tokens, num_heads)
            rate = flops / (result["forward"] + result["backward"]) / 1e12
            print(
                f"{row} {result['forward'] * 1e3:>9.1f} {result['backward'] * 1e3:>9.1f} {result['peak'] / 2**30:>8.2f} "
                f"{rate if kernel == 'triton' else float('nan'):>8.1f}",
                flush=True,
            )


def sweep(args: argparse.Namespace) -> None:
    """Time the Triton forward and backward under other launch shapes of one kernel at a time, the other kernels at
    their defaults. The winners go into ``kernels.LAUNCH``."""
    num_tokens, num_heads = args.tokens[0], args.heads[0]
    inputs = [tensor.detach() for tensor in operands(num_tokens, num_heads)]
    hopper = torch.cuda.get_device_capability()[0] >= 9
    tables = kernels.LAUNCH if hopper else kernels._LAUNCH_BEFORE_SM90
    print(f"{torch.cuda.get_device_name()}, {num_tokens} tokens, {num_heads} heads")
    for layer, name in itertools.product(args.layers, ("forward", "query", "key")):
        first_key = first_keys(num_tokens, layer)
        default, results = tables[name], []
        out, lse = kernels.attention_forward(*inputs, first_key, HEAD_DIM**-0.5)
        for shape in itertools.product((32, 64, 128), (32, 64, 128), (4, 8), (1, 2, 3)):
            tables[name] = kernels.Launch(*shape)
            try:
                times = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if name == "forward":
                        kernels.attention_forward(*inputs, first_key, HEAD_DIM**-0.5)
                    else:
                        kernels.attention_backward(out, lse, out, lse, *inputs, first_key, HEAD_DIM**-0.5, False)
                    times.append(elapsed(start))
            except triton.runtime.errors.OutOfResources:
                # The error's frames hold the launch's buffers in a cycle.
                gc.collect()
                continue
            results.append((min(times[1:]), shape))
        tables[name] = default
        for seconds, shape in sorted(results)[:6]:
            print(f"{layer:>6} {name:>8} (queries, keys, warps, stages)={shape} {seconds * 1e3:8.2f} ms", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768, 65536])
    parser.add_argument("--heads", type=int, nargs="+", default=[24, 12, 6])
    parser.add_argument("--layers", nargs="+", default=["window", "full"], choices=["window", "full"])
    parser.add_argument("--kernels", nargs="+", default=["dense", "triton"], choices=["dense", "triton"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    sweep(args) if args.sweep else compare(args)


if __name__ == "__main__":
    main()

"""Building blocks of the Inkling 2 decoder around its attention and MoE.

Rounding follows the serving kernels: norms, attention-residual reads, convolutions and gates compute in FP32 and
round once to the residual's dtype; only GEMMs run in that dtype. Functions that return FP32 say so. This holds in
eager execution. A compiled layer keeps FP32 where Inductor fuses two such operations and skips the rounding between
them, so its outputs differ from eager's at rounding level.

Activations are ``[1, T, ...]``: one packed row of documents. ``positions [T]`` counts tokens inside each document,
which is what keeps a convolution from reaching into the previous document. Under context parallelism each rank
holds an equal shard of the row, and a convolution reads the tokens before its shard from the rank that holds them.
"""

from collections.abc import Sequence

import torch
from torch import nn

from prime_rl.utils.cp import CPContext, gather_for_cp

DEPTH_MIX_EPS = 1e-6
L2_CLIP_EPS = 1e-6


def unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``x`` scaled to unit RMS over its last dim, in FP32."""
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


class Inkling2RMSNorm(nn.Module):
    """RMSNorm that takes an FP32 or residual-dtype input and rounds its FP32 result to ``dtype``."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (unweighted_rms_norm(x, self.eps) * self.weight.float()).to(dtype)


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Scale each head vector of ``x [T, heads, d]`` by ``scale [heads, d]``, then shrink it to L2 norm
    ``threshold`` if it is longer. Shorter vectors are unchanged."""
    values = x.float()
    if scale is not None:
        values = values * scale.float()
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return (values * (threshold / (norm + L2_CLIP_EPS)).clamp(max=1.0)).to(x.dtype)


def log_length_scale(positions: torch.Tensor, n_floor: int, alpha: float) -> torch.Tensor:
    """``1 + alpha * log(max((position + 1) / n_floor, 1))`` as FP32 ``[T, 1, 1]``: one up to ``n_floor`` tokens,
    growing past them."""
    return (1.0 + alpha * torch.log(((positions + 1).float() / n_floor).clamp(min=1.0))).view(-1, 1, 1)


def _tokens_before(x: torch.Tensor, count: int, cp_context: CPContext) -> torch.Tensor | None:
    """The ``count`` tokens before the context-parallel shard ``x [1, T, ...]``: the end of the previous rank's shard.
    The first shard gets the end of the last one, the way ``torch.roll`` wraps a whole row, and the document mask
    drops it there too. ``None`` without context parallelism."""
    if not cp_context.cp_enabled:
        return None
    ends = gather_for_cp(x[:, x.shape[1] - count :], cp_context.cp_group)
    previous_rank = (cp_context.cp_rank - 1) % cp_context.cp_world_size
    return ends[:, previous_rank * count : (previous_rank + 1) * count]


def _lagged(x: torch.Tensor, lag: int, positions: torch.Tensor, before: torch.Tensor | None) -> torch.Tensor:
    """``x[:, t - lag]`` where that token is in ``t``'s document, zero elsewhere. ``before`` holds the tokens before
    ``x`` when ``x`` is a shard of the row (see ``_tokens_before``)."""
    if lag == 0:
        return x
    in_document = (positions >= lag).view(1, -1, *([1] * (x.dim() - 2)))
    if before is None:
        shifted = torch.roll(x, shifts=lag, dims=1)
    else:
        shifted = torch.cat([before[:, before.shape[1] - lag :], x[:, : x.shape[1] - lag]], dim=1)
    return shifted * in_document


class StaticShortConv(nn.Module):
    """Causal depthwise convolution with an identity residual, scaled by ``(kernel + 1) ** -0.5``."""

    cp_context: CPContext = CPContext()

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        # The checkpoint's Conv1d layout: tap `kernel_size - 1` weights the current token.
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))

    def forward(self, x: torch.Tensor, source: torch.Tensor | None, positions: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float().squeeze(1)
        kernel_size = weight.shape[1]
        values = x.float()
        before = _tokens_before(values, kernel_size - 1, self.cp_context)
        total = values
        for lag in range(kernel_size):
            total = total + _lagged(values, lag, positions, before) * weight[:, kernel_size - 1 - lag]
        return (total * (kernel_size + 1) ** -0.5).to(x.dtype)


class DynamicShortConv(nn.Module):
    """Causal convolution whose taps are static per-channel weights plus per-token weights projected from
    ``source`` and shared within a head. Identity residual, scaled by ``(2 * kernel + 1) ** -0.5``."""

    cp_context: CPContext = CPContext()

    def __init__(self, channels: int, source_size: int, num_heads: int, kernel_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Row `lag` weights the token `lag` steps back.
        self.static_weight = nn.Parameter(torch.empty(kernel_size, channels))
        self.weight_proj = nn.Linear(source_size, num_heads * kernel_size, bias=False)

    def forward(self, x: torch.Tensor, source: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        kernel_size = self.static_weight.shape[0]
        token_taps = self.weight_proj(source).float().unflatten(-1, (self.num_heads, 1, kernel_size))
        static_taps = self.static_weight.float().unflatten(-1, (self.num_heads, -1))
        values = x.float().unflatten(-1, (self.num_heads, -1))
        before = _tokens_before(values, kernel_size - 1, self.cp_context)
        total = values
        for lag in range(kernel_size):
            total = total + _lagged(values, lag, positions, before) * (static_taps[lag] + token_taps[..., lag])
        return (total * (2 * kernel_size + 1) ** -0.5).flatten(-2).to(x.dtype)


def depth_mix(
    sources: Sequence[torch.Tensor], queries: torch.Tensor, source_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, ...]:
    """Attend over depth: one softmax mix of the unit-RMS ``sources``, each ``[1, T, H]``, per row of
    ``queries [m, H]``. ``source_mask [n]`` marks the sources that exist; the others get no weight.

    Returns ``m`` FP32 tensors ``[1, T, H]``. The RMS scale of a source is one number per token, so it is taken out of
    both sums over ``H``: a score is a reduction of the source itself and a mix one pass over all of them, which a
    compiled layer fuses with the norms around them. Nothing holds the sources in FP32, or their normed values. The form
    is ``_mix_sources`` of NVIDIA-NeMo/Automodel pull request 4914 (Apache-2.0, nemo_automodel/components/models/inkling2/
    layers.py), with a mask over the sources.
    """
    sources = [source.float() for source in sources]
    scales = [torch.rsqrt(source.square().mean(-1) + DEPTH_MIX_EPS) for source in sources]
    mixes = []
    for query in queries.float().unbind(0):
        scores = torch.stack([(source * query).sum(-1) * scale for source, scale in zip(sources, scales)], dim=-1)
        scores = scores * query.shape[-1] ** -0.5
        if source_mask is not None:
            scores = scores.masked_fill(~source_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unbind(-1)
        terms = [(weight * scale).unsqueeze(-1) * source for weight, scale, source in zip(weights, scales, sources)]
        mixes.append(sum(terms[1:], terms[0]))
    return tuple(mixes)


def block_sources(
    block_states: torch.Tensor, block_mask: torch.Tensor, *later: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """The sources of a depth mix and their mask: the block slots, then the ``later [1, T, H]`` states.

    The decoder passes ``block_states [1, T, slots, H]`` with one slot per block of the whole model and
    ``block_mask [slots]`` marking the finished ones, so that every layer sees the same shapes.
    """
    return (*block_states.unbind(2), *later), torch.cat([block_mask, block_mask.new_ones(len(later))])


class DepthMix(nn.Module):
    """A sub-layer's input: a learned mix of the finished blocks and the running sum of the current one."""

    def __init__(self, hidden_size: int, num_queries: int = 1) -> None:
        super().__init__()
        self.query = nn.Parameter(
            torch.zeros(hidden_size) if num_queries == 1 else torch.zeros(num_queries, hidden_size)
        )

    def forward(self, sources: Sequence[torch.Tensor], source_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``sources`` each ``[1, T, H]``, ``source_mask [n]``. Returns an FP32 ``[1, T, H]`` per query."""
        return depth_mix(sources, self.query.view(-1, self.query.shape[-1]), source_mask)


class MessageGate(nn.Module):
    """Blend a loop message into a sub-layer input: ``(1 - forget) * state + gain * message``, with both gates a
    tanh of a low-rank projection of the normed pair."""

    def __init__(self, hidden_size: int, rank: int, eps: float) -> None:
        super().__init__()
        self.down_proj = nn.Linear(2 * hidden_size, rank, bias=False)
        self.message_proj = nn.Linear(rank, hidden_size, bias=False)
        self.forget_proj = nn.Linear(rank, hidden_size, bias=False)
        self.eps = eps

    def forward(self, state: torch.Tensor, message: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Returns FP32."""
        state, message = state.float(), message.float()
        # Rounded before they are joined: a compiled layer would hold the joined pair in FP32.
        normed = torch.cat([unweighted_rms_norm(x, self.eps).to(dtype) for x in (state, message)], dim=-1)
        hidden = self.down_proj(normed).float()
        forget = torch.tanh(hidden @ self.forget_proj.weight.float().t())
        gain = torch.tanh(hidden @ self.message_proj.weight.float().t())
        return (1.0 - forget) * state + gain * message


class Inkling2FinalNorm(nn.Module):
    """The model's output read: a depth mix over every block and the last running sum, then RMSNorm."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, block_states: torch.Tensor, block_mask: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        sources, source_mask = block_sources(block_states, block_mask, current)
        (mixed,) = depth_mix(sources, self.query.unsqueeze(0), source_mask)
        return (unweighted_rms_norm(mixed, self.eps) * self.weight.float()).to(current.dtype)


__all__ = [
    "DepthMix",
    "DynamicShortConv",
    "Inkling2FinalNorm",
    "Inkling2RMSNorm",
    "MessageGate",
    "StaticShortConv",
    "block_sources",
    "depth_mix",
    "l2_clip",
    "log_length_scale",
    "unweighted_rms_norm",
]

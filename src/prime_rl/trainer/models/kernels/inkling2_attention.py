# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from NVIDIA-NeMo/Automodel pull request 4914, nemo_automodel/components/models/inkling2/triton_attention.py,
# and changed: one row instead of a batch, the sink stays outside (the log-sum-exp is an output with a gradient), the
# relative logits are built for a few KV heads at a time, reads are clamped instead of masked at the row's end, the
# key kernel reads its relative logits as vectors, and the query kernel clears the gradient columns it leaves out.
"""Triton kernels of Inkling 2's attention: grouped-query attention over one packed row with a relative bias.

A query sees the keys from its ``first_key`` to itself. The score of query ``i`` on key ``j`` is
``scale * q_i . k_j + r_i . table[:, i - j]``, where ``r_i`` are 16 relative states per query head and the table's
column is zero from its extent on. The sink and the gate are not part of the kernels.

The relative logits are read as tiles from a *profile*: every query's row of the product ``r @ table``, stored in key
order so that the tile of a block of queries by a block of keys is a rectangle. Stored by distance instead, the logits
of neighbouring queries for one key would sit one column apart, and a kernel would read a tile one element at a time.
A row of ``T`` tokens has ``T * heads * (extent + align)`` profile elements, 7 GB in BF16 for 128K tokens and 24 heads,
so they are built, used and dropped for as many KV heads at a time as ``PROFILE_BYTES`` allows, and only the relative
states are kept for backward. Nothing here waits on the device.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

# Profile bytes (and as many bytes of their gradient in backward) alive at a time. One KV head's share is the minimum.
PROFILE_BYTES = 1 << 30
# Elements kept free on both sides of a profile: the key kernel reads whole tiles and drops the pairs outside the
# extent, whose columns can lie before or after the pair's row.
_PROFILE_PAD = 512


@dataclass(frozen=True)
class Launch:
    """Tile (queries, keys) and launch options of one kernel."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


# Measured by NVIDIA on GB200 at head_dim 128. `tools/bench_inkling2_kernels.py --sweep` measures others.
LAUNCH = {
    "forward": Launch(64, 64, 4, 2),
    "query": Launch(64, 32, 4, 3),
    "key": Launch(128, 128, 8, 1),
}
# The key kernel's tile does not fit into the shared memory of GPUs before Hopper. Measured on an RTX 4000 Ada (SM89).
_LAUNCH_BEFORE_SM90 = LAUNCH | {"forward": Launch(128, 64, 8, 2), "key": Launch(32, 64, 8, 3)}
# FP32 operands (exact checks) take twice the shared memory per tile.
_LAUNCH_FP32 = Launch(32, 32, 4, 1)
# Query heads per program of the kernel that prepares the output gradient.
_TERM_ROWS = 64
# Triton compiles a kernel again for an argument that is one, or a multiple of 16: these differ from row to row and
# from chunk to chunk, and no tile's alignment depends on them.
_LAYOUT_ARGS = ["T", "E", "HEAD0", "KV0", "ROWS", "HEADS"]


def _launch(name: str, q: torch.Tensor) -> Launch:
    if q.dtype == torch.float32:
        return _LAUNCH_FP32
    hopper = torch.cuda.get_device_capability(q.device)[0] >= 9
    return (LAUNCH if hopper else _LAUNCH_BEFORE_SM90)[name]


@triton.jit
def _load_rows(base, rows, offs_d):
    return tl.load(base + rows[:, None] + offs_d[None, :])


@triton.jit
def _offsets(token, heads, head, D: tl.constexpr, WIDE: tl.constexpr):
    """Element offset of each token's head vector in a ``[T, heads, D]`` tensor."""
    if WIDE:
        token = token.to(tl.int64)
    return (token * heads + head) * D


@triton.jit
def _profile_rows(offs_m, head, ROWS, HEADS, EXTENT, ALIGN: tl.constexpr, PW: tl.constexpr, WIDE: tl.constexpr):
    """Per query, the profile offset that plus a key's index addresses the pair's logit.

    The rows are ordered by the query's residue in its block of ALIGN queries, then by block and head. In a row, key n
    is column n - block start + EXTENT (the extent rounded up to ALIGN), the same for every query of the block.
    """
    block = offs_m // ALIGN
    row = (offs_m % ALIGN) * ROWS + block * HEADS + head
    if WIDE:
        row = row.to(tl.int64)
    return row * PW - block * ALIGN + EXTENT


@triton.jit
def _key_blocks(lo, pid_m, T, E, BM: tl.constexpr, BN: tl.constexpr):
    """The key blocks of one query block: [first, near) without and [near, end) with relative logits."""
    kb_lo = tl.min(lo, axis=0) // BN
    q_last = tl.minimum(pid_m * BM + BM - 1, T - 1)
    kb_end = q_last // BN + 1
    # Key blocks before kb_near lie beyond the extent of every query of the block.
    kb_near = tl.minimum(tl.maximum(kb_lo, tl.maximum(pid_m * BM - E + 1, 0) // BN), kb_end)
    return kb_lo, kb_near, kb_end


@triton.jit
def _scores(q, k, PROF, prof_row, offs_m, offs_n, lo, E, scale, PRECISION: tl.constexpr, REL: tl.constexpr):
    """The scores [queries, keys] of one tile, which pairs are allowed, and which of those have a relative logit."""
    s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale
    allowed = (offs_n[None, :] >= lo[:, None]) & (offs_n[None, :] <= offs_m[:, None])
    in_rel = allowed
    if REL:
        in_rel = allowed & ((offs_m[:, None] - offs_n[None, :]) < E)
        r = tl.load(PROF + prof_row[:, None] + offs_n[None, :])
        s = s + tl.where(in_rel, r.to(tl.float32), 0.0)
    return s, allowed, in_rel


@triton.jit
def _forward_tile(
    q, K, V, PROF, prof_row, offs_m, lo, m_i, l_i, acc, kb, kvh, T, E, scale,
    KVH: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, PRECISION: tl.constexpr, WIDE: tl.constexpr,
    REL: tl.constexpr,
):  # fmt: skip
    """Fold one key block into the running softmax (base 2) and the value sum."""
    offs_n = kb * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    # Keys past the row's end read its last key. They are after every query of the row.
    col = _offsets(tl.minimum(offs_n, T - 1), KVH, kvh, D, WIDE)
    k = _load_rows(K, col, offs_d)
    s, allowed, _ = _scores(q, k, PROF, prof_row, offs_m, offs_n, lo, E, scale, PRECISION, REL)
    s = tl.where(allowed, s * 1.4426950408889634, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    v = _load_rows(V, col, offs_d)
    acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None], input_precision=PRECISION)
    return m_new, l_i, acc


@triton.jit(do_not_specialize=_LAYOUT_ARGS)
def _forward_kernel(
    Q, K, V, PROF, LO, OUT, LSE, T, E, scale, HEAD0, ROWS, HEADS,
    H: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    PRECISION: tl.constexpr, WIDE: tl.constexpr, EXTENT: tl.constexpr, ALIGN: tl.constexpr, PW: tl.constexpr,
):  # fmt: skip
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    h = HEAD0 + head
    kvh = h // (H // KVH)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < T
    # Queries past the row's end read its last query. No key is allowed for them, and they are not stored.
    q = _load_rows(Q, _offsets(tl.minimum(offs_m, T - 1), H, h, D, WIDE), offs_d)
    lo = tl.load(LO + offs_m, mask=row_ok, other=2147483647)
    # The softmax starts below every score, at a finite value: a tile without an allowed key must not give inf - inf.
    m_i = tl.full([BM], -1e30, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    prof_row = _profile_rows(offs_m, head, ROWS, HEADS, EXTENT, ALIGN, PW, WIDE)
    kb_lo, kb_near, kb_end = _key_blocks(lo, pid_m, T, E, BM, BN)
    for kb in range(kb_lo, kb_near):
        m_i, l_i, acc = _forward_tile(
            q, K, V, PROF, prof_row, offs_m, lo, m_i, l_i, acc, kb, kvh, T, E, scale,
            KVH, D, BN, PRECISION, WIDE, False,
        )  # fmt: skip
    for kb in range(kb_near, kb_end):
        m_i, l_i, acc = _forward_tile(
            q, K, V, PROF, prof_row, offs_m, lo, m_i, l_i, acc, kb, kvh, T, E, scale,
            KVH, D, BN, PRECISION, WIDE, True,
        )  # fmt: skip
    row = _offsets(offs_m, H, h, 1, WIDE)
    tl.store(OUT + (row * D)[:, None] + offs_d[None, :], acc / l_i[:, None], mask=row_ok[:, None])
    tl.store(LSE + row, (m_i + tl.math.log2(l_i)) * 0.6931471805599453, mask=row_ok)


@triton.jit
def _query_tile(
    q, do, K, V, PROF, DPROF, prof_row, offs_m, lo, lse, delta, dq, kb, kvh, T, E, scale,
    KVH: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, PRECISION: tl.constexpr, WIDE: tl.constexpr,
    REL: tl.constexpr,
):  # fmt: skip
    """Add one key block's share of the query gradient and write its relative-logit gradient."""
    offs_n = kb * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    col = _offsets(tl.minimum(offs_n, T - 1), KVH, kvh, D, WIDE)
    k = _load_rows(K, col, offs_d)
    v = _load_rows(V, col, offs_d)
    s, allowed, in_rel = _scores(q, k, PROF, prof_row, offs_m, offs_n, lo, E, scale, PRECISION, REL)
    p = tl.where(allowed, tl.math.exp2((s - lse[:, None]) * 1.4426950408889634), 0.0)
    dp = tl.dot(do, tl.trans(v), input_precision=PRECISION)
    ds = p * (dp - delta[:, None])
    dq = tl.dot(ds.to(k.dtype), k, dq, input_precision=PRECISION)
    if REL:
        # Each pair owns one profile element and this program owns its queries' rows. The whole tile is written as
        # aligned vectors: its other elements belong to no pair and are zero.
        tl.store(DPROF + prof_row[:, None] + offs_n[None, :], tl.where(in_rel, ds, 0.0).to(DPROF.dtype.element_ty))
    return dq


@triton.jit(do_not_specialize=_LAYOUT_ARGS)
def _backward_query_kernel(
    Q, K, V, PROF, LO, LSE, DO, DELTA, DQ, DPROF, T, E, scale, HEAD0, ROWS, HEADS,
    H: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    PRECISION: tl.constexpr, WIDE: tl.constexpr, EXTENT: tl.constexpr, ALIGN: tl.constexpr, PW: tl.constexpr,
):  # fmt: skip
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    h = HEAD0 + head
    kvh = h // (H // KVH)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < T
    row = _offsets(tl.minimum(offs_m, T - 1), H, h, 1, WIDE)
    q = _load_rows(Q, row * D, offs_d)
    do = _load_rows(DO, row * D, offs_d)
    lse = tl.load(LSE + row)
    delta = tl.load(DELTA + row)
    lo = tl.load(LO + offs_m, mask=row_ok, other=2147483647)
    dq = tl.zeros([BM, D], tl.float32)
    prof_row = _profile_rows(offs_m, head, ROWS, HEADS, EXTENT, ALIGN, PW, WIDE)
    kb_lo, kb_near, kb_end = _key_blocks(lo, pid_m, T, E, BM, BN)
    for kb in range(kb_lo, kb_near):
        dq = _query_tile(
            q, do, K, V, PROF, DPROF, prof_row, offs_m, lo, lse, delta, dq, kb, kvh, T, E, scale,
            KVH, D, BN, PRECISION, WIDE, False,
        )  # fmt: skip
    for kb in range(kb_near, kb_end):
        dq = _query_tile(
            q, do, K, V, PROF, DPROF, prof_row, offs_m, lo, lse, delta, dq, kb, kvh, T, E, scale,
            KVH, D, BN, PRECISION, WIDE, True,
        )  # fmt: skip
    tl.store(DQ + (row * D)[:, None] + offs_d[None, :], dq * scale, mask=row_ok[:, None])
    # The profile columns outside [kb_near, kb_end) belong to no pair. Column 0 of these rows holds key first * BN.
    first = (pid_m * BM // ALIGN * ALIGN - EXTENT) // BN
    last = first + PW // BN
    zeros = tl.zeros([BM, BN], DPROF.dtype.element_ty)
    for block in range(first, tl.minimum(tl.maximum(kb_near, first), last)):
        tl.store(DPROF + prof_row[:, None] + (block * BN + tl.arange(0, BN))[None, :], zeros)
    for block in range(tl.minimum(tl.maximum(kb_end, first), last), last):
        tl.store(DPROF + prof_row[:, None] + (block * BN + tl.arange(0, BN))[None, :], zeros)


@triton.jit
def _softmax_term_kernel(GRAD, GRAD_LSE, OUT, DO, DELTA, rows, D: tl.constexpr, R: tl.constexpr):
    """Per query head: sum(dO * O) - dLSE in FP32, and dO rounded to the dtype of the products."""
    row = tl.program_id(0) * R + tl.arange(0, R)
    row_ok = row < rows
    offsets = (row.to(tl.int64) * D)[:, None] + tl.arange(0, D)[None, :]
    grad = tl.load(GRAD + offsets, mask=row_ok[:, None], other=0.0)
    out = tl.load(OUT + offsets, mask=row_ok[:, None], other=0.0)
    delta = tl.sum(grad * out, axis=1) - tl.load(GRAD_LSE + row, mask=row_ok, other=0.0)
    tl.store(DELTA + row, delta, mask=row_ok)
    tl.store(DO + offsets, grad.to(DO.dtype.element_ty), mask=row_ok[:, None])


@triton.jit
def _key_tile(
    k, v, Q, DO, PROF, LO, LSE, DELTA, offs_n, dk, dv, qb, h, head, T, E, scale, ROWS, HEADS,
    H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, PRECISION: tl.constexpr, WIDE: tl.constexpr,
    EXTENT: tl.constexpr, ALIGN: tl.constexpr, PW: tl.constexpr, REL: tl.constexpr,
):  # fmt: skip
    """Add one query block's share of the key and value gradients."""
    offs_m = qb * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    row = _offsets(tl.minimum(offs_m, T - 1), H, h, 1, WIDE)
    q = _load_rows(Q, row * D, offs_d)
    do = _load_rows(DO, row * D, offs_d)
    lse = tl.load(LSE + row)
    delta = tl.load(DELTA + row)
    lo = tl.load(LO + offs_m, mask=offs_m < T, other=2147483647)
    prof_row = _profile_rows(offs_m, head, ROWS, HEADS, EXTENT, ALIGN, PW, WIDE)
    s, allowed, _ = _scores(q, k, PROF, prof_row, offs_m, offs_n, lo, E, scale, PRECISION, REL)
    p = tl.where(allowed, tl.math.exp2((s - lse[:, None]) * 1.4426950408889634), 0.0)
    dv = tl.dot(tl.trans(p).to(do.dtype), do, dv, input_precision=PRECISION)
    dp = tl.dot(do, tl.trans(v), input_precision=PRECISION)
    ds = p * (dp - delta[:, None])
    dk = tl.dot(tl.trans(ds).to(q.dtype), q, dk, input_precision=PRECISION)
    return dk, dv


@triton.jit(do_not_specialize=_LAYOUT_ARGS)
def _backward_key_kernel(
    Q, K, V, PROF, LO, LAST, LSE, DO, DELTA, DK, DV, T, E, scale, KV0, ROWS, HEADS,
    H: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    PRECISION: tl.constexpr, WIDE: tl.constexpr, EXTENT: tl.constexpr, ALIGN: tl.constexpr, PW: tl.constexpr,
):  # fmt: skip
    pid_n = tl.program_id(0)
    kv = tl.program_id(1)
    kvh = KV0 + kv
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    col_ok = offs_n < T
    col = _offsets(tl.minimum(offs_n, T - 1), KVH, kvh, D, WIDE)
    k = _load_rows(K, col, offs_d)
    v = _load_rows(V, col, offs_d)
    last = tl.load(LAST + offs_n, mask=col_ok, other=-1)
    # Queries that can see a key of this block lie between the block's first key and the last such query.
    qb_lo = pid_n * BN // BM
    qb_end = (tl.max(last, axis=0) + BM) // BM
    # Query blocks from this one on lie beyond the extent of every key of the block.
    qb_far = (pid_n * BN + BN - 1 + E + BM - 1) // BM
    qb_far = tl.minimum(tl.maximum(qb_far, qb_lo), qb_end)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    for g in range(H // KVH):
        head = kv * (H // KVH) + g
        h = kvh * (H // KVH) + g
        for qb in range(qb_lo, qb_far):
            dk, dv = _key_tile(
                k, v, Q, DO, PROF, LO, LSE, DELTA, offs_n, dk, dv, qb, h, head, T, E, scale, ROWS, HEADS,
                H, D, BM, PRECISION, WIDE, EXTENT, ALIGN, PW, True,
            )  # fmt: skip
        for qb in range(qb_far, qb_end):
            dk, dv = _key_tile(
                k, v, Q, DO, PROF, LO, LSE, DELTA, offs_n, dk, dv, qb, h, head, T, E, scale, ROWS, HEADS,
                H, D, BM, PRECISION, WIDE, EXTENT, ALIGN, PW, False,
            )  # fmt: skip
    tl.store(DK + col[:, None] + offs_d[None, :], dk * scale, mask=col_ok[:, None])
    tl.store(DV + col[:, None] + offs_d[None, :], dv, mask=col_ok[:, None])


@lru_cache(maxsize=None)
def _precision(dtype: torch.dtype) -> str:
    """Exact FP32 products for FP32 operands (checks); BF16 operands use their own."""
    return "ieee" if dtype == torch.float32 else "tf32"


@dataclass(frozen=True)
class _Profiles:
    """Layout of the relative logits of ``T`` queries: see ``_profile_rows``."""

    tokens: int
    features: int
    extent: int
    align: int

    @property
    def blocks(self) -> int:
        return -(-self.tokens // self.align)

    @property
    def padded_extent(self) -> int:
        return -(-self.extent // self.align) * self.align

    @property
    def width(self) -> int:
        return self.padded_extent + self.align

    def kv_heads_per_chunk(self, group: int, num_kv_heads: int, element_bytes: int) -> int:
        head_bytes = self.blocks * self.align * group * self.width * element_bytes
        return min(num_kv_heads, max(1, PROFILE_BYTES // head_bytes))

    def shifted_tables(self, table: torch.Tensor) -> torch.Tensor:
        """The table in key order for each residue, ``[align, features, width]``: column ``c`` of residue ``r`` is
        distance ``r + padded_extent - c``. Every residue's is a window of one padded table."""
        start = self.padded_extent - self.extent + 1
        flipped = F.pad(table.flip(-1), (self.align - 1 + start, self.width - self.extent - start))
        return flipped.unfold(-1, self.width, 1).flip(1).transpose(0, 1)

    def by_residue(self, rel_states: torch.Tensor) -> torch.Tensor:
        """``[T, heads, features]`` relative states as ``[align, blocks * heads, features]``, padded with zeros."""
        heads = rel_states.shape[1]
        padding = self.blocks * self.align - self.tokens
        if padding:
            rel_states = F.pad(rel_states, (0, 0, 0, 0, 0, padding))
        by_residue = rel_states.unflatten(0, (self.blocks, self.align)).transpose(0, 1)
        return by_residue.reshape(self.align, self.blocks * heads, self.features)

    def from_residue(self, grad: torch.Tensor, heads: int) -> torch.Tensor:
        """The inverse of ``by_residue``."""
        tokens_first = grad.view(self.align, self.blocks, heads, self.features).transpose(0, 1)
        return tokens_first.reshape(self.blocks * self.align, heads, self.features)[: self.tokens]

    def buffer(self, heads: int, like: torch.Tensor) -> torch.Tensor:
        """An uninitialized profile for ``heads`` query heads, with free elements around it."""
        shape = (self.align, self.blocks * heads, self.width)
        numel = shape[0] * shape[1] * shape[2]
        flat = torch.empty(numel + 2 * _PROFILE_PAD, dtype=like.dtype, device=like.device)
        return flat[_PROFILE_PAD : _PROFILE_PAD + numel].view(shape)

    def table_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Sum the residues' table gradients ``[align, width, features]`` into ``[features, extent]``."""
        residue = torch.arange(self.align, device=grad.device)[:, None]
        distance = torch.arange(self.extent, device=grad.device)[None, :]
        columns = residue + self.padded_extent - distance
        return grad.gather(1, columns.unsqueeze(-1).expand(-1, -1, self.features)).sum(0).t()


def _batched_product(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.bmm(a, b) if a.dtype == dtype else torch.bmm(a, b, out_dtype=dtype)


def _wide(q: torch.Tensor, profile: torch.Tensor) -> bool:
    """Whether an element offset needs 64 bits."""
    return max(q.numel(), profile.numel()) >= 2**31


def _launch_sizes(q: torch.Tensor, k: torch.Tensor, profile: torch.Tensor, layout: _Profiles, launch: Launch) -> dict:
    return dict(
        H=q.shape[1], KVH=k.shape[1], D=q.shape[2], BM=launch.block_m, BN=launch.block_n, PRECISION=_precision(q.dtype),
        WIDE=_wide(q, profile), EXTENT=layout.padded_extent, ALIGN=layout.align, PW=layout.width,
        num_warps=launch.num_warps, num_stages=launch.num_stages,
    )  # fmt: skip


def _layout(q: torch.Tensor, rel_states: torch.Tensor, table: torch.Tensor) -> _Profiles:
    """Profile layout: rows aligned to the larger tile of the two kernels that read and write whole rows of tiles."""
    launches = (_launch("forward", q), _launch("query", q))
    align = max(max(launch.block_m, launch.block_n) for launch in launches)
    return _Profiles(q.shape[0], rel_states.shape[-1], table.shape[-1], align)


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_states: torch.Tensor,
    table: torch.Tensor,
    first_key: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention output ``[T, Hq, D]`` and the log-sum-exp of every query's scores ``[T, Hq]``, both FP32.

    ``q [T, Hq, D]``, ``k`` and ``v [T, Hkv, D]``, ``rel_states [T, Hq, F]`` and ``table [F, E]`` share one dtype, BF16
    or FP32. Scores, softmax and sums are FP32; the products take the operands' dtype, and so do the probabilities in
    the value product and the relative logits. ``D`` is a power of two of at least 16."""
    q, k, v, rel_states = (x.contiguous() for x in (q, k, v, rel_states))
    first_key = first_key.to(torch.int32).contiguous()
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group = num_heads // num_kv_heads
    out = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float32, device=q.device)
    lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    launch, layout = _launch("forward", q), _layout(q, rel_states, table)
    tables = layout.shifted_tables(table)
    chunk = layout.kv_heads_per_chunk(group, num_kv_heads, q.element_size())
    for kv_start in range(0, num_kv_heads, chunk):
        heads = group * (min(num_kv_heads, kv_start + chunk) - kv_start)
        head0 = group * kv_start
        profile = layout.buffer(heads, q)
        torch.bmm(layout.by_residue(rel_states[:, head0 : head0 + heads]), tables, out=profile)
        _forward_kernel[(triton.cdiv(num_tokens, launch.block_m), heads)](
            q, k, v, profile, first_key, out, lse, num_tokens, layout.extent, scale, head0, profile.shape[1], heads,
            **_launch_sizes(q, k, profile, layout, launch),
        )  # fmt: skip
        del profile
    return out, lse


def attention_backward(
    grad_out: torch.Tensor,
    grad_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_states: torch.Tensor,
    table: torch.Tensor,
    first_key: torch.Tensor,
    scale: float,
    table_needs_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of ``q``, ``k``, ``v``, ``rel_states`` and ``table`` in their dtype (the table's is empty unless needed).

    ``out`` and ``lse`` are the forward's results. Each gradient accumulates in FP32 and is rounded once."""
    q, k, v, rel_states, lse = (x.contiguous() for x in (q, k, v, rel_states, lse))
    first_key = first_key.to(torch.int32).contiguous()
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group = num_heads // num_kv_heads
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    do = torch.empty_like(q)
    # The softmax's own term and the log-sum-exp's gradient, taken in FP32 before dO is rounded for the products.
    delta = torch.empty_like(lse)
    _softmax_term_kernel[(triton.cdiv(lse.numel(), _TERM_ROWS),)](
        grad_out.float().contiguous(), grad_lse.float().contiguous(), out.contiguous(), do, delta, lse.numel(),
        D=head_dim, R=_TERM_ROWS,
    )  # fmt: skip
    # The last query that sees each key. The first keys of the queries do not decrease.
    keys = torch.arange(num_tokens, device=q.device, dtype=torch.int32)
    last = torch.searchsorted(first_key, keys, right=True, out_int32=True) - 1

    query_launch, key_launch = _launch("query", q), _launch("key", q)
    layout = _layout(q, rel_states, table)
    tables = layout.shifted_tables(table).contiguous()
    d_rel_states = torch.empty_like(rel_states)
    d_table = torch.zeros(table.shape if table_needs_grad else (0,), dtype=torch.float32, device=q.device)
    chunk = layout.kv_heads_per_chunk(group, num_kv_heads, q.element_size())
    for kv_start in range(0, num_kv_heads, chunk):
        kv_heads = min(num_kv_heads, kv_start + chunk) - kv_start
        heads, head0 = group * kv_heads, group * kv_start
        head_slice = slice(head0, head0 + heads)
        by_residue = layout.by_residue(rel_states[:, head_slice])
        profile, d_profile = layout.buffer(heads, q), layout.buffer(heads, q)
        torch.bmm(by_residue, tables, out=profile)
        rows = profile.shape[1]
        _backward_query_kernel[(triton.cdiv(num_tokens, query_launch.block_m), heads)](
            q, k, v, profile, first_key, lse, do, delta, dq, d_profile, num_tokens, layout.extent, scale, head0, rows,
            heads, **_launch_sizes(q, k, profile, layout, query_launch),
        )  # fmt: skip
        _backward_key_kernel[(triton.cdiv(num_tokens, key_launch.block_n), kv_heads)](
            q, k, v, profile, first_key, last, lse, do, delta, dk, dv, num_tokens, layout.extent, scale, kv_start, rows,
            heads, **_launch_sizes(q, k, profile, layout, key_launch),
        )  # fmt: skip
        del profile
        d_rel_states[:, head_slice] = layout.from_residue(torch.bmm(d_profile, tables.transpose(1, 2)), heads)
        if table_needs_grad:
            d_residues = _batched_product(d_profile.transpose(1, 2), by_residue, torch.float32)
            d_table += layout.table_gradient(d_residues)
        del d_profile
    return dq, dk, dv, d_rel_states, d_table.to(table.dtype)


__all__ = ["LAUNCH", "PROFILE_BYTES", "Launch", "attention_backward", "attention_forward"]

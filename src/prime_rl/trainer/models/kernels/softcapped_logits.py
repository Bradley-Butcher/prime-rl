"""Triton kernels for the loss of softcapped logits, ``softcap * tanh(logits / softcap) / temperature``.

Both take a block of FP32 logits, ``[tokens, columns]`` with a range of the vocabulary as columns, and do all the
elementwise work on it in one pass: the operations one by one read and write tens of GB per 8192 tokens."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_COLUMNS_PER_STEP = 512


@triton.jit
def _accumulate_kernel(
    LOGITS, INV_T, LABELS, MAX, SUM, WEIGHTED, TARGET, first_column, columns, stride, softcap,
    WRITE_BACK: tl.constexpr, W: tl.constexpr,
):  # fmt: skip
    """Fold one token's columns into its running softmax: the maximum, sum(exp(x - max)) and sum(exp(x - max) * x)."""
    row = tl.program_id(0)
    base = row.to(tl.int64) * stride
    inv_t = tl.load(INV_T + row)
    target_column = tl.load(LABELS + row) - first_column
    m = tl.load(MAX + row)
    s = tl.load(SUM + row)
    t = tl.load(WEIGHTED + row)
    target = tl.load(TARGET + row)
    for start in range(0, columns, W):
        cols = start + tl.arange(0, W)
        mask = cols < columns
        x = tl.load(LOGITS + base + cols, mask=mask, other=0.0)
        x = softcap * libdevice.tanh(x / softcap) * inv_t
        if WRITE_BACK:
            tl.store(LOGITS + base + cols, x, mask=mask)
        target += tl.sum(tl.where(mask & (cols == target_column), x, 0.0), axis=0)
        m_new = tl.maximum(m, tl.max(tl.where(mask, x, float("-inf")), axis=0))
        # The first step folds an empty sum: -inf - m_new gives a zero factor.
        old = tl.exp(m - m_new)
        p = tl.where(mask, tl.exp(x - m_new), 0.0)
        s = s * old + tl.sum(p, axis=0)
        t = t * old + tl.sum(p * x, axis=0)
        m = m_new
    tl.store(MAX + row, m)
    tl.store(SUM + row, s)
    tl.store(WEIGHTED + row, t)
    tl.store(TARGET + row, target)


@triton.jit
def _gradient_kernel(
    LOGITS, INV_T, LABELS, LOGZ, SOFTMAX_SCALE, LABEL_GRAD, GRAD, first_column, columns, stride, softcap,
    W: tl.constexpr,
):  # fmt: skip
    """One token's logit gradient: ``softmax_scale * p + label_grad * one_hot(label)``, back through temperature and
    softcap."""
    row = tl.program_id(0)
    base = row.to(tl.int64) * stride
    inv_t = tl.load(INV_T + row)
    target_column = tl.load(LABELS + row) - first_column
    logz = tl.load(LOGZ + row)
    softmax_scale = tl.load(SOFTMAX_SCALE + row)
    label_grad = tl.load(LABEL_GRAD + row)
    for start in range(0, columns, W):
        cols = start + tl.arange(0, W)
        mask = cols < columns
        squashed = libdevice.tanh(tl.load(LOGITS + base + cols, mask=mask, other=0.0) / softcap)
        p = tl.exp(softcap * squashed * inv_t - logz)
        grad = softmax_scale * p + tl.where(cols == target_column, label_grad, 0.0)
        grad = grad * inv_t * (1.0 - squashed * squashed)
        tl.store(GRAD + row.to(tl.int64) * columns + cols, grad.to(GRAD.dtype.element_ty), mask=mask)


class SoftmaxState:
    """Per-token running softmax over vocabulary blocks, and each token's label logit (zero for a label in none)."""

    def __init__(self, tokens: int, device: torch.device) -> None:
        self.max = torch.full((tokens,), float("-inf"), device=device, dtype=torch.float32)
        self.sum, self.weighted, self.target = torch.zeros((3, tokens), device=device, dtype=torch.float32)

    def add(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        inv_temperature: torch.Tensor,
        first_column: int,
        softcap: float,
        write_back: bool = False,
    ) -> None:
        """Fold in the columns from ``first_column`` on. ``write_back`` leaves the capped, scaled logits in
        ``logits``."""
        tokens, columns = logits.shape
        _accumulate_kernel[(tokens,)](
            logits, inv_temperature.contiguous(), labels.contiguous(), self.max, self.sum, self.weighted, self.target,
            first_column, columns, logits.stride(0), softcap, WRITE_BACK=write_back, W=_COLUMNS_PER_STEP,
        )  # fmt: skip

    @property
    def logz(self) -> torch.Tensor:
        return self.max + torch.log(self.sum)


def softcapped_logit_gradient(
    logits: torch.Tensor,
    labels: torch.Tensor,
    inv_temperature: torch.Tensor,
    logz: torch.Tensor,
    softmax_scale: torch.Tensor,
    label_grad: torch.Tensor,
    first_column: int,
    softcap: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Gradient w.r.t. the uncapped ``logits [tokens, columns]``, in ``dtype``."""
    tokens, columns = logits.shape
    grad = torch.empty((tokens, columns), device=logits.device, dtype=dtype)
    # Autograd hands over broadcast gradients as expanded tensors.
    per_token = (x.contiguous() for x in (inv_temperature, labels, logz, softmax_scale, label_grad))
    _gradient_kernel[(tokens,)](
        logits, *per_token, grad, first_column, columns, logits.stride(0), softcap, W=_COLUMNS_PER_STEP
    )
    return grad


__all__ = ["SoftmaxState", "softcapped_logit_gradient"]

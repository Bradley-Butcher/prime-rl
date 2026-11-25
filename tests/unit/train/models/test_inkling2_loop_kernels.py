"""The Triton kernels of the looped Inkling 2 decoder against the dense FP32 reference and FP64."""

import pytest
import torch

from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig
from prime_rl.trainer.model import apply_ac, apply_compile
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopForCausalLM
from prime_rl.trainer.models.inkling2_loop.attention import Inkling2Attention, rel_attention, tiled_rel_attention
from prime_rl.trainer.models.kernels import inkling2_attention as kernels
from prime_rl.trainer.models.layers.expert_compute import BF16ExpertCompute
from prime_rl.trainer.models.layers.moe import MoE
from tests.unit.train.models.test_inkling2_loop import _randomized_model, _text_config

pytestmark = [pytest.mark.gpu]

WINDOW = EXTENT = 1024
# Documents of one token, around the window's length, and longer than it; none but the first starts on a tile. The row
# ends inside a tile.
DOCUMENTS = [1, 1023, 1024, 1, 1025, 70, 63, 65, 2, 2100]


@pytest.fixture
def nan_filled_buffers():
    """Every ``torch.empty`` is filled with NaN: an element a kernel leaves unwritten shows up in a result."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(False)


def _first_key(documents: list[int], window: int | None) -> torch.Tensor:
    lengths = torch.tensor(documents)
    first_key = torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
    if window is not None:
        first_key = torch.maximum(first_key, torch.arange(sum(documents)) - (window - 1))
    return first_key.cuda()


def _operands(num_tokens: int, num_heads: int, dtype: torch.dtype, scale: float = 1.0) -> list[torch.Tensor]:
    generator = torch.Generator("cuda").manual_seed(0)
    shapes = ((num_heads, 128), (num_heads // 3, 128), (num_heads // 3, 128), (num_heads, 16))
    tensors = [torch.randn(num_tokens, *shape, device="cuda", generator=generator) for shape in shapes]
    tensors.append(0.3 * torch.randn(16, EXTENT, device="cuda", generator=generator))
    tensors[0] *= scale
    return [tensor.to(dtype) for tensor in tensors]


def _dense(q, k, v, rel_states, rel_table, first_key, scale):
    return rel_attention(q, k, v, torch.einsum("thf,fe->the", rel_states, rel_table), first_key, scale)


def _outputs_and_gradients(kernel, operands, first_key, out_weight, lse_weight) -> list[torch.Tensor]:
    operands = [operand.detach().requires_grad_() for operand in operands]
    out, lse = kernel(*operands, first_key, 128**-0.5)
    loss = (out * out_weight.to(out.dtype)).sum() + (lse * lse_weight.to(lse.dtype)).sum()
    return [out.detach(), lse.detach(), *torch.autograd.grad(loss, operands)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("window", [None, WINDOW])
@pytest.mark.parametrize("num_heads", [24, 12, 6])
def test_tiled_attention_matches_the_references(nan_filled_buffers, num_heads, window, dtype):
    """Outputs and the gradients of every operand. FP32 operands: equal to FP64 and to the dense FP32 kernel up to
    FP32 rounding. BF16 operands: off the FP64 result by little more than the dense kernel is, which computes in FP32
    with the same BF16 bias."""
    num_tokens = sum(DOCUMENTS)
    first_key = _first_key(DOCUMENTS, window)
    operands = _operands(num_tokens, num_heads, dtype)
    weights = torch.randn(num_tokens, num_heads, 129, device="cuda").split([128, 1], dim=-1)
    weights = (weights[0], weights[1].squeeze(-1))

    actual = _outputs_and_gradients(tiled_rel_attention, operands, first_key, *weights)
    dense = _outputs_and_gradients(_dense, operands, first_key, *weights)
    exact = _outputs_and_gradients(_dense, [operand.double() for operand in operands], first_key, *weights)
    names = ("out", "lse", "q", "k", "v", "rel_states", "rel_table")
    for name, a, d, e in zip(names, actual, dense, exact):
        assert a.dtype == d.dtype and a.isfinite().all(), name
        limit = e.abs().max()
        if dtype == torch.float32:
            assert (a - e).abs().max() < 1e-5 * limit, name
            assert (a - d).abs().max() < 1e-5 * limit, name
        else:
            # Both round the bias and their results. The tiles also round the operands of their last products.
            assert (a - e).norm() < 2.5 * (d - e).norm(), name
            assert (a - e).norm() < 1e-2 * e.norm(), name


def test_each_query_sees_exactly_its_window(nan_filled_buffers):
    """The keys with a value gradient are the ones a query sees: the 1,024 up to itself that are in its document. Every
    distance up to the extent's last one gets its own column of the bias table."""
    documents = [700, 3000]
    num_tokens = sum(documents)
    first_key = _first_key(documents, WINDOW)
    operands = [operand.requires_grad_() for operand in _operands(num_tokens, 6, torch.float32)]
    out, lse = tiled_rel_attention(*operands, first_key, 128**-0.5)
    for query in (0, 699, 700, 701, 1722, 1723, 1724, 1791, 1792, num_tokens - 1):
        grad_v, grad_table = torch.autograd.grad(
            out[query].sum() + lse[query].sum(), operands[2:5:2], retain_graph=True
        )
        start = 0 if query < 700 else 700
        expected = torch.arange(max(start, query - 1023), query + 1, device="cuda")
        assert torch.equal(grad_v.abs().sum((1, 2)).nonzero().squeeze(-1), expected), query
        distances = grad_table.abs().sum(0).nonzero().squeeze(-1)
        assert torch.equal(distances, (query - expected).flip(0)), query


def _gpu_model(dtype: torch.dtype, kernel: str, **config) -> Inkling2LoopForCausalLM:
    """The random model of the CPU tests, with smaller products: in BF16 its own rounding errors, which every pass
    amplifies, would drown any difference."""
    model = _randomized_model(_text_config(**config)).to("cuda", dtype)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("proj.weight", "gate_proj", "up_proj", "down_proj", "rel_table")):
                param.mul_(0.25)
    for module in model.modules():
        if isinstance(module, Inkling2Attention):
            module.kernel = kernel
        if isinstance(module, MoE):
            module.router.selection_bias.data = module.router.selection_bias.float()
    return model


def _loss_and_gradients(model: Inkling2LoopForCausalLM) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Of packed documents, with the routing replayed: a rounding difference must not flip an expert."""
    torch.manual_seed(0)
    seq_lens = torch.tensor([23, 1, 40, 9], device="cuda")
    num_tokens = int(seq_lens.sum())
    tokens = torch.randint(2, 300, (2, 1, num_tokens), device="cuda")
    scores = torch.rand(1, num_tokens, model.model.num_layer_visits, 8, device="cuda")
    model.zero_grad(set_to_none=True)
    output = model(input_ids=tokens[0], labels=tokens[1], seq_lens=seq_lens, routed_experts=scores.argsort(-1)[..., :2])
    output["loss"].backward()
    gradients = {name: param.grad.clone() for name, param in model.named_parameters()}
    gradients = {name.replace("_checkpoint_wrapped_module.", ""): grad for name, grad in gradients.items()}
    return output["loss"].detach(), gradients


def _assert_same_gradients(actual, expected, tolerance: float) -> None:
    assert actual.keys() == expected.keys()
    for name, grad in expected.items():
        assert grad.any() and actual[name].isfinite().all(), name
        assert (actual[name].float() - grad.float()).norm() <= tolerance * grad.float().norm(), name


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_whole_model_matches_the_dense_kernel(nan_filled_buffers, dtype):
    """The loss and every weight's gradient of a model with loop passes, sliding and full layers. In BF16 the kernel
    moves them by less than BF16 itself does."""
    loss, gradients = _loss_and_gradients(_gpu_model(dtype, "triton"))
    expected_loss, expected_gradients = _loss_and_gradients(_gpu_model(dtype, "dense"))
    if dtype == torch.float32:
        torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=0)
        _assert_same_gradients(gradients, expected_gradients, 1e-3)
        return
    exact_loss, exact_gradients = _loss_and_gradients(_gpu_model(torch.float32, "dense"))
    assert (loss - exact_loss).abs() < 2 * (expected_loss - exact_loss).abs() + 1e-3 * exact_loss
    for name, exact in exact_gradients.items():
        error = (gradients[name] - exact).norm() / exact.norm()
        dense_error = (expected_gradients[name] - exact).norm() / exact.norm()
        assert error < 1.5 * dense_error + 0.01, (name, float(error), float(dense_error))


def _checkpointed_model(mode: str, dtype: torch.dtype, **config) -> Inkling2LoopForCausalLM:
    model = _gpu_model(dtype, "triton", **config)
    for module in model.modules():
        if isinstance(module, MoE):
            # A compiled layer cannot run the exact experts of the CPU tests.
            module.experts.set_compute(BF16ExpertCompute())
    if mode != "none":
        targets = ["prime_rl_attn"] if mode == "selective" else None
        apply_ac(model, ActivationCheckpointConfig(mode=mode, targets=targets))
    return model


@pytest.fixture
def forward_kernel_runs(monkeypatch) -> list:
    runs = []
    forward = kernels.attention_forward
    monkeypatch.setattr(kernels, "attention_forward", lambda *args: (runs.append(1), forward(*args))[1])
    return runs


def test_checkpointing_changes_nothing(forward_kernel_runs):
    """Full and selective checkpointing give the loss and gradients of no checkpointing. Selective checkpointing of
    the attention namespace runs the forward kernel once per layer visit, full checkpointing twice."""
    reference = None
    for mode in ("none", "full", "selective"):
        model = _checkpointed_model(mode, torch.bfloat16)
        visits = model.model.num_layer_visits
        forward_kernel_runs.clear()
        loss, gradients = _loss_and_gradients(model)
        assert len(forward_kernel_runs) == (2 * visits if mode == "full" else visits), mode
        reference = reference or (loss, gradients)
        torch.testing.assert_close(loss, reference[0], rtol=1e-6, atol=0)
        _assert_same_gradients(gradients, reference[1], 1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("mode, dtype", [("full", torch.float32), ("selective", torch.bfloat16)])
def test_compiled_layers_match_eager(forward_kernel_runs, mode, dtype):
    """Compiled with the trainer's settings the model gives the eager loss and gradients. In FP32 up to what the BF16
    experts make of a rounding difference. In BF16 a compiled layer skips roundings that eager does, and a gradient made
    of cancelling terms moves by more than the others: typically it is as far from the FP32 gradient as eager's."""
    config = dict(num_hidden_layers=6, loop_config=dict(core_start=2, core_end=6, num_passes=2, gate_rank=8))
    expected_loss, expected_gradients = _loss_and_gradients(_checkpointed_model("none", dtype, **config))
    model = _checkpointed_model(mode, dtype, **config)
    apply_compile(model, CompileConfig(fullgraph=True))
    _loss_and_gradients(model)
    forward_kernel_runs.clear()
    loss, gradients = _loss_and_gradients(model)
    visits = model.model.num_layer_visits
    assert len(forward_kernel_runs) == (2 * visits if mode == "full" else visits)
    if dtype == torch.float32:
        torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=0)
        _assert_same_gradients(gradients, expected_gradients, 0.03)
        return
    torch.testing.assert_close(loss, expected_loss, rtol=2e-3, atol=0)
    _, exact = _loss_and_gradients(_gpu_model(torch.float32, "dense", **config))
    error, eager_error = (
        torch.stack([(grads[name] - exact[name]).norm() / exact[name].norm() for name in exact]).median()
        for grads in (gradients, expected_gradients)
    )
    assert error < 1.2 * eager_error, (float(error), float(eager_error))

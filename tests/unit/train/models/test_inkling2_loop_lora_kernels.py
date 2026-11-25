"""The Triton attention kernels under a LoRA adapter: the operands they take are the adapted projections'.

Two branches met here, one that replaced the attention layer's core and one that wraps the projections around it. A core
that read a projection's base weights would serve and train a model without its attention adapter, and nothing would fail.

The CPU tests run the layer's Triton branch with the dense reference in place of the two kernels, so what they test is the
branch itself: which operands it hands over, and the autograd of the custom op. The GPU tests run the kernels.
"""

import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig, LoRAConfig
from prime_rl.trainer.lora import apply_lora_to_model
from prime_rl.trainer.model import apply_ac, apply_compile
from prime_rl.trainer.models import custom_causal_lm_from_config
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopForCausalLM
from prime_rl.trainer.models.inkling2_loop import attention as inkling2_attention
from prime_rl.trainer.models.inkling2_loop.attention import Inkling2Attention, rel_attention
from prime_rl.trainer.models.kernels import inkling2_attention as kernels
from prime_rl.trainer.models.layers.lora import LoRALinear, LoRAModule, experts
from prime_rl.trainer.models.layers.moe import MoE
from tests.unit.train.models.test_inkling2_loop import _logits, _randomized_model, _text_config
from tests.unit.train.models.test_inkling2_loop_lora import (
    ALPHA,
    ATTENTION_TARGETS,
    RANK,
    SEQ_LENS,
    TARGETS,
    exact_grouped_mm,
    randomize_adapters,
)


def _dense(q, k, v, rel_states, table, first_key, scale):
    return rel_attention(q, k, v, torch.einsum("thf,fe->the", rel_states, table), first_key, scale)


@pytest.fixture
def reference_kernels(monkeypatch) -> dict[str, list]:
    """The kernels' entry points, computed by the dense reference and its own backward. Returns the calls."""
    calls: dict[str, list] = {"forward": [], "backward": []}

    def forward(*operands):
        calls["forward"].append(operands)
        return _dense(*operands)

    def backward(grad_out, grad_lse, out, lse, q, k, v, rel_states, table, first_key, scale, table_needs_grad):
        calls["backward"].append((q, k, v, rel_states, table, table_needs_grad))
        rel = torch.einsum("thf,fe->the", rel_states, table)
        *grads, grad_rel = inkling2_attention._rel_attention_backward(
            grad_out, grad_lse, q, k, v, rel, first_key, scale
        )
        grad_table = torch.einsum("thf,the->fe", rel_states, grad_rel) if table_needs_grad else table.new_empty(0)
        return *grads, torch.einsum("the,fe->thf", grad_rel, table), grad_table

    monkeypatch.setattr(kernels, "attention_forward", forward)
    monkeypatch.setattr(kernels, "attention_backward", backward)
    return calls


def _use(model: nn.Module, kernel: str) -> nn.Module:
    """As ``setup_model`` does, after the adapters are on."""
    layers = [module for module in model.modules() if isinstance(module, Inkling2Attention)]
    assert len(layers) == model.config.num_hidden_layers
    for layer in layers:
        layer.kernel = kernel
    return model


def _adapted(targets: list[str], dtype: str | None = None) -> Inkling2LoopForCausalLM:
    model = _randomized_model(_text_config())
    apply_lora_to_model(model, LoRAConfig(rank=RANK, alpha=ALPHA, target_modules=targets, dtype=dtype))
    randomize_adapters(model)
    return model


def _merged(adapted: Inkling2LoopForCausalLM) -> Inkling2LoopForCausalLM:
    """The same model without adapters: each adapted projection's weight is ``base + scaling * B A``."""
    model = _randomized_model(_text_config())
    with torch.no_grad():
        for name, module in adapted.named_modules():
            if isinstance(module, LoRALinear):
                model.get_submodule(name).weight.add_(module.scaling * module.lora_B @ module.lora_A)
    return model


def _input_ids() -> torch.Tensor:
    vocab = _text_config().num_logits
    return torch.randint(0, vocab, (sum(SEQ_LENS),), generator=torch.Generator().manual_seed(7))


def _adapter_gradients(model: nn.Module, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Of a loss on every logit, with the routing replayed."""
    model.zero_grad(set_to_none=True)
    num_tokens = sum(SEQ_LENS)
    scores = torch.rand(1, num_tokens, model.model.num_layer_visits, 8, generator=torch.Generator().manual_seed(3))
    output = model(
        input_ids=_input_ids().unsqueeze(0).to(device),
        seq_lens=torch.tensor(SEQ_LENS, device=device),
        routed_experts=scores.argsort(-1)[..., :2].to(device),
    )
    output["logits"].float().square().mean().backward()
    for name, param in model.named_parameters():
        assert (param.grad is not None) == ("lora_" in name), name
    return {
        name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", ""): param.grad.float().clone()
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def _largest_error(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> tuple[float, str]:
    assert actual.keys() == expected.keys()
    return max(((actual[name] - grad).norm().item() / grad.norm().item(), name) for name, grad in expected.items())


@pytest.mark.parametrize("targets", [*([target] for target in ATTENTION_TARGETS), ATTENTION_TARGETS])
def test_the_kernel_branch_applies_the_attention_adapter(reference_kernels, targets):
    """Every attention projection's adapter reaches the output of the Triton branch, with the size the merged weights give
    it, and gets the dense branch's gradient."""
    input_ids = _input_ids()
    base_logits = _logits(_use(_randomized_model(_text_config()), "triton"), input_ids, SEQ_LENS)
    model = _use(_adapted(targets), "triton")
    reference_kernels["forward"].clear()
    logits = _logits(model, input_ids, SEQ_LENS)
    assert len(reference_kernels["forward"]) == model.model.num_layer_visits
    assert (logits - base_logits).abs().mean() > 0.05
    merged_logits = _logits(_use(_merged(model), "dense"), input_ids, SEQ_LENS)
    assert (logits - merged_logits).abs().max() < 2e-3

    gradients = _adapter_gradients(model)
    assert len(gradients) == 2 * len(targets) * model.config.num_hidden_layers
    assert all(gradient.any() for gradient in gradients.values())
    error, name = _largest_error(gradients, _adapter_gradients(_use(model, "dense")))
    assert error < 1e-3, name


def test_a_frozen_base_asks_for_no_table_gradient(reference_kernels):
    """The bias table is a base weight: its gradient is a pass over every profile, which a LoRA step need not pay."""
    model = _use(_adapted(TARGETS), "triton")
    _adapter_gradients(model)
    assert len(reference_kernels["backward"]) == model.model.num_layer_visits
    assert not any(call[-1] for call in reference_kernels["backward"])


def test_a_float32_adapter_hands_the_kernels_bfloat16_operands(reference_kernels):
    """``lora.dtype = "float32"`` over a BF16 base. The kernels refuse nothing and check nothing: operands of two dtypes
    would be read as another's bits."""
    config = _text_config()
    reference = _randomized_model(config).state_dict()
    model = custom_causal_lm_from_config(config, dtype=torch.bfloat16).eval()
    model.init_buffers_post_meta()
    with torch.no_grad():
        for name, weight in model.state_dict().items():
            weight.copy_(reference[name])
    apply_lora_to_model(model, LoRAConfig(rank=RANK, alpha=16.0, target_modules=TARGETS, dtype="float32"))
    randomize_adapters(model)
    gradients = _adapter_gradients(_use(model, "triton"))
    assert {operand.dtype for call in reference_kernels["forward"] for operand in call[:5]} == {torch.bfloat16}
    assert all(gradient.any() for gradient in gradients.values())
    assert {param.grad.dtype for param in model.parameters() if param.grad is not None} == {torch.float32}


def test_auto_takes_the_reference_where_the_kernels_do_not_run(reference_kernels):
    """On the CPU, with adapters as without."""
    _logits(_use(_adapted(TARGETS), "auto"), _input_ids(), SEQ_LENS)
    assert not reference_kernels["forward"]


def _gpu_model(dtype: torch.dtype, kernel: str, targets: list[str]) -> Inkling2LoopForCausalLM:
    """A frozen base in ``dtype`` under an FP32 adapter, with the smaller products of the other GPU tests: in BF16 the
    random model's own rounding errors, which every pass amplifies, would drown any difference."""
    model = _randomized_model(_text_config()).to("cuda", dtype)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("proj.weight", "gate_proj", "up_proj", "down_proj", "rel_table")):
                param.mul_(0.25)
    for module in model.modules():
        if isinstance(module, MoE):
            module.router.selection_bias.data = module.router.selection_bias.float()
    apply_lora_to_model(model, LoRAConfig(rank=RANK, alpha=ALPHA, target_modules=targets, dtype="float32"))
    randomize_adapters(model)
    return _use(model, kernel)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_kernels_give_the_adapters_the_dense_gradients(monkeypatch, dtype):
    """FP32: up to its rounding. BF16: off the FP32 reference by little more than the dense kernel is."""
    if dtype == torch.float32:
        monkeypatch.setattr(experts, "_grouped_mm", exact_grouped_mm)
    actual = _adapter_gradients(_gpu_model(dtype, "triton", TARGETS), "cuda")
    dense = _adapter_gradients(_gpu_model(dtype, "dense", TARGETS), "cuda")
    if dtype == torch.float32:
        error, name = _largest_error(actual, dense)
        assert error < 2e-3, name
        return
    monkeypatch.setattr(experts, "_grouped_mm", exact_grouped_mm)
    exact = _adapter_gradients(_gpu_model(torch.float32, "dense", TARGETS), "cuda")
    for name, gradient in exact.items():
        error, dense_error = ((grads[name] - gradient).norm() / gradient.norm() for grads in (actual, dense))
        assert error < 1.5 * dense_error + 0.02, (name, float(error), float(dense_error))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("mode", ["selective", "full"])
def test_compiled_layers_run_the_kernels_under_the_adapters(monkeypatch, mode):
    """As the trainer runs them: BF16 compute, checkpointed, compiled with ``fullgraph``. The kernel runs once a visit with
    selective checkpointing and twice with full, and the typical gradient is as far from the FP32 reference as eager's."""
    runs = []
    forward = kernels.attention_forward
    monkeypatch.setattr(kernels, "attention_forward", lambda *args: (runs.append(1), forward(*args))[1])
    eager = _adapter_gradients(_gpu_model(torch.bfloat16, "triton", TARGETS), "cuda")
    model = _gpu_model(torch.bfloat16, "triton", TARGETS)
    visits = model.model.num_layer_visits
    apply_ac(model, ActivationCheckpointConfig(mode=mode, targets=["prime_rl_attn"] if mode == "selective" else None))
    apply_compile(model, CompileConfig(fullgraph=True))
    _adapter_gradients(model, "cuda")
    runs.clear()
    compiled = _adapter_gradients(model, "cuda")
    assert len(runs) == (2 * visits if mode == "full" else visits)
    with monkeypatch.context() as exactly:
        exactly.setattr(experts, "_grouped_mm", exact_grouped_mm)
        exact = _adapter_gradients(_gpu_model(torch.float32, "dense", TARGETS), "cuda")
    error, eager_error = (
        torch.stack([(grads[name] - exact[name]).norm() / exact[name].norm() for name in exact]).median().item()
        for grads in (compiled, eager)
    )
    assert error < 1.3 * eager_error, (error, eager_error)


def test_every_adapted_projection_is_a_lora_module():
    """What the tests above rely on: no projection is left unwrapped by a name that the branch changed."""
    model = _adapted(TARGETS)
    wrapped = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, LoRAModule)}
    assert wrapped == set(TARGETS)

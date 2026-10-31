import pytest
import torch
from pydantic import ValidationError
from torch import nn

from prime_rl.configs.trainer import LoRAConfig, ModelConfig
from prime_rl.trainer.lora import setup_lora_state
from prime_rl.trainer.models.layers.lora import MultiLoRALinear, set_lora_num_tokens
from prime_rl.trainer.models.layers.lora.multi_moe import MultiLoRAGroupedExperts
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs

RANK = 8


def adapted(dtype: torch.dtype | None) -> tuple[MultiLoRALinear, MultiLoRAGroupedExperts]:
    """A linear and an expert LoRA over bfloat16 base weights."""
    torch.manual_seed(0)
    setup_lora_state(LoRAConfig(rank=RANK, alpha=16.0), torch.device("cpu"))
    linear = MultiLoRALinear(nn.Linear(16, 16, dtype=torch.bfloat16), rank=RANK, n_adapters=1, dtype=dtype)
    moe = MoE.from_args(
        MoEArgs(num_experts=2, top_k=1, load_balance_coeff=None), dim=16, hidden_dim=24, shared_expert=None
    )
    experts = MultiLoRAGroupedExperts(moe.experts.to(torch.bfloat16), rank=RANK, n_adapters=1, dtype=dtype)
    return linear, experts


@pytest.mark.parametrize("dtype, expected", [(torch.float32, torch.float32), (None, torch.bfloat16)])
def test_the_adapter_takes_its_own_dtype_over_a_bfloat16_base(dtype, expected):
    linear, experts = adapted(dtype)
    for module in (linear, experts):
        assert {param.dtype for _, param in module.named_parameters_for_adapter(0)} == {expected}
        assert {param.dtype for param in module.base_layer.parameters()} == {torch.bfloat16}
        assert module.base_dtype == torch.bfloat16

    # Outside FSDP's bf16 param cast, the layers cast the adapter for their bf16 compute themselves.
    with torch.no_grad():
        for module in (linear, experts):
            for _, param in module.named_parameters_for_adapter(0):
                nn.init.normal_(param, std=0.1)
    set_lora_num_tokens(torch.tensor([10], dtype=torch.int32))
    x = torch.randn(10, 16, dtype=torch.bfloat16)
    out = experts(linear(x), torch.tensor([4, 6]))
    assert out.dtype == torch.bfloat16
    out.float().square().sum().backward()
    for module in (linear, experts):
        assert {param.grad.dtype for _, param in module.named_parameters_for_adapter(0)} == {expected}


def test_an_adam_step_a_bfloat16_weight_rounds_away_moves_a_float32_one():
    """A first Adam step moves each weight by about lr; at |a| = 0.004 half a bf16 unit is about 1.5e-5."""
    float32 = nn.Parameter(torch.full((4,), 0.004))
    bfloat16 = nn.Parameter(torch.full((4,), 0.004, dtype=torch.bfloat16))
    before = (float32.detach().clone(), bfloat16.detach().clone())
    # One parameter group of both dtypes, as the trainer's AdamW holds an fp32 adapter beside bf16 parameters.
    optimizer = torch.optim.AdamW([float32, bfloat16], lr=1e-6, weight_decay=0.0, foreach=True)
    float32.grad, bfloat16.grad = torch.ones_like(float32), torch.ones_like(bfloat16)
    optimizer.step()

    assert torch.allclose(before[0] - float32.detach(), torch.full((4,), 1e-6), rtol=1e-3)
    assert torch.equal(bfloat16.detach(), before[1])


def test_the_adapter_is_exported_in_the_base_dtype():
    linear, experts = adapted(torch.float32)
    state = setup_lora_state(LoRAConfig(rank=RANK, alpha=16.0, dtype="float32"), torch.device("cpu"))
    state.register_module("model.layers.0.self_attn.q_proj", linear)
    state.register_module("model.layers.0.mlp.experts", experts)

    exported = state.adapter_state_dict()

    assert {tensor.dtype for tensor in exported.values()} == {torch.bfloat16}
    assert torch.equal(
        exported["model.layers.0.self_attn.q_proj.lora_A.weight"], linear.lora_A[0].detach().to(torch.bfloat16)
    )
    assert torch.equal(
        exported["model.layers.0.mlp.experts.1.up_proj.lora_A.weight"],
        experts.up_proj_lora_A[0].detach()[1].to(torch.bfloat16),
    )
    assert linear.lora_A[0].dtype == torch.float32  # the export casts a copy


def test_a_fully_trained_module_cannot_share_a_unit_with_an_adapter_of_another_dtype():
    lora = {"dtype": "float32", "modules_to_save": ["norm"]}
    with pytest.raises(ValidationError, match="modules_to_save"):
        ModelConfig(optimization_dtype="bfloat16", lora=lora)
    assert ModelConfig(optimization_dtype="float32", lora=lora).lora.dtype == "float32"
    assert ModelConfig(optimization_dtype="bfloat16", lora={"dtype": "float32"}).lora.dtype == "float32"

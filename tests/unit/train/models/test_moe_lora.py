import copy

import torch

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.lora import setup_lora_state
from prime_rl.trainer.models.layers.lora.multi_moe import MultiLoRAGroupedExperts
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs

RANK, ALPHA = 8, 16.0


def _moe() -> MoE:
    torch.manual_seed(0)
    moe = MoE.from_args(
        MoEArgs(num_experts=4, top_k=2, load_balance_coeff=None), dim=16, hidden_dim=24, shared_expert=None
    )
    for parameter in moe.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return moe


def test_expert_lora_matches_merged_experts_and_trains_only_the_adapter():
    """Without expert parallelism, an expert LoRA computes each expert with W + (alpha / r) B A, and only the
    adapter learns."""
    setup_lora_state(LoRAConfig(rank=RANK, alpha=ALPHA), torch.device("cpu"))
    base = _moe()
    adapted = copy.deepcopy(base)
    adapted.experts = MultiLoRAGroupedExperts(adapted.experts, rank=RANK, n_adapters=1, alpha=ALPHA)
    experts = adapted.experts
    with torch.no_grad():
        for _, parameter in experts.named_parameters_for_adapter(0):
            torch.nn.init.normal_(parameter, std=0.2)  # a trained adapter: B is no longer zero

    merged = copy.deepcopy(base)
    with torch.no_grad():
        for name in ("gate_proj", "up_proj", "down_proj"):
            lora_a, lora_b = getattr(experts, f"{name}_lora_A")[0], getattr(experts, f"{name}_lora_B")[0]
            getattr(merged.experts, name).add_(ALPHA / RANK * lora_b @ lora_a)

    x = torch.randn(2, 20, 16)
    out = adapted(x)
    torch.testing.assert_close(out, merged(x), rtol=0.05, atol=0.05)

    out.square().sum().backward()
    for name, parameter in experts.named_parameters_for_adapter(0):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert all(parameter.grad is None for parameter in experts.base_layer.parameters())

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.lora import save_lora_config, setup_lora_state
from prime_rl.trainer.models.layers.activations import ClampedSwiglu
from prime_rl.trainer.models.layers.lora import MultiLoRALinear
from prime_rl.trainer.models.layers.lora.multi_moe import (
    MultiLoRAGptOssGroupedExperts,
    MultiLoRAGroupedExperts,
    MultiLoRANonGatedGroupedExperts,
)
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM
from prime_rl.utils.weights import save_state_dict

RANK, ALPHA = 8, 16.0


def experts(**kwargs) -> GroupedExperts:
    base = GroupedExperts(16, 24, 3, **kwargs)
    for param in base.parameters():
        nn.init.normal_(param, std=0.1)
    return base.to(torch.bfloat16)


class Adapted(nn.Module):
    """One module of each LoRA kind over bf16 base weights, registered as ``apply_lora_to_model`` does."""

    def __init__(self, dtype: torch.dtype | None, converter=None):
        super().__init__()
        self.config = SimpleNamespace(_name_or_path="tiny")
        self.state = setup_lora_state(LoRAConfig(rank=RANK, alpha=ALPHA), torch.device("cpu"))
        if converter is not None:
            self.state.register_adapter_state_dict_converter(converter)
        kwargs = {"rank": RANK, "n_adapters": 1, "alpha": ALPHA, "dtype": dtype}
        self.layers = nn.ModuleDict(
            {
                "q_proj": MultiLoRALinear(nn.Linear(16, 32, dtype=torch.bfloat16), **kwargs),
                "experts": MultiLoRAGroupedExperts(experts(), **kwargs),
                "relu2_experts": MultiLoRANonGatedGroupedExperts(experts(expert_type="non_gated"), **kwargs),
                "gpt_oss_experts": MultiLoRAGptOssGroupedExperts(
                    experts(activation=ClampedSwiglu, bias=True), **kwargs
                ),
            }
        )
        prefixes = {"q_proj": "self_attn.q_proj", "experts": "mlp.experts"}
        for name, module in self.layers.items():
            self.state.register_module(f"model.layers.0.{prefixes.get(name, name)}", module)

    def adapter(self) -> dict[str, torch.Tensor]:
        return {name: param.detach().clone() for name, param in self.named_parameters() if "lora_" in name}

    def perturb(self) -> None:
        with torch.no_grad():
            for module in self.layers.values():
                for _, param in module.named_parameters_for_adapter(0):
                    nn.init.normal_(param, std=0.5)


def broadcast(model: Adapted, step_dir):
    """The adapter as the filesystem weight broadcast writes it."""
    save_state_dict(model.state.adapter_state_dict(), step_dir, save_sharded=False, adapter=True)
    save_lora_config(model, step_dir, rank=RANK, alpha=ALPHA, dropout=0.0)
    return step_dir / "adapter_model.safetensors"


@pytest.mark.parametrize("dtype", [None, torch.float32])
@pytest.mark.parametrize("converter", [None, NemotronHForCausalLM.convert_adapter_to_hf])
def test_a_warm_start_restores_the_broadcast_adapter_and_exports_it_bit_for_bit(tmp_path, dtype, converter):
    model = Adapted(dtype, converter)
    model.perturb()  # a trained adapter: lora_B is no longer zero
    path = broadcast(model, tmp_path / "step_7")
    saved = model.adapter()
    model.perturb()

    model.state.load_adapter(tmp_path / "step_7")

    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.dtype == (dtype or torch.bfloat16)
            # The broadcast is bf16, so an fp32 adapter comes back at the broadcast's precision.
            assert torch.equal(param.detach(), saved[name].to(torch.bfloat16).to(param.dtype)), name
    exported = broadcast(model, tmp_path / "step_0")
    assert exported.read_bytes() == path.read_bytes()
    if converter is not None:
        assert any(".mixer." in key for key in load_file(path))


def test_a_warm_start_refuses_an_adapter_that_does_not_fit(tmp_path):
    model = Adapted(torch.float32)
    path = broadcast(model, tmp_path)
    weights = load_file(path)

    def refused(changed: dict[str, torch.Tensor], match: str) -> None:
        save_file(changed, path)
        with pytest.raises(ValueError, match=match):
            model.state.load_adapter(path)

    linear = "model.layers.0.self_attn.q_proj.lora_A.weight"
    expert = "model.layers.0.mlp.experts.2.gate_proj.lora_B.weight"
    refused({key: value for key, value in weights.items() if key != expert}, f"1 missing \\({expert}\\)")
    refused(weights | {"model.layers.1.self_attn.q_proj.lora_A.weight": weights[linear].clone()}, "1 extra")
    refused(weights | {linear: weights[linear][:, :8].contiguous()}, "1 with another shape")

    save_file(weights, path)
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    (tmp_path / "adapter_config.json").write_text(json.dumps(config | {"r": 16}))
    with pytest.raises(ValueError, match="rank 16 and alpha 16.0"):
        model.state.load_adapter(path)
    (tmp_path / "adapter_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        model.state.load_adapter(tmp_path)

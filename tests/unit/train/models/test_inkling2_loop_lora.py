"""LoRA on Inkling 2 Mini: the exported adapter, applied the way vLLM applies it, gives the trainer's outputs.

A wrong mapping can load cleanly and serve a different model, so the adapter directory a broadcast writes is read back
here by ``served_checkpoint``, which shares no code with the export. It transcribes what vLLM (Apache-2.0, at commit
ab18334a4f60da74db7de89fe95594e5ed0cda36) does with such a directory: the scaling of ``lora/peft_helper.py``, the
names of ``lora/utils.py`` and ``models/inkling2/nvidia/lora.py``, and the weights those adapters belong to, ``qkvr``
(the packed attention projections), ``wo_ud``, and the experts' ``w13`` and ``w2``. vLLM adds an adapter's product to
the one of the weight it belongs to, so adding ``scaling * B @ A`` to the weight is the same model.

``tests/unit/inference/test_inkling2_loop_lora_serving.py`` runs vLLM's own code on such a directory, where that vLLM
is installed.
"""

import json
import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.lora import apply_lora_to_model, get_lora_state, save_lora_config
from prime_rl.trainer.models import custom_causal_lm_from_config
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopForCausalLM
from prime_rl.trainer.models.layers.lora import LoRAModule, experts
from prime_rl.utils.weights import save_state_dict
from tests.unit.train.models.test_inkling2_loop import _ExactExpertCompute, _logits, _randomized_model, _text_config

# A scale of 3: neither 1 nor the alpha over the w13 adapter's rank.
RANK, ALPHA = 8, 24.0
ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "rel_proj", "gate_proj", "sink_proj", "o_proj"]
TARGETS = [*ATTENTION_TARGETS, "experts"]
SEQ_LENS = [23, 40]

# vLLM's `PACKED_MODULES_MAPPING["qkvr"]`: the projections packed into one weight, in row order.
QKVR = ("wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv")


def exact_grouped_mm(x: Tensor, weight: Tensor, offsets: Tensor) -> Tensor:
    """``_grouped_mm`` in the input's own dtype, so a CPU forward is exact instead of BF16-rounded."""
    starts = [0, *offsets.tolist()]
    rows = [x[start:stop] @ weight[expert].T for expert, (start, stop) in enumerate(zip(starts, starts[1:]))]
    return torch.cat([*rows, x.new_zeros((x.shape[0] - starts[-1], weight.shape[1]))])


def randomize_adapters(model: nn.Module, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.4 / param.shape[-1] ** 0.5)


def export_adapter(model: nn.Module, step_dir: Path, rank: int = RANK, alpha: float = ALPHA) -> Path:
    """The adapter as the filesystem weight broadcast writes it."""
    state_dict = get_lora_state().adapter_state_dict()
    save_state_dict(state_dict, step_dir, save_sharded=False, adapter=True)
    save_lora_config(model, state_dict, step_dir, rank=rank, alpha=alpha, dropout=0.0)
    return step_dir


def rank_and_scaling(config: dict, module: str) -> tuple[int, float]:
    """``PEFTHelper.get_rank_and_scaling``: the first pattern that ends the module's name, from a dot, wins."""

    def match(patterns: dict, default: float) -> float:
        for pattern, value in patterns.items():
            if re.match(rf"(.*\.)?({pattern})$", module):
                return value
        return default

    rank = match(config.get("rank_pattern", {}), config["r"])
    return rank, match(config.get("alpha_pattern", {}), config["lora_alpha"]) / rank


def served_checkpoint(checkpoint: dict[str, Tensor], tensors: dict[str, Tensor], config: dict) -> dict[str, Tensor]:
    """The checkpoint with the adapter merged into the weights vLLM applies it to."""
    merged = {name: weight.clone() for name, weight in checkpoint.items()}
    for name in tensors:
        match = re.fullmatch(r"(model\.llm\.layers\.\d+\.(?:attn|mlp\.experts))\.(\w+)\.lora_A\.weight", name)
        if match is None:
            assert name.endswith(".lora_B.weight") and name.replace("lora_B", "lora_A") in tensors, name
            continue
        parent, module = match.groups()
        lora_a, lora_b = tensors[name], tensors[name.replace("lora_A", "lora_B")]
        rank, scaling = rank_and_scaling(config, f"{parent}.{module}")
        assert lora_a.shape[-2] == lora_b.shape[-1] == rank
        delta = scaling * lora_b @ lora_a
        if parent.endswith("experts"):
            # `Inkling2MoEWithLoRA`: one `[E, r, in]` / `[E, out, r]` pair on each of the two expert products.
            assert module in ("w13_weight", "w2_weight"), module
            weight = merged[f"{parent}.{module}"]
            for expert in range(weight.shape[0]):
                weight[expert] += delta[expert]
        elif module == "wo_ud":
            merged[f"{parent}.wo_ud.weight"] += delta
        else:
            # `qkvr`: the module's slice of the packed rows.
            sizes = [merged[f"{parent}.{projection}.weight"].shape[0] for projection in QKVR]
            qkvr = torch.cat([merged[f"{parent}.{projection}.weight"] for projection in QKVR])
            qkvr.split(sizes)[QKVR.index(module)].add_(delta)
            merged.update({f"{parent}.{projection}.weight": rows for projection, rows in zip(QKVR, qkvr.split(sizes))})
    return merged


class Served:
    """A trained adapter on disk, the checkpoint it adapts, and the trainer's logits with it."""

    def __init__(self, step_dir: Path):
        config = _text_config()
        model = _randomized_model(config)
        self.checkpoint = model.convert_to_hf({name: weight.clone() for name, weight in model.state_dict().items()})
        self.input_ids = torch.randint(
            0, config.num_logits, (sum(SEQ_LENS),), generator=torch.Generator().manual_seed(7)
        )
        self.base_logits = _logits(model, self.input_ids, SEQ_LENS)
        apply_lora_to_model(model, LoRAConfig(rank=RANK, alpha=ALPHA, target_modules=TARGETS))
        randomize_adapters(model)
        self.trainer_logits = _logits(model, self.input_ids, SEQ_LENS)
        export_adapter(model, step_dir)
        self.tensors = load_file(step_dir / "adapter_model.safetensors")
        self.config = json.loads((step_dir / "adapter_config.json").read_text())

    def logits(self, tensors: dict[str, Tensor] | None = None, config: dict | None = None) -> Tensor:
        """The logits of the checkpoint with the adapter, through the model's checkpoint conversion."""
        checkpoint = served_checkpoint(self.checkpoint, tensors or self.tensors, config or self.config)
        model = _randomized_model(_text_config(), seed=1)
        model.load_state_dict(model.convert_to_prime(checkpoint), strict=False)
        for module in model.modules():
            if hasattr(module, "set_compute"):
                module.set_compute(_ExactExpertCompute())
        return _logits(model, self.input_ids, SEQ_LENS)


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> Served:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(experts, "_grouped_mm", exact_grouped_mm)
        return Served(tmp_path_factory.mktemp("step_0"))


def test_the_exported_adapter_served_as_vllm_serves_it_gives_the_trainers_logits(served):
    assert (served.trainer_logits - served.base_logits).abs().mean() > 1.0
    assert (served.logits() - served.trainer_logits).abs().max() < 1e-3


def swap_rows(lora_b: Tensor) -> Tensor:
    return lora_b.unflatten(1, (-1, 2)).flip(2).flatten(1, 2)


def swapped(tensors: dict[str, Tensor], first: str, second: str) -> dict[str, Tensor]:
    """The adapters of two modules under each other's name."""
    names = {first: second, second: first}
    return {".".join(names.get(part, part) for part in name.split(".")): weight for name, weight in tensors.items()}


def edited(tensors: dict[str, Tensor], suffix: str, edit) -> dict[str, Tensor]:
    return {name: edit(weight) if name.endswith(suffix) else weight for name, weight in tensors.items()}


W13_B, W13_A = "experts.w13_weight.lora_B.weight", "experts.w13_weight.lora_A.weight"
ADAPTER_MISTAKES = {
    "gate and up swapped": lambda t: edited(t, W13_B, swap_rows),
    "rows concatenated instead of interleaved": lambda t: edited(
        t, W13_B, lambda b: torch.cat([b[:, 0::2], b[:, 1::2]], dim=1)
    ),
    "the two ranks of w13 under each other's B": lambda t: edited(t, W13_A, lambda a: a.roll(RANK, 1)),
    "experts in another order": lambda t: edited(t, W13_A, lambda a: a.flip(0)),
    "down experts in another order": lambda t: edited(t, "experts.w2_weight.lora_B.weight", lambda b: b.flip(0)),
    "query and output gate swapped": lambda t: swapped(t, "wq_du", "gate_attn"),
    "key and value swapped": lambda t: swapped(t, "wk_dv", "wv_dv"),
    "one attention projection left out": lambda t: {n: w for n, w in t.items() if ".attn_sink_proj." not in n},
    "two layers of the loop under each other's number": lambda t: swapped(t, "3", "5"),
}


@pytest.mark.parametrize("mistake", ADAPTER_MISTAKES)
def test_a_mistake_in_the_adapter_shows_in_the_logits(served, mistake):
    tensors = ADAPTER_MISTAKES[mistake](served.tensors)
    assert (served.logits(tensors) - served.trainer_logits).abs().max() > 0.05


@pytest.mark.parametrize(
    "mistake",
    [
        # The scale of w13 is alpha over twice the rank without its alpha.
        {"alpha_pattern": {}},
        {"lora_alpha": float(RANK)},
        {"alpha_pattern": {"mlp.experts.w13_weight": ALPHA}},
    ],
)
def test_a_wrong_scale_shows_in_the_logits(served, mistake):
    logits = served.logits(config={**served.config, **mistake})
    assert (logits - served.trainer_logits).abs().max() > 0.05


def test_the_exported_tensors_are_the_ones_vllm_names(served):
    config = _text_config()
    heads = {True: (6, 3, 8), False: (4, 2, 16)}  # (query heads, KV heads, head dim) of a sliding / a full layer.
    expected = {}
    for layer in range(config.num_hidden_layers):
        num_heads, num_kv_heads, head_dim = heads[layer in (0, 2, 4, 6)]
        rows = {
            "wq_du": num_heads * head_dim,
            "wr_du": num_heads * 4,
            "gate_attn": num_heads * head_dim,
            "attn_sink_proj": num_heads,
            "wk_dv": num_kv_heads * head_dim,
            "wv_dv": num_kv_heads * head_dim,
        }
        shapes = {f"attn.{name}": ((RANK, 64), (out, RANK)) for name, out in rows.items()}
        shapes["attn.wo_ud"] = ((RANK, num_heads * head_dim), (64, RANK))
        shapes["mlp.experts.w13_weight"] = ((8, 2 * RANK, 64), (8, 2 * 32, 2 * RANK))
        shapes["mlp.experts.w2_weight"] = ((8, RANK, 32), (8, 64, RANK))
        for module, (lora_a, lora_b) in shapes.items():
            expected[f"model.llm.layers.{layer}.{module}.lora_A.weight"] = lora_a
            expected[f"model.llm.layers.{layer}.{module}.lora_B.weight"] = lora_b
    assert {name: tuple(weight.shape) for name, weight in served.tensors.items()} == expected

    # vLLM refuses an adapter whose largest rank exceeds max_lora_rank, so the config must say w13's.
    assert (served.config["r"], served.config["lora_alpha"]) == (RANK, ALPHA)
    assert served.config["rank_pattern"] == {"mlp.experts.w13_weight": 2 * RANK}
    assert served.config["alpha_pattern"] == {"mlp.experts.w13_weight": 2 * ALPHA}


def test_an_attention_adapter_stays_at_the_trainers_rank(tmp_path):
    model = _randomized_model(_text_config())
    apply_lora_to_model(model, LoRAConfig(rank=RANK, alpha=ALPHA, target_modules=["q_proj"]))
    export_adapter(model, tmp_path)
    assert "rank_pattern" not in json.loads((tmp_path / "adapter_config.json").read_text())


@pytest.mark.parametrize(
    "pattern, named",
    [
        # The cross-family defaults name the message gates' `down_proj`.
        ("down_proj", "message_gates.q.down_proj"),
        ("message_gates", "message_gates.q.down_proj"),
        ("weight_proj", "k_conv.weight_proj"),
        ("k_conv", "self_attn.k_conv"),
        ("attn_conv", "attn_conv"),
        ("attention_input", "attention_input"),
        ("message_queries", "model.message_queries"),
        ("router", "mlp.router.gate"),
        ("lm_head", "lm_head"),
        ("embed_tokens", "embed_tokens"),
        ("mlp", "mlp.router.gate"),
        (r"self_attn\..*_scale", "self_attn.q_scale"),
        (r"layers\.3\..*proj$", "weight_proj"),
    ],
)
def test_targets_without_an_adapter_in_vllm_are_refused(pattern, named):
    model = Inkling2LoopForCausalLM(_text_config())
    with pytest.raises(ValueError, match="matches weights that 'inkling2_loop_text' cannot train") as error:
        apply_lora_to_model(model, LoRAConfig(target_modules=["q_proj", pattern]))
    assert named in str(error.value)
    assert "q_proj, k_proj, v_proj, rel_proj, gate_proj, sink_proj, o_proj" in str(error.value)
    assert not any(isinstance(module, LoRAModule) for module in model.modules())


def test_the_default_targets_are_refused_and_the_documented_ones_are_not():
    with pytest.raises(ValueError, match="message_gates"):
        apply_lora_to_model(Inkling2LoopForCausalLM(_text_config()), LoRAConfig())
    model = Inkling2LoopForCausalLM(_text_config())
    apply_lora_to_model(model, LoRAConfig(target_modules=TARGETS))
    adapted = {
        name.removeprefix("model.layers.0.")
        for name, module in model.named_modules()
        if isinstance(module, LoRAModule) and ".0." in name
    }
    assert adapted == {*(f"self_attn.{name}" for name in ATTENTION_TARGETS), "mlp.experts"}


def test_a_float32_adapter_over_the_bfloat16_base(tmp_path):
    """`optimization_dtype = "bfloat16"` with `lora.dtype = "float32"`: the queries that the checkpoint holds in FP32
    are the BF16 values the trainer computes with anyway, and the router's selection bias is a buffer and stays FP32."""
    config = _text_config()
    reference = _randomized_model(config)
    model = custom_causal_lm_from_config(config, dtype=torch.bfloat16).eval()
    model.init_buffers_post_meta()
    state_dict = reference.state_dict()
    with torch.no_grad():
        for name, weight in model.state_dict().items():
            weight.copy_(state_dict[name])  # As DCP loads a checkpoint tensor into a parameter of another dtype.
    apply_lora_to_model(model, LoRAConfig(rank=RANK, alpha=16.0, target_modules=TARGETS, dtype="float32"))
    randomize_adapters(model)

    for name, param in model.named_parameters():
        assert param.dtype == (torch.float32 if "lora_" in name else torch.bfloat16), name
        assert param.requires_grad == ("lora_" in name), name
    for name, buffer in model.named_buffers():
        if name.endswith("selection_bias"):
            assert torch.equal(buffer, state_dict[name]) and buffer.dtype == torch.float32
    for name in ("model.message_queries", "model.norm.query", "model.layers.3.attention_input.query"):
        assert torch.equal(model.get_parameter(name), state_dict[name].to(torch.bfloat16))

    # Outside FSDP's param dtype, the layers cast the adapter for their BF16 compute themselves.
    input_ids = torch.randint(0, config.num_logits, (40,), generator=torch.Generator().manual_seed(7))
    output = model(input_ids=input_ids.unsqueeze(0), seq_lens=torch.tensor([40]))["logits"]
    output.float().square().mean().backward()
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.grad.dtype == torch.float32 and param.grad.abs().sum() > 0, name
        else:
            assert param.grad is None, name

    export_adapter(model, tmp_path, alpha=16.0)
    exported = load_file(tmp_path / "adapter_model.safetensors")
    assert {weight.dtype for weight in exported.values()} == {torch.bfloat16}
    q_proj, expert_layer = model.model.layers[2].self_attn.q_proj, model.model.layers[2].mlp.experts
    assert torch.equal(
        exported["model.llm.layers.2.attn.wq_du.lora_B.weight"], q_proj.lora_B.detach().to(torch.bfloat16)
    )
    assert torch.equal(
        exported["model.llm.layers.2.mlp.experts.w2_weight.lora_A.weight"],
        expert_layer.down_proj_lora_A.detach().to(torch.bfloat16),
    )
    assert q_proj.lora_B.dtype == torch.float32  # The export casts a copy.


def test_an_adam_step_a_bfloat16_weight_rounds_away_moves_a_float32_one():
    """A first Adam step moves each weight by about lr; at |a| = 0.004 half a bf16 unit is about 1.5e-5."""
    float32 = nn.Parameter(torch.full((4,), 0.004))
    bfloat16 = nn.Parameter(torch.full((4,), 0.004, dtype=torch.bfloat16))
    before = (float32.detach().clone(), bfloat16.detach().clone())
    optimizer = torch.optim.AdamW([float32, bfloat16], lr=1e-6, weight_decay=0.0, foreach=True)
    float32.grad, bfloat16.grad = torch.ones_like(float32), torch.ones_like(bfloat16)
    optimizer.step()

    assert torch.allclose(before[0] - float32.detach(), torch.full((4,), 1e-6), rtol=1e-3)
    assert torch.equal(bfloat16.detach(), before[1])

"""The server's LoRA path for Inkling 2 Mini, on a CPU, with vLLM's own code around prime-rl's experts.

Runs only where the vLLM build that serves the model is installed (``vllm.models.inkling2``, not in the release that
prime-rl pins), next to the trainer. The adapter directory a broadcast writes goes through vLLM's worker LoRA manager:
its name parsing and mapping, the check of ``max_lora_rank``, the packing, the wrapping of ``qkvr``, ``wo_ud`` and the
MoE, the copy into a slot, and the deltas of ``Inkling2MoEWithLoRA`` and ``add_linear_lora`` around prime-rl's
``Inkling2RLMoE`` and ``Inkling2BF16Experts``. What comes out must be the trainer's layers' outputs.

vLLM's kernels need a GPU. The stand-ins below compute what they are documented to compute, in FP32 instead of BF16:
``select_experts``, ``grouped_gemm``, ``swiglu`` and ``combine``. vLLM's CPU implementation of its linear-layer LoRA
kernels runs as it is. Left to the GPU check (``tools/check_inkling2_lora_serving.py``): the kernels, the model around
the layers, CUDA graphs, and the server.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("vllm.models.inkling2.nvidia.lora", reason="the vLLM release prime-rl pins has no Inkling 2")
pytest.importorskip("prime_rl.trainer.models.inkling2_loop", reason="the trainer is not installed")

from vllm import envs  # noqa: E402
from vllm.config import LoRAConfig as VllmLoRAConfig  # noqa: E402
from vllm.distributed import parallel_state  # noqa: E402
from vllm.lora import layers as vllm_lora_layers  # noqa: E402
from vllm.lora import lora_model, lora_weights, model_manager  # noqa: E402
from vllm.lora.layers import BaseLayerWithLoRA  # noqa: E402
from vllm.lora.punica_wrapper import punica_selector  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager  # noqa: E402
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear  # noqa: E402
from vllm.model_executor.models.interfaces import SupportsLoRA  # noqa: E402
from vllm.models.inkling2.nvidia import lora as vllm_lora  # noqa: E402
from vllm.models.inkling2.nvidia.attention import Inkling2Attention  # noqa: E402
from vllm.models.inkling2.nvidia.lora import add_linear_lora  # noqa: E402
from vllm.models.inkling2.nvidia.model import Inkling2ForConditionalGeneration  # noqa: E402
from vllm.utils import torch_utils  # noqa: E402

from prime_rl.configs.trainer import LoRAConfig  # noqa: E402
from prime_rl.inference.vllm.models.inkling2_loop import model as serving  # noqa: E402
from prime_rl.inference.vllm.models.inkling2_loop.fp8_experts import Inkling2FP8Experts  # noqa: E402
from prime_rl.trainer.lora import apply_lora_to_model  # noqa: E402
from prime_rl.trainer.models.layers.lora import experts as trainer_experts  # noqa: E402
from tests.unit.train.models.test_inkling2_loop import _randomized_model, _text_config  # noqa: E402
from tests.unit.train.models.test_inkling2_loop_lora import (  # noqa: E402
    ALPHA,
    RANK,
    TARGETS,
    exact_grouped_mm,
    export_adapter,
    randomize_adapters,
)

TOKENS = 48
NUM_LAYERS = 4  # A sliding and a full layer before the loop core, and one of each in it.
PASSES = 3


def config():
    return _text_config(num_hidden_layers=NUM_LAYERS, local_layer_ids=[0, 2])


# Stand-ins for the kernels, after the contracts in vLLM's `ops/grouped_gemm.py`, `ops/moe_sm90.py` and `ops/moe.py`.


def grouped_gemm(x, weight, perm, offsets, num_rows, topk, gather, out=None):
    """`out[r] (+)= x[r] @ weight[e].T`; with `gather`, row `r` reads `x[perm[r] // topk]`. Without `out` the rows of
    no expert are left unset."""
    result = x.new_full((num_rows, weight.shape[1]), float("nan")) if out is None else out
    for expert in range(weight.shape[0]):
        rows = torch.arange(int(offsets[expert]), int(offsets[expert + 1]))
        product = x[perm[rows] // topk if gather else rows] @ weight[expert].T
        result[rows] = product if out is None else result[rows] + product
    return result


def swiglu(gate_up, weights, sorted_ids, gate_cap, up_cap):
    gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
    softcapped = gate_cap * torch.tanh(gate / gate_cap) * torch.sigmoid(gate) * up_cap * torch.tanh(up / up_cap)
    return softcapped * weights[:, None]


def combine(down, row_of_slot, topk, dtype):
    rows = torch.where(row_of_slot[:, None] >= 0, down[row_of_slot.clamp(min=0)], 0.0)
    return rows.unflatten(0, (-1, topk)).sum(1).to(dtype)


def select_experts(x, gate_weight, bias, topk, route_scale, expert_bounds, is_padding=None):
    scores = torch.sigmoid(x @ gate_weight.T)
    ids = torch.topk(scores + bias, topk, dim=1).indices
    weights = scores.gather(1, ids)
    weights = route_scale * weights / weights.sum(1, keepdim=True)
    return ids.to(torch.int32), weights.flatten(), torch.arange(ids.numel(), dtype=torch.int32)


class Layer(nn.Module):
    """The modules of one vLLM decoder layer that take adapters, laid out as ``Inkling2Attention.__init__`` does."""

    def __init__(self, text_config, layer_idx: int, experts: nn.Module) -> None:
        super().__init__()
        sliding = layer_idx in (0, 2)
        heads, kv_heads, head_dim = (6, 3, 8) if sliding else (4, 2, 16)
        sizes = [heads * head_dim, heads * 4, heads * head_dim, heads, kv_heads * head_dim, kv_heads * head_dim]
        self.q_side_splits = [*sizes[:4], -sum(sizes[:4]) % 8]
        self.split_sizes = [*sizes, -sum(sizes) % 8]
        assert self.split_sizes[-1] > 0  # The padding slice, which has no adapter.
        self.attn = nn.Module()
        self.attn.qkvr = MergedColumnParallelLinear(text_config.hidden_size, self.split_sizes, bias=False)
        self.attn.wo_ud = RowParallelLinear(heads * head_dim, text_config.hidden_size, bias=False, reduce_results=False)
        visits = range(layer_idx, layer_idx + PASSES) if layer_idx >= 2 else (layer_idx,)
        self.mlp = serving.Inkling2RLMoE(text_config, visits, experts)


class Skeleton(nn.Module, SupportsLoRA):
    """The decoder layers' adapted modules under the names they have in ``Inkling2LoopForRL``, with its loading and
    vLLM's LoRA declarations."""

    packed_modules_mapping = Inkling2ForConditionalGeneration.packed_modules_mapping
    embedding_modules = Inkling2ForConditionalGeneration.embedding_modules
    hf_to_vllm_mapper = Inkling2ForConditionalGeneration.hf_to_vllm_mapper
    _renames = Inkling2ForConditionalGeneration._renames
    _shards = Inkling2ForConditionalGeneration._shards
    load_weights = serving.Inkling2LoopForRL.load_weights
    _trainer_values = serving.Inkling2LoopForRL._trainer_values

    def __init__(self, experts_cls=serving.Inkling2BF16Experts) -> None:
        super().__init__()
        self.config = self.text_config = config()
        self.visual = None
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            Layer(self.text_config, layer_idx, experts_cls(self.text_config)) for layer_idx in range(NUM_LAYERS)
        )

    def _refresh_derived(self) -> None:
        """Derived tensors belong to the modules left out here."""


class TrainerLayers:
    """The trainer's model, its checkpoint, and an adapter of it on disk."""

    def __init__(self, step_dir, seed: int) -> None:
        self.model = _randomized_model(config(), seed=seed)
        state_dict = {name: weight.clone() for name, weight in self.model.state_dict().items()}
        self.checkpoint = self.model.convert_to_hf(state_dict)
        apply_lora_to_model(self.model, LoRAConfig(rank=RANK, alpha=ALPHA, target_modules=TARGETS))
        randomize_adapters(self.model, seed=seed + 10)
        self.adapter = export_adapter(self.model, step_dir)

    def served_weights(self) -> list[tuple[str, Tensor]]:
        """The checkpoint's tensors of the modules above, in FP32 as the trainer's forward here."""
        names = ("wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv", "wo_ud")
        return [
            (name, weight)
            for name, weight in self.checkpoint.items()
            if ".mlp.gate." in name or ".mlp.experts." in name or any(f".attn.{part}." in name for part in names)
        ]

    def projections(self, layer_idx: int, x_q: Tensor, x_k: Tensor, x_v: Tensor) -> list[Tensor]:
        """In the order vLLM packs them."""
        attn = self.model.model.layers[layer_idx].self_attn
        return [attn.q_proj(x_q), attn.rel_proj(x_q), attn.gate_proj(x_q), attn.sink_proj(x_q), attn.k_proj(x_k), attn.v_proj(x_v)]  # fmt: skip

    def moe(self, layer_idx: int, x: Tensor) -> tuple[Tensor, Tensor]:
        mlp = self.model.model.layers[layer_idx].mlp
        return mlp(x.unsqueeze(0))[0], mlp.router(x)[1]


class Server:
    def __init__(self, model: Skeleton, max_lora_rank: int = 2 * RANK) -> None:
        self.model = model
        vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64),
            model_config=SimpleNamespace(
                get_vocab_size=lambda: 320, hf_config=SimpleNamespace(get_text_config=lambda: SimpleNamespace())
            ),
            lora_config=VllmLoRAConfig(max_lora_rank=max_lora_rank, max_loras=1, lora_dtype=torch.float32),
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
        )
        vllm_config.model_config.is_encoder_decoder = False
        self.manager = LRUCacheWorkerLoRAManager(vllm_config, torch.device("cpu"), {})
        self.manager.create_lora_manager(model, vllm_config)

    def load(self, adapter_dir) -> None:
        self.manager.add_adapter(LoRARequest("adapter", 1, str(adapter_dir), load_inplace=True))

    def adapter(self, token_adapters: list[int]) -> None:
        """The adapter of each token (0: none)."""
        mapping = vllm_lora_layers.LoRAMapping(
            index_mapping=tuple(token_adapters), prompt_mapping=(token_adapters[-1],), is_prefill=True
        )
        self.manager.set_active_adapters(set(), mapping)

    def projections(self, layer_idx: int, x_q: Tensor, x_k: Tensor, x_v: Tensor) -> list[Tensor]:
        layer = self.model.model.layers[layer_idx]
        attention = SimpleNamespace(
            qkvr=layer.attn.qkvr, split_sizes=layer.split_sizes, q_side_splits=layer.q_side_splits
        )
        return Inkling2Attention._project(attention, x_q, x_k, x_v)

    def output(self, layer_idx: int, x: Tensor) -> Tensor:
        wo_ud = self.model.model.layers[layer_idx].attn.wo_ud
        out = x @ wo_ud.weight.T
        add_linear_lora(wo_ud, out, x)
        return out


@pytest.fixture(scope="module")
def vllm_on_cpu():
    with pytest.MonkeyPatch.context() as monkeypatch:
        for module in (serving, vllm_lora):
            monkeypatch.setattr(module, "grouped_gemm", grouped_gemm)
        monkeypatch.setattr(serving, "swiglu", swiglu)
        monkeypatch.setattr(serving, "combine", combine)
        monkeypatch.setattr(serving, "select_experts", select_experts)
        monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", True, raising=False)
        # A machine without a GPU has no platform, which names the implementation of the linear layers' LoRA kernels.
        cpu = SimpleNamespace(get_punica_wrapper=lambda: "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU")
        monkeypatch.setattr(punica_selector, "current_platform", cpu)
        for module in (lora_model, lora_weights, model_manager, torch_utils):
            monkeypatch.setattr(module, "PIN_MEMORY", False)
        monkeypatch.setattr(trainer_experts, "_grouped_mm", exact_grouped_mm)
        # One rank: all the layers ask of vLLM's tensor-parallel group.
        monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0))
        yield


@pytest.fixture(scope="module")
def trainers(vllm_on_cpu, tmp_path_factory) -> list[TrainerLayers]:
    return [TrainerLayers(tmp_path_factory.mktemp(f"step_{seed}"), seed) for seed in (0, 1)]


@pytest.fixture
def server(trainers) -> Server:
    model = Skeleton().float()
    model.load_weights(trainers[0].served_weights())
    return Server(model)


def inputs(seed: int, width: int = 64) -> Tensor:
    return torch.randn((TOKENS, width), generator=torch.Generator().manual_seed(seed))


def difference(served: list[Tensor], trained: list[Tensor]) -> float:
    return max(float((left - right).abs().max()) for left, right in zip(served, trained, strict=True))


@torch.no_grad()
def assert_serves(server: Server, trainer: TrainerLayers, tolerance: float = 2e-4) -> None:
    """Every adapted module of every layer, on every loop pass, gives the trainer's output."""
    x_q, x_k, x_v = inputs(1), inputs(2), inputs(3)
    for layer_idx, layer in enumerate(trainer.model.model.layers):
        # One input for all projections (a sliding layer's first pass), and one per stream (every other visit).
        for streams in ((x_q, x_q, x_q), (x_q, x_k, x_v)):
            served, trained = server.projections(layer_idx, *streams), trainer.projections(layer_idx, *streams)
            assert difference(served, trained) < tolerance
        x = inputs(4, layer.self_attn.o_proj.in_features)
        assert difference([server.output(layer_idx, x)], [layer.self_attn.o_proj(x)]) < tolerance
        trained, routed = trainer.moe(layer_idx, x_q)
        mlp = server.model.model.layers[layer_idx].mlp
        captures = getattr(mlp, "base_layer", mlp).routing_captures
        for pass_idx, capture in enumerate(captures):
            captured = []
            capture.capture_fn = captured.append
            assert difference([mlp(x_q, pass_idx=pass_idx)], [trained]) < tolerance
            assert torch.equal(captured[0].sort(1).values.long(), routed.sort(1).values)
            capture.capture_fn = None


@torch.no_grad()
def assert_serves_the_base(server: Server, trainer: TrainerLayers) -> None:
    x = inputs(1)
    for layer_idx, layer in enumerate(trainer.model.model.layers):
        attn = layer.self_attn
        trained = [getattr(attn, name).base_layer(x) for name in ("q_proj", "rel_proj", "gate_proj", "sink_proj", "k_proj", "v_proj")]  # fmt: skip
        assert difference(server.projections(layer_idx, x, x, x), trained) < 2e-4


def test_an_adapter_from_disk_serves_the_trainers_layers(server, trainers):
    server.load(trainers[0].adapter)
    for name, module in server.model.named_modules():
        if name.endswith((".qkvr", ".wo_ud", ".mlp")):
            assert isinstance(module, BaseLayerWithLoRA), name
    server.adapter([1] * TOKENS)
    assert_serves(server, trainers[0])
    server.adapter([0] * TOKENS)
    assert_serves_the_base(server, trainers[0])
    with pytest.raises(AssertionError):
        assert_serves(server, trainers[0], tolerance=0.05)


def test_a_hot_swap_and_a_base_update_under_the_adapter_load_in_place(server, trainers):
    addresses = {name: param.data_ptr() for name, param in server.model.named_parameters()}
    server.load(trainers[0].adapter)
    server.adapter([1] * TOKENS)
    other = trainers[1]
    server.model.load_weights(other.served_weights())
    with pytest.raises(AssertionError):  # The other base under the first adapter.
        assert_serves(server, other, tolerance=0.05)
    server.load(other.adapter)
    server.adapter([1] * TOKENS)
    assert_serves(server, other)
    assert addresses == {name: param.data_ptr() for name, param in server.model.named_parameters()}


@torch.no_grad()
def test_tokens_of_the_base_and_padding_tokens_get_no_delta(server, trainers, monkeypatch):
    server.load(trainers[0].adapter)
    x = inputs(1)
    mlp = server.model.model.layers[2].mlp
    server.adapter([0] * TOKENS)
    base = mlp(x)
    server.adapter([1] * TOKENS)
    adapted = mlp(x)
    assert (base - adapted).abs().amax(1).min() > 1e-2
    server.adapter([1] * 10 + [0] * (TOKENS - 10))
    mixed = mlp(x)
    assert torch.allclose(mixed[:10], adapted[:10], atol=1e-5) and torch.allclose(mixed[10:], base[10:], atol=1e-5)
    is_padding = torch.arange(TOKENS) >= 40
    monkeypatch.setattr(serving, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(serving, "get_forward_context", lambda: SimpleNamespace(is_padding=is_padding))
    padded = mlp(x)
    assert torch.allclose(padded[:10], adapted[:10], atol=1e-5) and padded[40:].abs().max() == 0


def test_a_server_with_too_small_a_rank_refuses_the_expert_adapter(trainers):
    server = Server(Skeleton().float(), max_lora_rank=RANK)
    with pytest.raises(ValueError, match=f"LoRA rank {2 * RANK} is greater than max_lora_rank {RANK}"):
        server.load(trainers[0].adapter)


def test_a_server_that_would_drop_part_of_an_adapter_is_refused(vllm_on_cpu):
    lora_config = VllmLoRAConfig(max_lora_rank=16, target_modules=["qkvr", "wo_ud"])
    with pytest.raises(ValueError, match="takes no lora_target_modules"):
        serving.Inkling2LoopForRL(vllm_config=SimpleNamespace(lora_config=lora_config))


class FP8Experts(nn.Module):
    """Stands for ``Inkling2FP8Experts``, which needs sizes in whole 128 x 128 blocks."""

    supports_lora = Inkling2FP8Experts.supports_lora

    def __init__(self, text_config) -> None:
        super().__init__()


def test_fp8_experts_take_no_expert_adapter(trainers):
    server = Server(Skeleton(FP8Experts))
    assert not isinstance(server.model.model.layers[0].mlp, BaseLayerWithLoRA)
    with pytest.raises(ValueError, match="mlp_w13"):
        server.load(trainers[0].adapter)
    with pytest.raises(NotImplementedError, match="takes no LoRA adapters"):
        server.model.model.layers[0].mlp(inputs(1), lora=object())

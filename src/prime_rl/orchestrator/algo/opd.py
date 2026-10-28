from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import OPDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_reference_logprobs
from prime_rl.orchestrator.trajectories import iter_trainable_branches

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


class OPDAlgorithm(Algorithm):
    """On-policy distillation. Needs a teacher: the frozen reference model the
    per-token reverse KL is computed against.

    The policy samples its own rollouts; at ship time each sample's full
    context is prefill-scored under the teacher (``ref_logprobs`` on the
    wire), and the trainer evaluates the KL against the live policy. No
    credit is assigned — rollouts keep ``advantages=None`` and samples ship no
    advantage stream; ``group_size`` only fans out sampling.

    With ``teacher_context``, the teacher reads that text at the end of each
    sample's system message, and the logprobs over the sample's own tokens
    become the reference (on-policy context distillation)."""

    def __init__(self, config: OPDAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.teacher = config.teacher
        self.teacher_context = config.teacher_context
        self.renderer_config = config.renderer
        self.teacher_clients: InferenceClient | None = None  # frozen teacher endpoint, connected in setup()
        self.context_ids: list[int] = []  # teacher_context on the policy's tokenizer, built in setup()
        self.message_ends: set[int] = set()  # the renderer's end-of-message tokens, built in setup()

    async def setup(self) -> None:
        self.teacher_clients = await self.connect(self.teacher)
        if self.teacher_context is not None:
            from renderers.base import create_renderer, load_tokenizer

            tokenizer = load_tokenizer(self.clients.model_name)
            self.context_ids = tokenizer.encode(self.teacher_context, add_special_tokens=False)
            self.message_ends = set(create_renderer(tokenizer, self.renderer_config).get_stop_token_ids())

    def context_at(self, branch: vf.Branch) -> int | None:
        """Where ``teacher_context`` goes in the branch's tokens: at the token
        that closes its system message. None without a context."""
        if not self.context_ids:
            return None
        system = branch.nodes[0]
        if system.message.role != "system":
            raise ValueError(
                f"teacher_context: the sample opens with a {system.message.role} message, not a system one"
            )
        closes = [index for index, token in enumerate(system.token_ids) if token in self.message_ends]
        if not closes:
            raise ValueError("teacher_context: the sample's system message has no end-of-message token")
        return closes[-1]

    async def score_episode(self, episode: vf.Episode) -> None:
        teacher = self.teacher_clients
        assert teacher is not None, "teacher not connected — Algorithm.setup() must run first"
        branches = [
            branch
            for _, trace in iter_trainable_traces([episode], self.drop_truncated)
            for branch, _ in iter_trainable_branches(trace)
        ]
        context = self.context_ids
        places = [self.context_at(branch) for branch in branches]
        scores = await asyncio.gather(
            *(
                teacher.score(
                    branch.token_ids if at is None else branch.token_ids[:at] + context + branch.token_ids[at:]
                )
                for branch, at in zip(branches, places, strict=True)
            )
        )
        for branch, at, logprobs in zip(branches, places, scores, strict=True):
            # Every token before the context is the sample's own, scored on the same prefix; drop only the context's.
            assign_reference_logprobs(branch, logprobs if at is None else logprobs[:at] + logprobs[at + len(context) :])

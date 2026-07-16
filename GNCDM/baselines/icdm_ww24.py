"""PyTorch reimplementation of the core ICDM-WWW'24 inductive encoder.

The official repository is tied to torch 1.13 and DGL 1.1.  This module keeps
the model's response-aware graph induction in the project's current PyTorch
runtime: Q-graph propagation, separate right/wrong response channels, source
attention, and a GLIF-style cognitive interaction function.

It intentionally does not copy the official source.  In particular, graph
aggregation is expressed with ``index_add_`` and dense Math1 Q matrices so the
baseline can run without DGL.  Experiment-level protocol adaptation lives in
``experiments/_core/run_icdm_ww24.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class InteractionGraph:
    """A response graph represented by aligned user, item, and score tensors."""

    user: torch.Tensor
    item: torch.Tensor
    score: torch.Tensor

    def __post_init__(self):
        lengths = {self.user.numel(), self.item.numel(), self.score.numel()}
        if len(lengths) != 1:
            raise ValueError("user, item, and score must have identical lengths")

    def __len__(self) -> int:
        return self.user.numel()

    @property
    def device(self) -> torch.device:
        return self.user.device

    def to(self, device: torch.device | str) -> InteractionGraph:
        return InteractionGraph(
            self.user.to(device),
            self.item.to(device),
            self.score.to(device),
        )

    def subset(self, mask: torch.Tensor) -> InteractionGraph:
        return InteractionGraph(self.user[mask], self.item[mask], self.score[mask])


def concat_graphs(*graphs: InteractionGraph) -> InteractionGraph:
    """Concatenate non-empty response graphs without changing edge order."""

    graphs = tuple(graph for graph in graphs if len(graph) > 0)
    if not graphs:
        raise ValueError("at least one non-empty graph is required")
    return InteractionGraph(
        torch.cat([graph.user for graph in graphs]),
        torch.cat([graph.item for graph in graphs]),
        torch.cat([graph.score for graph in graphs]),
    )


@dataclass(frozen=True)
class EncodedState:
    """Node representations produced from a leakage-free response context."""

    student: torch.Tensor
    item: torch.Tensor
    concept: torch.Tensor


def _mean_by_index(values: torch.Tensor, index: torch.Tensor, size: int):
    out = values.new_zeros((size, values.shape[-1]))
    count = values.new_zeros((size, 1))
    if index.numel() > 0:
        out.index_add_(0, index, values)
        count.index_add_(0, index, values.new_ones((index.numel(), 1)))
    return out / count.clamp_min(1.0), count.squeeze(-1)


class SourceAttention(nn.Module):
    """Fuse several graph views while masking unavailable neighborhood views."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, sources: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        if sources.ndim != 3 or available.shape != sources.shape[:2]:
            raise ValueError("sources must be [N,S,D] and available must be [N,S]")
        logits = self.score(torch.tanh(self.proj(sources))).squeeze(-1)
        logits = logits.masked_fill(~available, -1e4)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(sources * weights.unsqueeze(-1), dim=1)


class ICDMWW24(nn.Module):
    """Inductive graph cognitive diagnosis model adapted for fixed Math1 topology.

    ``known_user_mask`` separates users observed during parameter learning from
    validation/test users.  Unseen users use a learned population prior plus
    representations induced solely from their support edges.
    """

    def __init__(
        self,
        n_user: int,
        n_item: int,
        n_know: int,
        q_matrix,
        known_user_mask: torch.Tensor,
        dim: int = 64,
        k_hop: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        q = torch.as_tensor(q_matrix, dtype=torch.float32)
        if q.shape != (n_item, n_know):
            raise ValueError(f"Q shape {tuple(q.shape)} != {(n_item, n_know)}")
        known_user_mask = torch.as_tensor(known_user_mask, dtype=torch.bool)
        if known_user_mask.shape != (n_user,):
            raise ValueError("known_user_mask must have shape [n_user]")

        self.n_user = n_user
        self.n_item = n_item
        self.n_know = n_know
        self.dim = dim
        self.k_hop = k_hop

        q_item_norm = q / q.sum(dim=1, keepdim=True).clamp_min(1.0)
        diagnostic_q = q.clone()
        diagnostic_q[diagnostic_q.sum(dim=1) == 0] = 1.0
        diagnostic_q_norm = diagnostic_q / diagnostic_q.sum(dim=1, keepdim=True)
        self.register_buffer("q_matrix", q)
        self.register_buffer("q_item_norm", q_item_norm)
        self.register_buffer("diagnostic_q", diagnostic_q)
        self.register_buffer("diagnostic_q_norm", diagnostic_q_norm)
        q_t = q.t()
        q_concept_norm = q_t / q_t.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("q_item_sparse", q_item_norm.to_sparse_coo().coalesce())
        self.register_buffer("q_concept_sparse", q_concept_norm.to_sparse_coo().coalesce())
        self.register_buffer("known_user_mask", known_user_mask)

        self.student_embedding = nn.Embedding(n_user, dim)
        self.general_student = nn.Parameter(torch.zeros(1, dim))
        self.item_right_embedding = nn.Embedding(n_item, dim)
        self.item_wrong_embedding = nn.Embedding(n_item, dim)
        self.concept_embedding = nn.Embedding(n_know, dim)

        self.right_norm = nn.LayerNorm(dim)
        self.wrong_norm = nn.LayerNorm(dim)
        self.concept_norm = nn.LayerNorm(dim)
        self.student_attention = SourceAttention(dim)
        self.item_attention = SourceAttention(dim)
        self.dropout = nn.Dropout(dropout)

        self.student_to_knowledge = nn.Linear(dim, n_know)
        self.item_to_knowledge = nn.Linear(dim, n_know)
        self.concept_to_knowledge = nn.Linear(dim, n_know)
        self.discrimination = nn.Embedding(n_item, 1)
        self.predictor = nn.Sequential(
            nn.Linear(n_know, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.student_embedding.weight)
        nn.init.xavier_normal_(self.item_right_embedding.weight)
        nn.init.xavier_normal_(self.item_wrong_embedding.weight)
        nn.init.xavier_normal_(self.concept_embedding.weight)
        nn.init.zeros_(self.general_student)
        nn.init.zeros_(self.discrimination.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _q_propagation(self):
        right = self.item_right_embedding.weight
        wrong = self.item_wrong_embedding.weight
        concept = self.concept_embedding.weight
        for _ in range(self.k_hop):
            concept_to_item = torch.sparse.mm(self.q_item_sparse, concept)
            next_right = self.right_norm(right + concept_to_item)
            next_wrong = self.wrong_norm(wrong + concept_to_item)
            concept_message = 0.5 * (
                torch.sparse.mm(self.q_concept_sparse, right)
                + torch.sparse.mm(self.q_concept_sparse, wrong)
            )
            next_concept = self.concept_norm(concept + concept_message)
            right, wrong, concept = (
                self.dropout(next_right),
                self.dropout(next_wrong),
                self.dropout(next_concept),
            )
        return right, wrong, concept

    def _student_prior(self):
        population = self.general_student.expand(self.n_user, -1)
        known = self.student_embedding.weight + population
        return torch.where(self.known_user_mask[:, None], known, population)

    def encode(self, context: InteractionGraph) -> EncodedState:
        """Encode all nodes using only the supplied observed response edges."""

        if context.device != self.q_matrix.device:
            raise ValueError("context and model must be on the same device")
        right, wrong, concept = self._q_propagation()
        item_concept = torch.sparse.mm(self.q_item_sparse, concept)

        right_mask = context.score >= 0.5
        wrong_mask = ~right_mask
        right_user, right_count = _mean_by_index(
            right[context.item[right_mask]], context.user[right_mask], self.n_user
        )
        wrong_user, wrong_count = _mean_by_index(
            wrong[context.item[wrong_mask]], context.user[wrong_mask], self.n_user
        )
        concept_user, concept_count = _mean_by_index(
            item_concept[context.item], context.user, self.n_user
        )

        prior = self._student_prior()
        student_sources = torch.stack([prior, right_user, wrong_user, concept_user], dim=1)
        student_available = torch.stack(
            [
                torch.ones_like(right_count, dtype=torch.bool),
                right_count > 0,
                wrong_count > 0,
                concept_count > 0,
            ],
            dim=1,
        )
        student = self.student_attention(student_sources, student_available)

        item_sources = torch.stack([right, wrong, item_concept], dim=1)
        item_available = torch.ones(
            (self.n_item, item_sources.shape[1]),
            dtype=torch.bool,
            device=item_sources.device,
        )
        item = self.item_attention(item_sources, item_available)
        return EncodedState(student=student, item=item, concept=concept)

    def predict(self, user: torch.Tensor, item: torch.Tensor, state: EncodedState):
        """Predict correctness probabilities for user-item pairs."""

        q = self.diagnostic_q[item]
        concept_view = self.diagnostic_q_norm[item] @ state.concept
        concept_gate = torch.sigmoid(self.concept_to_knowledge(concept_view))
        student_knowledge = self.student_to_knowledge(state.student[user])
        item_knowledge = self.item_to_knowledge(state.item[item])
        discrimination = torch.sigmoid(self.discrimination(item))
        diagnostic_state = discrimination * (
            torch.sigmoid(student_knowledge * concept_gate)
            - torch.sigmoid(item_knowledge * concept_gate)
        )
        diagnostic_state = diagnostic_state * q
        return torch.sigmoid(self.predictor(diagnostic_state).squeeze(-1))

    def mastery(self, context: InteractionGraph) -> torch.Tensor:
        state = self.encode(context)
        return torch.sigmoid(self.student_to_knowledge(state.student))

    def l2_regularization(self, item: torch.Tensor) -> torch.Tensor:
        right = self.item_right_embedding(item)
        wrong = self.item_wrong_embedding(item)
        return 0.5 * (right.square().mean() + wrong.square().mean())

    def clip_monotonic_weights(self):
        """Apply the non-negative interaction-layer constraint used by NCD models."""

        with torch.no_grad():
            for module in self.predictor:
                if isinstance(module, nn.Linear):
                    module.weight.clamp_(min=0.0)

    def loss(
        self,
        target: InteractionGraph,
        context: InteractionGraph,
        weight_reg: float = 1e-3,
    ) -> torch.Tensor:
        state = self.encode(context)
        prediction = self.predict(target.user, target.item, state)
        bce = F.binary_cross_entropy(prediction, target.score)
        return bce + weight_reg * self.l2_regularization(target.item)

"""Frozen query encoder and anchor-relative uplift scorer used by CARP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


def attention_mask_mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool token representations while excluding padding tokens."""
    weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-9)


class FrozenQueryEncoder:
    """Qwen encoder used once per complete query; encoder parameters stay frozen."""

    def __init__(self, model_name: str, device: str | None = None, max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], batch_size: int = 16) -> Tensor:
        vectors: list[Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            ).to(self.device)
            output = self.model(**encoded)
            vectors.append(attention_mask_mean_pool(output.last_hidden_state, encoded['attention_mask']).cpu())
        if not vectors:
            return torch.empty((0, self.hidden_size), dtype=torch.float32)
        return torch.cat(vectors, dim=0)


class PriorResidualUpliftHead(nn.Module):
    """Predict non-anchor uplift with a training-set prior and a centered residual.

    The residual is centered at a fixed reference embedding, so the prior remains
    the prediction at that reference instead of being absorbed by the MLP bias.
    """

    def __init__(
        self,
        feature_dim: int,
        prior: Tensor,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if prior.ndim != 1:
            raise ValueError('prior must be a one-dimensional tensor')
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, prior.numel(), bias=False)
        nn.init.zeros_(self.fc2.weight)
        self.register_buffer('prior', prior.detach().clone().float())

    def residual(self, features: Tensor) -> Tensor:
        return self.fc2(self.dropout(torch.nn.functional.gelu(self.fc1(features))))

    def forward(self, features: Tensor, reference_features: Tensor) -> Tensor:
        if reference_features.ndim == 1:
            reference_features = reference_features.unsqueeze(0)
        reference = self.residual(reference_features)
        return self.prior.unsqueeze(0) + self.residual(features) - reference


@dataclass(frozen=True)
class UpliftCheckpointInfo:
    anchor: str
    methods: list[str]
    encoder_name: str
    max_length: int
    hidden_dim: int
    reference_text: str


def build_full_uplifts(anchor: str, methods: Sequence[str], non_anchor_uplifts: Tensor) -> Tensor:
    """Insert the fixed zero anchor coordinate into predicted non-anchor uplifts."""
    non_anchor = [method for method in methods if method != anchor]
    if non_anchor_uplifts.shape[-1] != len(non_anchor):
        raise ValueError('uplift tensor and methods have incompatible dimensions')
    result = torch.zeros((*non_anchor_uplifts.shape[:-1], len(methods)), dtype=non_anchor_uplifts.dtype)
    positions = [index for index, method in enumerate(methods) if method != anchor]
    result[..., positions] = non_anchor_uplifts
    return result

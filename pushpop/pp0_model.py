"""Tiny decoder-only transformer for PP0."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class TinyTransformerConfig:
    vocab_size: int
    context_length: int
    d_model: int = 128
    d_mlp: int = 256
    n_layers: int = 2
    n_heads: int = 4

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.d_model <= 0 or self.d_mlp <= 0:
            raise ValueError("d_model and d_mlp must be positive")
        if self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("n_layers and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    def to_dict(self) -> dict[str, int]:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "d_mlp": self.d_mlp,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "TinyTransformerConfig":
        return cls(**data)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = x.shape
        query = self.q_proj(x).view(batch_size, sequence_length, self.n_heads, self.d_head)
        key = self.k_proj(x).view(batch_size, sequence_length, self.n_heads, self.d_head)
        value = self.v_proj(x).view(batch_size, sequence_length, self.n_heads, self.d_head)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_model)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_mlp)
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(config.d_mlp, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_ln = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.config.context_length}"
            )

        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states: list[torch.Tensor] | None = [x] if return_hidden_states else None
        for block in self.blocks:
            x = block(x)
            if hidden_states is not None:
                hidden_states.append(x)
        x = self.final_ln(x)
        if hidden_states is not None:
            hidden_states.append(x)

        logits = self.unembed(x)
        if hidden_states is not None:
            return logits, tuple(hidden_states)
        return logits

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

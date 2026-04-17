"""Tiny decoder-only transformer for PP0."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
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


@dataclass(frozen=True, slots=True)
class ResidualIntervention:
    layer_name: str
    position_index: int
    replacement_vector: torch.Tensor
    mode: str = "replace"

    def __post_init__(self) -> None:
        if self.position_index < 0:
            raise ValueError("position_index must be non-negative")
        if self.replacement_vector.ndim != 1:
            raise ValueError("replacement_vector must be one-dimensional")
        if self.mode not in {"replace", "add"}:
            raise ValueError("mode must be one of: replace, add")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_head_outputs: bool = False,
        return_attention_patterns: bool = False,
        head_interventions: dict[int, Sequence[ResidualIntervention]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        head_context = torch.matmul(attention, value).transpose(1, 2).contiguous()

        has_head_interventions = bool(head_interventions) and any(
            interventions for interventions in head_interventions.values()
        )
        if not return_head_outputs and not return_attention_patterns and not has_head_interventions:
            output = head_context.view(batch_size, sequence_length, d_model)
            return self.out_proj(output)

        auxiliary: dict[str, torch.Tensor] = {}
        if return_head_outputs or has_head_interventions:
            out_weight_by_head = (
                self.out_proj.weight.view(d_model, self.n_heads, self.d_head)
                .permute(1, 0, 2)
                .contiguous()
            )
            head_outputs = torch.einsum("bshd,hmd->bshm", head_context, out_weight_by_head)
            if head_interventions:
                for head_index, interventions in head_interventions.items():
                    head_outputs[:, :, head_index, :] = _apply_residual_interventions(
                        head_outputs[:, :, head_index, :],
                        interventions,
                    )

            attention_update = head_outputs.sum(dim=2)
            if self.out_proj.bias is not None:
                attention_update = attention_update + self.out_proj.bias.view(1, 1, -1)
            if return_head_outputs:
                auxiliary["head_outputs"] = head_outputs
        else:
            attention_update = self.out_proj(head_context.view(batch_size, sequence_length, d_model))

        if return_attention_patterns:
            auxiliary["attention_patterns"] = attention

        if not return_head_outputs and not return_attention_patterns:
            return attention_update
        return attention_update, auxiliary


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

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_component_outputs: bool = False,
        return_attention_patterns: bool = False,
        attention_interventions: Sequence[ResidualIntervention] | None = None,
        attention_head_interventions: dict[int, Sequence[ResidualIntervention]] | None = None,
        mlp_interventions: Sequence[ResidualIntervention] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attention_output = self.attention(
            self.ln_1(x),
            return_head_outputs=return_component_outputs,
            return_attention_patterns=return_attention_patterns,
            head_interventions=attention_head_interventions,
        )
        if return_component_outputs or return_attention_patterns:
            attention_update, attention_auxiliary = attention_output
        else:
            attention_update = attention_output
        attention_update = _apply_residual_interventions(
            attention_update,
            attention_interventions or (),
        )
        x = x + attention_update

        mlp_update = self.mlp(self.ln_2(x))
        mlp_update = _apply_residual_interventions(mlp_update, mlp_interventions or ())
        x = x + mlp_update

        if return_component_outputs or return_attention_patterns:
            block_outputs: dict[str, torch.Tensor] = {}
            if return_component_outputs:
                block_outputs["attn"] = attention_update
                block_outputs["attn_heads"] = attention_auxiliary["head_outputs"]
                block_outputs["mlp"] = mlp_update
            if return_attention_patterns:
                block_outputs["attn_patterns"] = attention_auxiliary["attention_patterns"]
            return x, block_outputs
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
        return_component_outputs: bool = False,
        return_attention_patterns: bool = False,
        interventions: Sequence[ResidualIntervention] | None = None,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, tuple[torch.Tensor, ...]]
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
        | tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]
    ):
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.config.context_length}"
            )

        interventions_by_layer = _group_interventions_by_layer(interventions)
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = _apply_residual_interventions(x, interventions_by_layer.get("embed", ()))
        hidden_states: list[torch.Tensor] | None = [x] if return_hidden_states else None
        auxiliary_outputs: dict[str, torch.Tensor] | None = (
            {} if return_component_outputs or return_attention_patterns else None
        )
        for block_index, block in enumerate(self.blocks):
            block_output = block(
                x,
                return_component_outputs=return_component_outputs,
                return_attention_patterns=return_attention_patterns,
                attention_interventions=interventions_by_layer.get(f"block_{block_index}.attn", ()),
                attention_head_interventions={
                    head_index: interventions_by_layer.get(
                        f"block_{block_index}.attn_head_{head_index}",
                        (),
                    )
                    for head_index in range(block.attention.n_heads)
                },
                mlp_interventions=interventions_by_layer.get(f"block_{block_index}.mlp", ()),
            )
            if return_component_outputs or return_attention_patterns:
                x, block_component_outputs = block_output
                assert auxiliary_outputs is not None
                if return_component_outputs:
                    auxiliary_outputs[f"block_{block_index}.attn"] = block_component_outputs["attn"]
                    for head_index in range(block.attention.n_heads):
                        auxiliary_outputs[f"block_{block_index}.attn_head_{head_index}"] = (
                            block_component_outputs["attn_heads"][:, :, head_index, :]
                        )
                    auxiliary_outputs[f"block_{block_index}.mlp"] = block_component_outputs["mlp"]
                if return_attention_patterns:
                    auxiliary_outputs[f"block_{block_index}.attn_pattern"] = block_component_outputs[
                        "attn_patterns"
                    ]
            else:
                x = block_output
            x = _apply_residual_interventions(
                x,
                interventions_by_layer.get(f"block_{block_index}", ()),
            )
            if hidden_states is not None:
                hidden_states.append(x)
        x = self.final_ln(x)
        x = _apply_residual_interventions(x, interventions_by_layer.get("final_ln", ()))
        if hidden_states is not None:
            hidden_states.append(x)

        logits = self.unembed(x)
        if hidden_states is not None and auxiliary_outputs is not None:
            return logits, tuple(hidden_states), auxiliary_outputs
        if hidden_states is not None:
            return logits, tuple(hidden_states)
        if auxiliary_outputs is not None:
            return logits, auxiliary_outputs
        return logits

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


def _group_interventions_by_layer(
    interventions: Sequence[ResidualIntervention] | None,
) -> dict[str, tuple[ResidualIntervention, ...]]:
    if not interventions:
        return {}

    grouped: dict[str, list[ResidualIntervention]] = defaultdict(list)
    for intervention in interventions:
        grouped[intervention.layer_name].append(intervention)
    return {
        layer_name: tuple(layer_interventions)
        for layer_name, layer_interventions in grouped.items()
    }


def _apply_residual_interventions(
    x: torch.Tensor,
    interventions: Sequence[ResidualIntervention],
) -> torch.Tensor:
    if not interventions:
        return x

    x = x.clone()
    d_model = x.shape[-1]
    for intervention in interventions:
        if intervention.position_index >= x.shape[1]:
            continue
        replacement_vector = intervention.replacement_vector.to(device=x.device, dtype=x.dtype)
        if replacement_vector.shape != (d_model,):
            raise ValueError(
                f"replacement_vector for {intervention.layer_name} must have shape {(d_model,)}, "
                f"got {tuple(replacement_vector.shape)}"
            )
        if intervention.mode == "replace":
            x[:, intervention.position_index, :] = replacement_vector
        else:
            x[:, intervention.position_index, :] += replacement_vector
    return x

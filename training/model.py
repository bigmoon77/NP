from __future__ import annotations

import torch
from torch import nn


class MessagePassingBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2 + 7, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        rest_positions: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source = edge_index[0]
        target = edge_index[1]
        current_delta = positions[source] - positions[target]
        rest_delta = rest_positions[source] - rest_positions[target]
        rest_length = torch.linalg.vector_norm(rest_delta, dim=-1, keepdim=True)
        edge_features = torch.cat((current_delta, rest_delta, rest_length), dim=-1)
        message_input = torch.cat(
            (node_features[source], node_features[target], edge_features), dim=-1
        )
        messages = self.message_mlp(message_input)

        aggregate = torch.zeros_like(node_features)
        aggregate.index_add_(0, target, messages)
        degree = torch.zeros(
            (node_features.shape[0], 1),
            dtype=node_features.dtype,
            device=node_features.device,
        )
        degree.index_add_(
            0,
            target,
            torch.ones((target.shape[0], 1), dtype=node_features.dtype, device=node_features.device),
        )
        aggregate = aggregate / degree.clamp_min(1.0)
        update = self.update_mlp(torch.cat((node_features, aggregate), dim=-1))
        return self.norm(node_features + update)


class MeshGraphSimulator(nn.Module):
    """A resolution-independent graph network predicting per-node acceleration."""

    def __init__(self, hidden_size: int = 64, message_passing_steps: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [MessagePassingBlock(hidden_size) for _ in range(message_passing_steps)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(
        self,
        rest_positions: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((positions - rest_positions, velocities, rest_positions), dim=-1)
        hidden = self.encoder(features)
        for block in self.blocks:
            hidden = block(hidden, positions, rest_positions, edge_index)
        return self.decoder(hidden)

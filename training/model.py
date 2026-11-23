from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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
        length_scale: float,
    ) -> torch.Tensor:
        source = edge_index[0]
        target = edge_index[1]
        current_delta = (positions[source] - positions[target]) / length_scale
        rest_delta = (rest_positions[source] - rest_positions[target]) / length_scale
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
            0, target,
            torch.ones((target.shape[0], 1), dtype=node_features.dtype, device=node_features.device),
        )
        aggregate = aggregate / degree.clamp_min(1.0)
        update = self.update_mlp(torch.cat((node_features, aggregate), dim=-1))
        return self.norm(node_features + update)


class MeshGraphSimulator(nn.Module):
    """Resolution-independent quasi-static graph model predicting displacement and stress."""

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
            nn.Linear(hidden_size, 4),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(
        self,
        rest_positions: torch.Tensor,
        positions: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        length_scale: float = 1.0,
        displacement_scale: float = 1.0,
        stress_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centered_rest = rest_positions - torch.mean(rest_positions, dim=0, keepdim=True)
        type_features = F.one_hot(node_types.long(), num_classes=3).to(positions.dtype)
        features = torch.cat(
            ((positions - rest_positions) / length_scale, centered_rest / length_scale, type_features),
            dim=-1,
        )
        hidden = self.encoder(features)
        for block in self.blocks:
            hidden = block(hidden, positions, rest_positions, edge_index, length_scale)
        output = self.decoder(hidden)
        displacement = output[:, :3] * displacement_scale
        stress = output[:, 3] * stress_scale
        return displacement, stress

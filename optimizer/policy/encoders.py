"""Learned encoders: program graph, hardware, candidate configuration.

The :class:`GraphEncoder` is a residual message-passing network over the
:class:`~compiler.ir.ProgramGraph` (no string round-trip anywhere): node
features flow along dataflow edges in both directions, then mean+max pooling
yields ``z_program``.  Shared by the performance model (trained end-to-end
with the regression loss) and the RL policy (trained with PPO).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from compiler.ir import NODE_FEATURE_DIM, ProgramGraph
from compiler.transformations.space import CANDIDATE_FEATURE_DIM
from hardware.gpu_info import HARDWARE_FEATURE_DIM


def _mlp(dims: list[int], act: type[nn.Module] = nn.GELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class GraphEncoder(nn.Module):
    """Message-passing encoder: (node_features, adjacency) → embedding."""

    def __init__(self, hidden: int = 64, out_dim: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.embed = _mlp([NODE_FEATURE_DIM, hidden, hidden])
        self.rounds = nn.ModuleList(
            [_mlp([3 * hidden, hidden, hidden]) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.out = _mlp([2 * hidden, hidden, out_dim])
        self.out_dim = out_dim

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """node_feats: (B, N, F); adj: (B, N, N) with adj[b, i, j]=1 for edge i→j.
        Returns (B, out_dim)."""
        fwd = adj / adj.sum(-1, keepdim=True).clamp_min(1.0)          # producer→consumer
        bwd = adj.transpose(1, 2) / adj.transpose(1, 2).sum(-1, keepdim=True).clamp_min(1.0)
        h = self.embed(node_feats)
        for mlp, norm in zip(self.rounds, self.norms):
            msg_in = torch.bmm(bwd, h)    # aggregate from producers
            msg_out = torch.bmm(fwd, h)   # aggregate from consumers
            h = norm(h + mlp(torch.cat([h, msg_in, msg_out], dim=-1)))
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.out(pooled)

    def encode_graph(self, graph: ProgramGraph, device: str = "cpu") -> torch.Tensor:
        """Convenience single-graph forward (1, out_dim)."""
        feats = torch.from_numpy(graph.node_features()).unsqueeze(0).to(device)
        adj = torch.from_numpy(graph.adjacency()).unsqueeze(0).to(device)
        return self.forward(feats, adj)


class HardwareEncoder(nn.Module):
    """GPU spec feature vector → embedding."""

    def __init__(self, hidden: int = 32, out_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp([HARDWARE_FEATURE_DIM, hidden, out_dim])
        self.out_dim = out_dim

    def forward(self, hw_feats: torch.Tensor) -> torch.Tensor:
        return self.net(hw_feats)


class CandidateEncoder(nn.Module):
    """Structured configuration encoding → embedding."""

    def __init__(self, hidden: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.net = _mlp([CANDIDATE_FEATURE_DIM, hidden, out_dim])
        self.out_dim = out_dim

    def forward(self, cand_feats: torch.Tensor) -> torch.Tensor:
        return self.net(cand_feats)


def pad_graph_batch(graphs: list[ProgramGraph], device: str = "cpu"
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch variable-size graphs into padded (B, N, F) + (B, N, N) tensors."""
    n_max = max(len(g) for g in graphs)
    feats = np.zeros((len(graphs), n_max, NODE_FEATURE_DIM), dtype=np.float32)
    adj = np.zeros((len(graphs), n_max, n_max), dtype=np.float32)
    for i, g in enumerate(graphs):
        n = len(g)
        feats[i, :n] = g.node_features()
        adj[i, :n, :n] = g.adjacency()
    return (torch.from_numpy(feats).to(device), torch.from_numpy(adj).to(device))

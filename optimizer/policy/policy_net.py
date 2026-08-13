"""Actor-critic policy over the structured transformation space.

State = program graph ⊕ hardware ⊕ current configuration ⊕ search scalars,
fused by a small transformer over the four embedding tokens.  Heads:

- discrete action logits over the global transformation catalog (masked),
- a Gaussian head for continuous parameters (active when a task's space
  declares any; degenerate otherwise),
- state value for PPO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from compiler.transformations.actions import NUM_ACTIONS
from optimizer.policy.encoders import (
    CandidateEncoder,
    GraphEncoder,
    HardwareEncoder,
    _mlp,
)

SCALAR_DIM = 3  # steps_left, log2(current/baseline), log2(best/baseline)


@dataclass
class PolicyOutput:
    action: torch.Tensor        # (B,) chosen discrete action
    log_prob: torch.Tensor      # (B,)
    entropy: torch.Tensor       # (B,)
    value: torch.Tensor         # (B,)
    logits: torch.Tensor        # (B, NUM_ACTIONS) masked logits


class PolicyNet(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, n_continuous: int = 1) -> None:
        super().__init__()
        self.graph_enc = GraphEncoder(hidden=d_model, out_dim=d_model)
        self.hw_enc = HardwareEncoder(out_dim=d_model)
        self.cand_enc = CandidateEncoder(out_dim=d_model)
        self.scalar_enc = _mlp([SCALAR_DIM, d_model, d_model])
        self.token_type = nn.Parameter(torch.zeros(4, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.0, activation="gelu")
        self.fuse = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.action_head = nn.Linear(d_model, NUM_ACTIONS)
        self.value_head = nn.Linear(d_model, 1)
        # Continuous head (mean, per-dim learned log-std); used only when the
        # task's parameter space declares continuous parameters.
        self.cont_mu = nn.Linear(d_model, n_continuous)
        self.cont_logstd = nn.Parameter(torch.full((n_continuous,), -0.5))

    def _fused(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack([
            self.graph_enc(obs["node_feats"], obs["adj"]),
            self.hw_enc(obs["hw"]),
            self.cand_enc(obs["cand"]),
            self.scalar_enc(obs["scalars"]),
        ], dim=1) + self.token_type.unsqueeze(0)
        return self.fuse(tokens).mean(dim=1)

    def forward(self, obs: dict[str, torch.Tensor],
                action: torch.Tensor | None = None,
                deterministic: bool = False) -> PolicyOutput:
        h = self._fused(obs)
        logits = self.action_head(h)
        mask = obs["mask"].bool()
        logits = logits.masked_fill(~mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return PolicyOutput(
            action=action,
            log_prob=dist.log_prob(action),
            entropy=dist.entropy(),
            value=self.value_head(h).squeeze(-1),
            logits=logits,
        )

    def continuous(self, obs: dict[str, torch.Tensor]) -> torch.distributions.Normal:
        """Gaussian over continuous parameters (unit-interval parameterization)."""
        h = self._fused(obs)
        return torch.distributions.Normal(torch.sigmoid(self.cont_mu(h)),
                                          self.cont_logstd.exp())


def obs_to_tensors(obs_list: list[dict[str, np.ndarray]], device: str = "cpu"
                   ) -> dict[str, torch.Tensor]:
    """Stack per-episode observation dicts into batched tensors."""
    out: dict[str, torch.Tensor] = {}
    for key in ("node_feats", "adj", "hw", "cand", "scalars", "mask"):
        out[key] = torch.stack(
            [torch.as_tensor(np.asarray(o[key], dtype=np.float32)) for o in obs_list]
        ).to(device)
    return out

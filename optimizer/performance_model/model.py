"""Learned performance surrogate: deep ensemble with uncertainty.

Each ensemble member embeds (program graph, hardware, candidate config) and
predicts log-latency, log-memory, and compile-success probability.  The
ensemble mean is the prediction; the spread across members is the epistemic
uncertainty that drives exploration (UCB / Thompson sampling).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from compiler.transformations.space import Config
from hardware.gpu_info import HardwareSpec
from optimizer.performance_model.features import (
    FeatureBatch,
    TrainingRow,
    rows_to_batch,
)
from optimizer.policy.encoders import CandidateEncoder, GraphEncoder, HardwareEncoder, _mlp

logger = logging.getLogger(__name__)


class SurrogateNet(nn.Module):
    """One ensemble member."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.graph_enc = GraphEncoder(hidden=hidden, out_dim=hidden)
        self.hw_enc = HardwareEncoder(out_dim=32)
        self.cand_enc = CandidateEncoder(out_dim=hidden)
        joint = hidden + 32 + hidden
        self.trunk = _mlp([joint, 2 * hidden, 2 * hidden])
        self.head_time = nn.Linear(2 * hidden, 1)
        self.head_mem = nn.Linear(2 * hidden, 1)
        self.head_compile = nn.Linear(2 * hidden, 1)

    def forward(self, b: FeatureBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.cat([
            self.graph_enc(b.node_feats, b.adj),
            self.hw_enc(b.hw_feats),
            self.cand_enc(b.cand_feats),
        ], dim=-1)
        h = F.gelu(self.trunk(z))
        return (self.head_time(h).squeeze(-1),
                self.head_mem(h).squeeze(-1),
                self.head_compile(h).squeeze(-1))


def _member_loss(net: SurrogateNet, batch: FeatureBatch) -> torch.Tensor:
    log_ms, log_mem, compile_logit = net(batch)
    loss = F.binary_cross_entropy_with_logits(compile_logit, batch.compiled)
    mask_t = torch.isfinite(batch.log_ms)
    if mask_t.any():
        loss = loss + F.mse_loss(log_ms[mask_t], batch.log_ms[mask_t])
    mask_m = torch.isfinite(batch.log_mem)
    if mask_m.any():
        loss = loss + 0.2 * F.mse_loss(log_mem[mask_m], batch.log_mem[mask_m])
    return loss


@dataclass
class SurrogatePrediction:
    mean_ms: float
    std_ms: float
    compile_prob: float
    memory_bytes: float
    mu_log_ms: float
    std_log_ms: float


class PerformanceModel:
    """High-level train/predict API over the deep ensemble."""

    def __init__(self, n_members: int = 4, hidden: int = 64, device: str = "cpu",
                 seed: int = 0) -> None:
        self.device = device
        self.n_members = n_members
        self.hidden = hidden
        torch.manual_seed(seed)
        self.members = [SurrogateNet(hidden).to(device) for _ in range(n_members)]
        self.opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in self.members]
        self.rows: list[TrainingRow] = []
        self._rng = random.Random(seed)
        self.trained = False

    # ----------------------------------------------------------------- fit
    def add_rows(self, rows: list[TrainingRow]) -> None:
        self.rows.extend(rows)

    def fit(self, epochs: int = 20, batch_size: int = 128,
            min_rows: int = 8) -> dict[str, float]:
        """(Re)train every member on bootstrap resamples of the corpus."""
        if len(self.rows) < min_rows:
            logger.info("surrogate: only %d rows, skipping fit", len(self.rows))
            return {"loss": float("nan"), "rows": len(self.rows)}
        last = 0.0
        for member, opt in zip(self.members, self.opts):
            member.train()
            data = [self._rng.choice(self.rows) for _ in range(len(self.rows))]
            for _ in range(epochs):
                self._rng.shuffle(data)
                for i in range(0, len(data), batch_size):
                    batch = rows_to_batch(data[i:i + batch_size]).to(self.device)
                    opt.zero_grad()
                    loss = _member_loss(member, batch)
                    loss.backward()
                    opt.step()
                    last = float(loss.detach())
        self.trained = True
        return {"loss": last, "rows": len(self.rows)}

    # ------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, task: str, shape: tuple[int, ...], configs: list[Config],
                hardware: HardwareSpec, member: int | None = None
                ) -> list[SurrogatePrediction]:
        """Ensemble prediction (or a single member's, for Thompson sampling)."""
        rows = [TrainingRow(task=task, shape=tuple(shape), config=dict(c),
                            gpu_name=hardware.name, compiled=True,
                            latency_ms=None, memory_bytes=None) for c in configs]
        batch = rows_to_batch(rows, hardware=hardware).to(self.device)
        nets = self.members if member is None else [self.members[member]]
        times, mems, compiles = [], [], []
        for net in nets:
            net.eval()
            t, m, c = net(batch)
            times.append(t)
            mems.append(m)
            compiles.append(torch.sigmoid(c))
        t = torch.stack(times)            # (K, B)
        mu = t.mean(0)
        std = t.std(0) if t.shape[0] > 1 else torch.zeros_like(mu)
        mem = torch.stack(mems).mean(0)
        pc = torch.stack(compiles).mean(0)
        out = []
        for i in range(len(configs)):
            mean_ms = float(torch.exp(mu[i]))
            out.append(SurrogatePrediction(
                mean_ms=mean_ms,
                std_ms=mean_ms * float(std[i]),   # first-order log→linear
                compile_prob=float(pc[i]),
                memory_bytes=float(torch.expm1(mem[i]).clamp_min(0)),
                mu_log_ms=float(mu[i]),
                std_log_ms=float(std[i]),
            ))
        return out

    # --------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "n_members": self.n_members, "hidden": self.hidden,
            "states": [m.state_dict() for m in self.members],
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> PerformanceModel:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(n_members=ckpt["n_members"], hidden=ckpt["hidden"], device=device)
        for m, state in zip(model.members, ckpt["states"]):
            m.load_state_dict(state)
        model.trained = True
        return model

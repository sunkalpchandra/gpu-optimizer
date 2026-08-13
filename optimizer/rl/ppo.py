"""PPO with GAE over transformation-episode transitions.

Transitions arrive asynchronously from the search loop (the RL searcher
records them as benchmark rewards come back); once a horizon's worth is
buffered, :meth:`PPOTrainer.update` runs clipped-objective minibatch epochs
with entropy regularization and approximate-KL early stopping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from optimizer.policy.policy_net import PolicyNet, obs_to_tensors

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    horizon: int = 128            # transitions per update


@dataclass
class Transition:
    obs: dict
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass
class RolloutBuffer:
    transitions: list[Transition] = field(default_factory=list)

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions.clear()

    def compute_gae(self, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        """Returns (advantages, returns).  Transitions are stored in episode
        order; ``done`` marks boundaries so bootstrap value is 0 there and the
        next transition's value elsewhere."""
        n = len(self.transitions)
        adv = np.zeros(n, dtype=np.float32)
        last = 0.0
        for i in reversed(range(n)):
            t = self.transitions[i]
            if t.done or i == n - 1:
                next_value = 0.0
                last = 0.0 if t.done else last
            else:
                next_value = self.transitions[i + 1].value
            delta = t.reward + gamma * next_value * (0.0 if t.done else 1.0) - t.value
            last = delta + gamma * lam * (0.0 if t.done else 1.0) * last
            adv[i] = last
        returns = adv + np.array([t.value for t in self.transitions], dtype=np.float32)
        return adv, returns


class PPOTrainer:
    def __init__(self, policy: PolicyNet, config: PPOConfig | None = None,
                 device: str = "cpu") -> None:
        self.policy = policy.to(device)
        self.config = config or PPOConfig()
        self.device = device
        self.opt = torch.optim.Adam(policy.parameters(),
                                    lr=self.config.learning_rate, eps=1e-5)
        self.buffer = RolloutBuffer()
        self.updates_done = 0
        self.stats: dict[str, float] = {}

    @property
    def ready(self) -> bool:
        return len(self.buffer) >= self.config.horizon

    def update(self) -> dict[str, float]:
        cfg = self.config
        trans = self.buffer.transitions
        adv, returns = self.buffer.compute_gae(cfg.gamma, cfg.gae_lambda)
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = obs_to_tensors([t.obs for t in trans], self.device)
        actions = torch.tensor([t.action for t in trans], device=self.device)
        old_logp = torch.tensor([t.log_prob for t in trans], device=self.device)
        adv_t = torch.tensor(adv, device=self.device)
        ret_t = torch.tensor(returns, device=self.device)

        n = len(trans)
        idx = np.arange(n)
        pol_losses, val_losses, kls = [], [], []
        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(idx)
            stop = False
            for s in range(0, n, cfg.minibatch_size):
                mb = idx[s:s + cfg.minibatch_size]
                mb_obs = {k: v[mb] for k, v in obs.items()}
                out = self.policy(mb_obs, action=actions[mb])
                logratio = out.log_prob - old_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = float(((ratio - 1) - logratio).mean())
                kls.append(approx_kl)

                pg1 = -adv_t[mb] * ratio
                pg2 = -adv_t[mb] * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((out.value - ret_t[mb]) ** 2).mean()
                loss = (pg_loss - cfg.entropy_coef * out.entropy.mean()
                        + cfg.value_coef * v_loss)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                               cfg.max_grad_norm)
                self.opt.step()
                pol_losses.append(float(pg_loss.detach()))
                val_losses.append(float(v_loss.detach()))
                if approx_kl > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break

        self.buffer.clear()
        self.updates_done += 1
        self.stats = {
            "policy_loss": float(np.mean(pol_losses)) if pol_losses else 0.0,
            "value_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "transitions": float(n),
        }
        logger.info("PPO update %d: %s", self.updates_done, self.stats)
        return self.stats

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(),
                    "config": self.config.__dict__,
                    "updates_done": self.updates_done}, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.updates_done = int(ckpt.get("updates_done", 0))

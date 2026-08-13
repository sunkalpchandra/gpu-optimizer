"""Online PPO searcher over kernel-transformation episodes.

Each episode starts from a concrete configuration and the policy edits it
step by step through the structured action catalog (inc/dec a parameter,
switch precision/strategy, STOP).  Rewards are measured latency improvements
(log₂ ratio), so the return of an episode telescopes to the total speedup it
achieved.  Transitions stream into the PPO trainer as benchmark results come
back through the standard search loop; the policy improves *during* search.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from benchmarks.harness import BenchmarkResult
from compiler.transformations.actions import (
    STOP_ACTION_INDEX,
    apply_action,
    valid_action_mask,
)
from compiler.transformations.space import Candidate, Config, encode_config
from optimizer.performance_model.features import graph_for
from optimizer.policy.policy_net import PolicyNet, obs_to_tensors
from optimizer.rl.ppo import PPOConfig, PPOTrainer, Transition
from optimizer.search.base import SearchContext, Searcher
from optimizer.search.factory import register_searcher

logger = logging.getLogger(__name__)

STEP_PENALTY = 0.02
FAIL_REWARD = -0.5
REWARD_CLIP = 4.0


@dataclass
class _Episode:
    config: Config
    current_ms: float | None = None      # measured latency of `config`
    best_ms: float = float("inf")
    steps: int = 0
    pending_id: str | None = None        # candidate awaiting benchmark
    pending_is_start: bool = False
    pending_obs: dict | None = None
    pending_action: int = 0
    pending_logp: float = 0.0
    pending_value: float = 0.0
    transitions: list[Transition] = field(default_factory=list)


class RLSearcher(Searcher):
    name = "rl"

    def __init__(self, ctx: SearchContext, ppo: PPOConfig | None = None,
                 max_episode_steps: int = 12, trainer: PPOTrainer | None = None,
                 deterministic: bool = False) -> None:
        super().__init__(ctx)
        self.max_episode_steps = max_episode_steps
        self.deterministic = deterministic
        self.trainer = trainer or PPOTrainer(PolicyNet(), ppo or PPOConfig())
        self.policy = self.trainer.policy
        # static observation parts
        graph = graph_for(ctx.task.name, ctx.shape, "float32")
        self._node_feats = graph.node_features()
        self._adj = graph.adjacency()
        self._hw = ctx.hardware.feature_vector()
        # caches / episode state
        self._cache: dict[str, tuple[bool, float]] = {}   # id → (ok, latency_ms)
        self._ref_ms: float | None = None                 # reward reference scale
        self._best: tuple[Config, float] | None = None
        self._episodes: list[_Episode] = []

    # --------------------------------------------------------------- obs
    def _obs(self, ep: _Episode) -> dict:
        cur = ep.current_ms or self._ref_ms or 1.0
        ref = self._ref_ms or cur
        best = ep.best_ms if math.isfinite(ep.best_ms) else cur
        scalars = np.array([
            (self.max_episode_steps - ep.steps) / self.max_episode_steps,
            float(np.clip(math.log2(cur / ref), -4, 4)) / 4.0,
            float(np.clip(math.log2(best / ref), -4, 4)) / 4.0,
        ], dtype=np.float32)
        return {
            "node_feats": self._node_feats, "adj": self._adj, "hw": self._hw,
            "cand": encode_config(ep.config).astype(np.float32),
            "scalars": scalars,
            "mask": valid_action_mask(self.ctx.space, ep.config).astype(np.float32),
        }

    # ------------------------------------------------------------ propose
    def propose(self, n: int) -> list[Candidate]:
        out: list[Candidate] = []
        guard = 0
        while len(out) < n and guard < 50 * n:
            guard += 1
            ep = self._idle_episode()
            cand = self._advance(ep)
            if cand is not None:
                out.append(cand)
        return out

    def _idle_episode(self) -> _Episode:
        for ep in self._episodes:
            if ep.pending_id is None:
                return ep
        ep = _Episode(config=self._start_config())
        self._episodes.append(ep)
        return ep

    def _start_config(self) -> Config:
        rng = self.ctx.rng
        if self._best is None or rng.random() < 0.4:
            if not self._episodes and self._best is None:
                return self.ctx.task.default_config(self.ctx.shape)
            return self.ctx.space.sample(rng)
        return dict(self._best[0])

    def _advance(self, ep: _Episode) -> Candidate | None:
        """Step the episode with the policy until it needs a measurement."""
        # ensure the start config is measured first
        if ep.current_ms is None:
            cand = self.ctx.candidate(ep.config, "rl")
            cached = self._cache.get(cand.candidate_id)
            if cached is None:
                ep.pending_id = cand.candidate_id
                ep.pending_is_start = True
                return cand
            ok, ms = cached
            if not ok:
                self._reset(ep)
                return self._advance(ep)
            ep.current_ms = ms
            ep.best_ms = min(ep.best_ms, ms)

        while True:
            obs = self._obs(ep)
            with torch.no_grad():
                out = self.policy(obs_to_tensors([obs]),
                                  deterministic=self.deterministic)
            action = int(out.action.item())
            logp, value = float(out.log_prob.item()), float(out.value.item())

            if action == STOP_ACTION_INDEX:
                ep.transitions.append(Transition(obs, action, logp, value,
                                                 reward=0.0, done=True))
                self._finish(ep)
                return None  # propose()'s loop will start a fresh episode

            new_cfg = apply_action(self.ctx.space, ep.config, action)
            cand = self.ctx.candidate(new_cfg, "rl",
                                      parent=self.ctx.candidate(ep.config, "rl"))
            cached = self._cache.get(cand.candidate_id)
            if cached is None:
                ep.pending_id = cand.candidate_id
                ep.pending_is_start = False
                ep.pending_obs = obs
                ep.pending_action = action
                ep.pending_logp = logp
                ep.pending_value = value
                return cand
            # cached: apply immediately and keep stepping
            self._apply_step(ep, obs, action, logp, value, *cached, new_cfg)
            if ep.steps >= self.max_episode_steps:
                self._finish(ep)
                return None

    # ------------------------------------------------------------ observe
    def observe(self, results: list[tuple[Candidate, BenchmarkResult, float]]) -> None:
        for cand, result, _reward in results:
            ok = result.ok
            ms = result.latency.median_ms if ok else float("inf")
            self._cache[cand.candidate_id] = (ok, ms)
            if ok and self._ref_ms is None:
                self._ref_ms = ms
            if ok and (self._best is None or ms < self._best[1]):
                self._best = (dict(cand.config), ms)

            ep = next((e for e in self._episodes
                       if e.pending_id == cand.candidate_id), None)
            if ep is None:
                continue
            ep.pending_id = None
            if ep.pending_is_start:
                if ok:
                    ep.current_ms = ms
                    ep.best_ms = min(ep.best_ms, ms)
                else:
                    self._reset(ep)
                continue
            self._apply_step(ep, ep.pending_obs, ep.pending_action,
                             ep.pending_logp, ep.pending_value, ok, ms,
                             self._pending_config(ep, cand))
            if ep.steps >= self.max_episode_steps:
                self._finish(ep)

        if self.trainer.ready and not self.deterministic:
            self.trainer.update()

    # ------------------------------------------------------------ helpers
    def _pending_config(self, ep: _Episode, cand: Candidate) -> Config:
        return dict(cand.config)

    def _apply_step(self, ep: _Episode, obs: dict, action: int, logp: float,
                    value: float, ok: bool, ms: float, new_cfg: Config) -> None:
        if ok:
            reward = math.log2(ep.current_ms / ms) - STEP_PENALTY
            ep.config = new_cfg
            ep.current_ms = ms
            ep.best_ms = min(ep.best_ms, ms)
        else:
            reward = FAIL_REWARD  # stay at current config
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
        ep.steps += 1
        done = ep.steps >= self.max_episode_steps
        ep.transitions.append(Transition(obs, action, logp, value, reward, done))

    def _finish(self, ep: _Episode) -> None:
        if ep.transitions:
            ep.transitions[-1].done = True
            for t in ep.transitions:
                self.trainer.buffer.add(t)
        self._remove(ep)

    def _reset(self, ep: _Episode) -> None:
        self._remove(ep)

    def _remove(self, ep: _Episode) -> None:
        if ep in self._episodes:
            self._episodes.remove(ep)

    @property
    def best_config(self) -> Config | None:
        return dict(self._best[0]) if self._best else None


@register_searcher("rl")
def _build(ctx: SearchContext, rl_params: dict | None = None,
           max_episode_steps: int = 12, **_ignored) -> RLSearcher:
    rl_params = dict(rl_params or {})
    rl_params.pop("algorithm", None)  # "ppo" is the only implementation
    known = {f for f in PPOConfig.__dataclass_fields__}
    unknown = set(rl_params) - known
    if unknown:
        raise ValueError(f"unknown rl params: {sorted(unknown)}")
    return RLSearcher(ctx, ppo=PPOConfig(**rl_params),
                      max_episode_steps=max_episode_steps)

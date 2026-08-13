"""The flagship hybrid optimizer: RL policy + evolutionary population +
learned surrogate + uncertainty-aware acquisition + real benchmarks.

Per proposal batch:

1. A reserved fraction advances the PPO searcher's transformation episodes
   (their pending candidates must reach hardware or the episodes stall — the
   surrogate never vetoes them).
2. The rest comes from a virtual pool (evolutionary offspring + elite
   neighborhoods + random immigrants) ranked by the surrogate's UCB
   acquisition ``−μ_log(t) + β·σ_log(t) + λ·log(p_compile)``, so most bad
   candidates die on paper instead of on the GPU.

Every real measurement feeds all three learners: the GA pool, the PPO
trainer (via its result cache/episodes), and the surrogate ensemble.
"""

from __future__ import annotations

import logging
import math

from benchmarks.harness import BenchmarkResult
from compiler.transformations.space import Candidate, Config
from optimizer.evolutionary.ga import EvolutionarySearcher, GAParams
from optimizer.performance_model.features import result_to_row
from optimizer.performance_model.model import PerformanceModel
from optimizer.rl.ppo import PPOConfig
from optimizer.rl.searcher import RLSearcher
from optimizer.search.base import SearchContext, Searcher
from optimizer.search.factory import register_searcher

logger = logging.getLogger(__name__)


class HybridSearcher(Searcher):
    name = "hybrid"

    def __init__(
        self,
        ctx: SearchContext,
        model: PerformanceModel | None = None,
        rl_params: dict | None = None,
        ga_params: GAParams | None = None,
        rl_frac: float = 0.3,
        pool_multiplier: int = 6,
        beta: float = 1.0,
        compile_weight: float = 0.5,
        warm_start: int = 16,
        retrain_every: int = 16,
        fit_epochs: int = 10,
        max_episode_steps: int = 10,
    ) -> None:
        super().__init__(ctx)
        self.model = model or PerformanceModel(seed=ctx.seed)
        rl_cfg = {k: v for k, v in (rl_params or {}).items() if k != "algorithm"}
        self.rl = RLSearcher(ctx, ppo=PPOConfig(**rl_cfg) if rl_cfg else None,
                             max_episode_steps=max_episode_steps)
        self.ga = EvolutionarySearcher(ctx, ga_params or GAParams(population_size=24))
        self.rl_frac = rl_frac
        self.pool_multiplier = pool_multiplier
        self.beta = beta
        self.compile_weight = compile_weight
        self.warm_start = warm_start
        self.retrain_every = retrain_every
        self.fit_epochs = fit_epochs
        self._seen: set[str] = set()
        self._observed = 0
        self._since_fit = 0
        self._pruned_on_paper = 0   # candidates rejected by the surrogate

    # -------------------------------------------------------------- propose
    def propose(self, n: int) -> list[Candidate]:
        out: list[Candidate] = []

        n_rl = max(1, int(round(n * self.rl_frac))) if n > 1 else 1
        for cand in self.rl.propose(n_rl):
            if cand.candidate_id not in self._seen:
                self._seen.add(cand.candidate_id)
                out.append(cand)

        n_rest = n - len(out)
        if n_rest <= 0:
            return out

        pool = self._gather_pool(n_rest * self.pool_multiplier)
        if not pool:
            return out
        if self.model.trained and self._observed >= self.warm_start:
            chosen = self._acquire(pool, n_rest)
        else:
            chosen = pool[:n_rest]
        for cand in chosen:
            self._seen.add(cand.candidate_id)
            out.append(cand)
        return out

    def _gather_pool(self, size: int) -> list[Candidate]:
        pool: dict[str, Candidate] = {}
        for cand in self.ga.propose(max(size // 2, 4)):
            if cand.candidate_id not in self._seen:
                pool[cand.candidate_id] = cand
        # elite neighborhoods (local exploitation)
        if self.ga.best is not None:
            for cfg in self.ctx.space.neighbors(self.ga.best[0].config):
                cand = self.ctx.candidate(cfg, "evolutionary", parent=self.ga.best[0])
                if cand.candidate_id not in self._seen:
                    pool.setdefault(cand.candidate_id, cand)
        # random immigrants (global exploration)
        while len(pool) < size:
            cand = self.ctx.candidate(self.ctx.space.sample(self.ctx.rng), "random")
            if cand.candidate_id in self._seen or cand.candidate_id in pool:
                continue
            pool[cand.candidate_id] = cand
        return list(pool.values())

    def _acquire(self, pool: list[Candidate], n: int) -> list[Candidate]:
        preds = self.model.predict(self.ctx.task.name, self.ctx.shape,
                                   [c.config for c in pool], self.ctx.hardware)
        scored = []
        for cand, p in zip(pool, preds):
            acq = (-p.mu_log_ms + self.beta * p.std_log_ms
                   + self.compile_weight * math.log(max(p.compile_prob, 1e-3)))
            cand.predicted_ms = p.mean_ms
            cand.predicted_std = p.std_ms
            scored.append((acq, cand))
        scored.sort(key=lambda t: t[0], reverse=True)
        self._pruned_on_paper += max(0, len(scored) - n)
        return [c for _a, c in scored[:n]]

    # -------------------------------------------------------------- observe
    def observe(self, results: list[tuple[Candidate, BenchmarkResult, float]]) -> None:
        self.ga.observe(results)
        self.rl.observe(results)   # advances episodes; caches foreign results
        self.model.add_rows([result_to_row(r) for _c, r, _rw in results])
        self._observed += len(results)
        self._since_fit += len(results)
        if self._observed >= self.warm_start and (
                not self.model.trained or self._since_fit >= self.retrain_every):
            self.model.fit(epochs=self.fit_epochs)
            self._since_fit = 0

    # ------------------------------------------------------------- reporting
    @property
    def stats(self) -> dict:
        return {
            "observed": self._observed,
            "pruned_on_paper": self._pruned_on_paper,
            "surrogate_rows": len(self.model.rows),
            "ppo_updates": self.rl.trainer.updates_done,
            "ga_pool": len(self.ga.pool),
        }

    @property
    def best_config(self) -> Config | None:
        return dict(self.ga.best[0].config) if self.ga.best else None


@register_searcher("hybrid")
def _build(ctx: SearchContext, rl_params: dict | None = None,
           population_size: int = 24, rl_frac: float = 0.3, beta: float = 1.0,
           warm_start: int = 16, retrain_every: int = 16,
           pool_multiplier: int = 6, max_episode_steps: int = 10,
           **_ignored) -> HybridSearcher:
    return HybridSearcher(
        ctx, rl_params=rl_params, ga_params=GAParams(population_size=population_size),
        rl_frac=rl_frac, beta=beta, warm_start=warm_start,
        retrain_every=retrain_every, pool_multiplier=pool_multiplier,
        max_episode_steps=max_episode_steps)

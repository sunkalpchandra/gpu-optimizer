"""Uncertainty-aware model-based search (Bayesian-optimization style).

A deep-ensemble surrogate ranks a large virtual candidate pool; only the
acquisition top-k reach the real benchmark.  Acquisition:

- ``ucb``:      −μ_log(t) + β·σ_log(t) + λ·log(p_compile)
- ``thompson``: one random ensemble member scores each proposal round

so the searcher deliberately spends part of its budget on configurations the
model is *uncertain* about rather than greedily trusting the network.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from benchmarks.harness import BenchmarkResult
from compiler.transformations.space import Candidate, Config
from optimizer.performance_model.features import result_to_row
from optimizer.performance_model.model import PerformanceModel
from optimizer.search.base import SearchContext, Searcher
from optimizer.search.factory import register_searcher

logger = logging.getLogger(__name__)


class BayesianSearcher(Searcher):
    name = "bayesian"

    def __init__(
        self,
        ctx: SearchContext,
        model: PerformanceModel | None = None,
        warm_start: int = 16,
        pool_size: int = 512,
        beta: float = 1.0,
        compile_weight: float = 0.5,
        acquisition: str = "ucb",
        retrain_every: int = 16,
        fit_epochs: int = 12,
        surrogate_checkpoint: str | None = None,
    ) -> None:
        super().__init__(ctx)
        if acquisition not in ("ucb", "thompson"):
            raise ValueError(f"unknown acquisition {acquisition!r}")
        self.surrogate_checkpoint = surrogate_checkpoint
        if model is None and surrogate_checkpoint and Path(surrogate_checkpoint).exists():
            model = PerformanceModel.load(surrogate_checkpoint)
            logger.info("bayesian: warm-started surrogate from %s (%d rows)",
                        surrogate_checkpoint, len(model.rows))
        self.model = model or PerformanceModel(seed=ctx.seed)
        self._pretrained = self.model.trained
        self.warm_start = warm_start
        self.pool_size = pool_size
        self.beta = beta
        self.compile_weight = compile_weight
        self.acquisition = acquisition
        self.retrain_every = retrain_every
        self.fit_epochs = fit_epochs
        self._seen: set[str] = set()
        self._observed = 0
        self._since_fit = 0
        self._elite: list[tuple[Candidate, float]] = []   # (candidate, reward)

    # -------------------------------------------------------------- propose
    def propose(self, n: int) -> list[Candidate]:
        cold = self._observed < self.warm_start and not self._pretrained
        if cold or not self.model.trained:
            return self._random_batch(n)

        pool = self._virtual_pool()
        if not pool:
            return self._random_batch(n)
        member = None
        if self.acquisition == "thompson":
            member = self.ctx.rng.randrange(len(self.model.members))
        preds = self.model.predict(self.ctx.task.name, self.ctx.shape,
                                   pool, self.ctx.hardware, member=member)
        scored = []
        for cfg, p in zip(pool, preds):
            acq = (-p.mu_log_ms + self.beta * p.std_log_ms
                   + self.compile_weight * math.log(max(p.compile_prob, 1e-3)))
            scored.append((acq, cfg, p))
        scored.sort(key=lambda t: t[0], reverse=True)

        out: list[Candidate] = []
        for _acq, cfg, p in scored[: n]:
            cand = self.ctx.candidate(cfg, "bo")
            cand.predicted_ms = p.mean_ms
            cand.predicted_std = p.std_ms
            self._seen.add(cand.candidate_id)
            out.append(cand)
        return out

    def observe(self, results: list[tuple[Candidate, BenchmarkResult, float]]) -> None:
        self.model.add_rows([result_to_row(r) for _c, r, _rw in results])
        for cand, result, reward in results:
            self._seen.add(cand.candidate_id)
            if result.ok:
                self._elite.append((cand, reward))
        self._elite.sort(key=lambda cr: cr[1], reverse=True)
        self._elite = self._elite[:10]
        self._observed += len(results)
        self._since_fit += len(results)
        if self._observed >= self.warm_start and (
                not self.model.trained or self._since_fit >= self.retrain_every):
            self.model.fit(epochs=self.fit_epochs)
            self._since_fit = 0

    def finalize(self) -> None:
        if self.surrogate_checkpoint:
            self.model.save(self.surrogate_checkpoint)
            logger.info("bayesian: saved surrogate to %s (%d rows)",
                        self.surrogate_checkpoint, len(self.model.rows))

    # -------------------------------------------------------------- helpers
    def _random_batch(self, n: int) -> list[Candidate]:
        out = []
        for _ in range(n * 4):
            if len(out) >= n:
                break
            cand = self.ctx.candidate(self.ctx.space.sample(self.ctx.rng), "bo")
            if cand.candidate_id not in self._seen:
                self._seen.add(cand.candidate_id)
                out.append(cand)
        return out

    def _virtual_pool(self) -> list[Config]:
        """Random samples + neighborhoods of the current elite, unseen only."""
        pool: dict[str, Config] = {}
        for _ in range(self.pool_size):
            cfg = self.ctx.space.sample(self.ctx.rng)
            cand = self.ctx.candidate(cfg, "bo")
            if cand.candidate_id not in self._seen:
                pool[cand.candidate_id] = cfg
        for elite_cand, _r in self._elite:
            for cfg in self.ctx.space.neighbors(elite_cand.config):
                cand = self.ctx.candidate(cfg, "bo")
                if cand.candidate_id not in self._seen:
                    pool[cand.candidate_id] = cfg
        return list(pool.values())


@register_searcher("bayesian")
def _build(ctx: SearchContext, warm_start: int = 16, pool_size: int = 512,
           beta: float = 1.0, acquisition: str = "ucb", retrain_every: int = 16,
           surrogate_checkpoint: str | None = None, **_ignored) -> BayesianSearcher:
    return BayesianSearcher(ctx, warm_start=warm_start, pool_size=pool_size,
                            beta=beta, acquisition=acquisition,
                            retrain_every=retrain_every,
                            surrogate_checkpoint=surrogate_checkpoint)

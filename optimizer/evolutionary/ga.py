"""Steady-state evolutionary search over structured kernel configurations.

Tournament selection over every evaluated candidate, uniform crossover,
step/reset mutation over the ordered choice lists, elite preservation (the
evaluated pool never forgets its best members), plus random immigrants for
diversity.  Fitness is the task reward, so correctness and compile failures
push genomes out of the breeding pool automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.harness import BenchmarkResult
from compiler.transformations.space import Candidate, Config
from optimizer.search.base import SearchContext, Searcher
from optimizer.search.factory import register_searcher


@dataclass
class GAParams:
    population_size: int = 32       # size of the initial random generation
    tournament_k: int = 3
    mutation_rate: float = 0.3      # per-parameter mutation probability
    step_mutation_prob: float = 0.7  # step to a neighbor vs random reset
    crossover_rate: float = 0.7     # else clone-and-mutate a single parent
    immigrant_frac: float = 0.15    # fresh random genomes per batch
    max_spawn_attempts: int = 12    # tries to avoid proposing duplicates


class EvolutionarySearcher(Searcher):
    name = "evolutionary"

    def __init__(self, ctx: SearchContext, params: GAParams | None = None) -> None:
        super().__init__(ctx)
        self.params = params or GAParams()
        self.pool: list[tuple[Candidate, float]] = []   # evaluated genomes
        self._seen: set[str] = set()
        self._bootstrapped = 0

    # -------------------------------------------------------------- propose
    def propose(self, n: int) -> list[Candidate]:
        out: list[Candidate] = []
        # generation 0: random bootstrap (seeded with the naive default too)
        while self._bootstrapped < self.params.population_size and len(out) < n:
            cfg = (self.ctx.task.default_config(self.ctx.shape)
                   if self._bootstrapped == 0
                   else self.ctx.space.sample(self.ctx.rng))
            cand = self.ctx.candidate(cfg, "evolutionary")
            if cand.candidate_id not in self._seen:
                self._seen.add(cand.candidate_id)
                out.append(cand)
            self._bootstrapped += 1
        while len(out) < n and self.pool:
            child = self._spawn()
            if child is not None:
                out.append(child)
            else:
                break  # cannot generate anything new
        return out

    def observe(self, results: list[tuple[Candidate, BenchmarkResult, float]]) -> None:
        for cand, _result, reward in results:
            self.pool.append((cand, reward))

    # ------------------------------------------------------------ genetics
    def _spawn(self) -> Candidate | None:
        rng = self.ctx.rng
        p = self.params
        for _ in range(p.max_spawn_attempts):
            if rng.random() < p.immigrant_frac:
                cfg: Config = self.ctx.space.sample(rng)
                parent = None
            elif rng.random() < p.crossover_rate and len(self.pool) >= 2:
                a = self._tournament()
                b = self._tournament()
                cfg = self._crossover(a.config, b.config)
                cfg = self._mutate(cfg)
                parent = a
            else:
                a = self._tournament()
                cfg = self._mutate(dict(a.config))
                parent = a
            cand = self.ctx.candidate(cfg, "evolutionary", parent=parent)
            if cand.candidate_id not in self._seen:
                self._seen.add(cand.candidate_id)
                return cand
        return None

    def _tournament(self) -> Candidate:
        rng = self.ctx.rng
        k = min(self.params.tournament_k, len(self.pool))
        contenders = rng.sample(self.pool, k)
        return max(contenders, key=lambda cf: cf[1])[0]

    def _crossover(self, a: Config, b: Config) -> Config:
        return {k: (a[k] if self.ctx.rng.random() < 0.5 else b[k]) for k in a}

    def _mutate(self, cfg: Config) -> Config:
        rng = self.ctx.rng
        p = self.params
        out = dict(cfg)
        mutated = False
        for spec in self.ctx.space.discrete_specs:
            if rng.random() >= p.mutation_rate:
                continue
            choices = spec.choices  # type: ignore[assignment]
            if rng.random() < p.step_mutation_prob:
                idx = spec.index_of(out[spec.name])
                step = rng.choice((-1, 1))
                new_idx = min(max(idx + step, 0), len(choices) - 1)
                out[spec.name] = choices[new_idx]
            else:
                out[spec.name] = rng.choice(choices)
            mutated = mutated or out[spec.name] != cfg[spec.name]
        if not mutated:  # guarantee at least one gene changes
            spec = rng.choice(self.ctx.space.discrete_specs)
            choices = spec.choices  # type: ignore[assignment]
            alternatives = [c for c in choices if c != out[spec.name]]
            if alternatives:
                out[spec.name] = rng.choice(alternatives)
        return out

    # ------------------------------------------------------------- summary
    @property
    def best(self) -> tuple[Candidate, float] | None:
        return max(self.pool, key=lambda cf: cf[1]) if self.pool else None


@register_searcher("evolutionary")
def _build(ctx: SearchContext, population_size: int = 32, tournament_k: int = 3,
           mutation_rate: float = 0.3, crossover_rate: float = 0.7,
           immigrant_frac: float = 0.15, **_ignored) -> EvolutionarySearcher:
    return EvolutionarySearcher(ctx, GAParams(
        population_size=population_size, tournament_k=tournament_k,
        mutation_rate=mutation_rate, crossover_rate=crossover_rate,
        immigrant_frac=immigrant_frac))

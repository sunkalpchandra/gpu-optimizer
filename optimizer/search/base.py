"""Searcher interface shared by every optimization strategy.

A searcher proposes batches of candidates and observes their measured
results; the :class:`~optimizer.search.loop.SearchLoop` owns benchmarking,
reward computation, deduplication, and persistence.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from benchmarks.base import Task
from benchmarks.harness import BenchmarkResult
from compiler.transformations.space import Candidate, ParamSpace
from hardware.gpu_info import HardwareSpec


@dataclass
class SearchContext:
    """Everything a searcher may condition on."""

    task: Task
    shape: tuple[int, ...]
    hardware: HardwareSpec
    seed: int = 0
    space: ParamSpace = field(init=False)
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.shape = tuple(int(s) for s in self.shape)
        self.space = self.task.param_space(self.shape)
        self.rng = random.Random(self.seed)

    def candidate(self, config: dict, provenance: str,
                  parent: Candidate | None = None) -> Candidate:
        return Candidate(task=self.task.name, shape=self.shape, config=dict(config),
                         parent_id=parent.candidate_id if parent else None,
                         provenance=provenance)


class Searcher(ABC):
    """Propose/observe interface implemented by every strategy."""

    name: str = "abstract"

    def __init__(self, ctx: SearchContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def propose(self, n: int) -> list[Candidate]:
        """Return up to ``n`` candidates to benchmark next.  An empty list
        signals the searcher has exhausted its space."""

    def observe(self, results: list[tuple[Candidate, BenchmarkResult, float]]) -> None:
        """Feed back (candidate, result, reward) tuples.  Optional."""

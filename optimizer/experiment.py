"""Reproducible experiment configuration and runner.

Experiments are fully described by a YAML file (see ``configs/``): task,
shapes, algorithm + hyperparameters, benchmark settings, reward weights,
engine selection, and seed.  ``run_experiment`` executes one search per shape
and returns the outcomes; identical configs produce identical simulated runs
and statistically equivalent hardware runs.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from benchmarks import get_task
from benchmarks.db import BenchmarkDB, default_db_path
from benchmarks.harness import (
    BenchmarkEngine,
    BenchmarkSettings,
    SimulatedBenchmarkEngine,
    make_engine,
)
from hardware.gpu_info import (
    DEFAULT_SIMULATED_GPU,
    detect_hardware,
    simulated_hardware,
)
from optimizer.rewards.reward import RewardConfig
from optimizer.search.base import SearchContext
from optimizer.search.factory import make_searcher
from optimizer.search.loop import ProgressCallback, SearchLoop, SearchOutcome

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    task: str = "matmul"
    shapes: list[tuple[int, ...]] = field(default_factory=lambda: [(1024, 1024, 1024)])
    algorithm: str = "random"
    seed: int = 0
    engine: str = "auto"                 # auto | cuda | simulated
    gpu: str = DEFAULT_SIMULATED_GPU     # catalog key when simulated
    max_evaluations: int = 200
    batch_size: int = 8
    benchmark: dict[str, Any] = field(default_factory=dict)   # warmup, iterations
    search: dict[str, Any] = field(default_factory=dict)      # algorithm hyperparams
    rl: dict[str, Any] = field(default_factory=dict)          # PPO hyperparams
    reward: dict[str, Any] = field(default_factory=dict)      # RewardConfig fields
    db_path: str | None = None           # None → repo-default; "" → no persistence
    name: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        cfg = cls(**raw)
        cfg.shapes = [tuple(int(x) for x in s) for s in cfg.shapes]
        if not cfg.name:
            cfg.name = Path(path).stem
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        d = {**self.__dict__, "shapes": [list(s) for s in self.shapes]}
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_engine(cfg: ExperimentConfig) -> BenchmarkEngine:
    settings = BenchmarkSettings(
        warmup=int(cfg.benchmark.get("warmup", 20)),
        iterations=int(cfg.benchmark.get("iterations", 100)),
        seed=cfg.seed,
    )
    if cfg.engine == "cuda":
        return make_engine(detect_hardware(), settings)
    if cfg.engine == "simulated":
        return SimulatedBenchmarkEngine(simulated_hardware(cfg.gpu), settings)
    # auto
    if torch.cuda.is_available():
        return make_engine(detect_hardware(), settings)
    return SimulatedBenchmarkEngine(simulated_hardware(cfg.gpu), settings)


def run_experiment(cfg: ExperimentConfig,
                   callback: ProgressCallback | None = None,
                   db: BenchmarkDB | None = None) -> list[SearchOutcome]:
    """Execute the experiment: one search per shape."""
    set_all_seeds(cfg.seed)
    engine = build_engine(cfg)
    if db is None and cfg.db_path != "":
        db = BenchmarkDB(cfg.db_path or default_db_path())

    task = get_task(cfg.task)
    outcomes: list[SearchOutcome] = []
    for shape in cfg.shapes:
        ctx = SearchContext(task=task, shape=tuple(shape),
                            hardware=engine.hardware, seed=cfg.seed)
        searcher = make_searcher(cfg.algorithm, ctx, engine=engine,
                                 rl_params=cfg.rl, **cfg.search)
        loop = SearchLoop(
            ctx, searcher, engine, db=db,
            reward_config=RewardConfig(**cfg.reward),
            max_evaluations=cfg.max_evaluations,
            batch_size=cfg.batch_size,
            callback=callback,
        )
        outcome = loop.run()
        logger.info("\n%s", outcome.summary())
        outcomes.append(outcome)
    return outcomes

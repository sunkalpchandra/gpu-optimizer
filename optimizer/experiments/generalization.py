"""Generalization experiments: does what the system learns transfer?

Four studies, each producing honest metrics into a JSON + Markdown report:

- **Shape interpolation** — surrogate trained on some matmul sizes, evaluated
  on *unseen in-between* sizes (rank correlation + top-1 regret).
- **Shape extrapolation** — trained on small problems, evaluated on larger.
- **Workload transfer** — trained on one task family, evaluated on unseen
  related workloads.
- **Hardware transfer** — trained/searched on GPU A, evaluated on GPU B
  (simulated engines make this runnable anywhere; on multi-GPU hosts the
  same code runs against real engines).

Plus a **search-efficiency** comparison of every algorithm under an equal
benchmark budget.  All results carry the engine label; simulated numbers are
reported as simulated.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks import get_task
from benchmarks.harness import BenchmarkEngine, BenchmarkSettings, SimulatedBenchmarkEngine
from hardware.gpu_info import simulated_hardware
from optimizer.experiment import set_all_seeds
from optimizer.performance_model.features import TrainingRow, result_to_row
from optimizer.performance_model.model import PerformanceModel
from optimizer.search.base import SearchContext
from optimizer.search.factory import make_searcher
from optimizer.search.loop import SearchLoop

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ helpers

def _sample_rows(engine: BenchmarkEngine, task_name: str, shape: tuple[int, ...],
                 n: int, seed: int) -> list[TrainingRow]:
    task = get_task(task_name)
    space = task.param_space(shape)
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        cand = task.make_candidate(shape, space.sample(rng), provenance="random")
        rows.append(result_to_row(engine.benchmark_candidate(task, cand)))
    return rows


def _rank_metrics(model: PerformanceModel, engine: BenchmarkEngine,
                  task_name: str, shape: tuple[int, ...], n_eval: int,
                  seed: int) -> dict:
    """Spearman rank correlation + top-1 regret on fresh random configs."""
    rows = [r for r in _sample_rows(engine, task_name, shape, n_eval, seed)
            if r.latency_ms is not None]
    if len(rows) < 5:
        return {"n": len(rows), "spearman": None, "top1_regret_pct": None}
    preds = model.predict(task_name, shape, [r.config for r in rows],
                          engine.hardware)
    actual = [r.latency_ms for r in rows]
    predicted = [p.mean_ms for p in preds]

    def ranks(xs: list[float]) -> list[int]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rk = [0] * len(xs)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk

    ra, rp = ranks(actual), ranks(predicted)
    n = len(ra)
    rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(ra, rp)) / (n * (n * n - 1))
    picked = actual[min(range(n), key=lambda i: predicted[i])]
    best = min(actual)
    return {"n": n, "spearman": round(rho, 4),
            "top1_regret_pct": round(100 * (picked - best) / best, 2)}


@dataclass
class GeneralizationReport:
    engine_label: str
    gpu: str
    seed: int
    started_at: float = field(default_factory=time.time)
    interpolation: dict = field(default_factory=dict)
    extrapolation: dict = field(default_factory=dict)
    workload_transfer: dict = field(default_factory=dict)
    hardware_transfer: dict = field(default_factory=dict)
    search_efficiency: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)

    def to_markdown(self) -> str:
        sim_note = ("> **All numbers below are from the deterministic simulated "
                    "engine (no CUDA GPU present) — development results, not "
                    "hardware measurements.**\n\n"
                    if self.engine_label == "simulated" else "")
        out = [f"# Generalization report\n\n{sim_note}"
               f"Engine: `{self.engine_label}` · GPU: {self.gpu} · seed {self.seed}\n"]

        def table(title: str, rows: dict, cols: list[str]) -> None:
            if not rows:
                return
            out.append(f"\n## {title}\n")
            out.append("| case | " + " | ".join(cols) + " |")
            out.append("|---" * (len(cols) + 1) + "|")
            for case, m in rows.items():
                out.append("| " + case + " | "
                           + " | ".join(str(m.get(c, "—")) for c in cols) + " |")

        table("Shape interpolation (surrogate on unseen mid-range sizes)",
              self.interpolation, ["n", "spearman", "top1_regret_pct"])
        table("Shape extrapolation (small → large)",
              self.extrapolation, ["n", "spearman", "top1_regret_pct"])
        table("Workload transfer (train family → unseen workloads)",
              self.workload_transfer, ["n", "spearman", "top1_regret_pct"])
        table("Hardware transfer",
              self.hardware_transfer,
              ["transferred_ms", "native_ms", "penalty_pct", "top_k_needed",
               "spearman"])
        table("Search efficiency (equal budget)",
              self.search_efficiency,
              ["best_ms", "speedup_vs_torch", "evals_to_best", "compile_rate"])
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------- studies

def study_shape_generalization(engine: BenchmarkEngine, *, train_shapes,
                               test_shapes, task_name: str = "matmul",
                               rows_per_shape: int = 60, n_eval: int = 40,
                               seed: int = 0, epochs: int = 25) -> dict:
    model = PerformanceModel(seed=seed)
    for i, shape in enumerate(train_shapes):
        model.add_rows(_sample_rows(engine, task_name, tuple(shape),
                                    rows_per_shape, seed + i))
    model.fit(epochs=epochs)
    return {
        f"{task_name}{tuple(s)}": _rank_metrics(model, engine, task_name,
                                                tuple(s), n_eval, seed + 100 + i)
        for i, s in enumerate(test_shapes)
    }


def study_workload_transfer(engine: BenchmarkEngine, *, seed: int = 0,
                            rows_per_case: int = 60, n_eval: int = 40,
                            epochs: int = 25) -> dict:
    train_cases = [("matmul", (1024, 1024, 1024)), ("vecadd", (1 << 24,)),
                   ("reduction", (1 << 24,)), ("fused_elementwise", (1 << 24,))]
    test_cases = [("softmax", (4096, 1024)), ("layernorm", (4096, 1024)),
                  ("attention", (16, 1024, 64))]
    model = PerformanceModel(seed=seed)
    for i, (t, s) in enumerate(train_cases):
        model.add_rows(_sample_rows(engine, t, s, rows_per_case, seed + i))
    model.fit(epochs=epochs)
    return {f"{t}{s}": _rank_metrics(model, engine, t, s, n_eval, seed + 200 + i)
            for i, (t, s) in enumerate(test_cases)}


def study_hardware_transfer(*, gpu_a: str = "A100-SXM4-40GB",
                            gpu_b: str = "RTX-4090", task_name: str = "matmul",
                            shape=(2048, 2048, 2048), budget: int = 80,
                            rows: int = 160, n_eval: int = 40, top_k: int = 5,
                            seed: int = 0, settings: BenchmarkSettings | None = None
                            ) -> dict:
    """Simulated cross-GPU study: search/train on A, evaluate on B.

    Transfer protocol is top-K: GPU A's best configs are tried on B in rank
    order until one compiles there — a single top-1 config can legitimately
    fail on a GPU with less shared memory, and that hazard is part of what
    this study measures (K needed is reported).
    """
    settings = settings or BenchmarkSettings(warmup=2, iterations=20)
    eng_a = SimulatedBenchmarkEngine(simulated_hardware(gpu_a), settings,
                                     check_correctness_emulated=False)
    eng_b = SimulatedBenchmarkEngine(simulated_hardware(gpu_b), settings,
                                     check_correctness_emulated=False)
    task = get_task(task_name)
    shape = tuple(shape)

    def search(engine) -> SearchLoop:
        ctx = SearchContext(task=task, shape=shape, hardware=engine.hardware,
                            seed=seed)
        s = make_searcher("evolutionary", ctx, population_size=20)
        return SearchLoop(ctx, s, engine, db=None, max_evaluations=budget,
                          batch_size=8).run()

    out_a, out_b = search(eng_a), search(eng_b)
    native_b = out_b.best_latency_ms
    ranked_a = sorted((h for h in out_a.history if h["latency_ms"] is not None),
                      key=lambda h: h["latency_ms"])
    trans_ms, k_used = float("inf"), None
    for k, h in enumerate(ranked_a[:top_k], start=1):
        r = eng_b.benchmark_candidate(
            task, task.make_candidate(shape, h["config"], provenance="transfer"))
        if r.ok:
            trans_ms, k_used = r.latency.median_ms, k
            break
    transferred_ok = k_used is not None

    # surrogate transfer: trained only on A-data, ranked on B
    model = PerformanceModel(seed=seed)
    model.add_rows(_sample_rows(eng_a, task_name, shape, rows, seed))
    model.fit(epochs=30)
    rk = _rank_metrics(model, eng_b, task_name, shape, n_eval, seed + 300)

    key = f"{task_name}{shape}: {gpu_a}→{gpu_b}"
    penalty = (100 * (trans_ms - native_b) / native_b if transferred_ok else None)
    return {key: {
        "transferred_ms": round(trans_ms, 4) if transferred_ok else "all top-k failed",
        "native_ms": round(native_b, 4),
        "penalty_pct": round(penalty, 2) if penalty is not None else "—",
        "top_k_needed": k_used,
        "spearman": rk["spearman"],
    }}


def study_search_efficiency(engine: BenchmarkEngine, *, task_name: str = "matmul",
                            shape=(2048, 2048, 2048), budget: int = 80,
                            seed: int = 0,
                            algorithms=("random", "evolutionary", "bayesian",
                                        "rl", "hybrid")) -> dict:
    task = get_task(task_name)
    shape = tuple(shape)
    out = {}
    for algo in algorithms:
        set_all_seeds(seed)
        ctx = SearchContext(task=task, shape=shape, hardware=engine.hardware,
                            seed=seed)
        s = make_searcher(algo, ctx, population_size=20, warm_start=16)
        o = SearchLoop(ctx, s, engine, db=None, max_evaluations=budget,
                       batch_size=8).run()
        evals_to_best = next((h["iteration"] for h in o.history
                              if h["best_so_far_ms"] == o.best_latency_ms), None)
        out[algo] = {
            "best_ms": round(o.best_latency_ms, 4),
            "speedup_vs_torch": round(o.speedup, 3),
            "evals_to_best": evals_to_best,
            "compile_rate": round(o.compile_success_rate, 3),
        }
    return out


# ------------------------------------------------------------------- runner

def run_all(out_dir: str | Path = "reports", seed: int = 0, fast: bool = False,
            engine: BenchmarkEngine | None = None) -> GeneralizationReport:
    """Run every study and write reports/<ts>-generalization.{json,md}."""
    set_all_seeds(seed)
    if engine is None:
        settings = BenchmarkSettings(warmup=2, iterations=20)
        engine = SimulatedBenchmarkEngine(simulated_hardware(), settings,
                                          check_correctness_emulated=False)
    # NOTE: the surrogate needs ~80 rows/shape and ~25 epochs before its
    # cross-shape ranking is trustworthy (below that it can even invert);
    # "fast" stays above that threshold, it only trims evaluation counts.
    rows = 60 if fast else 90
    n_eval = 20 if fast else 40
    budget = 40 if fast else 80
    epochs = 20 if fast else 30

    report = GeneralizationReport(engine_label=engine.label,
                                  gpu=engine.hardware.name, seed=seed)
    logger.info("study: shape interpolation")
    report.interpolation = study_shape_generalization(
        engine, train_shapes=[(512, 512, 512), (1024, 1024, 1024),
                              (2048, 2048, 2048)],
        test_shapes=[(768, 768, 768), (1536, 1536, 1536)],
        rows_per_shape=rows, n_eval=n_eval, seed=seed, epochs=epochs)
    logger.info("study: shape extrapolation")
    report.extrapolation = study_shape_generalization(
        engine, train_shapes=[(256, 256, 256), (512, 512, 512),
                              (1024, 1024, 1024)],
        test_shapes=[(3072, 3072, 3072), (4096, 4096, 4096)],
        rows_per_shape=rows, n_eval=n_eval, seed=seed + 1, epochs=epochs)
    logger.info("study: workload transfer")
    report.workload_transfer = study_workload_transfer(
        engine, seed=seed + 2, rows_per_case=rows, n_eval=n_eval, epochs=epochs)
    logger.info("study: hardware transfer")
    report.hardware_transfer = study_hardware_transfer(
        seed=seed + 3, budget=budget, rows=rows * 2, n_eval=n_eval)
    logger.info("study: search efficiency")
    report.search_efficiency = study_search_efficiency(
        engine, budget=budget, seed=seed + 4)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}-generalization.json").write_text(report.to_json())
    (out / f"{stamp}-generalization.md").write_text(report.to_markdown())
    logger.info("wrote reports to %s", out)
    return report

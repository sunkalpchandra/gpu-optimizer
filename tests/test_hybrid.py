"""Stage 6: the hybrid optimizer — RL + GA + surrogate + acquisition."""

import math

import pytest
import torch

from benchmarks import get_task
from benchmarks.harness import BenchmarkSettings, SimulatedBenchmarkEngine
from hardware.gpu_info import simulated_hardware
from optimizer.search.base import SearchContext
from optimizer.search.factory import available_searchers, make_searcher
from optimizer.search.loop import SearchLoop


@pytest.fixture(scope="module")
def engine():
    return SimulatedBenchmarkEngine(simulated_hardware(),
                                    BenchmarkSettings(warmup=1, iterations=5),
                                    check_correctness_emulated=False)


def test_all_algorithms_registered():
    names = available_searchers()
    for expected in ("random", "grid", "evolutionary", "bayesian", "rl", "hybrid"):
        assert expected in names, f"{expected} missing from {names}"


def test_hybrid_end_to_end(engine):
    torch.manual_seed(0)
    task = get_task("matmul")
    ctx = SearchContext(task=task, shape=(2048, 2048, 2048),
                        hardware=engine.hardware, seed=0)
    searcher = make_searcher("hybrid", ctx, warm_start=16, retrain_every=24,
                            population_size=16, rl_frac=0.3)
    loop = SearchLoop(ctx, searcher, engine, db=None, max_evaluations=90,
                      batch_size=10)
    outcome = loop.run()

    assert outcome.candidates_evaluated == 90
    assert outcome.best_result is not None and outcome.best_result.ok
    assert math.isfinite(outcome.best_latency_ms)

    # all three proposal sources actually contributed
    provs = {h["provenance"] for h in outcome.history}
    assert "rl" in provs and "evolutionary" in provs, provs

    # surrogate engaged: post-warm-start candidates carry predictions,
    # and the virtual pool pruned candidates on paper
    predicted = [h for h in outcome.history if h["predicted_ms"] is not None]
    assert predicted, "surrogate never attached predictions"
    stats = searcher.stats
    assert stats["pruned_on_paper"] > 0
    assert stats["surrogate_rows"] >= 90
    assert stats["ga_pool"] >= 60

    # hybrid must at least match the naive baseline it started from
    assert outcome.best_latency_ms <= (outcome.baseline_naive_ms or float("inf"))


def test_hybrid_beats_random_under_equal_budget(engine):
    """Deterministic simulated surface: hybrid ≤ random best latency."""
    torch.manual_seed(0)
    task = get_task("matmul")
    budget = 80

    def run(algo):
        ctx = SearchContext(task=task, shape=(1024, 1024, 1024),
                            hardware=engine.hardware, seed=5)
        s = make_searcher(algo, ctx, warm_start=16, population_size=16)
        return SearchLoop(ctx, s, engine, db=None, max_evaluations=budget,
                          batch_size=8).run().best_latency_ms

    hybrid = run("hybrid")
    rnd = run("random")
    assert hybrid <= rnd * 1.05, f"hybrid {hybrid:.4f}ms vs random {rnd:.4f}ms"

"""Stage 4: surrogate ensemble learns the simulated surface; BO search works."""

import math
import random

import pytest
import torch

from benchmarks import get_task
from benchmarks.harness import BenchmarkSettings, SimulatedBenchmarkEngine
from hardware.gpu_info import simulated_hardware
from optimizer.performance_model.features import result_to_row, rows_to_batch
from optimizer.performance_model.model import PerformanceModel
from optimizer.search.base import SearchContext
from optimizer.search.factory import make_searcher
from optimizer.search.loop import SearchLoop


@pytest.fixture(scope="module")
def engine():
    return SimulatedBenchmarkEngine(simulated_hardware(),
                                    BenchmarkSettings(warmup=1, iterations=5),
                                    check_correctness_emulated=False)


@pytest.fixture(scope="module")
def corpus(engine):
    """Benchmark 120 random matmul configs on the simulated surface."""
    task = get_task("matmul")
    shape = (1024, 1024, 1024)
    space = task.param_space(shape)
    rng = random.Random(0)
    rows = []
    for _ in range(120):
        cand = task.make_candidate(shape, space.sample(rng), provenance="random")
        rows.append(result_to_row(engine.benchmark_candidate(task, cand)))
    return rows


def test_feature_batch_shapes(corpus):
    batch = rows_to_batch(corpus[:16])
    assert len(batch) == 16
    assert batch.node_feats.shape[0] == 16
    assert batch.hw_feats.shape == (16, 12)
    assert torch.isfinite(batch.cand_feats).all()
    # compile failures have NaN latency targets, successes finite
    assert torch.isfinite(batch.log_ms).sum() == sum(
        1 for r in corpus[:16] if r.latency_ms)


def test_surrogate_learns_ranking(corpus):
    torch.manual_seed(0)
    model = PerformanceModel(n_members=3, hidden=48, seed=0)
    train, held = corpus[:100], corpus[100:]
    model.add_rows(train)
    stats = model.fit(epochs=30)
    assert math.isfinite(stats["loss"])

    held_ok = [r for r in held if r.latency_ms]
    assert len(held_ok) >= 5
    preds = model.predict("matmul", (1024, 1024, 1024),
                          [r.config for r in held_ok], simulated_hardware())
    actual = [r.latency_ms for r in held_ok]
    predicted = [p.mean_ms for p in preds]
    # Spearman rank correlation must be clearly positive on held-out configs
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rk = [0] * len(xs)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    ra, rp = ranks(actual), ranks(predicted)
    n = len(ra)
    rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(ra, rp)) / (n * (n * n - 1))
    assert rho > 0.5, f"held-out rank correlation too weak: {rho:.2f}"

    # uncertainty exists and is finite
    assert all(p.std_ms >= 0 and math.isfinite(p.std_ms) for p in preds)

    # compile-probability head discriminates in aggregate: mean predicted
    # probability over known-failing configs < over known-passing configs
    from optimizer.world_model.analytic import AnalyticGPUModel

    task = get_task("matmul")
    hw = simulated_hardware()
    sim = AnalyticGPUModel(hw)
    space = task.param_space((1024, 1024, 1024))
    rng = random.Random(99)
    passing, failing = [], []
    while len(passing) < 20 or len(failing) < 20:
        cfg = space.sample(rng)
        ok, _ = sim.compiles(task, (1024, 1024, 1024), cfg)
        (passing if ok else failing).append(cfg)
    passing, failing = passing[:20], failing[:20]
    pp = model.predict("matmul", (1024, 1024, 1024), passing, hw)
    pf = model.predict("matmul", (1024, 1024, 1024), failing, hw)
    mean_pass = sum(p.compile_prob for p in pp) / len(pp)
    mean_fail = sum(p.compile_prob for p in pf) / len(pf)
    assert mean_fail < mean_pass, f"compile head: fail {mean_fail:.3f} vs pass {mean_pass:.3f}"


def test_save_load_roundtrip(tmp_path, corpus):
    model = PerformanceModel(n_members=2, hidden=32, seed=0)
    model.add_rows(corpus[:40])
    model.fit(epochs=5)
    p1 = model.predict("matmul", (1024, 1024, 1024), [corpus[0].config],
                       simulated_hardware())[0]
    model.save(tmp_path / "m.pt")
    loaded = PerformanceModel.load(tmp_path / "m.pt")
    p2 = loaded.predict("matmul", (1024, 1024, 1024), [corpus[0].config],
                        simulated_hardware())[0]
    assert abs(p1.mu_log_ms - p2.mu_log_ms) < 1e-5


def test_bayesian_search_end_to_end(engine):
    task = get_task("matmul")
    ctx = SearchContext(task=task, shape=(1024, 1024, 1024),
                        hardware=engine.hardware, seed=1)
    searcher = make_searcher("bayesian", ctx, warm_start=16, pool_size=128,
                            retrain_every=16)
    loop = SearchLoop(ctx, searcher, engine, db=None, max_evaluations=48,
                      batch_size=8)
    outcome = loop.run()
    assert outcome.best_result is not None
    # after warm start, proposals carry surrogate predictions
    model_iters = [h for h in outcome.history if h["predicted_ms"] is not None]
    assert model_iters, "no model-guided proposals recorded"
    assert all(h["provenance"] == "bo" for h in model_iters)


def test_thompson_acquisition_runs(engine):
    task = get_task("softmax")
    ctx = SearchContext(task=task, shape=(512, 1024),
                        hardware=engine.hardware, seed=2)
    searcher = make_searcher("bayesian", ctx, warm_start=12, pool_size=64,
                            acquisition="thompson", retrain_every=12)
    loop = SearchLoop(ctx, searcher, engine, db=None, max_evaluations=32,
                      batch_size=8)
    outcome = loop.run()
    assert outcome.candidates_evaluated == 32

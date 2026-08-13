"""Simulated engine behavior, analytic model sanity, DB round trips.

The simulated engine is development-only; these tests verify it is
deterministic, honestly labeled, and produces a response surface with the
qualitative structure real GPUs exhibit.
"""

import pytest

from benchmarks import get_task
from benchmarks.db import BenchmarkDB
from benchmarks.harness import (
    BenchmarkSettings,
    SimulatedBenchmarkEngine,
    make_engine,
)
from hardware.gpu_info import detect_environment, simulated_hardware
from optimizer.world_model.analytic import AnalyticGPUModel


@pytest.fixture(scope="module")
def hw():
    return simulated_hardware("A100-SXM4-40GB")


@pytest.fixture(scope="module")
def engine(hw):
    return SimulatedBenchmarkEngine(hw, BenchmarkSettings(warmup=2, iterations=20))


def test_environment_detection_runs():
    env = detect_environment()
    assert env.python_version
    assert isinstance(env.cuda_available, bool)
    assert env.summary()


def test_analytic_model_deterministic(hw):
    task = get_task("matmul")
    model = AnalyticGPUModel(hw)
    cfg = task.default_config((1024, 1024, 1024))
    a = model.latency_ms(task, (1024, 1024, 1024), cfg)
    b = model.latency_ms(task, (1024, 1024, 1024), cfg)
    assert a == b > 0


def test_analytic_model_orderings(hw):
    """Qualitative structure: known-good configs beat known-bad ones."""
    model = AnalyticGPUModel(hw)

    fe = get_task("fused_elementwise")
    shape = (1 << 24,)
    fused = {"BLOCK_SIZE": 1024, "num_warps": 4, "strategy": "fused", "dtype": "float32"}
    unfused = {**fused, "strategy": "unfused"}
    assert model.latency_ms(fe, shape, fused) < model.latency_ms(fe, shape, unfused)

    red = get_task("reduction")
    rshape = (1 << 24,)
    two_pass = {"BLOCK_SIZE": 1024, "num_warps": 4, "strategy": "two_pass",
                "dtype": "float32"}
    loop = {**two_pass, "strategy": "loop"}
    assert model.latency_ms(red, rshape, two_pass) < model.latency_ms(red, rshape, loop)

    mm = get_task("matmul")
    mshape = (2048, 2048, 2048)
    good = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8,
            "num_warps": 8, "num_stages": 4, "dtype": "float16"}
    tiny = {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16, "GROUP_M": 1,
            "num_warps": 8, "num_stages": 2, "dtype": "float16"}
    assert model.latency_ms(mm, mshape, good) < model.latency_ms(mm, mshape, tiny)
    # precision matters on tensor-core hardware
    good_fp32 = {**good, "dtype": "float32"}
    assert model.latency_ms(mm, mshape, good) < model.latency_ms(mm, mshape, good_fp32)


def test_simulated_compile_failure(hw):
    model = AnalyticGPUModel(hw)
    mm = get_task("matmul")
    huge = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8,
            "num_warps": 2, "num_stages": 5, "dtype": "float32"}
    ok, log = model.compiles(mm, (1024, 1024, 1024), huge)
    assert not ok and "shared memory" in log


def test_simulated_engine_result_labeling_and_determinism(engine):
    task = get_task("vecadd")
    cand = task.make_candidate((1 << 20,), task.default_config((1 << 20,)))
    r1 = engine.benchmark_candidate(task, cand)
    r2 = engine.benchmark_candidate(task, cand)
    assert r1.engine == "simulated"          # honesty label
    assert r1.status == "ok" and r1.correct
    assert r1.correctness_mode == "emulated"
    assert r1.latency.median_ms == r2.latency.median_ms
    assert r1.latency.p99_ms >= r1.latency.p50_ms > 0


def test_simulated_engine_compile_error_path(engine):
    task = get_task("matmul")
    bad = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8,
           "num_warps": 2, "num_stages": 5, "dtype": "float32"}
    cand = task.make_candidate((512, 512, 512), bad)
    r = engine.benchmark_candidate(task, cand)
    assert r.status == "compile_error"
    assert r.latency_ms == float("inf")


def test_simulated_baseline(engine):
    task = get_task("softmax")
    r = engine.benchmark_baseline(task, (512, 1024))
    assert r.engine == "simulated" and r.status == "ok"
    assert r.latency.median_ms > 0


def test_make_engine_falls_back_without_cuda(hw):
    import torch

    eng = make_engine(hw)
    if not torch.cuda.is_available():
        assert eng.label == "simulated"


def test_db_roundtrip(tmp_path, engine):
    db = BenchmarkDB(tmp_path / "t.sqlite")
    task = get_task("vecadd")
    cand = task.make_candidate((1 << 20,), task.default_config((1 << 20,)),
                               provenance="random")
    res = engine.benchmark_candidate(task, cand)

    db.create_run("run1", "vecadd", (1 << 20,), "random", engine.label, "test-gpu")
    rid = db.insert_result(res, run_id="run1")
    assert rid > 0
    db.insert_iteration("run1", 0, cand.candidate_id, res.latency.median_ms,
                        reward=1.0, best_so_far_ms=res.latency.median_ms, status="ok")
    db.update_run("run1", status="finished", best_candidate_id=cand.candidate_id,
                  best_latency_ms=res.latency.median_ms, candidates_evaluated=1)

    rows = db.fetch_results(task="vecadd")
    assert len(rows) == 1
    assert rows[0]["engine"] == "simulated"
    assert rows[0]["candidate_id"] == cand.candidate_id

    run = db.fetch_run("run1")
    assert run["status"] == "finished"
    iters = db.fetch_iterations("run1")
    assert len(iters) == 1

    ov = db.overview()
    assert ov["total_results"] == 1
    assert ov["results_by_engine"] == {"simulated": 1}

    with pytest.raises(ValueError):
        db.update_run("run1", nonsense_column=1)

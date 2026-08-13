"""Stage 2 integration: kernel → (simulated) benchmark → score pipeline."""

import itertools
import math

import pytest

from benchmarks import get_task
from benchmarks.db import BenchmarkDB
from benchmarks.harness import BenchmarkSettings, SimulatedBenchmarkEngine
from hardware.gpu_info import simulated_hardware
from optimizer.experiment import ExperimentConfig, run_experiment
from optimizer.rewards.reward import RewardConfig, compute_reward
from optimizer.search.base import SearchContext
from optimizer.search.factory import available_searchers, make_searcher
from optimizer.search.loop import SearchLoop


@pytest.fixture()
def engine():
    return SimulatedBenchmarkEngine(simulated_hardware(),
                                    BenchmarkSettings(warmup=2, iterations=10))


def test_reward_ordering(engine):
    task = get_task("vecadd")
    shape = (1 << 20,)
    good = engine.benchmark_candidate(task, task.make_candidate(shape, task.default_config(shape)))
    bad_cfg = {"BLOCK_SIZE": 128, "num_warps": 8, "dtype": "float32"}
    bad = engine.benchmark_candidate(task, task.make_candidate(shape, bad_cfg))
    baseline = engine.benchmark_baseline(task, shape).latency.median_ms
    assert compute_reward(good, baseline) > compute_reward(bad, baseline)

    # failed candidates are penalized, never rewarded
    mm = get_task("matmul")
    fail = engine.benchmark_candidate(mm, mm.make_candidate(
        (512, 512, 512),
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8,
         "num_warps": 2, "num_stages": 5, "dtype": "float32"}))
    assert fail.status == "compile_error"
    assert compute_reward(fail, baseline) == -RewardConfig().compile_penalty


def test_reward_modes():
    with pytest.raises(ValueError):
        RewardConfig(mode="bogus")


def test_random_search_loop_end_to_end(engine, tmp_path):
    db = BenchmarkDB(tmp_path / "s.sqlite")
    task = get_task("vecadd")
    ctx = SearchContext(task=task, shape=(1 << 22,), hardware=engine.hardware, seed=3)
    searcher = make_searcher("random", ctx)
    events = []
    loop = SearchLoop(ctx, searcher, engine, db=db, max_evaluations=25,
                      batch_size=5, callback=events.append)
    outcome = loop.run()

    assert outcome.candidates_evaluated == 25
    assert outcome.best_result is not None and outcome.best_result.ok
    assert math.isfinite(outcome.best_latency_ms)
    assert 0 < outcome.compile_success_rate <= 1.0
    assert outcome.engine == "simulated"

    # events: baseline + iterations + done
    kinds = [e["type"] for e in events]
    assert kinds[0] == "baseline" and kinds[-1] == "done"
    assert kinds.count("iteration") == 25

    # persistence
    run = db.fetch_run(outcome.run_id)
    assert run["status"] == "finished"
    assert run["candidates_evaluated"] == 25
    assert len(db.fetch_iterations(outcome.run_id)) == 25
    # results: 25 candidates + torch baseline + naive
    assert len(db.fetch_results(run_id=outcome.run_id)) == 27

    # best-so-far is monotone non-increasing
    curve = [e["best_so_far_ms"] for e in events if e["type"] == "iteration"]
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(curve))


def test_grid_search_exhausts(engine):
    task = get_task("vecadd")
    ctx = SearchContext(task=task, shape=(1 << 20,), hardware=engine.hardware, seed=0)
    searcher = make_searcher("grid", ctx)
    loop = SearchLoop(ctx, searcher, engine, db=None,
                      max_evaluations=10_000, batch_size=16)
    outcome = loop.run()
    assert outcome.candidates_evaluated == ctx.space.size()


def test_experiment_config_roundtrip(tmp_path):
    p = tmp_path / "e.yaml"
    cfg = ExperimentConfig(task="vecadd", shapes=[(1 << 20,)], algorithm="random",
                           engine="simulated", max_evaluations=5, db_path="")
    cfg.to_yaml(p)
    loaded = ExperimentConfig.from_yaml(p)
    assert loaded.task == "vecadd" and loaded.shapes == [(1 << 20,)]

    (tmp_path / "bad.yaml").write_text("task: vecadd\nbogus_key: 1\n")
    with pytest.raises(ValueError):
        ExperimentConfig.from_yaml(tmp_path / "bad.yaml")


def test_run_experiment_reproducible(tmp_path):
    cfg = ExperimentConfig(task="softmax", shapes=[(512, 1024)], algorithm="random",
                           engine="simulated", max_evaluations=12, batch_size=4,
                           benchmark={"warmup": 1, "iterations": 5}, db_path="")
    o1 = run_experiment(cfg)[0]
    o2 = run_experiment(cfg)[0]
    assert o1.best_candidate.config == o2.best_candidate.config
    assert o1.best_latency_ms == o2.best_latency_ms


def test_available_searchers_lists_baselines():
    names = available_searchers()
    assert "random" in names and "grid" in names

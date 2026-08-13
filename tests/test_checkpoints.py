"""Warm-start persistence: surrogate and PPO policy checkpoints across runs."""

import math

import pytest
import torch

from benchmarks import get_task
from benchmarks.harness import BenchmarkSettings, SimulatedBenchmarkEngine
from hardware.gpu_info import simulated_hardware
from optimizer.performance_model.model import PerformanceModel
from optimizer.search.base import SearchContext
from optimizer.search.factory import make_searcher
from optimizer.search.loop import SearchLoop


@pytest.fixture(scope="module")
def engine():
    return SimulatedBenchmarkEngine(simulated_hardware(),
                                    BenchmarkSettings(warmup=1, iterations=5),
                                    check_correctness_emulated=False)


def _run(engine, algo: str, seed: int, budget: int, **params):
    ctx = SearchContext(task=get_task("matmul"), shape=(1024, 1024, 1024),
                        hardware=engine.hardware, seed=seed)
    searcher = make_searcher(algo, ctx, **params)
    outcome = SearchLoop(ctx, searcher, engine, db=None, max_evaluations=budget,
                         batch_size=8).run()
    return searcher, outcome


def test_surrogate_checkpoint_roundtrip_with_corpus(engine, tmp_path):
    ckpt = tmp_path / "surrogate.pt"
    s1, _ = _run(engine, "bayesian", seed=0, budget=40,
                 warm_start=16, retrain_every=16,
                 surrogate_checkpoint=str(ckpt))
    assert ckpt.exists()
    assert s1.model.trained and len(s1.model.rows) >= 40

    # a fresh searcher loads weights AND corpus, and skips its cold start
    s2, o2 = _run(engine, "bayesian", seed=1, budget=16,
                  warm_start=16, retrain_every=999,
                  surrogate_checkpoint=str(ckpt))
    assert s2._pretrained
    assert len(s2.model.rows) >= 40 + 16  # prior corpus + this run's rows
    # pretrained model guides from the very first batch: predictions attached
    predicted = [h for h in o2.history if h["predicted_ms"] is not None]
    assert predicted, "warm-started searcher never used its surrogate"
    assert predicted[0]["iteration"] <= 8  # first batch, not after warm_start


def test_policy_checkpoint_roundtrip(engine, tmp_path):
    torch.manual_seed(0)
    ckpt = tmp_path / "policy.pt"
    s1, _ = _run(engine, "rl", seed=0, budget=80,
                 rl_params={"horizon": 32, "minibatch_size": 16},
                 max_episode_steps=6, policy_checkpoint=str(ckpt))
    assert ckpt.exists()
    updates_before = s1.trainer.updates_done
    assert updates_before >= 1

    s2, o2 = _run(engine, "rl", seed=1, budget=24,
                  rl_params={"horizon": 32, "minibatch_size": 16},
                  max_episode_steps=6, policy_checkpoint=str(ckpt))
    # warm start restored the update counter and kept working
    assert s2.trainer.updates_done >= updates_before
    assert o2.best_result is not None and math.isfinite(o2.best_latency_ms)

    # weights actually round-trip: fresh load matches what was saved
    m1 = PerformanceModel  # unrelated import guard; keep torch honest below
    del m1
    from optimizer.policy.policy_net import PolicyNet
    from optimizer.rl.ppo import PPOTrainer

    probe = PPOTrainer(PolicyNet())
    probe.load(ckpt)
    saved = probe.policy.state_dict()
    current = s2.trainer.policy.state_dict()
    assert all(torch.equal(saved[k], current[k]) for k in saved)


def test_hybrid_persists_both(engine, tmp_path):
    sckpt, pckpt = tmp_path / "s.pt", tmp_path / "p.pt"
    _s1, _ = _run(engine, "hybrid", seed=0, budget=48,
                  warm_start=16, retrain_every=24, population_size=12,
                  surrogate_checkpoint=str(sckpt), policy_checkpoint=str(pckpt))
    assert sckpt.exists() and pckpt.exists()

    s2, o2 = _run(engine, "hybrid", seed=2, budget=16,
                  warm_start=16, retrain_every=999, population_size=12,
                  surrogate_checkpoint=str(sckpt), policy_checkpoint=str(pckpt))
    assert s2._pretrained
    predicted = [h for h in o2.history if h["predicted_ms"] is not None]
    assert predicted, "hybrid warm start did not engage the surrogate immediately"


def test_missing_checkpoint_is_not_an_error(engine, tmp_path):
    s, o = _run(engine, "bayesian", seed=0, budget=8, warm_start=16,
                surrogate_checkpoint=str(tmp_path / "does-not-exist-yet.pt"))
    assert o.candidates_evaluated == 8
    assert not s._pretrained
    assert (tmp_path / "does-not-exist-yet.pt").exists()  # created on finalize

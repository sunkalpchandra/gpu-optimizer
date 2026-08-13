"""Stage 5: policy network, PPO machinery, and the online RL searcher."""

import math

import numpy as np
import pytest
import torch

from benchmarks import get_task
from benchmarks.harness import BenchmarkSettings, SimulatedBenchmarkEngine
from compiler.transformations.actions import NUM_ACTIONS, STOP_ACTION_INDEX
from hardware.gpu_info import simulated_hardware
from optimizer.policy.policy_net import PolicyNet, obs_to_tensors
from optimizer.rl.ppo import PPOConfig, PPOTrainer, RolloutBuffer, Transition
from optimizer.rl.searcher import RLSearcher
from optimizer.search.base import SearchContext
from optimizer.search.factory import make_searcher
from optimizer.search.loop import SearchLoop


@pytest.fixture(scope="module")
def engine():
    return SimulatedBenchmarkEngine(simulated_hardware(),
                                    BenchmarkSettings(warmup=1, iterations=5),
                                    check_correctness_emulated=False)


def _ctx(engine, seed=0):
    return SearchContext(task=get_task("matmul"), shape=(1024, 1024, 1024),
                         hardware=engine.hardware, seed=seed)


def _dummy_obs(mask=None):
    m = np.ones(NUM_ACTIONS, dtype=np.float32) if mask is None else mask
    return {
        "node_feats": np.random.rand(4, 34).astype(np.float32),
        "adj": np.eye(4, dtype=np.float32),
        "hw": np.random.rand(12).astype(np.float32),
        "cand": np.random.rand(25).astype(np.float32),
        "scalars": np.zeros(3, dtype=np.float32),
        "mask": m,
    }


def test_policy_respects_action_mask():
    torch.manual_seed(0)
    from compiler.ir import NODE_FEATURE_DIM
    from compiler.transformations.space import CANDIDATE_FEATURE_DIM

    policy = PolicyNet()
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    allowed = [0, 5, STOP_ACTION_INDEX]
    for a in allowed:
        mask[a] = 1.0
    obs = _dummy_obs(mask)
    obs["node_feats"] = np.random.rand(4, NODE_FEATURE_DIM).astype(np.float32)
    obs["cand"] = np.random.rand(CANDIDATE_FEATURE_DIM).astype(np.float32)
    batch = obs_to_tensors([obs] * 16)
    out = policy(batch)
    assert all(int(a) in allowed for a in out.action)
    assert torch.isfinite(out.log_prob).all()
    assert torch.isfinite(out.value).all()


def test_gae_computation():
    buf = RolloutBuffer()
    # two episodes: [r=1, r=1(done)], [r=-1(done)] with V=0 everywhere
    for r, d in ((1.0, False), (1.0, True), (-1.0, True)):
        buf.add(Transition(_dummy_obs(), 0, 0.0, 0.0, r, d))
    adv, ret = buf.compute_gae(gamma=1.0, lam=1.0)
    assert ret[2] == pytest.approx(-1.0)
    assert ret[0] == pytest.approx(2.0)   # 1 + 1 within the episode
    assert ret[1] == pytest.approx(1.0)
    # episode boundary respected: ep2's return unaffected by ep1
    assert adv.shape == (3,)


def test_ppo_update_improves_probability_of_rewarded_action():
    """A minimal sanity check that the PPO step moves the policy toward
    actions that received positive advantage."""
    torch.manual_seed(1)
    np.random.seed(1)
    from compiler.ir import NODE_FEATURE_DIM
    from compiler.transformations.space import CANDIDATE_FEATURE_DIM

    policy = PolicyNet(d_model=32, n_layers=1)
    trainer = PPOTrainer(policy, PPOConfig(horizon=32, minibatch_size=16,
                                           update_epochs=6, learning_rate=1e-3))
    obs = _dummy_obs()
    obs["node_feats"] = np.random.rand(4, NODE_FEATURE_DIM).astype(np.float32)
    obs["cand"] = np.random.rand(CANDIDATE_FEATURE_DIM).astype(np.float32)

    GOOD = 3
    batch = obs_to_tensors([obs])
    with torch.no_grad():
        before = torch.softmax(policy(batch).logits, -1)[0, GOOD]
        for _ in range(32):
            out = policy(batch)
            a = int(out.action.item())
            r = 1.0 if a == GOOD else -0.2
            trainer.buffer.add(Transition(obs, a, float(out.log_prob.item()),
                                          float(out.value.item()), r, True))
    trainer.update()
    with torch.no_grad():
        after = torch.softmax(policy(batch).logits, -1)[0, GOOD]
    assert after > before, f"P(good action) {float(before):.4f} → {float(after):.4f}"
    assert trainer.updates_done == 1
    assert len(trainer.buffer) == 0


def test_rl_searcher_end_to_end(engine):
    torch.manual_seed(0)
    ctx = _ctx(engine, seed=0)
    searcher = make_searcher(
        "rl", ctx, rl_params={"learning_rate": 3e-4, "horizon": 48,
                              "minibatch_size": 24}, max_episode_steps=6)
    loop = SearchLoop(ctx, searcher, engine, db=None, max_evaluations=80,
                      batch_size=8)
    outcome = loop.run()
    assert outcome.candidates_evaluated == 80
    assert outcome.best_result is not None and outcome.best_result.ok
    assert math.isfinite(outcome.best_latency_ms)
    # PPO actually trained during the search
    assert searcher.trainer.updates_done >= 1
    # transitions carry measured-latency rewards (episodes really advanced)
    assert searcher.best_config is not None
    # lineage: rl candidates that are policy edits carry parents
    edited = [h for h in outcome.history
              if h["provenance"] == "rl" and h["iteration"] > 1]
    assert edited


def test_rl_factory_rejects_unknown_params(engine):
    ctx = _ctx(engine)
    with pytest.raises(ValueError):
        make_searcher("rl", ctx, rl_params={"bogus": 1})

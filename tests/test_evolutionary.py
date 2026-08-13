"""Stage 3: evolutionary optimizer unit + comparative tests (simulated)."""

import pytest

from benchmarks import get_task
from benchmarks.harness import BenchmarkSettings, SimulatedBenchmarkEngine
from hardware.gpu_info import simulated_hardware
from optimizer.evolutionary.ga import EvolutionarySearcher, GAParams
from optimizer.search.base import SearchContext
from optimizer.search.factory import make_searcher
from optimizer.search.loop import SearchLoop


@pytest.fixture()
def engine():
    return SimulatedBenchmarkEngine(simulated_hardware(),
                                    BenchmarkSettings(warmup=1, iterations=5))


def _ctx(engine, seed=0, task="matmul", shape=(1024, 1024, 1024)):
    return SearchContext(task=get_task(task), shape=shape,
                         hardware=engine.hardware, seed=seed)


def test_mutation_always_changes_something(engine):
    ctx = _ctx(engine)
    s = EvolutionarySearcher(ctx, GAParams(mutation_rate=0.0))
    cfg = ctx.task.default_config(ctx.shape)
    for _ in range(10):
        assert s._mutate(cfg) != cfg


def test_crossover_mixes_parents(engine):
    ctx = _ctx(engine)
    s = EvolutionarySearcher(ctx)
    a = {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16, "GROUP_M": 1,
         "num_warps": 2, "num_stages": 2, "dtype": "float32"}
    b = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8,
         "num_warps": 8, "num_stages": 5, "dtype": "float16"}
    child = s._crossover(a, b)
    assert set(child) == set(a)
    assert all(child[k] in (a[k], b[k]) for k in child)


def test_no_duplicate_proposals(engine):
    ctx = _ctx(engine, task="vecadd", shape=(1 << 20,))
    s = EvolutionarySearcher(ctx, GAParams(population_size=8))
    seen = set()
    for _ in range(12):
        for cand in s.propose(8):
            assert cand.candidate_id not in seen
            seen.add(cand.candidate_id)
            s.observe([(cand, _fake_ok_result(cand), 0.1)])


def _fake_ok_result(cand):
    from benchmarks.harness import BenchmarkResult, LatencyStats

    return BenchmarkResult(
        candidate_id=cand.candidate_id, task=cand.task, shape=cand.shape,
        config=cand.config, engine="simulated", status="ok", gpu_name="t",
        correct=True, latency=LatencyStats.from_samples([1.0, 1.0, 1.0]))


def test_evolutionary_beats_random_on_structured_surface(engine):
    """Same budget, same simulated surface: evolution should find a better
    (or equal) matmul config than pure random sampling."""
    budget = 96

    def run(algo: str, seed: int) -> float:
        ctx = _ctx(engine, seed=seed)
        searcher = make_searcher(algo, ctx, population_size=24)
        loop = SearchLoop(ctx, searcher, engine, db=None,
                          max_evaluations=budget, batch_size=8)
        return loop.run().best_latency_ms

    evo = min(run("evolutionary", s) for s in (0, 1))
    rnd = min(run("random", s) for s in (0, 1))
    assert evo <= rnd * 1.02, f"evolutionary {evo:.4f}ms vs random {rnd:.4f}ms"


def test_lineage_recorded(engine):
    ctx = _ctx(engine, task="softmax", shape=(512, 1024))
    searcher = make_searcher("evolutionary", ctx, population_size=8)
    loop = SearchLoop(ctx, searcher, engine, db=None, max_evaluations=40,
                      batch_size=8)
    outcome = loop.run()
    children = [h for h in outcome.history if h["provenance"] == "evolutionary"]
    assert children, "no evolutionary candidates recorded"
    # after bootstrap, offspring should carry parent lineage in the tree
    tree_edges = [c for c in searcher.pool if c[0].parent_id is not None]
    assert tree_edges, "no parent lineage recorded in evolved candidates"

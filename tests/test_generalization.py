"""Stage 7 smoke tests: generalization studies run and report honestly."""

import pytest

from benchmarks.harness import BenchmarkSettings, SimulatedBenchmarkEngine
from hardware.gpu_info import simulated_hardware
from optimizer.experiments.generalization import (
    GeneralizationReport,
    study_hardware_transfer,
    study_search_efficiency,
    study_shape_generalization,
)


@pytest.fixture(scope="module")
def engine():
    return SimulatedBenchmarkEngine(simulated_hardware(),
                                    BenchmarkSettings(warmup=1, iterations=5),
                                    check_correctness_emulated=False)


def test_interpolation_study(engine):
    # budget matters: below ~160 training rows / 25 epochs the ensemble is
    # undertrained and its ranking is untrustworthy (see run_all NOTE)
    res = study_shape_generalization(
        engine,
        train_shapes=[(512, 512, 512), (1024, 1024, 1024)],
        test_shapes=[(768, 768, 768)],
        rows_per_shape=80, n_eval=24, seed=0, epochs=25)
    metrics = res["matmul(768, 768, 768)"]
    assert metrics["n"] >= 5
    assert metrics["spearman"] is not None and metrics["spearman"] > 0.3, (
        f"unseen-shape ranking should carry signal: {metrics}")
    assert metrics["top1_regret_pct"] is not None


def test_hardware_transfer_study():
    res = study_hardware_transfer(budget=32, rows=80, n_eval=16, seed=0,
                                  settings=BenchmarkSettings(warmup=1, iterations=5))
    (key, m), = res.items()
    assert "A100" in key and "RTX-4090" in key
    assert m["native_ms"] > 0
    # top-K transfer: some A-elite config must run on B (K reported)
    assert m["transferred_ms"] != "all top-k failed"
    assert 1 <= m["top_k_needed"] <= 5


def test_search_efficiency_study(engine):
    res = study_search_efficiency(engine, budget=24, seed=0,
                                  algorithms=("random", "evolutionary"))
    assert set(res) == {"random", "evolutionary"}
    for m in res.values():
        assert m["best_ms"] > 0 and m["evals_to_best"] is not None


def test_report_markdown_labels_simulated(engine):
    r = GeneralizationReport(engine_label="simulated", gpu=engine.hardware.name,
                             seed=0)
    r.search_efficiency = {"random": {"best_ms": 1.0, "speedup_vs_torch": 1.0,
                                      "evals_to_best": 1, "compile_rate": 1.0}}
    md = r.to_markdown()
    assert "simulated engine" in md and "not" in md and "hardware measurements" in md

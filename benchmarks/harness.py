"""Benchmark harness: measure candidates and baselines, honestly.

Two engines implement the same interface:

- :class:`CudaBenchmarkEngine` — real hardware measurements with proper CUDA
  event timing, warmup, and synchronization.  Engine label ``cuda``.
- :class:`SimulatedBenchmarkEngine` — the deterministic analytic model for
  development on machines without CUDA, **plus** real CPU emulation of the
  candidate algorithm for correctness checking.  Engine label ``simulated``.

The engine label travels with every result into the database, the API, and
the UI; simulated numbers are never presented as hardware measurements.
"""

from __future__ import annotations

import logging
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
import torch

from benchmarks.base import Task
from compiler.transformations.space import Candidate
from compiler.validation.correctness import CorrectnessReport, check_correctness
from hardware.gpu_info import HardwareSpec

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_COMPILE_ERROR = "compile_error"
STATUS_RUNTIME_ERROR = "runtime_error"
STATUS_INCORRECT = "incorrect"


@dataclass
class LatencyStats:
    mean_ms: float
    median_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    std_ms: float
    min_ms: float
    iterations: int

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float]) -> "LatencyStats":
        arr = np.asarray(samples_ms, dtype=np.float64)
        return cls(
            mean_ms=float(arr.mean()),
            median_ms=float(np.median(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p99_ms=float(np.percentile(arr, 99)),
            std_ms=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            min_ms=float(arr.min()),
            iterations=int(arr.size),
        )


@dataclass
class BenchmarkResult:
    candidate_id: str
    task: str
    shape: tuple[int, ...]
    config: dict
    engine: str                      # "cuda" | "simulated"
    status: str                      # ok | compile_error | runtime_error | incorrect
    gpu_name: str
    correct: bool = False
    correctness_mode: str = "none"   # device | emulated | none
    correctness: CorrectnessReport | None = None
    latency: LatencyStats | None = None
    throughput_gflops: float = 0.0
    memory_bytes: int = 0
    compile_time_s: float = 0.0
    warmup: int = 0
    provenance: str = "manual"
    parent_id: str | None = None
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def latency_ms(self) -> float:
        """Median latency, or +inf for failed candidates (safe to rank on)."""
        return self.latency.median_ms if (self.ok and self.latency) else float("inf")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d


@dataclass
class BenchmarkSettings:
    warmup: int = 20
    iterations: int = 100
    seed: int = 0


class BenchmarkEngine(ABC):
    """Common interface for real and simulated measurement."""

    label: str = "abstract"

    def __init__(self, hardware: HardwareSpec,
                 settings: BenchmarkSettings | None = None) -> None:
        self.hardware = hardware
        self.settings = settings or BenchmarkSettings()

    @abstractmethod
    def benchmark_candidate(self, task: Task, candidate: Candidate) -> BenchmarkResult:
        """Compile, verify, and measure one candidate."""

    @abstractmethod
    def benchmark_baseline(self, task: Task, shape: Sequence[int],
                           dtype: str = "float32") -> BenchmarkResult:
        """Measure the PyTorch (library) baseline for speedup reporting."""


# ---------------------------------------------------------------------------
# Real hardware engine
# ---------------------------------------------------------------------------

class CudaBenchmarkEngine(BenchmarkEngine):
    """Measures on the actual GPU with CUDA-event timing."""

    label = "cuda"

    def __init__(self, hardware: HardwareSpec,
                 settings: BenchmarkSettings | None = None) -> None:
        super().__init__(hardware, settings)
        if not torch.cuda.is_available():
            raise RuntimeError("CudaBenchmarkEngine requires a CUDA device")

    def _time_callable(self, fn) -> LatencyStats:
        s = self.settings
        for _ in range(s.warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(s.iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(s.iterations)]
        for i in range(s.iterations):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        samples = [starts[i].elapsed_time(ends[i]) for i in range(s.iterations)]
        return LatencyStats.from_samples(samples)

    def benchmark_candidate(self, task: Task, candidate: Candidate) -> BenchmarkResult:
        from compiler.triton.runner import KernelBuildError, build_callable, warm_compile

        shape, config = candidate.shape, candidate.config
        base = dict(candidate_id=candidate.candidate_id, task=task.name, shape=shape,
                    config=dict(config), engine=self.label, gpu_name=self.hardware.name,
                    provenance=candidate.provenance, parent_id=candidate.parent_id,
                    warmup=self.settings.warmup)
        inputs = task.make_inputs(shape, candidate.dtype, "cuda", self.settings.seed)

        try:
            kc = build_callable(task.name, shape, config, inputs)
            compile_time = warm_compile(kc)
        except KernelBuildError as e:
            return BenchmarkResult(**base, status=STATUS_COMPILE_ERROR, error=str(e))

        # Correctness before any reward-bearing measurement.
        try:
            out = kc.fn()
            torch.cuda.synchronize()
            ref = task.reference(*inputs)
            rtol, atol = task.tolerance(candidate.dtype)
            report = check_correctness(out, ref, rtol, atol)
        except Exception as e:
            return BenchmarkResult(**base, status=STATUS_RUNTIME_ERROR,
                                   compile_time_s=compile_time, error=str(e))
        if not report.passed:
            return BenchmarkResult(**base, status=STATUS_INCORRECT, correct=False,
                                   correctness_mode="device", correctness=report,
                                   compile_time_s=compile_time,
                                   error=f"correctness: {report.summary()}")

        torch.cuda.reset_peak_memory_stats()
        try:
            stats = self._time_callable(kc.fn)
        except Exception as e:
            return BenchmarkResult(**base, status=STATUS_RUNTIME_ERROR,
                                   compile_time_s=compile_time, error=str(e))
        mem = int(torch.cuda.max_memory_allocated())
        gflops = task.flops(shape) / (stats.median_ms * 1e-3) / 1e9

        return BenchmarkResult(**base, status=STATUS_OK, correct=True,
                               correctness_mode="device", correctness=report,
                               latency=stats, throughput_gflops=gflops,
                               memory_bytes=mem, compile_time_s=compile_time)

    def benchmark_baseline(self, task: Task, shape: Sequence[int],
                           dtype: str = "float32") -> BenchmarkResult:
        shape = tuple(int(x) for x in shape)
        inputs = task.make_inputs(shape, dtype, "cuda", self.settings.seed)
        fn = lambda: task.baseline(*inputs)  # noqa: E731
        stats = self._time_callable(fn)
        gflops = task.flops(shape) / (stats.median_ms * 1e-3) / 1e9
        return BenchmarkResult(
            candidate_id=f"baseline-torch-{task.name}", task=task.name, shape=shape,
            config={"dtype": dtype, "baseline": "torch"}, engine=self.label,
            gpu_name=self.hardware.name, status=STATUS_OK, correct=True,
            correctness_mode="device", latency=stats, throughput_gflops=gflops,
            warmup=self.settings.warmup, provenance="baseline",
        )


# ---------------------------------------------------------------------------
# Simulated engine (development only)
# ---------------------------------------------------------------------------

class SimulatedBenchmarkEngine(BenchmarkEngine):
    """Deterministic analytic latency + real CPU emulation for correctness.

    DEVELOPMENT ONLY.  Every result carries ``engine="simulated"``.
    """

    label = "simulated"

    def __init__(self, hardware: HardwareSpec,
                 settings: BenchmarkSettings | None = None,
                 check_correctness_emulated: bool = True) -> None:
        super().__init__(hardware, settings)
        from optimizer.world_model.analytic import AnalyticGPUModel

        self.model = AnalyticGPUModel(hardware)
        self.check_correctness_emulated = check_correctness_emulated
        self._correctness_cache: dict[tuple, CorrectnessReport] = {}

    # deterministic per-candidate iteration noise (measurement-like jitter)
    def _samples(self, base_ms: float, key: str) -> list[float]:
        rng = np.random.default_rng(abs(hash((key, self.settings.seed))) % 2**32)
        jitter = rng.normal(1.0, 0.015, size=self.settings.iterations)
        tail = rng.random(self.settings.iterations) < 0.02
        jitter[tail] *= rng.uniform(1.05, 1.25, size=int(tail.sum()))
        return list(base_ms * np.clip(jitter, 0.97, None))

    def _emulated_correctness(self, task: Task, candidate: Candidate
                              ) -> CorrectnessReport:
        # numerics depend only on algorithm-relevant parts of the config
        algo_keys = ("dtype", "strategy", "BLOCK_K", "BLOCK_N", "BLOCK_SIZE", "BLOCK_M")
        cache_key = (task.name, candidate.shape,
                     tuple((k, candidate.config.get(k)) for k in algo_keys))
        if cache_key in self._correctness_cache:
            return self._correctness_cache[cache_key]

        from compiler.validation.emulation import emulate

        # cap the shape for CPU tractability while keeping the algorithm intact
        shape = _emulation_shape(task.name, candidate.shape)
        inputs = task.make_inputs(shape, candidate.dtype, "cpu", self.settings.seed)
        out = emulate(task.name, shape, candidate.config, inputs)
        ref = task.reference(*inputs)
        rtol, atol = task.tolerance(candidate.dtype)
        report = check_correctness(out, ref, rtol, atol)
        self._correctness_cache[cache_key] = report
        return report

    def benchmark_candidate(self, task: Task, candidate: Candidate) -> BenchmarkResult:
        shape, config = candidate.shape, candidate.config
        base = dict(candidate_id=candidate.candidate_id, task=task.name, shape=shape,
                    config=dict(config), engine=self.label, gpu_name=self.hardware.name,
                    provenance=candidate.provenance, parent_id=candidate.parent_id,
                    warmup=self.settings.warmup)

        ok, log = self.model.compiles(task, shape, config)
        if not ok:
            return BenchmarkResult(**base, status=STATUS_COMPILE_ERROR, error=log)

        report: CorrectnessReport | None = None
        mode = "none"
        if self.check_correctness_emulated:
            try:
                report = self._emulated_correctness(task, candidate)
                mode = "emulated"
            except Exception as e:
                return BenchmarkResult(**base, status=STATUS_RUNTIME_ERROR,
                                       error=f"emulation failed: {e}")
            if not report.passed:
                return BenchmarkResult(**base, status=STATUS_INCORRECT, correct=False,
                                       correctness_mode=mode, correctness=report,
                                       error=f"correctness: {report.summary()}")

        base_ms = self.model.latency_ms(task, shape, config)
        stats = LatencyStats.from_samples(self._samples(base_ms, candidate.candidate_id))
        gflops = task.flops(shape) / (stats.median_ms * 1e-3) / 1e9
        # deterministic pseudo compile time in a plausible range
        ct = 0.5 + 2.5 * (abs(hash(candidate.candidate_id)) % 1000) / 1000.0

        return BenchmarkResult(**base, status=STATUS_OK,
                               correct=bool(report.passed) if report else True,
                               correctness_mode=mode, correctness=report,
                               latency=stats, throughput_gflops=gflops,
                               memory_bytes=self.model.memory_bytes(task, shape, config),
                               compile_time_s=ct)

    def benchmark_baseline(self, task: Task, shape: Sequence[int],
                           dtype: str = "float32") -> BenchmarkResult:
        """Simulated library baseline: near-roofline execution (eff. 0.92)."""
        shape = tuple(int(x) for x in shape)
        flops = task.flops(shape)
        byts = task.bytes_moved(shape, dtype, None)
        hw = self.hardware
        peak = hw.fp16_tflops if dtype in ("float16", "bfloat16") else hw.fp32_tflops
        base_ms = max(flops / (peak * 1e12), byts / (hw.memory_bandwidth_gbs * 1e9)) * 1e3
        base_ms = base_ms / 0.92 + 0.004
        stats = LatencyStats.from_samples(self._samples(base_ms, f"baseline-{task.name}-{shape}"))
        return BenchmarkResult(
            candidate_id=f"baseline-torch-{task.name}", task=task.name, shape=shape,
            config={"dtype": dtype, "baseline": "torch"}, engine=self.label,
            gpu_name=self.hardware.name, status=STATUS_OK, correct=True,
            correctness_mode="none", latency=stats,
            throughput_gflops=flops / (stats.median_ms * 1e-3) / 1e9,
            warmup=self.settings.warmup, provenance="baseline",
        )


def _emulation_shape(task: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Shrink huge shapes for CPU emulation while preserving tiling behavior."""
    if task == "matmul":
        m, n, k = shape
        return (min(m, 512), min(n, 512), min(k, 512))
    if task == "attention":
        bh, s, d = shape
        return (min(bh, 4), min(s, 512), d)
    if task in ("softmax", "layernorm"):
        m, n = shape
        return (min(m, 512), n)
    return (min(shape[0], 1 << 20),)


def make_engine(hardware: HardwareSpec, settings: BenchmarkSettings | None = None,
                force_simulated: bool = False) -> BenchmarkEngine:
    """Pick the real engine when CUDA exists, else the labeled simulated one."""
    if torch.cuda.is_available() and not force_simulated:
        return CudaBenchmarkEngine(hardware, settings)
    logger.warning("CUDA unavailable or simulation forced: using SimulatedBenchmarkEngine "
                   "(all results labeled 'simulated')")
    return SimulatedBenchmarkEngine(hardware, settings)

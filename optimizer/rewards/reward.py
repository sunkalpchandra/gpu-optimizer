"""Reward computation from benchmark results.

Conceptually: ``reward = speedup_term − correctness_penalty −
compilation_penalty − memory_penalty``, with every term configurable.
Failed candidates can never out-reward a working one, and a faster but
*incorrect* kernel earns the correctness penalty, not its speedup.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.harness import (
    STATUS_COMPILE_ERROR,
    STATUS_INCORRECT,
    STATUS_OK,
    STATUS_RUNTIME_ERROR,
    BenchmarkResult,
)


@dataclass
class RewardConfig:
    """Weights and shape of the reward signal."""

    mode: str = "log_speedup"          # "speedup" (baseline/t − 1) | "log_speedup" (log2)
    correctness_penalty: float = 1.0    # subtracted for wrong results
    compile_penalty: float = 0.5        # subtracted for compilation failures
    runtime_penalty: float = 0.8        # subtracted for launch/runtime crashes
    memory_limit_bytes: int | None = None
    memory_penalty: float = 0.5         # subtracted when exceeding the memory limit
    clip: float | None = 8.0            # symmetric reward clip (None to disable)

    def __post_init__(self) -> None:
        if self.mode not in ("speedup", "log_speedup"):
            raise ValueError(f"unknown reward mode {self.mode!r}")


def compute_reward(result: BenchmarkResult, baseline_ms: float,
                   config: RewardConfig | None = None) -> float:
    """Score one benchmark result against the task baseline latency."""
    import math

    cfg = config or RewardConfig()
    if baseline_ms <= 0:
        raise ValueError("baseline_ms must be positive")

    if result.status == STATUS_COMPILE_ERROR:
        r = -cfg.compile_penalty
    elif result.status == STATUS_RUNTIME_ERROR:
        r = -cfg.runtime_penalty
    elif result.status == STATUS_INCORRECT:
        r = -cfg.correctness_penalty
    elif result.status == STATUS_OK:
        speedup = baseline_ms / result.latency.median_ms
        r = math.log2(speedup) if cfg.mode == "log_speedup" else speedup - 1.0
        if (cfg.memory_limit_bytes is not None
                and result.memory_bytes > cfg.memory_limit_bytes):
            r -= cfg.memory_penalty
    else:
        raise ValueError(f"unknown result status {result.status!r}")

    if cfg.clip is not None:
        r = max(-cfg.clip, min(cfg.clip, r))
    return float(r)

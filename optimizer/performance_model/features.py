"""Feature assembly for the learned performance model.

Each training row (a benchmark result, from the live search or the results
DB) becomes: program graph tensors + hardware feature vector + candidate
encoding + targets (log latency, log memory, compile success).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from benchmarks import get_task
from benchmarks.harness import BenchmarkResult
from compiler.ir import ProgramGraph
from compiler.transformations.space import encode_config
from hardware.gpu_info import DEFAULT_SIMULATED_GPU, KNOWN_GPUS, HardwareSpec


@dataclass
class TrainingRow:
    task: str
    shape: tuple[int, ...]
    config: dict
    gpu_name: str
    compiled: bool
    latency_ms: float | None      # None unless status == ok
    memory_bytes: int | None


def result_to_row(r: BenchmarkResult) -> TrainingRow:
    return TrainingRow(
        task=r.task, shape=tuple(r.shape), config=dict(r.config),
        gpu_name=r.gpu_name, compiled=r.status != "compile_error",
        latency_ms=r.latency.median_ms if r.ok and r.latency else None,
        memory_bytes=r.memory_bytes if r.ok else None,
    )


def db_row_to_row(d: dict[str, Any]) -> TrainingRow:
    """Convert a raw ``results`` table row."""
    return TrainingRow(
        task=d["task"], shape=tuple(json.loads(d["shape"])),
        config=json.loads(d["config"]), gpu_name=d.get("gpu_name") or "",
        compiled=d["status"] != "compile_error",
        latency_ms=d["latency_median_ms"] if d["status"] == "ok" else None,
        memory_bytes=d["memory_bytes"] if d["status"] == "ok" else None,
    )


def spec_by_name(name: str) -> HardwareSpec:
    for spec in KNOWN_GPUS.values():
        base = spec.name.replace(" (simulated)", "").lower()
        if base and (base in name.lower() or name.lower() in spec.name.lower()):
            return spec
    return KNOWN_GPUS[DEFAULT_SIMULATED_GPU]


_GRAPH_CACHE: dict[tuple, ProgramGraph] = {}


def graph_for(task: str, shape: tuple[int, ...], dtype: str) -> ProgramGraph:
    ir_dtype = "float32" if dtype in ("tf32", "float32") else dtype
    key = (task, shape, ir_dtype)
    if key not in _GRAPH_CACHE:
        _GRAPH_CACHE[key] = get_task(task).graph(shape, ir_dtype)
    return _GRAPH_CACHE[key]


@dataclass
class FeatureBatch:
    node_feats: torch.Tensor      # (B, N, F)
    adj: torch.Tensor             # (B, N, N)
    hw_feats: torch.Tensor        # (B, H)
    cand_feats: torch.Tensor      # (B, C)
    log_ms: torch.Tensor          # (B,) target, NaN where unavailable
    log_mem: torch.Tensor         # (B,) target, NaN where unavailable
    compiled: torch.Tensor        # (B,) 0/1

    def to(self, device: str) -> "FeatureBatch":
        return FeatureBatch(*(t.to(device) for t in (
            self.node_feats, self.adj, self.hw_feats, self.cand_feats,
            self.log_ms, self.log_mem, self.compiled)))

    def __len__(self) -> int:
        return self.node_feats.shape[0]


def rows_to_batch(rows: list[TrainingRow], hardware: HardwareSpec | None = None
                  ) -> FeatureBatch:
    """Assemble a batch.  ``hardware`` overrides per-row GPU lookup (used at
    prediction time for the *current* target)."""
    from optimizer.policy.encoders import pad_graph_batch

    graphs = [graph_for(r.task, r.shape, str(r.config.get("dtype", "float32")))
              for r in rows]
    node_feats, adj = pad_graph_batch(graphs)
    hw = np.stack([(hardware or spec_by_name(r.gpu_name)).feature_vector()
                   for r in rows])
    cand = np.stack([encode_config(r.config) for r in rows])
    log_ms = np.array([math.log(r.latency_ms) if r.latency_ms else math.nan
                       for r in rows], dtype=np.float32)
    log_mem = np.array([math.log1p(r.memory_bytes) if r.memory_bytes else math.nan
                        for r in rows], dtype=np.float32)
    compiled = np.array([1.0 if r.compiled else 0.0 for r in rows], dtype=np.float32)
    return FeatureBatch(
        node_feats, adj,
        torch.from_numpy(hw.astype(np.float32)),
        torch.from_numpy(cand.astype(np.float32)),
        torch.from_numpy(log_ms), torch.from_numpy(log_mem),
        torch.from_numpy(compiled),
    )

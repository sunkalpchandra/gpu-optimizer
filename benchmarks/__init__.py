"""Benchmark task registry."""

from __future__ import annotations

from benchmarks.attention.task import AttentionTask
from benchmarks.base import Task
from benchmarks.fused_elementwise.task import FusedElementwiseTask
from benchmarks.layernorm.task import LayerNormTask
from benchmarks.matmul.task import MatmulTask
from benchmarks.reduction.task import ReductionTask
from benchmarks.softmax.task import SoftmaxTask
from benchmarks.vecadd.task import VecAddTask

TASKS: dict[str, Task] = {
    t.name: t
    for t in (
        MatmulTask(),
        VecAddTask(),
        ReductionTask(),
        SoftmaxTask(),
        LayerNormTask(),
        FusedElementwiseTask(),
        AttentionTask(),
    )
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name]

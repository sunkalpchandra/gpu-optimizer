"""Full-array sum reduction task: y = sum(x).

fp32 only: the ``atomic`` strategy relies on float32 atomics, and long
reductions in half precision would need a different accumulation contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from benchmarks.base import Task
from compiler.ir import ProgramGraph
from compiler.transformations.space import Config, ParamSpace, ParamSpec


class ReductionTask(Task):
    name = "reduction"
    supported_dtypes = ("float32",)

    def default_shapes(self) -> list[tuple[int, ...]]:
        return [(1 << 20,), (1 << 24,), (1 << 26,)]

    def make_inputs(self, shape: Sequence[int], dtype: str, device: str,
                    seed: int = 0) -> tuple[torch.Tensor, ...]:
        (n,) = shape
        # Scale keeps the total O(sqrt(N)·scale) so fp32 accumulation stays exact-ish.
        return (self._gen((n,), dtype, device, seed, scale=1.0),)

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        (x,) = inputs
        return x.double().sum().to(x.dtype).reshape(1)

    def baseline(self, *inputs: torch.Tensor) -> torch.Tensor:
        (x,) = inputs
        return x.sum().reshape(1)

    def param_space(self, shape: Sequence[int]) -> ParamSpace:
        return ParamSpace(
            [
                ParamSpec("BLOCK_SIZE", choices=(256, 512, 1024, 2048, 4096, 8192)),
                ParamSpec("num_warps", choices=(1, 2, 4, 8, 16)),
                ParamSpec("strategy", choices=("loop", "atomic", "two_pass")),
                ParamSpec("dtype", choices=("float32",)),
            ],
            name=f"{self.name}-{tuple(shape)}",
        )

    def default_config(self, shape: Sequence[int]) -> Config:
        return {"BLOCK_SIZE": 1024, "num_warps": 4, "strategy": "two_pass",
                "dtype": "float32"}

    def flops(self, shape: Sequence[int]) -> float:
        return float(shape[0])

    def bytes_moved(self, shape: Sequence[int], dtype: str,
                    config: Config | None = None) -> float:
        (n,) = shape
        base = 4.0 * n  # read everything once
        if config is not None and config.get("strategy") == "two_pass":
            block = int(config.get("BLOCK_SIZE", 1024))
            partials = max(1, (n + block - 1) // block)
            base += 2 * 4.0 * partials  # write + re-read partial sums
        return base

    def graph(self, shape: Sequence[int], dtype: str = "float32") -> ProgramGraph:
        (n,) = shape
        g = ProgramGraph(f"reduction_{n}")
        x = g.input((n,), dtype)
        s = g.add("reduce_sum", (1,), inputs=(x,), dtype=dtype, reduction_axes=(0,))
        g.add("output", (1,), inputs=(s,), dtype=dtype)
        return g

    def tolerance(self, dtype: str) -> tuple[float, float]:
        # Summing 2^26 values: order-dependent rounding accumulates.
        return (1e-3, 1e-2)

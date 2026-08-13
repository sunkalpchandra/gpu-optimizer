"""Matrix multiplication task: C = A @ B."""

from __future__ import annotations

from typing import Sequence

import torch

from benchmarks.base import Task
from compiler.ir import ProgramGraph
from compiler.transformations.space import Config, ParamSpace, ParamSpec

_DSIZE = {"float32": 4, "tf32": 4, "float16": 2, "bfloat16": 2}


class MatmulTask(Task):
    name = "matmul"
    supported_dtypes = ("float32", "tf32", "float16", "bfloat16")

    def default_shapes(self) -> list[tuple[int, ...]]:
        return [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]

    def make_inputs(self, shape: Sequence[int], dtype: str, device: str,
                    seed: int = 0) -> tuple[torch.Tensor, ...]:
        m, n, k = shape
        # Scale so C entries stay O(1) regardless of K → stable tolerances.
        s = float(k) ** -0.25
        a = self._gen((m, k), dtype, device, seed, scale=s)
        b = self._gen((k, n), dtype, device, seed + 1, scale=s)
        return a, b

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b = inputs
        return (a.float() @ b.float()).to(a.dtype)

    def baseline(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b = inputs
        return a @ b  # cuBLAS / native torch kernel in storage dtype

    def param_space(self, shape: Sequence[int]) -> ParamSpace:
        return ParamSpace(
            [
                ParamSpec("BLOCK_M", choices=(16, 32, 64, 128)),
                ParamSpec("BLOCK_N", choices=(16, 32, 64, 128)),
                ParamSpec("BLOCK_K", choices=(16, 32, 64)),
                ParamSpec("GROUP_M", choices=(1, 4, 8)),
                ParamSpec("num_warps", choices=(2, 4, 8)),
                ParamSpec("num_stages", choices=(2, 3, 4, 5)),
                ParamSpec("dtype", choices=("float32", "tf32", "float16", "bfloat16")),
            ],
            name=f"{self.name}-{tuple(shape)}",
        )

    def default_config(self, shape: Sequence[int]) -> Config:
        return {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 1,
                "num_warps": 4, "num_stages": 2, "dtype": "float32"}

    def flops(self, shape: Sequence[int]) -> float:
        m, n, k = shape
        return 2.0 * m * n * k

    def bytes_moved(self, shape: Sequence[int], dtype: str,
                    config: Config | None = None) -> float:
        m, n, k = shape
        return float(_DSIZE[dtype]) * (m * k + k * n + m * n)

    def graph(self, shape: Sequence[int], dtype: str = "float32") -> ProgramGraph:
        m, n, k = shape
        ir_dtype = "float32" if dtype == "tf32" else dtype
        g = ProgramGraph(f"matmul_{m}x{n}x{k}")
        a = g.input((m, k), ir_dtype)
        b = g.input((k, n), ir_dtype)
        c = g.add("matmul", (m, n), inputs=(a, b), dtype=ir_dtype, reduction_axes=(1,))
        g.add("output", (m, n), inputs=(c,), dtype=ir_dtype)
        return g

    def tolerance(self, dtype: str) -> tuple[float, float]:
        # Long-K accumulation: slightly looser than elementwise defaults.
        return {"float32": (1e-4, 1e-4), "tf32": (2e-2, 2e-3),
                "float16": (2e-2, 2e-3), "bfloat16": (5e-2, 1e-2)}[dtype]

"""Row-wise softmax task: y[i] = softmax(x[i])."""

from __future__ import annotations

from typing import Sequence

import torch

from benchmarks.base import Task
from compiler.ir import ProgramGraph
from compiler.transformations.space import Config, ParamSpace, ParamSpec

_DSIZE = {"float32": 4, "float16": 2, "bfloat16": 2}


class SoftmaxTask(Task):
    name = "softmax"
    supported_dtypes = ("float32", "float16", "bfloat16")

    def default_shapes(self) -> list[tuple[int, ...]]:
        return [(4096, 1024), (8192, 2048), (2048, 8192)]

    def make_inputs(self, shape: Sequence[int], dtype: str, device: str,
                    seed: int = 0) -> tuple[torch.Tensor, ...]:
        m, n = shape
        return (self._gen((m, n), dtype, device, seed),)

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        (x,) = inputs
        return torch.softmax(x.float(), dim=-1).to(x.dtype)

    def baseline(self, *inputs: torch.Tensor) -> torch.Tensor:
        (x,) = inputs
        return torch.softmax(x, dim=-1)

    def param_space(self, shape: Sequence[int]) -> ParamSpace:
        return ParamSpace(
            [
                ParamSpec("BLOCK_N", choices=(128, 256, 512, 1024, 2048, 4096)),
                ParamSpec("num_warps", choices=(1, 2, 4, 8, 16)),
                ParamSpec("num_stages", choices=(1, 2, 3, 4)),
                ParamSpec("dtype", choices=("float32", "float16", "bfloat16")),
            ],
            name=f"{self.name}-{tuple(shape)}",
        )

    def default_config(self, shape: Sequence[int]) -> Config:
        return {"BLOCK_N": 1024, "num_warps": 4, "num_stages": 2, "dtype": "float32"}

    def flops(self, shape: Sequence[int]) -> float:
        m, n = shape
        return 5.0 * m * n  # max, sub, exp, sum, div

    def bytes_moved(self, shape: Sequence[int], dtype: str,
                    config: Config | None = None) -> float:
        m, n = shape
        passes = 1.0
        if config is not None and int(config.get("BLOCK_N", n)) < n:
            passes = 2.0  # online two-pass when the row doesn't fit one tile
        return (passes + 1.0) * m * n * _DSIZE[dtype]

    def graph(self, shape: Sequence[int], dtype: str = "float32") -> ProgramGraph:
        m, n = shape
        g = ProgramGraph(f"softmax_{m}x{n}")
        x = g.input((m, n), dtype)
        mx = g.add("reduce_max", (m, 1), inputs=(x,), dtype=dtype, reduction_axes=(1,))
        sh = g.add("sub", (m, n), inputs=(x, mx), dtype=dtype)
        ex = g.add("exp", (m, n), inputs=(sh,), dtype=dtype)
        sm = g.add("reduce_sum", (m, 1), inputs=(ex,), dtype=dtype, reduction_axes=(1,))
        y = g.add("div", (m, n), inputs=(ex, sm), dtype=dtype)
        g.add("output", (m, n), inputs=(y,), dtype=dtype)
        return g

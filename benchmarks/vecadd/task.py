"""Vector addition task: C = A + B."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from benchmarks.base import Task
from compiler.ir import ProgramGraph
from compiler.transformations.space import Config, ParamSpace, ParamSpec

_DSIZE = {"float32": 4, "float16": 2, "bfloat16": 2}


class VecAddTask(Task):
    name = "vecadd"
    supported_dtypes = ("float32", "float16", "bfloat16")

    def default_shapes(self) -> list[tuple[int, ...]]:
        return [(1 << 20,), (1 << 24,), (1 << 26,)]

    def make_inputs(self, shape: Sequence[int], dtype: str, device: str,
                    seed: int = 0) -> tuple[torch.Tensor, ...]:
        (n,) = shape
        return (self._gen((n,), dtype, device, seed),
                self._gen((n,), dtype, device, seed + 1))

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b = inputs
        return a + b

    def param_space(self, shape: Sequence[int]) -> ParamSpace:
        return ParamSpace(
            [
                ParamSpec("BLOCK_SIZE", choices=(128, 256, 512, 1024, 2048, 4096)),
                ParamSpec("num_warps", choices=(1, 2, 4, 8)),
                ParamSpec("dtype", choices=("float32", "float16", "bfloat16")),
            ],
            name=f"{self.name}-{tuple(shape)}",
        )

    def default_config(self, shape: Sequence[int]) -> Config:
        return {"BLOCK_SIZE": 1024, "num_warps": 4, "dtype": "float32"}

    def flops(self, shape: Sequence[int]) -> float:
        return float(shape[0])

    def bytes_moved(self, shape: Sequence[int], dtype: str,
                    config: Config | None = None) -> float:
        return 3.0 * shape[0] * _DSIZE[dtype]

    def graph(self, shape: Sequence[int], dtype: str = "float32") -> ProgramGraph:
        (n,) = shape
        g = ProgramGraph(f"vecadd_{n}")
        a = g.input((n,), dtype)
        b = g.input((n,), dtype)
        c = g.add("add", (n,), inputs=(a, b), dtype=dtype)
        g.add("output", (n,), inputs=(c,), dtype=dtype)
        return g

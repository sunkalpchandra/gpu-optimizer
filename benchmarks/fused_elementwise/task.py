"""Fused elementwise task: y = relu(x * scale + bias).

The ``strategy`` parameter exposes operation fusion to the optimizer:
``fused`` runs one kernel; ``unfused`` launches mul / add / relu separately
(three round trips to DRAM), matching what a naive op-by-op executor does.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from benchmarks.base import Task
from compiler.ir import ProgramGraph
from compiler.transformations.space import Config, ParamSpace, ParamSpec

_DSIZE = {"float32": 4, "float16": 2, "bfloat16": 2}
_SCALE = 0.7071


class FusedElementwiseTask(Task):
    name = "fused_elementwise"
    supported_dtypes = ("float32", "float16", "bfloat16")

    def default_shapes(self) -> list[tuple[int, ...]]:
        return [(1 << 22,), (1 << 24,), (1 << 26,)]

    def make_inputs(self, shape: Sequence[int], dtype: str, device: str,
                    seed: int = 0) -> tuple[torch.Tensor, ...]:
        (n,) = shape
        x = self._gen((n,), dtype, device, seed)
        bias = self._gen((n,), dtype, device, seed + 1, scale=0.5)
        return x, bias

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        x, bias = inputs
        return torch.relu(x.float() * _SCALE + bias.float()).to(x.dtype)

    def baseline(self, *inputs: torch.Tensor) -> torch.Tensor:
        x, bias = inputs
        return torch.relu(x * _SCALE + bias)

    @property
    def scale(self) -> float:
        return _SCALE

    def param_space(self, shape: Sequence[int]) -> ParamSpace:
        return ParamSpace(
            [
                ParamSpec("BLOCK_SIZE", choices=(128, 256, 512, 1024, 2048, 4096)),
                ParamSpec("num_warps", choices=(1, 2, 4, 8)),
                ParamSpec("strategy", choices=("fused", "unfused")),
                ParamSpec("dtype", choices=("float32", "float16", "bfloat16")),
            ],
            name=f"{self.name}-{tuple(shape)}",
        )

    def default_config(self, shape: Sequence[int]) -> Config:
        return {"BLOCK_SIZE": 1024, "num_warps": 4, "strategy": "unfused",
                "dtype": "float32"}

    def flops(self, shape: Sequence[int]) -> float:
        return 3.0 * shape[0]

    def bytes_moved(self, shape: Sequence[int], dtype: str,
                    config: Config | None = None) -> float:
        (n,) = shape
        d = _DSIZE[dtype]
        if config is not None and config.get("strategy") == "unfused":
            # mul: r+w, add: 2r+w, relu: r+w  → 7 array traversals
            return 7.0 * n * d
        return 3.0 * n * d  # x + bias in, y out

    def graph(self, shape: Sequence[int], dtype: str = "float32") -> ProgramGraph:
        (n,) = shape
        g = ProgramGraph(f"fused_elementwise_{n}")
        x = g.input((n,), dtype)
        b = g.input((n,), dtype)
        xs = g.add("scale", (n,), inputs=(x,), dtype=dtype, factor=_SCALE)
        xb = g.add("add", (n,), inputs=(xs, b), dtype=dtype)
        y = g.add("relu", (n,), inputs=(xb,), dtype=dtype)
        g.add("output", (n,), inputs=(y,), dtype=dtype)
        return g

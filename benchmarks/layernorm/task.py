"""LayerNorm task: y = LayerNorm(x) * weight + bias over the last dim."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from benchmarks.base import Task
from compiler.ir import ProgramGraph
from compiler.transformations.space import Config, ParamSpace, ParamSpec

_DSIZE = {"float32": 4, "float16": 2, "bfloat16": 2}
_EPS = 1e-5


class LayerNormTask(Task):
    name = "layernorm"
    supported_dtypes = ("float32", "float16", "bfloat16")

    def default_shapes(self) -> list[tuple[int, ...]]:
        return [(4096, 1024), (8192, 2048), (16384, 512)]

    def make_inputs(self, shape: Sequence[int], dtype: str, device: str,
                    seed: int = 0) -> tuple[torch.Tensor, ...]:
        m, n = shape
        x = self._gen((m, n), dtype, device, seed)
        w = self._gen((n,), dtype, device, seed + 1, scale=0.5)
        b = self._gen((n,), dtype, device, seed + 2, scale=0.5)
        return x, w, b

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        x, w, b = inputs
        y = F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), _EPS)
        return y.to(x.dtype)

    def baseline(self, *inputs: torch.Tensor) -> torch.Tensor:
        x, w, b = inputs
        return F.layer_norm(x, (x.shape[-1],), w, b, _EPS)

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
        return 8.0 * m * n  # mean, var, normalize, affine

    def bytes_moved(self, shape: Sequence[int], dtype: str,
                    config: Config | None = None) -> float:
        m, n = shape
        passes = 1.0
        if config is not None and int(config.get("BLOCK_N", n)) < n:
            passes = 2.0  # Welford pass + normalize pass when the row is tiled
        return (passes + 1.0) * m * n * _DSIZE[dtype] + 2 * n * _DSIZE[dtype]

    def graph(self, shape: Sequence[int], dtype: str = "float32") -> ProgramGraph:
        m, n = shape
        g = ProgramGraph(f"layernorm_{m}x{n}")
        x = g.input((m, n), dtype)
        w = g.input((n,), dtype)
        b = g.input((n,), dtype)
        mu = g.add("reduce_mean", (m, 1), inputs=(x,), dtype=dtype, reduction_axes=(1,))
        xc = g.add("sub", (m, n), inputs=(x, mu), dtype=dtype)
        sq = g.add("mul", (m, n), inputs=(xc, xc), dtype=dtype)
        var = g.add("reduce_mean", (m, 1), inputs=(sq,), dtype=dtype, reduction_axes=(1,))
        rstd = g.add("rsqrt", (m, 1), inputs=(var,), dtype=dtype)
        xn = g.add("mul", (m, n), inputs=(xc, rstd), dtype=dtype)
        xw = g.add("mul", (m, n), inputs=(xn, w), dtype=dtype)
        y = g.add("add", (m, n), inputs=(xw, b), dtype=dtype)
        g.add("output", (m, n), inputs=(y,), dtype=dtype)
        return g

    def tolerance(self, dtype: str) -> tuple[float, float]:
        return {"float32": (1e-4, 1e-4), "float16": (2e-2, 2e-3),
                "bfloat16": (5e-2, 1e-2)}[dtype]

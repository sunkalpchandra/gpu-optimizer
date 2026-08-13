"""Simplified scaled-dot-product attention: O = softmax(Q Kᵀ / √D) V.

Non-causal, batch and heads folded into the leading dimension.  The candidate
kernel is a Flash-Attention-style single pass with online softmax; the
reference is plain PyTorch, the baseline is
``torch.nn.functional.scaled_dot_product_attention``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from benchmarks.base import Task
from compiler.ir import ProgramGraph
from compiler.transformations.space import Config, ParamSpace, ParamSpec

_DSIZE = {"float32": 4, "float16": 2, "bfloat16": 2}


class AttentionTask(Task):
    name = "attention"
    supported_dtypes = ("float32", "float16", "bfloat16")

    def default_shapes(self) -> list[tuple[int, ...]]:
        # (BH, S, D): batch*heads, sequence length, head dim
        return [(16, 1024, 64), (32, 2048, 64), (16, 4096, 128)]

    def make_inputs(self, shape: Sequence[int], dtype: str, device: str,
                    seed: int = 0) -> tuple[torch.Tensor, ...]:
        bh, s, d = shape
        q = self._gen((bh, s, d), dtype, device, seed)
        k = self._gen((bh, s, d), dtype, device, seed + 1)
        v = self._gen((bh, s, d), dtype, device, seed + 2)
        return q, k, v

    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        q, k, v = inputs
        qf, kf, vf = q.float(), k.float(), v.float()
        scores = qf @ kf.transpose(-2, -1) / math.sqrt(q.shape[-1])
        return (torch.softmax(scores, dim=-1) @ vf).to(q.dtype)

    def baseline(self, *inputs: torch.Tensor) -> torch.Tensor:
        q, k, v = inputs
        return F.scaled_dot_product_attention(q, k, v)

    def param_space(self, shape: Sequence[int]) -> ParamSpace:
        return ParamSpace(
            [
                ParamSpec("BLOCK_M", choices=(16, 32, 64, 128)),
                ParamSpec("BLOCK_N", choices=(16, 32, 64, 128)),
                ParamSpec("num_warps", choices=(2, 4, 8)),
                ParamSpec("num_stages", choices=(1, 2, 3)),
                ParamSpec("dtype", choices=("float32", "float16", "bfloat16")),
            ],
            name=f"{self.name}-{tuple(shape)}",
        )

    def default_config(self, shape: Sequence[int]) -> Config:
        return {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1,
                "dtype": "float32"}

    def flops(self, shape: Sequence[int]) -> float:
        bh, s, d = shape
        return 4.0 * bh * s * s * d + 5.0 * bh * s * s

    def bytes_moved(self, shape: Sequence[int], dtype: str,
                    config: Config | None = None) -> float:
        bh, s, d = shape
        return 4.0 * bh * s * d * _DSIZE[dtype]  # Q, K, V in; O out (flash-style)

    def graph(self, shape: Sequence[int], dtype: str = "float32") -> ProgramGraph:
        bh, s, d = shape
        g = ProgramGraph(f"attention_{bh}x{s}x{d}")
        q = g.input((bh, s, d), dtype)
        k = g.input((bh, s, d), dtype)
        v = g.input((bh, s, d), dtype)
        qk = g.add("matmul", (bh, s, s), inputs=(q, k), dtype=dtype, reduction_axes=(2,))
        sc = g.add("scale", (bh, s, s), inputs=(qk,), dtype=dtype,
                   factor=1.0 / math.sqrt(d))
        mx = g.add("reduce_max", (bh, s, 1), inputs=(sc,), dtype=dtype, reduction_axes=(2,))
        sh = g.add("sub", (bh, s, s), inputs=(sc, mx), dtype=dtype)
        ex = g.add("exp", (bh, s, s), inputs=(sh,), dtype=dtype)
        sm = g.add("reduce_sum", (bh, s, 1), inputs=(ex,), dtype=dtype, reduction_axes=(2,))
        p = g.add("div", (bh, s, s), inputs=(ex, sm), dtype=dtype)
        o = g.add("matmul", (bh, s, d), inputs=(p, v), dtype=dtype, reduction_axes=(2,))
        g.add("output", (bh, s, d), inputs=(o,), dtype=dtype)
        return g

    def tolerance(self, dtype: str) -> tuple[float, float]:
        return {"float32": (1e-3, 1e-3), "float16": (2e-2, 2e-3),
                "bfloat16": (5e-2, 1e-2)}[dtype]

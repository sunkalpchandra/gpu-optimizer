"""Builds executable callables from candidate configurations.

``build_callable`` maps (task, shape, config, inputs) → a zero-argument
function that launches the configured Triton kernel(s) and returns the output
tensor.  The first invocation triggers Triton JIT compilation; compilation
failures surface as :class:`KernelBuildError` with the compiler log attached.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from compiler.transformations.space import Config
from compiler.triton import kernels as K

TRITON_AVAILABLE = K.TRITON_AVAILABLE


class KernelBuildError(RuntimeError):
    """Raised when a candidate cannot be compiled/launched; log attached."""

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(message)
        self.log = log


@dataclass
class KernelCallable:
    """A ready-to-run candidate kernel."""

    fn: Callable[[], torch.Tensor]
    description: str
    launch: dict = field(default_factory=dict)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _require_triton() -> None:
    if not TRITON_AVAILABLE:
        raise KernelBuildError(
            "Triton is not available on this machine; real kernel execution "
            "requires Linux + CUDA + `pip install triton`."
        )


def build_callable(task: str, shape: Sequence[int], config: Config,
                   inputs: tuple[torch.Tensor, ...]) -> KernelCallable:
    """Construct the launcher for one candidate. Raises KernelBuildError."""
    _require_triton()
    builders = {
        "matmul": _build_matmul,
        "vecadd": _build_vecadd,
        "reduction": _build_reduction,
        "softmax": _build_softmax,
        "layernorm": _build_layernorm,
        "fused_elementwise": _build_fused_elementwise,
        "attention": _build_attention,
    }
    if task not in builders:
        raise KernelBuildError(f"no Triton builder for task {task!r}")
    try:
        return builders[task](tuple(int(s) for s in shape), config, inputs)
    except KernelBuildError:
        raise
    except Exception as e:  # config-dependent launch construction errors
        raise KernelBuildError(f"builder failed: {e}", log=str(e)) from e


def warm_compile(kc: KernelCallable) -> float:
    """Run once to force JIT compilation; returns compile+first-run seconds.

    Compilation errors (including shared-memory OutOfResources) are raised as
    :class:`KernelBuildError`.
    """
    import time

    t0 = time.perf_counter()
    try:
        kc.fn()
        torch.cuda.synchronize()
    except Exception as e:
        raise KernelBuildError(f"compilation/first launch failed: {e}", log=str(e)) from e
    return time.perf_counter() - t0


# ---------------------------------------------------------------- builders

def _build_matmul(shape, config, inputs) -> KernelCallable:
    a, b = inputs
    m, n, k = shape
    bm, bn, bk = int(config["BLOCK_M"]), int(config["BLOCK_N"]), int(config["BLOCK_K"])
    gm = int(config["GROUP_M"])
    warps, stages = int(config["num_warps"]), int(config["num_stages"])
    allow_tf32 = config.get("dtype") == "tf32"
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    grid = (_cdiv(m, bm) * _cdiv(n, bn),)

    def fn() -> torch.Tensor:
        K.matmul_kernel[grid](
            a, b, out, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            ALLOW_TF32=allow_tf32,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
            num_warps=warps, num_stages=stages,
        )
        return out

    return KernelCallable(fn, "triton matmul (grouped)", {"grid": list(grid), **config})


def _build_vecadd(shape, config, inputs) -> KernelCallable:
    x, y = inputs
    (n,) = shape
    bs, warps = int(config["BLOCK_SIZE"]), int(config["num_warps"])
    out = torch.empty_like(x)
    grid = (_cdiv(n, bs),)

    def fn() -> torch.Tensor:
        K.vecadd_kernel[grid](x, y, out, n, BLOCK_SIZE=bs, num_warps=warps)
        return out

    return KernelCallable(fn, "triton vecadd", {"grid": list(grid), **config})


def _build_reduction(shape, config, inputs) -> KernelCallable:
    (x,) = inputs
    (n,) = shape
    bs, warps = int(config["BLOCK_SIZE"]), int(config["num_warps"])
    strategy = str(config["strategy"])
    out = torch.empty(1, dtype=torch.float32, device=x.device)

    if strategy == "loop":
        def fn() -> torch.Tensor:
            K.reduce_loop_kernel[(1,)](x, out, n, BLOCK_SIZE=bs, num_warps=warps)
            return out
    elif strategy == "atomic":
        def fn() -> torch.Tensor:
            out.zero_()
            K.reduce_atomic_kernel[(_cdiv(n, bs),)](x, out, n, BLOCK_SIZE=bs,
                                                    num_warps=warps)
            return out
    elif strategy == "two_pass":
        num_blocks = _cdiv(n, bs)
        partials = torch.empty(num_blocks, dtype=torch.float32, device=x.device)

        def fn() -> torch.Tensor:
            K.reduce_partial_kernel[(num_blocks,)](x, partials, n, BLOCK_SIZE=bs,
                                                   num_warps=warps)
            K.reduce_loop_kernel[(1,)](partials, out, num_blocks, BLOCK_SIZE=bs,
                                       num_warps=warps)
            return out
    else:
        raise KernelBuildError(f"unknown reduction strategy {strategy!r}")

    return KernelCallable(fn, f"triton reduction ({strategy})", dict(config))


def _build_softmax(shape, config, inputs) -> KernelCallable:
    (x,) = inputs
    m, n = shape
    bn = int(config["BLOCK_N"])
    warps, stages = int(config["num_warps"]), int(config["num_stages"])
    single = bn >= n
    xc = x.contiguous()
    out = torch.empty_like(xc)

    def fn() -> torch.Tensor:
        K.softmax_kernel[(m,)](xc, out, m, n, xc.stride(0),
                               BLOCK_N=bn, SINGLE_TILE=single,
                               num_warps=warps, num_stages=stages)
        return out

    return KernelCallable(fn, f"triton softmax ({'single-tile' if single else 'online'})",
                          {"SINGLE_TILE": single, **config})


def _build_layernorm(shape, config, inputs) -> KernelCallable:
    x, w, b = inputs
    m, n = shape
    bn = int(config["BLOCK_N"])
    warps, stages = int(config["num_warps"]), int(config["num_stages"])
    single = bn >= n
    xc = x.contiguous()
    out = torch.empty_like(xc)

    def fn() -> torch.Tensor:
        K.layernorm_kernel[(m,)](xc, w, b, out, m, n, xc.stride(0), 1e-5,
                                 BLOCK_N=bn, SINGLE_TILE=single,
                                 num_warps=warps, num_stages=stages)
        return out

    return KernelCallable(fn, f"triton layernorm ({'single-tile' if single else 'two-pass'})",
                          {"SINGLE_TILE": single, **config})


def _build_fused_elementwise(shape, config, inputs) -> KernelCallable:
    x, bias = inputs
    (n,) = shape
    bs, warps = int(config["BLOCK_SIZE"]), int(config["num_warps"])
    strategy = str(config["strategy"])
    from benchmarks.fused_elementwise.task import _SCALE as scale
    out = torch.empty_like(x)
    grid = (_cdiv(n, bs),)

    if strategy == "fused":
        def fn() -> torch.Tensor:
            K.fused_scale_bias_relu_kernel[grid](x, bias, out, n, scale,
                                                 BLOCK_SIZE=bs, num_warps=warps)
            return out
    else:  # unfused: three kernels, three DRAM round trips
        t1 = torch.empty_like(x)
        t2 = torch.empty_like(x)

        def fn() -> torch.Tensor:
            K.scale_kernel[grid](x, t1, n, scale, BLOCK_SIZE=bs, num_warps=warps)
            K.vecadd_kernel[grid](t1, bias, t2, n, BLOCK_SIZE=bs, num_warps=warps)
            K.relu_kernel[grid](t2, out, n, BLOCK_SIZE=bs, num_warps=warps)
            return out

    return KernelCallable(fn, f"triton elementwise ({strategy})", dict(config))


def _build_attention(shape, config, inputs) -> KernelCallable:
    q, k, v = inputs
    bh, s, d = shape
    if d & (d - 1) or d < 16:
        raise KernelBuildError(f"head dim {d} must be a power of two >= 16")
    bm, bn = int(config["BLOCK_M"]), int(config["BLOCK_N"])
    warps, stages = int(config["num_warps"]), int(config["num_stages"])
    allow_tf32 = False  # fp32 stays IEEE; fp16/bf16 use tensor cores natively
    sm_scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q)
    grid = (_cdiv(s, bm), bh)

    def fn() -> torch.Tensor:
        K.attention_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            s, sm_scale,
            ALLOW_TF32=allow_tf32,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=d,
            num_warps=warps, num_stages=stages,
        )
        return out

    return KernelCallable(fn, "triton flash-attention (non-causal)",
                          {"grid": list(grid), "BLOCK_D": d, **config})


# -------------------------------------------------------------- source view

_KERNEL_FOR_TASK = {
    "matmul": "matmul_kernel",
    "vecadd": "vecadd_kernel",
    "reduction": {"loop": "reduce_loop_kernel", "atomic": "reduce_atomic_kernel",
                  "two_pass": "reduce_partial_kernel"},
    "softmax": "softmax_kernel",
    "layernorm": "layernorm_kernel",
    "fused_elementwise": {"fused": "fused_scale_bias_relu_kernel",
                          "unfused": "scale_kernel"},
    "attention": "attention_kernel",
}


def render_source(task: str, config: Config) -> str:
    """Kernel source + resolved meta-parameters, for inspection/UI display.

    Works without Triton installed by reading this package's kernel file.
    """
    import re
    from pathlib import Path

    entry = _KERNEL_FOR_TASK.get(task)
    if isinstance(entry, dict):
        entry = entry.get(str(config.get("strategy", "")), next(iter(entry.values())))
    if entry is None:
        return f"# no kernel registered for task {task!r}"

    text = (Path(__file__).parent / "kernels.py").read_text()
    match = re.search(rf"(@triton\.jit\n    def {entry}\(.*?)(?=\n    @triton\.jit|\Z)",
                      text, re.DOTALL)
    body = match.group(1) if match else f"# source for {entry} not found"
    # De-indent the class-level nesting for display
    body = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in body.splitlines())
    header = "\n".join(f"#   {k} = {v}" for k, v in sorted(config.items()))
    return f"# Triton kernel: {entry}\n# Candidate configuration:\n{header}\n\n{body}\n"

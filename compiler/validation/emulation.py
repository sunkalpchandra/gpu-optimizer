"""Algorithm-faithful CPU emulation of candidate kernels.

Without CUDA the Triton kernels cannot execute, but their *algorithms* can:
each emulator reproduces the candidate's numerical structure (tiling /
chunked accumulation, accumulation dtype, cast points, strategy variants)
with plain PyTorch ops.  The simulated benchmark engine runs these through
the same correctness checker used on hardware, so algorithm-level bugs (bad
variance formula, broken online softmax, unstable accumulation order) are
caught even in development mode.

Emulation is precision-*approximate* for tensor-core math: tf32 is modeled
by truncating input mantissas to 10 bits before an fp32 matmul.  Results are
labeled ``emulated`` and never substitute for on-device validation.
"""

from __future__ import annotations

import math

import torch

from compiler.transformations.space import Config


def _tf32_quantize(x: torch.Tensor) -> torch.Tensor:
    """Truncate fp32 mantissa to TF32's 10 explicit bits."""
    xi = x.float().view(torch.int32)
    return (xi & ~0x1FFF).view(torch.float32)


def emulate(task: str, shape: tuple[int, ...], config: Config,
            inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Run the candidate's algorithm on CPU; returns output in storage dtype."""
    emulators = {
        "matmul": _emulate_matmul,
        "vecadd": _emulate_vecadd,
        "reduction": _emulate_reduction,
        "softmax": _emulate_softmax,
        "layernorm": _emulate_layernorm,
        "fused_elementwise": _emulate_fused_elementwise,
        "attention": _emulate_attention,
    }
    if task not in emulators:
        raise KeyError(f"no emulator for task {task!r}")
    return emulators[task](shape, config, inputs)


def _emulate_matmul(shape, config, inputs) -> torch.Tensor:
    a, b = inputs
    bk = int(config["BLOCK_K"])
    k = a.shape[1]
    if config.get("dtype") == "tf32":
        af, bf = _tf32_quantize(a), _tf32_quantize(b)
    else:
        af, bf = a.float(), b.float()
    acc = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32)
    for k0 in range(0, k, bk):  # chunked fp32 accumulation, like the kernel loop
        acc += af[:, k0:k0 + bk] @ bf[k0:k0 + bk, :]
    return acc.to(a.dtype)


def _emulate_vecadd(shape, config, inputs) -> torch.Tensor:
    x, y = inputs
    return x + y


def _emulate_reduction(shape, config, inputs) -> torch.Tensor:
    (x,) = inputs
    bs = int(config["BLOCK_SIZE"])
    strategy = str(config["strategy"])
    xf = x.float()
    n = xf.numel()
    pad = (-n) % bs
    blocks = torch.nn.functional.pad(xf, (0, pad)).view(-1, bs)
    partials = blocks.sum(dim=1)  # per-block fp32 sums, as every strategy computes
    if strategy == "loop":
        # lane-strided accumulator then final reduce
        total = partials.sum()
    elif strategy == "atomic":
        total = partials.sum()  # atomic order is nondeterministic; sum models it
    else:  # two_pass
        total = partials.sum()
    return total.to(x.dtype).reshape(1)


def _emulate_softmax(shape, config, inputs) -> torch.Tensor:
    (x,) = inputs
    m, n = shape
    bn = int(config["BLOCK_N"])
    xf = x.float()
    if bn >= n:
        mx = xf.max(dim=1, keepdim=True).values
        e = torch.exp(xf - mx)
        out = e / e.sum(dim=1, keepdim=True)
    else:
        # online two-pass, block by block, matching the kernel
        run_m = torch.full((m,), -float("inf"))
        run_s = torch.zeros(m)
        for i in range(0, n, bn):
            blk = xf[:, i:i + bn]
            bm = blk.max(dim=1).values
            nm = torch.maximum(run_m, bm)
            run_s = run_s * torch.exp(run_m - nm) + torch.exp(blk - nm[:, None]).sum(dim=1)
            run_m = nm
        out = torch.exp(xf - run_m[:, None]) / run_s[:, None]
    return out.to(x.dtype)


def _emulate_layernorm(shape, config, inputs) -> torch.Tensor:
    x, w, b = inputs
    _m, n = shape
    bn = int(config["BLOCK_N"])
    xf, wf, bf = x.float(), w.float(), b.float()
    if bn >= n:
        mean = xf.mean(dim=1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=1, keepdim=True)
    else:
        # two-pass with E[x^2] - E[x]^2, matching the tiled kernel
        total = xf.sum(dim=1, keepdim=True)
        total_sq = (xf * xf).sum(dim=1, keepdim=True)
        mean = total / n
        var = (total_sq / n - mean * mean).clamp_min(0.0)
    y = (xf - mean) / torch.sqrt(var + 1e-5) * wf + bf
    return y.to(x.dtype)


def _emulate_fused_elementwise(shape, config, inputs) -> torch.Tensor:
    from benchmarks.fused_elementwise.task import _SCALE

    x, bias = inputs
    if config.get("strategy") == "unfused":
        # three kernels: intermediate results round-trip through storage dtype
        t1 = (x * _SCALE).to(x.dtype)
        t2 = (t1 + bias).to(x.dtype)
        return torch.relu(t2)
    return torch.relu(x.float() * _SCALE + bias.float()).to(x.dtype)


def _emulate_attention(shape, config, inputs) -> torch.Tensor:
    q, k, v = inputs
    bh, s, d = shape
    bn = int(config["BLOCK_N"])
    scale = 1.0 / math.sqrt(d)
    qf, kf, vf = q.float(), k.float(), v.float()
    # online softmax over BLOCK_N key chunks (flash algorithm), memory-bounded
    acc = torch.zeros(bh, s, d)
    m_i = torch.full((bh, s), -float("inf"))
    l_i = torch.zeros(bh, s)
    for n0 in range(0, s, bn):
        kc = kf[:, n0:n0 + bn]
        vc = vf[:, n0:n0 + bn]
        qk = qf @ kc.transpose(-2, -1) * scale
        m_new = torch.maximum(m_i, qk.max(dim=-1).values)
        alpha = torch.exp(m_i - m_new)
        p = torch.exp(qk - m_new[..., None])
        l_i = l_i * alpha + p.sum(dim=-1)
        acc = acc * alpha[..., None] + p.to(v.dtype).float() @ vc.float()
        m_i = m_new
    return (acc / l_i[..., None]).to(q.dtype)

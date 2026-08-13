"""Triton kernel implementations for every benchmark task.

These are the *actual* kernels compiled and run on CUDA hardware.  Tiling,
warp counts, pipeline stages, and precision arrive as ``tl.constexpr``
meta-parameters / launch options from the candidate configuration — the
search space edits these, never free-form source.

This module imports cleanly only where Triton is installed (Linux + CUDA).
Everything is wrapped so that machines without Triton can still import the
package; ``TRITON_AVAILABLE`` gates all use.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - platform dependent
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:

    # ------------------------------------------------------------- matmul
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        ALLOW_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """C = A @ B with grouped program ordering for L2 reuse."""
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = acc.to(c_ptr.dtype.element_ty)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    # ------------------------------------------------------------- vecadd
    @triton.jit
    def vecadd_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    # ---------------------------------------------------------- reduction
    @triton.jit
    def reduce_partial_kernel(x_ptr, partial_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """Each program sums one BLOCK_SIZE slice into partial_ptr[pid]."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs, mask=offs < n_elements, other=0.0).to(tl.float32)
        tl.store(partial_ptr + pid, tl.sum(x, axis=0))

    @triton.jit
    def reduce_atomic_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """Each program atomically adds its slice sum into out_ptr[0]."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs, mask=offs < n_elements, other=0.0).to(tl.float32)
        tl.atomic_add(out_ptr, tl.sum(x, axis=0))

    @triton.jit
    def reduce_loop_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """Single program strided loop over the whole array."""
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(0, tl.cdiv(n_elements, BLOCK_SIZE)):
            offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            acc += tl.load(x_ptr + offs, mask=offs < n_elements, other=0.0).to(tl.float32)
        tl.store(out_ptr, tl.sum(acc, axis=0))

    # ------------------------------------------------------------ softmax
    @triton.jit
    def softmax_kernel(
        x_ptr, out_ptr, M, N, stride_m,
        BLOCK_N: tl.constexpr, SINGLE_TILE: tl.constexpr,
    ):
        """Row softmax; online two-pass when the row exceeds one tile."""
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_m
        o_row = out_ptr + row * stride_m
        if SINGLE_TILE:
            offs = tl.arange(0, BLOCK_N)
            mask = offs < N
            x = tl.load(x_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
            m = tl.max(x, axis=0)
            e = tl.exp(x - m)
            s = tl.sum(e, axis=0)
            tl.store(o_row + offs, (e / s).to(out_ptr.dtype.element_ty), mask=mask)
        else:
            m = -float("inf")
            s = 0.0
            for i in range(0, tl.cdiv(N, BLOCK_N)):
                offs = i * BLOCK_N + tl.arange(0, BLOCK_N)
                x = tl.load(x_row + offs, mask=offs < N, other=-float("inf")).to(tl.float32)
                bm = tl.max(x, axis=0)
                nm = tl.maximum(m, bm)
                s = s * tl.exp(m - nm) + tl.sum(tl.exp(x - nm), axis=0)
                m = nm
            for i in range(0, tl.cdiv(N, BLOCK_N)):
                offs = i * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = offs < N
                x = tl.load(x_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
                y = tl.exp(x - m) / s
                tl.store(o_row + offs, y.to(out_ptr.dtype.element_ty), mask=mask)

    # ---------------------------------------------------------- layernorm
    @triton.jit
    def layernorm_kernel(
        x_ptr, w_ptr, b_ptr, out_ptr, M, N, stride_m, eps,
        BLOCK_N: tl.constexpr, SINGLE_TILE: tl.constexpr,
    ):
        """Row LayerNorm with affine transform; fp32 statistics."""
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_m
        o_row = out_ptr + row * stride_m
        if SINGLE_TILE:
            offs = tl.arange(0, BLOCK_N)
            mask = offs < N
            x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
            mean = tl.sum(x, axis=0) / N
            diff = tl.where(mask, x - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / N
            rstd = 1.0 / tl.sqrt(var + eps)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * w + b
            tl.store(o_row + offs, y.to(out_ptr.dtype.element_ty), mask=mask)
        else:
            total = 0.0
            total_sq = 0.0
            for i in range(0, tl.cdiv(N, BLOCK_N)):
                offs = i * BLOCK_N + tl.arange(0, BLOCK_N)
                x = tl.load(x_row + offs, mask=offs < N, other=0.0).to(tl.float32)
                total += tl.sum(x, axis=0)
                total_sq += tl.sum(x * x, axis=0)
            mean = total / N
            var = total_sq / N - mean * mean
            var = tl.maximum(var, 0.0)
            rstd = 1.0 / tl.sqrt(var + eps)
            for i in range(0, tl.cdiv(N, BLOCK_N)):
                offs = i * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = offs < N
                x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
                w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                y = (x - mean) * rstd * w + b
                tl.store(o_row + offs, y.to(out_ptr.dtype.element_ty), mask=mask)

    # -------------------------------------------------- fused elementwise
    @triton.jit
    def fused_scale_bias_relu_kernel(
        x_ptr, bias_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr
    ):
        """y = relu(x * scale + bias) in a single pass."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        b = tl.load(bias_ptr + offs, mask=mask).to(tl.float32)
        y = tl.maximum(x * scale + b, 0.0)
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def scale_kernel(x_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * scale, mask=mask)

    @triton.jit
    def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)

    # ---------------------------------------------------------- attention
    @triton.jit
    def attention_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr,
        stride_qb, stride_qs, stride_qd,
        stride_kb, stride_ks, stride_kd,
        stride_vb, stride_vs, stride_vd,
        stride_ob, stride_os, stride_od,
        S, sm_scale,
        ALLOW_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Flash-style non-causal attention with online softmax.

        Grid: (cdiv(S, BLOCK_M), batch*heads).  BLOCK_D equals the head dim.
        """
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        m_mask = offs_m[:, None] < S

        q_ptrs = (q_ptr + pid_b * stride_qb + offs_m[:, None] * stride_qs
                  + offs_d[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=m_mask, other=0.0)

        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for start_n in range(0, S, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n[:, None] < S
            k = tl.load(
                k_ptr + pid_b * stride_kb + offs_n[:, None] * stride_ks
                + offs_d[None, :] * stride_kd,
                mask=n_mask, other=0.0,
            )
            qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * sm_scale
            qk = tl.where(offs_n[None, :] < S, qk, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v = tl.load(
                v_ptr + pid_b * stride_vb + offs_n[:, None] * stride_vs
                + offs_d[None, :] * stride_vd,
                mask=n_mask, other=0.0,
            )
            acc += tl.dot(p.to(v.dtype), v, allow_tf32=ALLOW_TF32)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = (o_ptr + pid_b * stride_ob + offs_m[:, None] * stride_os
                  + offs_d[None, :] * stride_od)
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=m_mask)

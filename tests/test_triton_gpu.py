"""Real-hardware tests: compile and validate every Triton kernel on CUDA.

These run only where CUDA + Triton exist (they skip cleanly elsewhere).
On a GPU machine this is the ground-truth Stage-1 verification:
kernel → compile → run → correctness for a spread of configurations.
"""

import random

import pytest
import torch

from benchmarks import TASKS, get_task
from compiler.triton.runner import TRITON_AVAILABLE, render_source

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE),
    reason="requires CUDA + Triton",
)

GPU_SHAPES = {
    "matmul": (384, 320, 288),
    "vecadd": (1_000_003,),
    "reduction": (1_000_003,),
    "softmax": (256, 1000),
    "layernorm": (256, 1000),
    "fused_elementwise": (1_000_003,),
    "attention": (4, 512, 64),
}


@requires_gpu
@pytest.mark.parametrize("name", sorted(TASKS))
def test_kernel_correct_on_gpu_default_config(name):
    from benchmarks.harness import BenchmarkSettings, CudaBenchmarkEngine
    from hardware.gpu_info import detect_hardware

    task = get_task(name)
    shape = GPU_SHAPES[name]
    engine = CudaBenchmarkEngine(detect_hardware(),
                                 BenchmarkSettings(warmup=3, iterations=10))
    cand = task.make_candidate(shape, task.default_config(shape))
    res = engine.benchmark_candidate(task, cand)
    assert res.status == "ok", f"{name}: {res.error}"
    assert res.correct and res.correctness_mode == "device"
    assert res.latency.median_ms > 0
    assert res.engine == "cuda"


@requires_gpu
@pytest.mark.parametrize("name", sorted(TASKS))
def test_kernel_correct_on_gpu_random_configs(name):
    from benchmarks.harness import BenchmarkSettings, CudaBenchmarkEngine
    from hardware.gpu_info import detect_hardware

    task = get_task(name)
    shape = GPU_SHAPES[name]
    engine = CudaBenchmarkEngine(detect_hardware(),
                                 BenchmarkSettings(warmup=2, iterations=5))
    space = task.param_space(shape)
    rng = random.Random(0)
    ok, tried = 0, 0
    for _ in range(6):
        cand = task.make_candidate(shape, space.sample(rng), provenance="random")
        res = engine.benchmark_candidate(task, cand)
        tried += 1
        # compile errors are legitimate (resource limits); wrong answers are not
        assert res.status in ("ok", "compile_error"), f"{name}: {res.error}"
        ok += res.status == "ok"
    assert ok >= 1, f"{name}: no random config compiled ({tried} tried)"


def test_render_source_works_without_triton():
    src = render_source("matmul", {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32,
                                   "GROUP_M": 4, "num_warps": 4, "num_stages": 3,
                                   "dtype": "float16"})
    assert "@triton.jit" in src
    assert "matmul_kernel" in src
    assert "BLOCK_M = 64" in src

    src_red = render_source("reduction", {"strategy": "atomic", "BLOCK_SIZE": 1024})
    assert "reduce_atomic_kernel" in src_red

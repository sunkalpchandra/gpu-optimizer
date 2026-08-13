"""Tasks: deterministic inputs, sane analytics, and — critically — CPU
emulation of every candidate algorithm matching the trusted reference."""

import random

import pytest
import torch

from benchmarks import TASKS, get_task
from compiler.validation.correctness import check_correctness
from compiler.validation.emulation import emulate

SMALL_SHAPES = {
    "matmul": (96, 80, 72),          # deliberately not multiples of block sizes
    "vecadd": (10_007,),
    "reduction": (100_003,),
    "softmax": (37, 500),
    "layernorm": (37, 500),
    "fused_elementwise": (10_007,),
    "attention": (3, 130, 32),
}


@pytest.mark.parametrize("name", sorted(TASKS))
def test_task_basics(name):
    task = get_task(name)
    shape = SMALL_SHAPES[name]
    assert task.flops(shape) > 0
    assert task.bytes_moved(shape, "float32") > 0
    g = task.graph(shape)
    assert len(g) >= 3
    space = task.param_space(shape)
    cfg = task.default_config(shape)
    space.validate(cfg)
    assert space.size() > 10

    # deterministic inputs
    i1 = task.make_inputs(shape, "float32", "cpu", seed=7)
    i2 = task.make_inputs(shape, "float32", "cpu", seed=7)
    i3 = task.make_inputs(shape, "float32", "cpu", seed=8)
    for a, b in zip(i1, i2):
        assert torch.equal(a, b)
    assert not all(torch.equal(a, b) for a, b in zip(i1, i3))

    ref = task.reference(*i1)
    assert torch.isfinite(ref.float()).all()
    base = task.baseline(*i1)
    assert base.shape == ref.shape


@pytest.mark.parametrize("name", sorted(TASKS))
def test_emulation_matches_reference_random_configs(name):
    """Every emulated candidate algorithm must satisfy the same dtype-aware
    correctness bar we apply on hardware."""
    task = get_task(name)
    shape = SMALL_SHAPES[name]
    space = task.param_space(shape)
    rng = random.Random(123)
    for trial in range(8):
        cfg = space.sample(rng)
        dtype = str(cfg.get("dtype", "float32"))
        inputs = task.make_inputs(shape, dtype, "cpu", seed=trial)
        out = emulate(name, shape, cfg, inputs)
        ref = task.reference(*inputs)
        rtol, atol = task.tolerance(dtype)
        report = check_correctness(out, ref, rtol, atol)
        assert report.passed, (f"{name} cfg={cfg}: {report.summary()}")


def test_emulation_reduction_strategies_differ_only_numerically():
    task = get_task("reduction")
    shape = (65_537,)
    inputs = task.make_inputs(shape, "float32", "cpu", seed=0)
    outs = []
    for strategy in ("loop", "atomic", "two_pass"):
        cfg = {"BLOCK_SIZE": 1024, "num_warps": 4, "strategy": strategy,
               "dtype": "float32"}
        outs.append(emulate("reduction", shape, cfg, inputs))
    for o in outs[1:]:
        assert torch.allclose(outs[0], o, rtol=1e-4, atol=1e-4)


def test_emulation_online_softmax_matches_single_tile():
    task = get_task("softmax")
    shape = (16, 1000)
    inputs = task.make_inputs(shape, "float32", "cpu", seed=0)
    single = emulate("softmax", shape, {"BLOCK_N": 1024, "num_warps": 4,
                                        "num_stages": 2, "dtype": "float32"}, inputs)
    online = emulate("softmax", shape, {"BLOCK_N": 128, "num_warps": 4,
                                        "num_stages": 2, "dtype": "float32"}, inputs)
    assert torch.allclose(single, online, rtol=1e-5, atol=1e-6)


def test_correctness_checker_flags_bad_output():
    task = get_task("vecadd")
    shape = (1024,)
    inputs = task.make_inputs(shape, "float32", "cpu", seed=0)
    ref = task.reference(*inputs)
    bad = ref.clone()
    bad[100] += 1.0
    rtol, atol = task.tolerance("float32")
    assert not check_correctness(bad, ref, rtol, atol).passed
    assert not check_correctness(ref * float("nan"), ref, rtol, atol).passed
    assert check_correctness(ref.clone(), ref, rtol, atol).passed

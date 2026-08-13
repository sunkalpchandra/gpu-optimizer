"""Hardware-resource constraint estimates for candidate configurations.

Shared by the real runner (pre-launch sanity check) and the simulated engine
(compile-failure modeling).  Estimates intentionally mirror how Triton
allocates shared memory for software-pipelined loads: roughly
``num_stages × (bytes staged per iteration)``.
"""

from __future__ import annotations

from compiler.transformations.space import Config
from hardware.gpu_info import HardwareSpec

_DSIZE = {"float32": 4, "tf32": 4, "float16": 2, "bfloat16": 2}


def dtype_size(dtype: str) -> int:
    return _DSIZE[dtype]


def estimate_shared_mem_bytes(task: str, shape: tuple[int, ...], config: Config) -> int:
    """Approximate shared-memory footprint of one thread block."""
    d = _DSIZE[str(config.get("dtype", "float32"))]
    stages = int(config.get("num_stages", 1))
    if task == "matmul":
        bm, bn, bk = int(config["BLOCK_M"]), int(config["BLOCK_N"]), int(config["BLOCK_K"])
        return stages * (bm * bk + bk * bn) * d
    if task == "attention":
        bm, bn = int(config["BLOCK_M"]), int(config["BLOCK_N"])
        bd = int(config.get("BLOCK_D", shape[-1]))
        # K and V tiles are staged; Q tile is register/smem resident once.
        return stages * 2 * (bn * bd) * d + bm * bd * d
    if task in ("softmax", "layernorm"):
        bn = int(config["BLOCK_N"])
        return min(int(config.get("num_stages", 1)), 2) * bn * 4  # fp32 row tile
    # 1-D elementwise/reduction kernels stage a single block
    block = int(config.get("BLOCK_SIZE", 1024))
    return block * d


def exceeds_shared_mem(task: str, shape: tuple[int, ...], config: Config,
                       hw: HardwareSpec) -> bool:
    """True if the config cannot fit the target's per-SM shared memory."""
    return estimate_shared_mem_bytes(task, shape, config) > hw.shared_mem_per_sm_kb * 1024


def register_pressure_score(task: str, config: Config) -> float:
    """Heuristic 0..1+ register-pressure indicator (1 ≈ at the limit).

    Used only as a *feature* (simulator inefficiency + surrogate input),
    never as a hard reject — real compilation is the arbiter on hardware.
    """
    warps = int(config.get("num_warps", 4))
    if "BLOCK_M" in config and "BLOCK_N" in config:
        acc_elems = int(config["BLOCK_M"]) * int(config["BLOCK_N"])
        # fp32 accumulator registers per thread vs ~255 architectural limit
        per_thread = acc_elems / max(warps * 32, 1)
        return per_thread / 255.0
    block = int(config.get("BLOCK_SIZE", config.get("BLOCK_N", 1024)))
    return (block / max(warps * 32, 1)) / 255.0

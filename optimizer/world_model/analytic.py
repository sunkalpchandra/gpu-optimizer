"""Deterministic analytic GPU cost model — the *simulated* benchmark engine.

DEVELOPMENT USE ONLY.  On machines without CUDA this model stands in for the
GPU so the full search/RL/dashboard stack can be exercised end-to-end.  Every
number it produces is labeled ``simulated`` throughout the system and must
never be presented as a hardware measurement.

The model is a roofline estimate multiplied by configuration-dependent
inefficiency terms chosen to mimic well-known GPU behavior (occupancy,
tile quantization, register pressure, pipelining, precision), plus a smooth
deterministic perturbation so the response surface has task-specific local
structure that a searcher must actually discover.  Determinism: the same
(task, shape, config, hardware) always yields the same latency.
"""

from __future__ import annotations

import hashlib
import math

from benchmarks.base import Task
from compiler.transformations.constraints import (
    dtype_size,
    exceeds_shared_mem,
    register_pressure_score,
)
from compiler.transformations.space import Config
from hardware.gpu_info import HardwareSpec


def _unit_hash(*parts: object) -> float:
    """Deterministic pseudo-random float in [0, 1) from the given parts."""
    payload = "|".join(repr(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


class AnalyticGPUModel:
    """Roofline-with-inefficiencies latency model for one hardware target."""

    def __init__(self, hardware: HardwareSpec) -> None:
        self.hardware = hardware

    # ------------------------------------------------------------ compile
    def compiles(self, task: Task, shape: tuple[int, ...], config: Config) -> tuple[bool, str]:
        """Mirror the dominant real-world compile failure: shared-memory OOM."""
        if exceeds_shared_mem(task.name, shape, config, self.hardware):
            return False, ("simulated CompilationError: out of shared memory "
                           f"(config {config} exceeds "
                           f"{self.hardware.shared_mem_per_sm_kb} KB/SM)")
        if register_pressure_score(task.name, config) > 2.0:
            return False, "simulated CompilationError: register allocation failure"
        return True, ""

    # ------------------------------------------------------------ latency
    def latency_ms(self, task: Task, shape: tuple[int, ...], config: Config) -> float:
        hw = self.hardware
        dtype = str(config.get("dtype", "float32"))
        flops = task.flops(shape)
        bytes_moved = task.bytes_moved(shape, dtype, config)

        # Peak throughputs for this precision.
        if dtype in ("float16", "bfloat16"):
            peak_tflops = hw.fp16_tflops
        elif dtype == "tf32" and hw.supports_tf32:
            peak_tflops = hw.fp16_tflops / 2.0
        else:
            peak_tflops = hw.fp32_tflops

        compute_ms = flops / (peak_tflops * 1e12) * 1e3
        memory_ms = bytes_moved / (hw.memory_bandwidth_gbs * 1e9) * 1e3
        base_ms = max(compute_ms, memory_ms)

        eff = self._efficiency(task, shape, config)
        # Smooth deterministic perturbation (±8%) creates local structure.
        wiggle = 0.92 + 0.16 * _unit_hash(task.name, shape, sorted(config.items()),
                                          hw.name)
        # Fixed kernel-launch overhead floor.
        launches = 3 if config.get("strategy") == "unfused" else (
            2 if config.get("strategy") == "two_pass" else 1)
        overhead_ms = 0.004 * launches

        return base_ms / max(eff, 0.02) * wiggle + overhead_ms

    # --------------------------------------------------------- efficiency
    def _efficiency(self, task: Task, shape: tuple[int, ...], config: Config) -> float:
        """0..1 multiplier modeling how well the config uses the machine."""
        hw = self.hardware
        eff = 0.9

        # --- occupancy: enough thread blocks to fill all SMs
        blocks = self._grid_size(task.name, shape, config)
        waves = blocks / max(hw.sm_count, 1)
        if waves < 1.0:
            eff *= 0.25 + 0.65 * waves          # underfilled GPU
        else:
            # tail effect: partial last wave hurts small grids
            frac = waves - math.floor(waves)
            if waves < 8 and frac > 0:
                eff *= 1.0 - 0.25 * (1.0 - frac) / max(waves, 1.0)

        # --- thread count per block vs tile size
        warps = int(config.get("num_warps", 4))
        threads = warps * hw.warp_size
        tile = self._tile_elems(task.name, shape, config)
        elems_per_thread = tile / max(threads, 1)
        if elems_per_thread < 1.0:
            eff *= 0.55 + 0.45 * elems_per_thread   # more threads than work
        elif elems_per_thread > 256:
            eff *= 256.0 / elems_per_thread          # ILP can't cover latency

        # --- register pressure: spilling cliff
        rp = register_pressure_score(task.name, config)
        if rp > 1.0:
            eff *= max(0.15, 1.0 - 0.7 * (rp - 1.0))

        # --- software pipelining: matmul-like kernels want >=3 stages
        stages = int(config.get("num_stages", 1))
        if task.name in ("matmul", "attention"):
            eff *= {1: 0.55, 2: 0.75, 3: 0.92, 4: 1.0, 5: 0.98}.get(stages, 0.9)
            # tl.dot efficiency: tensor-core tiles want BLOCK_K >= 32
            bk = int(config.get("BLOCK_K", 32))
            if bk < 32:
                eff *= 0.8
            # L2 reuse via grouped ordering (matmul only)
            if task.name == "matmul":
                gm = int(config.get("GROUP_M", 1))
                eff *= {1: 0.82, 4: 0.96, 8: 1.0}.get(gm, 0.9)

        # --- tile quantization: padded work on the boundary
        eff *= self._quantization_factor(task.name, shape, config)

        # --- memory coalescing proxy for 1-D kernels: tiny blocks hurt
        if "BLOCK_SIZE" in config:
            bs = int(config["BLOCK_SIZE"])
            if bs < 256:
                eff *= 0.75 + 0.25 * bs / 256.0

        # --- strategy-specific effects
        strategy = str(config.get("strategy", ""))
        if strategy == "loop":
            eff *= 0.02   # single thread block: catastrophic serialization
        elif strategy == "atomic":
            eff *= 0.85   # contention on the single accumulator

        return min(eff, 1.0)

    # ------------------------------------------------------------ helpers
    def _grid_size(self, task: str, shape: tuple[int, ...], config: Config) -> int:
        if task == "matmul":
            m, n, _ = shape
            return math.ceil(m / int(config["BLOCK_M"])) * math.ceil(n / int(config["BLOCK_N"]))
        if task == "attention":
            bh, s, _ = shape
            return bh * math.ceil(s / int(config["BLOCK_M"]))
        if task in ("softmax", "layernorm"):
            return shape[0]
        n = shape[0]
        if config.get("strategy") == "loop":
            return 1
        return math.ceil(n / int(config.get("BLOCK_SIZE", 1024)))

    def _tile_elems(self, task: str, shape: tuple[int, ...], config: Config) -> int:
        if task == "matmul":
            return int(config["BLOCK_M"]) * int(config["BLOCK_N"])
        if task == "attention":
            return int(config["BLOCK_M"]) * shape[-1]
        if task in ("softmax", "layernorm"):
            return min(int(config["BLOCK_N"]), shape[1])
        return int(config.get("BLOCK_SIZE", 1024))

    def _quantization_factor(self, task: str, shape: tuple[int, ...],
                             config: Config) -> float:
        def util(dim: int, block: int) -> float:
            blocks = math.ceil(dim / block)
            return dim / (blocks * block)

        if task == "matmul":
            m, n, k = shape
            return (util(m, int(config["BLOCK_M"])) * util(n, int(config["BLOCK_N"]))
                    * util(k, int(config["BLOCK_K"])))
        if task == "attention":
            _, s, _ = shape
            return util(s, int(config["BLOCK_M"])) * util(s, int(config["BLOCK_N"]))
        if task in ("softmax", "layernorm"):
            n = shape[1]
            bn = int(config["BLOCK_N"])
            return util(n, bn) if bn < n else n / bn if bn > n else 1.0
        n = shape[0]
        return util(n, int(config.get("BLOCK_SIZE", 1024)))

    # -------------------------------------------------------------- memory
    def memory_bytes(self, task: Task, shape: tuple[int, ...], config: Config) -> int:
        """Rough peak device-memory footprint of one execution."""
        dtype = str(config.get("dtype", "float32"))
        d = dtype_size(dtype)
        io_elems = task.bytes_moved(shape, dtype, None) / d
        extra = 0
        if config.get("strategy") == "two_pass":
            extra = math.ceil(shape[0] / int(config.get("BLOCK_SIZE", 1024))) * 4
        if config.get("strategy") == "unfused":
            extra = int(io_elems) * d  # temporaries
        return int(io_elems * d + extra)

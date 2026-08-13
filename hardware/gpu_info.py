"""Hardware detection and representation.

Produces a :class:`HardwareSpec` describing the target GPU.  On CUDA machines
the spec is read from the driver; on machines without CUDA a spec from the
known-GPU catalog can be used to drive the *simulated* benchmark engine (all
results produced against a catalog spec are labeled ``simulated``).

The spec exposes ``feature_vector()`` — the hardware embedding input consumed
by the policy and the performance model, enabling hardware-conditional
optimization (the same program may map to different strategies on different
GPUs).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareSpec:
    """Static description of a GPU target."""

    name: str
    compute_capability: tuple[int, int]
    sm_count: int
    memory_gb: float
    memory_bandwidth_gbs: float          # peak DRAM bandwidth
    fp32_tflops: float                    # peak FP32 (non-tensor-core)
    fp16_tflops: float                    # peak FP16/BF16 tensor-core throughput
    shared_mem_per_sm_kb: int
    registers_per_sm: int
    warp_size: int = 32
    l2_cache_mb: float = 4.0
    supports_bf16: bool = True
    supports_tf32: bool = True
    is_simulated: bool = False            # True → catalog spec, not detected hardware

    def feature_vector(self) -> np.ndarray:
        """Fixed-size hardware embedding input (normalized log-scale features)."""
        cc = self.compute_capability[0] + self.compute_capability[1] / 10.0
        return np.array(
            [
                cc / 10.0,
                self.sm_count / 150.0,
                np.log2(max(self.memory_gb, 1e-3)) / 8.0,
                np.log2(max(self.memory_bandwidth_gbs, 1.0)) / 12.0,
                np.log2(max(self.fp32_tflops, 1e-2)) / 8.0,
                np.log2(max(self.fp16_tflops, 1e-2)) / 10.0,
                self.shared_mem_per_sm_kb / 256.0,
                self.registers_per_sm / 65536.0,
                self.warp_size / 32.0,
                np.log2(max(self.l2_cache_mb, 0.5)) / 7.0,
                1.0 if self.supports_bf16 else 0.0,
                1.0 if self.supports_tf32 else 0.0,
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["compute_capability"] = list(self.compute_capability)
        return d


HARDWARE_FEATURE_DIM = 12

# Catalog of well-known GPUs, used for the simulated engine and for
# hardware-transfer experiments.  Numbers are public spec-sheet values.
KNOWN_GPUS: dict[str, HardwareSpec] = {
    "A100-SXM4-40GB": HardwareSpec(
        name="NVIDIA A100-SXM4-40GB (simulated)",
        compute_capability=(8, 0), sm_count=108, memory_gb=40.0,
        memory_bandwidth_gbs=1555.0, fp32_tflops=19.5, fp16_tflops=312.0,
        shared_mem_per_sm_kb=164, registers_per_sm=65536, l2_cache_mb=40.0,
        is_simulated=True,
    ),
    "H100-SXM5-80GB": HardwareSpec(
        name="NVIDIA H100-SXM5-80GB (simulated)",
        compute_capability=(9, 0), sm_count=132, memory_gb=80.0,
        memory_bandwidth_gbs=3350.0, fp32_tflops=67.0, fp16_tflops=989.0,
        shared_mem_per_sm_kb=228, registers_per_sm=65536, l2_cache_mb=50.0,
        is_simulated=True,
    ),
    "RTX-4090": HardwareSpec(
        name="NVIDIA GeForce RTX 4090 (simulated)",
        compute_capability=(8, 9), sm_count=128, memory_gb=24.0,
        memory_bandwidth_gbs=1008.0, fp32_tflops=82.6, fp16_tflops=165.2,
        shared_mem_per_sm_kb=100, registers_per_sm=65536, l2_cache_mb=72.0,
        is_simulated=True,
    ),
    "T4": HardwareSpec(
        name="NVIDIA T4 (simulated)",
        compute_capability=(7, 5), sm_count=40, memory_gb=16.0,
        memory_bandwidth_gbs=320.0, fp32_tflops=8.1, fp16_tflops=65.0,
        shared_mem_per_sm_kb=64, registers_per_sm=65536, l2_cache_mb=4.0,
        supports_bf16=False, supports_tf32=False, is_simulated=True,
    ),
    "V100-SXM2-16GB": HardwareSpec(
        name="NVIDIA V100-SXM2-16GB (simulated)",
        compute_capability=(7, 0), sm_count=80, memory_gb=16.0,
        memory_bandwidth_gbs=900.0, fp32_tflops=15.7, fp16_tflops=125.0,
        shared_mem_per_sm_kb=96, registers_per_sm=65536, l2_cache_mb=6.0,
        supports_bf16=False, supports_tf32=False, is_simulated=True,
    ),
}

DEFAULT_SIMULATED_GPU = "A100-SXM4-40GB"


@dataclass
class EnvironmentReport:
    """What the current machine actually provides.  Printed at startup."""

    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    compute_capability: tuple[int, int] | None
    triton_available: bool
    triton_version: str | None
    device_count: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Python {self.python_version}",
            f"PyTorch {self.torch_version} | CUDA available: {self.cuda_available}"
            + (f" (CUDA {self.cuda_version})" if self.cuda_version else ""),
            f"Triton: {'v' + self.triton_version if self.triton_available else 'not available'}",
        ]
        if self.cuda_available and self.gpu_name:
            cc = self.compute_capability
            lines.append(f"GPU: {self.gpu_name} (sm_{cc[0]}{cc[1]}, {self.device_count} device(s))")
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)


def detect_environment() -> EnvironmentReport:
    """Probe Python/PyTorch/CUDA/Triton once at startup."""
    import platform

    import torch

    cuda = torch.cuda.is_available()
    gpu_name = None
    cc = None
    device_count = 0
    if cuda:
        device_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)

    try:
        import triton  # noqa: F401

        triton_ok, triton_ver = True, triton.__version__
    except Exception:  # pragma: no cover - depends on platform
        triton_ok, triton_ver = False, None

    notes = []
    if not cuda:
        notes.append("no CUDA GPU: real benchmarking disabled; simulated engine available "
                     "(results will be labeled 'simulated')")
    if cuda and not triton_ok:
        notes.append("CUDA present but Triton missing: `pip install triton`")

    return EnvironmentReport(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=cuda,
        cuda_version=getattr(torch.version, "cuda", None),
        gpu_name=gpu_name,
        compute_capability=cc,
        triton_available=triton_ok,
        triton_version=triton_ver,
        device_count=device_count,
        notes=notes,
    )


def detect_hardware(device_index: int = 0) -> HardwareSpec:
    """Build a :class:`HardwareSpec` for a real CUDA device.

    Bandwidth/TFLOPS peaks are not exposed by the driver, so we look the card
    up in the catalog by name when possible and fall back to conservative
    estimates derived from queried properties.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("detect_hardware() requires CUDA; use simulated_hardware() instead")

    props = torch.cuda.get_device_properties(device_index)
    name = props.name
    for spec in KNOWN_GPUS.values():
        if spec.name.replace(" (simulated)", "").lower() in name.lower() or (
            name.lower() in spec.name.lower()
        ):
            logger.info("matched GPU %s to catalog entry %s", name, spec.name)
            return HardwareSpec(**{**spec.to_dict(), "name": name, "is_simulated": False,
                                   "compute_capability": (props.major, props.minor)})

    # Conservative fallback from queried properties.
    logger.info("GPU %s not in catalog; deriving estimated spec", name)
    return HardwareSpec(
        name=name,
        compute_capability=(props.major, props.minor),
        sm_count=props.multi_processor_count,
        memory_gb=props.total_memory / 2**30,
        # Rough peaks; only used for roofline context, never for reported results.
        memory_bandwidth_gbs=800.0,
        fp32_tflops=15.0,
        fp16_tflops=100.0,
        shared_mem_per_sm_kb=props.shared_memory_per_multiprocessor // 1024,
        registers_per_sm=props.regs_per_multiprocessor,
        warp_size=props.warp_size,
        supports_bf16=props.major >= 8,
        supports_tf32=props.major >= 8,
        is_simulated=False,
    )


def simulated_hardware(catalog_key: str = DEFAULT_SIMULATED_GPU) -> HardwareSpec:
    """A catalog GPU spec for the simulated engine (clearly labeled)."""
    if catalog_key not in KNOWN_GPUS:
        raise KeyError(f"unknown GPU {catalog_key!r}; options: {sorted(KNOWN_GPUS)}")
    return KNOWN_GPUS[catalog_key]

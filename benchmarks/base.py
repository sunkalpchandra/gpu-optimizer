"""Benchmark task abstraction.

A :class:`Task` bundles everything the optimizer needs to know about one
tensor-program workload:

- trusted reference implementation (PyTorch),
- input construction (seeded, dtype-aware),
- the tunable :class:`ParamSpace` of its Triton kernel,
- analytic FLOP/byte counts (roofline features + simulated engine),
- its :class:`ProgramGraph` representation for the learned encoders,
- dtype-aware correctness tolerances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from compiler.ir import ProgramGraph
from compiler.transformations.space import Candidate, Config, ParamSpace

TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "tf32": torch.float32,          # fp32 storage; tf32 tensor-core math
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Baseline tolerances per storage dtype; tasks may loosen for numerically
# harder ops (long reductions, tensor-core matmul).
DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    "float32": (1e-4, 1e-5),
    "tf32": (1e-2, 1e-3),
    "float16": (1e-2, 1e-3),
    "bfloat16": (2e-2, 1e-2),
}


class Task(ABC):
    """One optimization workload (matmul, softmax, ...)."""

    name: str = "abstract"
    supported_dtypes: tuple[str, ...] = ("float32", "float16", "bfloat16")

    # ----------------------------------------------------------- problem
    @abstractmethod
    def default_shapes(self) -> list[tuple[int, ...]]:
        """Representative problem shapes for this task."""

    @abstractmethod
    def make_inputs(
        self, shape: Sequence[int], dtype: str, device: str, seed: int = 0
    ) -> tuple[torch.Tensor, ...]:
        """Construct deterministic inputs for one problem instance."""

    @abstractmethod
    def reference(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Trusted PyTorch implementation (the correctness oracle).

        May upcast internally for numerical fidelity."""

    def baseline(self, *inputs: torch.Tensor) -> torch.Tensor:
        """The fast PyTorch implementation whose latency defines speedup.

        Defaults to :meth:`reference`; override where the production kernel
        differs from the oracle (e.g. native-dtype cuBLAS matmul)."""
        return self.reference(*inputs)

    # ------------------------------------------------------------- search
    @abstractmethod
    def param_space(self, shape: Sequence[int]) -> ParamSpace:
        """Tunable kernel parameters for this problem shape."""

    @abstractmethod
    def default_config(self, shape: Sequence[int]) -> Config:
        """A safe naive configuration (the Triton-naive baseline)."""

    def make_candidate(self, shape: Sequence[int], config: Config,
                       provenance: str = "manual") -> Candidate:
        space = self.param_space(shape)
        space.validate(config)
        return Candidate(task=self.name, shape=tuple(shape), config=dict(config),
                         provenance=provenance)

    # ---------------------------------------------------------- analytics
    @abstractmethod
    def flops(self, shape: Sequence[int]) -> float:
        """Useful floating-point operations for one execution."""

    @abstractmethod
    def bytes_moved(self, shape: Sequence[int], dtype: str,
                    config: Config | None = None) -> float:
        """Minimum DRAM traffic in bytes (config-dependent where fusion or
        multi-pass strategies change traffic)."""

    # ------------------------------------------------------ representation
    @abstractmethod
    def graph(self, shape: Sequence[int], dtype: str = "float32") -> ProgramGraph:
        """The program-graph representation for the learned encoders."""

    # ---------------------------------------------------------- validation
    def tolerance(self, dtype: str) -> tuple[float, float]:
        """(rtol, atol) for correctness checks at this storage dtype."""
        return DEFAULT_TOLERANCES[dtype]

    def output_dtype(self, dtype: str) -> torch.dtype:
        return TORCH_DTYPES[dtype]

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _gen(shape: Sequence[int], dtype: str, device: str, seed: int,
             scale: float = 1.0) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed)
        t = torch.randn(*shape, generator=g, dtype=torch.float32) * scale
        return t.to(device=device, dtype=TORCH_DTYPES[dtype])

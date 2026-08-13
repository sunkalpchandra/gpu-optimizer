"""Candidate correctness verification against trusted references.

Every candidate output is compared to the task's reference implementation
with dtype-aware tolerances.  A faster-but-wrong kernel must never earn
reward: failures propagate into the benchmark record and the reward function
applies the correctness penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CorrectnessReport:
    passed: bool
    max_abs_err: float
    max_rel_err: float
    rtol: float
    atol: float
    detail: str = ""

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"{status} (max_abs={self.max_abs_err:.3e}, "
                f"max_rel={self.max_rel_err:.3e}, rtol={self.rtol:g}, atol={self.atol:g})")


def check_correctness(candidate: torch.Tensor, reference: torch.Tensor,
                      rtol: float, atol: float) -> CorrectnessReport:
    """Elementwise |c - r| <= atol + rtol * |r| check (torch.allclose semantics),
    with NaN/Inf treated as automatic failure."""
    if candidate.shape != reference.shape:
        return CorrectnessReport(False, float("inf"), float("inf"), rtol, atol,
                                 f"shape mismatch {tuple(candidate.shape)} vs "
                                 f"{tuple(reference.shape)}")
    c = candidate.detach().float()
    r = reference.detach().float()
    if not torch.isfinite(c).all():
        return CorrectnessReport(False, float("inf"), float("inf"), rtol, atol,
                                 "non-finite values in candidate output")

    abs_err = (c - r).abs()
    max_abs = float(abs_err.max()) if abs_err.numel() else 0.0
    denom = r.abs().clamp_min(1e-12)
    max_rel = float((abs_err / denom).max()) if abs_err.numel() else 0.0
    ok = bool((abs_err <= atol + rtol * r.abs()).all())
    return CorrectnessReport(ok, max_abs, max_rel, rtol, atol)

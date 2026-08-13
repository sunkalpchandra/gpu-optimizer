"""The kernel → compile → benchmark → score orchestration loop.

Owns everything strategy-independent: baseline measurement, deduplication,
reward computation, persistence (results / runs / iterations tables), best
tracking, and progress callbacks.  Any :class:`Searcher` plugs in.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from benchmarks.db import BenchmarkDB
from benchmarks.harness import BenchmarkEngine, BenchmarkResult
from compiler.transformations.space import Candidate
from optimizer.rewards.reward import RewardConfig, compute_reward
from optimizer.search.base import SearchContext, Searcher

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict], None]


@dataclass
class SearchOutcome:
    run_id: str
    task: str
    shape: tuple[int, ...]
    algorithm: str
    engine: str
    best_candidate: Candidate | None
    best_result: BenchmarkResult | None
    baseline_torch_ms: float
    baseline_naive_ms: float | None
    candidates_evaluated: int
    compile_success_rate: float
    duration_s: float
    history: list[dict] = field(default_factory=list)

    @property
    def best_latency_ms(self) -> float:
        return self.best_result.latency.median_ms if self.best_result else float("inf")

    @property
    def speedup(self) -> float:
        """Speedup of the best found kernel vs the PyTorch baseline."""
        if not self.best_result:
            return 0.0
        return self.baseline_torch_ms / self.best_latency_ms

    def summary(self) -> str:
        lines = [
            f"run {self.run_id} [{self.algorithm}] {self.task} {list(self.shape)} "
            f"on {self.engine}",
            f"  baseline (torch): {self.baseline_torch_ms:.4f} ms",
        ]
        if self.baseline_naive_ms:
            lines.append(f"  baseline (triton naive): {self.baseline_naive_ms:.4f} ms")
        if self.best_result:
            lines.append(f"  best: {self.best_latency_ms:.4f} ms "
                         f"({self.speedup:.2f}x vs torch)  "
                         f"config={self.best_candidate.config}")
        else:
            lines.append("  no successful candidate found")
        lines.append(f"  evaluated {self.candidates_evaluated} candidates, "
                     f"compile success {self.compile_success_rate:.1%}, "
                     f"{self.duration_s:.1f}s")
        return "\n".join(lines)


class SearchLoop:
    """Run one search strategy against one task instance."""

    def __init__(
        self,
        ctx: SearchContext,
        searcher: Searcher,
        engine: BenchmarkEngine,
        db: BenchmarkDB | None = None,
        reward_config: RewardConfig | None = None,
        max_evaluations: int = 200,
        batch_size: int = 8,
        callback: ProgressCallback | None = None,
        run_id: str | None = None,
    ) -> None:
        self.ctx = ctx
        self.searcher = searcher
        self.engine = engine
        self.db = db
        self.reward_config = reward_config or RewardConfig()
        self.max_evaluations = max_evaluations
        self.batch_size = batch_size
        self.callback = callback or (lambda event: None)
        self.run_id = run_id or f"{ctx.task.name}-{searcher.name}-{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------ run
    def run(self) -> SearchOutcome:
        ctx, task = self.ctx, self.ctx.task
        t0 = time.time()
        dtype0 = str(task.default_config(ctx.shape).get("dtype", "float32"))

        baseline = self.engine.benchmark_baseline(task, ctx.shape, dtype0)
        baseline_ms = baseline.latency.median_ms
        naive = self.engine.benchmark_candidate(
            task, ctx.candidate(task.default_config(ctx.shape), "baseline-naive"))
        naive_ms = naive.latency.median_ms if naive.ok else None

        if self.db:
            self.db.create_run(self.run_id, task.name, ctx.shape, self.searcher.name,
                               self.engine.label, self.engine.hardware.name)
            self.db.insert_result(baseline, run_id=self.run_id)
            self.db.insert_result(naive, run_id=self.run_id)
            self.db.update_run(self.run_id, baseline_torch_ms=baseline_ms,
                               baseline_naive_ms=naive_ms)
        self.callback({"type": "baseline", "run_id": self.run_id,
                       "torch_ms": baseline_ms, "naive_ms": naive_ms,
                       "engine": self.engine.label})

        seen: set[str] = set()
        best: tuple[Candidate, BenchmarkResult] | None = None
        if naive.ok:
            best = (ctx.candidate(task.default_config(ctx.shape), "baseline-naive"),
                    naive)
        evaluated = 0
        compiled = 0
        history: list[dict] = []

        while evaluated < self.max_evaluations:
            want = min(self.batch_size, self.max_evaluations - evaluated)
            batch = self.searcher.propose(want)
            if not batch:
                logger.info("searcher %s exhausted after %d evals",
                            self.searcher.name, evaluated)
                break
            observations = []
            for cand in batch:
                if cand.candidate_id in seen:
                    continue
                seen.add(cand.candidate_id)
                result = self.engine.benchmark_candidate(task, cand)
                reward = compute_reward(result, baseline_ms, self.reward_config)
                evaluated += 1
                compiled += result.status != "compile_error"
                if result.ok and (best is None or result.latency.median_ms
                                  < best[1].latency.median_ms):
                    best = (cand, result)
                observations.append((cand, result, reward))

                best_ms = best[1].latency.median_ms if best else None
                record = {
                    "type": "iteration", "run_id": self.run_id,
                    "iteration": evaluated, "candidate_id": cand.candidate_id,
                    "config": cand.config, "provenance": cand.provenance,
                    "status": result.status,
                    "latency_ms": result.latency.median_ms if result.ok else None,
                    "predicted_ms": cand.predicted_ms,
                    "predicted_std": cand.predicted_std,
                    "reward": reward, "best_so_far_ms": best_ms,
                }
                history.append(record)
                self.callback(record)
                if self.db:
                    self.db.insert_result(result, run_id=self.run_id)
                    self.db.insert_iteration(
                        self.run_id, evaluated, cand.candidate_id,
                        actual_ms=result.latency.median_ms if result.ok else None,
                        reward=reward, best_so_far_ms=best_ms, status=result.status,
                        predicted_ms=cand.predicted_ms,
                        predicted_std=cand.predicted_std, note=cand.provenance)
            if observations:
                self.searcher.observe(observations)

        rate = compiled / evaluated if evaluated else 0.0
        outcome = SearchOutcome(
            run_id=self.run_id, task=task.name, shape=ctx.shape,
            algorithm=self.searcher.name, engine=self.engine.label,
            best_candidate=best[0] if best else None,
            best_result=best[1] if best else None,
            baseline_torch_ms=baseline_ms, baseline_naive_ms=naive_ms,
            candidates_evaluated=evaluated, compile_success_rate=rate,
            duration_s=time.time() - t0, history=history,
        )
        if self.db:
            self.db.update_run(
                self.run_id, status="finished",
                best_candidate_id=best[0].candidate_id if best else None,
                best_latency_ms=outcome.best_latency_ms if best else None,
                candidates_evaluated=evaluated, compile_success_rate=rate,
                finished_at=time.time())
        self.callback({"type": "done", "run_id": self.run_id,
                       "best_ms": outcome.best_latency_ms,
                       "speedup": outcome.speedup})
        return outcome

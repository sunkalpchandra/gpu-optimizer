"""FastAPI backend for the dashboard.

Serves overview stats, run/iteration/search-tree data from the benchmark DB,
kernel source rendering, hardware info, and can launch searches in a
background thread so the UI can watch them live (the iterations endpoint
supports incremental polling).  If ``frontend/dist`` exists it is served
statically, so one process hosts the whole demo.

Run:  uvicorn server.api.main:app --reload
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from benchmarks import TASKS, get_task
from benchmarks.db import BenchmarkDB, default_db_path
from compiler.triton.runner import render_source
from hardware.gpu_info import detect_environment
from optimizer.experiment import ExperimentConfig, build_engine, run_experiment
from optimizer.search.factory import available_searchers

app = FastAPI(title="gpu-optimizer", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_db = BenchmarkDB(default_db_path())
_env = detect_environment()
_active_runs: dict[str, dict[str, Any]] = {}   # run_id → {thread, status, error}


def db() -> BenchmarkDB:
    return _db


# ------------------------------------------------------------------ status

@app.get("/api/status")
def status() -> dict:
    ov = db().overview()
    best_speedup = None
    for row in ov["best_by_task"]:
        if row.get("baseline_torch_ms") and row.get("best_ms"):
            s = row["baseline_torch_ms"] / row["best_ms"]
            best_speedup = max(best_speedup or 0.0, s)
    kernels_optimized = len({(r["task"], r["shape"]) for r in ov["best_by_task"]})
    return {
        "environment": {
            "python": _env.python_version,
            "torch": _env.torch_version,
            "cuda_available": _env.cuda_available,
            "cuda_version": _env.cuda_version,
            "gpu_name": _env.gpu_name,
            "triton_available": _env.triton_available,
            "triton_version": _env.triton_version,
            "notes": _env.notes,
        },
        "overview": {
            "best_speedup": best_speedup,
            "kernels_optimized": kernels_optimized,
            "benchmarks_completed": ov["total_results"],
            "successful_benchmarks": ov["ok_results"],
            "compile_success_rate": ov["compile_success_rate"],
            "runs": ov["runs"],
            "results_by_engine": ov["results_by_engine"],
        },
        "simulated_data_present": ov["results_by_engine"].get("simulated", 0) > 0,
        "active_runs": [rid for rid, info in _active_runs.items()
                        if info["status"] == "running"],
    }


@app.get("/api/tasks")
def tasks() -> list[dict]:
    out = []
    for name, task in sorted(TASKS.items()):
        shapes = task.default_shapes()
        out.append({
            "name": name,
            "default_shapes": [list(s) for s in shapes],
            "space_size": task.param_space(shapes[0]).size(),
            "supported_dtypes": list(task.supported_dtypes),
        })
    return out


@app.get("/api/algorithms")
def algorithms() -> list[str]:
    return available_searchers()


@app.get("/api/gpu")
def gpu() -> dict:
    from hardware.gpu_info import KNOWN_GPUS, detect_hardware, simulated_hardware

    import torch

    if torch.cuda.is_available():
        hw = detect_hardware()
    else:
        hw = simulated_hardware()
    return {
        "current": hw.to_dict(),
        "is_simulated": hw.is_simulated,
        "catalog": {k: v.to_dict() for k, v in KNOWN_GPUS.items()},
        "cuda_available": _env.cuda_available,
    }


# -------------------------------------------------------------------- runs

@app.get("/api/runs")
def runs(limit: int = Query(100, le=500)) -> list[dict]:
    rows = db().fetch_runs(limit=limit)
    for r in rows:
        r["shape"] = json.loads(r["shape"])
        r["config"] = json.loads(r["config"])
        info = _active_runs.get(r["run_id"])
        if info and info["status"] == "error":
            r["status"] = "error"
            r["error"] = info.get("error", "")
    return rows


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    r = db().fetch_run(run_id)
    if not r:
        raise HTTPException(404, f"run {run_id} not found")
    r["shape"] = json.loads(r["shape"])
    r["config"] = json.loads(r["config"])
    return r


@app.get("/api/runs/{run_id}/iterations")
def run_iterations(run_id: str, after: int = Query(0, ge=0)) -> list[dict]:
    """Iterations with iteration index > ``after`` (incremental polling)."""
    rows = db().fetch_iterations(run_id)
    return [r for r in rows if r["iteration"] > after]


@app.get("/api/runs/{run_id}/tree")
def run_tree(run_id: str) -> dict:
    """Search tree: every candidate of the run with its lineage edge."""
    rows = db().fetch_results(run_id=run_id)
    nodes = []
    for r in reversed(rows):  # chronological
        nodes.append({
            "candidate_id": r["candidate_id"],
            "parent_id": r["parent_id"],
            "config": json.loads(r["config"]),
            "status": r["status"],
            "provenance": r["provenance"],
            "latency_ms": r["latency_median_ms"],
            "engine": r["engine"],
        })
    return {"run_id": run_id, "nodes": nodes}


@app.get("/api/candidates/{candidate_id}/source")
def candidate_source(candidate_id: str) -> dict:
    rows = db().fetch_results(candidate_id=candidate_id, limit=1)
    row = rows[0] if rows else None
    if row is None:
        raise HTTPException(404, f"candidate {candidate_id} not found")
    config = json.loads(row["config"])
    return {
        "candidate_id": candidate_id,
        "task": row["task"],
        "shape": json.loads(row["shape"]),
        "config": config,
        "engine": row["engine"],
        "status": row["status"],
        "latency_ms": row["latency_median_ms"],
        "source": render_source(row["task"], config),
    }


# ---------------------------------------------------------------- reports

@app.get("/api/reports")
def reports() -> list[dict]:
    rep_dir = Path(__file__).resolve().parents[2] / "reports"
    out = []
    if rep_dir.exists():
        for p in sorted(rep_dir.glob("*-generalization.json"), reverse=True):
            try:
                out.append({"name": p.stem, **json.loads(p.read_text())})
            except (OSError, json.JSONDecodeError):
                continue
    return out


# --------------------------------------------------------------- optimize

class OptimizeRequest(BaseModel):
    task: str = "matmul"
    shape: list[int] = Field(default_factory=lambda: [1024, 1024, 1024])
    algorithm: str = "hybrid"
    max_evaluations: int = Field(120, ge=1, le=5000)
    batch_size: int = Field(8, ge=1, le=64)
    seed: int = 0
    engine: str = "auto"          # auto | cuda | simulated
    warmup: int = Field(10, ge=0)
    iterations: int = Field(50, ge=3)


@app.post("/api/optimize")
def optimize(req: OptimizeRequest) -> dict:
    try:
        task = get_task(req.task)
    except KeyError as e:
        raise HTTPException(400, str(e)) from e
    if req.algorithm not in available_searchers():
        raise HTTPException(400, f"unknown algorithm {req.algorithm!r}")
    run_id = f"{req.task}-{req.algorithm}-{uuid.uuid4().hex[:8]}"
    cfg = ExperimentConfig(
        task=req.task, shapes=[tuple(req.shape)], algorithm=req.algorithm,
        seed=req.seed, engine=req.engine, max_evaluations=req.max_evaluations,
        batch_size=req.batch_size,
        benchmark={"warmup": req.warmup, "iterations": req.iterations},
    )
    # validate shape early (raises before the thread starts)
    task.param_space(tuple(req.shape))

    def work() -> None:
        try:
            engine = build_engine(cfg)
            from optimizer.rewards.reward import RewardConfig
            from optimizer.search.base import SearchContext
            from optimizer.search.factory import make_searcher
            from optimizer.search.loop import SearchLoop

            ctx = SearchContext(task=task, shape=tuple(req.shape),
                                hardware=engine.hardware, seed=req.seed)
            searcher = make_searcher(req.algorithm, ctx, engine=engine,
                                     rl_params=cfg.rl, **cfg.search)
            SearchLoop(ctx, searcher, engine, db=db(),
                       reward_config=RewardConfig(),
                       max_evaluations=req.max_evaluations,
                       batch_size=req.batch_size, run_id=run_id).run()
            _active_runs[run_id]["status"] = "finished"
        except Exception as e:  # surfaced through /api/runs
            _active_runs[run_id]["status"] = "error"
            _active_runs[run_id]["error"] = str(e)

    t = threading.Thread(target=work, name=f"run-{run_id}", daemon=True)
    _active_runs[run_id] = {"thread": t, "status": "running", "error": None}
    t.start()
    return {"run_id": run_id, "status": "started"}


# ------------------------------------------------------- static frontend

_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():  # pragma: no cover - depends on build artifacts
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")

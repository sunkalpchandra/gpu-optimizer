"""SQLite persistence for benchmark results, search runs, and iterations.

The results table doubles as the replay/training corpus for the learned
performance model; runs + iterations power the dashboard (live run view,
search tree, performance graphs).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from benchmarks.harness import BenchmarkResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    run_id TEXT,
    task TEXT NOT NULL,
    shape TEXT NOT NULL,
    config TEXT NOT NULL,
    engine TEXT NOT NULL,
    status TEXT NOT NULL,
    correct INTEGER NOT NULL DEFAULT 0,
    correctness_mode TEXT NOT NULL DEFAULT 'none',
    max_abs_err REAL,
    latency_median_ms REAL,
    latency_mean_ms REAL,
    latency_p50_ms REAL,
    latency_p90_ms REAL,
    latency_p99_ms REAL,
    latency_std_ms REAL,
    latency_min_ms REAL,
    iterations INTEGER DEFAULT 0,
    throughput_gflops REAL DEFAULT 0,
    memory_bytes INTEGER DEFAULT 0,
    compile_time_s REAL DEFAULT 0,
    warmup INTEGER DEFAULT 0,
    gpu_name TEXT,
    provenance TEXT,
    parent_id TEXT,
    error TEXT DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_task ON results(task, engine, status);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_candidate ON results(candidate_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    shape TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    engine TEXT NOT NULL,
    gpu_name TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    config TEXT NOT NULL DEFAULT '{}',
    best_candidate_id TEXT,
    best_latency_ms REAL,
    baseline_torch_ms REAL,
    baseline_naive_ms REAL,
    candidates_evaluated INTEGER DEFAULT 0,
    compile_success_rate REAL,
    started_at REAL NOT NULL,
    finished_at REAL
);

CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    predicted_ms REAL,
    predicted_std REAL,
    actual_ms REAL,
    reward REAL,
    best_so_far_ms REAL,
    status TEXT,
    note TEXT DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_iter_run ON iterations(run_id, iteration);
"""


class BenchmarkDB:
    """Thin, threadsafe-enough wrapper over SQLite (WAL, per-op connections)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30.0)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------- results
    def insert_result(self, r: BenchmarkResult, run_id: str | None = None) -> int:
        lat = r.latency
        with self._conn() as con:
            cur = con.execute(
                """INSERT INTO results
                   (candidate_id, run_id, task, shape, config, engine, status, correct,
                    correctness_mode, max_abs_err,
                    latency_median_ms, latency_mean_ms, latency_p50_ms, latency_p90_ms,
                    latency_p99_ms, latency_std_ms, latency_min_ms, iterations,
                    throughput_gflops, memory_bytes, compile_time_s, warmup,
                    gpu_name, provenance, parent_id, error, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r.candidate_id, run_id, r.task, json.dumps(list(r.shape)),
                    json.dumps(r.config, default=str), r.engine, r.status,
                    int(r.correct), r.correctness_mode,
                    r.correctness.max_abs_err if r.correctness else None,
                    lat.median_ms if lat else None, lat.mean_ms if lat else None,
                    lat.p50_ms if lat else None, lat.p90_ms if lat else None,
                    lat.p99_ms if lat else None, lat.std_ms if lat else None,
                    lat.min_ms if lat else None, lat.iterations if lat else 0,
                    r.throughput_gflops, r.memory_bytes, r.compile_time_s, r.warmup,
                    r.gpu_name, r.provenance, r.parent_id, r.error, r.timestamp,
                ),
            )
            return int(cur.lastrowid)

    def fetch_results(self, task: str | None = None, engine: str | None = None,
                      run_id: str | None = None, status: str | None = None,
                      candidate_id: str | None = None,
                      limit: int | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM results WHERE 1=1"
        args: list[Any] = []
        for col, val in (("task", task), ("engine", engine),
                         ("run_id", run_id), ("status", status),
                         ("candidate_id", candidate_id)):
            if val is not None:
                q += f" AND {col} = ?"
                args.append(val)
        q += " ORDER BY id DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self._conn() as con:
            return [dict(row) for row in con.execute(q, args)]

    def count_results(self, engine: str | None = None) -> int:
        q = "SELECT COUNT(*) FROM results"
        args: list[Any] = []
        if engine:
            q += " WHERE engine = ?"
            args.append(engine)
        with self._conn() as con:
            return int(con.execute(q, args).fetchone()[0])

    # ---------------------------------------------------------------- runs
    def create_run(self, run_id: str, task: str, shape: tuple[int, ...],
                   algorithm: str, engine: str, gpu_name: str,
                   config: dict | None = None) -> None:
        with self._conn() as con:
            con.execute(
                """INSERT INTO runs (run_id, task, shape, algorithm, engine, gpu_name,
                                     config, started_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (run_id, task, json.dumps(list(shape)), algorithm, engine, gpu_name,
                 json.dumps(config or {}, default=str), time.time()),
            )

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {"status", "best_candidate_id", "best_latency_ms", "baseline_torch_ms",
                   "baseline_naive_ms", "candidates_evaluated", "compile_success_rate",
                   "finished_at"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"cannot update run fields: {bad}")
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as con:
            con.execute(f"UPDATE runs SET {cols} WHERE run_id = ?",
                        (*fields.values(), run_id))

    def fetch_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as con:
            return [dict(r) for r in con.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))]

    def fetch_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as con:
            row = con.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    # ----------------------------------------------------------- iterations
    def insert_iteration(self, run_id: str, iteration: int, candidate_id: str,
                         actual_ms: float | None, reward: float | None,
                         best_so_far_ms: float | None, status: str,
                         predicted_ms: float | None = None,
                         predicted_std: float | None = None, note: str = "") -> None:
        with self._conn() as con:
            con.execute(
                """INSERT INTO iterations (run_id, iteration, candidate_id, predicted_ms,
                                           predicted_std, actual_ms, reward,
                                           best_so_far_ms, status, note, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, iteration, candidate_id, predicted_ms, predicted_std,
                 actual_ms, reward, best_so_far_ms, status, note, time.time()),
            )

    def fetch_iterations(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as con:
            return [dict(r) for r in con.execute(
                "SELECT * FROM iterations WHERE run_id = ? ORDER BY iteration, id",
                (run_id,))]

    # ------------------------------------------------------------- overview
    def overview(self) -> dict[str, Any]:
        with self._conn() as con:
            total = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            ok = con.execute("SELECT COUNT(*) FROM results WHERE status='ok'").fetchone()[0]
            compiled = con.execute(
                "SELECT COUNT(*) FROM results WHERE status NOT IN ('compile_error')"
            ).fetchone()[0]
            runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            by_engine = {r["engine"]: r["n"] for r in con.execute(
                "SELECT engine, COUNT(*) AS n FROM results GROUP BY engine")}
            best = [dict(r) for r in con.execute(
                """SELECT r.task, r.shape, r.engine, MIN(r.latency_median_ms) AS best_ms,
                          ru.baseline_torch_ms
                   FROM results r LEFT JOIN runs ru ON r.run_id = ru.run_id
                   WHERE r.status='ok' GROUP BY r.task, r.shape, r.engine""")]
        return {
            "total_results": total,
            "ok_results": ok,
            "compile_success_rate": (compiled / total) if total else None,
            "runs": runs,
            "results_by_engine": by_engine,
            "best_by_task": best,
        }


def default_db_path() -> Path:
    """Repo-relative default database location (no hardcoded absolute paths)."""
    return Path(__file__).resolve().parent.parent / "data" / "benchmark_db" / "results.sqlite"

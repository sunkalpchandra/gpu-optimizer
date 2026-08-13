#!/usr/bin/env python
"""Export a static snapshot of the dashboard data for GitHub Pages.

Reads the results DB through the same code paths the FastAPI backend uses and
writes flat JSON files under ``frontend/public/demo/`` that the frontend's
static mode consumes (see ``frontend/src/api.ts``).  The snapshot is honest by
construction: it contains the same engine labels the live API serves, and the
static UI shows an explicit "recorded snapshot" banner.

Usage:
    python scripts/export_demo_snapshot.py [--out frontend/public/demo] [--runs N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException

from server.api import main as api


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=None, separators=(",", ":"), default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="frontend/public/demo")
    ap.add_argument("--runs", type=int, default=12, help="most recent runs to include")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    status = api.status()
    status["active_runs"] = []  # a snapshot has no live runs
    dump(out / "status.json", status)
    dump(out / "tasks.json", api.tasks())
    dump(out / "algorithms.json", api.algorithms())
    dump(out / "gpu.json", api.gpu())
    dump(out / "reports.json", api.reports())

    runs = [r for r in api.runs(limit=args.runs) if r["status"] == "finished"]
    dump(out / "runs.json", runs)

    candidates: set[str] = set()
    for run in runs:
        rid = run["run_id"]
        dump(out / f"run-{rid}.json", api.run_detail(rid))
        dump(out / f"run-{rid}-iterations.json", api.run_iterations(rid, after=0))
        tree = api.run_tree(rid)
        dump(out / f"run-{rid}-tree.json", tree)
        candidates.update(n["candidate_id"] for n in tree["nodes"])

    exported = 0
    for cid in sorted(candidates):
        try:
            dump(out / f"candidate-{cid}.json", api.candidate_source(cid))
            exported += 1
        except HTTPException:
            continue

    n_files = len(list(out.glob("*.json")))
    size_kb = sum(p.stat().st_size for p in out.glob("*.json")) / 1024
    print(f"exported {len(runs)} runs, {exported} candidate sources "
          f"({n_files} files, {size_kb:.0f} KB) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

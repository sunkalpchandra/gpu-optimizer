#!/usr/bin/env python
"""Train (or refresh) a performance-model checkpoint from the results DB.

Usage:
    python scripts/train_surrogate.py [--db PATH] [--out checkpoints/surrogate.pt]
                                      [--task NAME] [--epochs 40] [--holdout 0.15]

Reports held-out rank correlation honestly; refuses to save a model whose
held-out Spearman is not clearly positive unless --force is given.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.db import BenchmarkDB, default_db_path
from optimizer.performance_model.features import db_row_to_row
from optimizer.performance_model.model import PerformanceModel


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(xs: list[float]) -> list[int]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rk = [0] * len(xs)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="results DB (default: repo DB)")
    ap.add_argument("--out", default="checkpoints/surrogate.pt")
    ap.add_argument("--task", default=None, help="restrict to one task")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="save even if held-out ranking is weak")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    db = BenchmarkDB(args.db or default_db_path())
    raw = db.fetch_results(task=args.task)
    rows = [db_row_to_row(r) for r in raw if r["provenance"] != "baseline"]
    if len(rows) < 60:
        print(f"only {len(rows)} usable rows in {db.path}; need >= 60 "
              f"(below that the ensemble's ranking is untrustworthy)")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_hold = max(10, int(len(rows) * args.holdout))
    held, train = rows[:n_hold], rows[n_hold:]

    model = PerformanceModel(n_members=args.members, seed=args.seed)
    model.add_rows(train)
    stats = model.fit(epochs=args.epochs)
    print(f"trained on {len(train)} rows, final loss {stats['loss']:.4f}")

    held_ok = [r for r in held if r.latency_ms is not None]
    by_case: dict[tuple, list] = {}
    for r in held_ok:
        by_case.setdefault((r.task, r.shape), []).append(r)
    rhos = []
    for (task, shape), case_rows in by_case.items():
        if len(case_rows) < 8:
            continue
        from optimizer.performance_model.features import spec_by_name

        hw = spec_by_name(case_rows[0].gpu_name)
        preds = model.predict(task, shape, [r.config for r in case_rows], hw)
        rho = spearman([r.latency_ms for r in case_rows], [p.mean_ms for p in preds])
        rhos.append(rho)
        print(f"held-out {task}{shape}: n={len(case_rows)} spearman={rho:.3f}")
    mean_rho = sum(rhos) / len(rhos) if rhos else float("nan")
    print(f"mean held-out spearman: {mean_rho:.3f}")

    if not rhos or mean_rho < 0.3:
        if not args.force:
            print("held-out ranking too weak to trust; not saving (--force to override)")
            return 1
        print("weak ranking, saving anyway (--force)")

    model.rows = rows  # persist the full corpus with the weights
    model.save(args.out)
    print(f"saved {args.out} ({len(rows)} corpus rows, {args.members} members)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

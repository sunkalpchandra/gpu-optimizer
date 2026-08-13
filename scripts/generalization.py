#!/usr/bin/env python
"""Run the generalization experiment suite and write reports/.

Usage:
    python scripts/generalization.py [--fast] [--seed N] [--out reports]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.experiments.generalization import run_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="reduced budgets (smoke run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    report = run_all(out_dir=args.out, seed=args.seed, fast=args.fast)
    print(report.to_markdown())


if __name__ == "__main__":
    main()

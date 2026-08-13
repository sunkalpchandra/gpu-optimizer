#!/usr/bin/env python
"""Run a search experiment from a YAML config.

Usage:
    python scripts/run_search.py configs/matmul_hybrid.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.experiment import ExperimentConfig, run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="path to YAML experiment config")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = ExperimentConfig.from_yaml(args.config)
    for outcome in run_experiment(cfg):
        print(outcome.summary())


if __name__ == "__main__":
    main()

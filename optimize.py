#!/usr/bin/env python
"""gpu-optimizer demo CLI.

    python optimize.py --task matmul --size 4096

Detects the environment, measures the PyTorch baseline, runs the requested
search (hybrid by default), and reports the best kernel found — honestly
labeling simulated results when no CUDA GPU is present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def default_shape(task: str, size: int, rows: int) -> tuple[int, ...]:
    """Map a single --size to a task-appropriate shape."""
    if task == "matmul":
        return (size, size, size)
    if task in ("vecadd", "reduction", "fused_elementwise"):
        return (size,)
    if task in ("softmax", "layernorm"):
        return (rows, size)
    if task == "attention":
        return (16, size, 64)
    raise ValueError(f"unknown task {task!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="matmul",
                    help="matmul | vecadd | reduction | softmax | layernorm | "
                         "fused_elementwise | attention")
    ap.add_argument("--size", type=int, default=1024,
                    help="problem size (matmul: N for NxNxN; 1-D tasks: element "
                         "count; softmax/layernorm: row length; attention: seq len)")
    ap.add_argument("--rows", type=int, default=4096,
                    help="row count for softmax/layernorm shapes")
    ap.add_argument("--shape", default=None,
                    help="explicit comma-separated shape, overrides --size")
    ap.add_argument("--algorithm", default="hybrid",
                    help="random | grid | evolutionary | bayesian | rl | hybrid")
    ap.add_argument("--budget", type=int, default=150, help="candidate evaluations")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", default="auto", choices=("auto", "cuda", "simulated"))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--no-db", action="store_true", help="skip persistence")
    ap.add_argument("--warm", action="store_true",
                    help="load/save per-task surrogate + policy checkpoints under "
                         "checkpoints/, so successive runs start smarter")
    ap.add_argument("--surrogate", default=None, metavar="PATH",
                    help="surrogate checkpoint to warm-start from and save back to")
    ap.add_argument("--policy", default=None, metavar="PATH",
                    help="PPO policy checkpoint to warm-start from and save back to")
    ap.add_argument("--save-kernel", default=None, metavar="PATH",
                    help="write the best kernel's annotated source here")
    ap.add_argument("--show-source", action="store_true",
                    help="print the best kernel source at the end")
    args = ap.parse_args()

    from rich.console import Console

    console = Console(highlight=False)

    from benchmarks import get_task
    from compiler.triton.runner import render_source
    from hardware.gpu_info import detect_environment
    from optimizer.experiment import ExperimentConfig, run_experiment

    env = detect_environment()
    get_task(args.task)  # validate the task name before printing the header
    shape = (tuple(int(x) for x in args.shape.split(","))
             if args.shape else default_shape(args.task, args.size, args.rows))

    console.print("\n[bold]GPU Optimizer[/bold]")
    console.print("─" * 44)
    console.print(f"Task: [bold]{args.task.upper()}[/bold]")
    console.print(f"Shape: {' × '.join(str(s) for s in shape)}")
    console.print(f"Algorithm: {args.algorithm}")
    for line in env.summary().splitlines():
        console.print(f"  {line}")

    search_params: dict = {}
    surrogate = args.surrogate or (f"checkpoints/{args.task}-surrogate.pt"
                                   if args.warm else None)
    policy = args.policy or (f"checkpoints/{args.task}-policy.pt" if args.warm else None)
    if surrogate and args.algorithm in ("bayesian", "hybrid"):
        search_params["surrogate_checkpoint"] = surrogate
    if policy and args.algorithm in ("rl", "hybrid"):
        search_params["policy_checkpoint"] = policy
    if (surrogate or policy) and not search_params:
        console.print(f"[yellow]note: --warm/--surrogate/--policy have no effect on "
                      f"algorithm {args.algorithm!r}[/yellow]")

    cfg = ExperimentConfig(
        task=args.task, shapes=[shape], algorithm=args.algorithm, seed=args.seed,
        engine=args.engine, max_evaluations=args.budget, batch_size=args.batch_size,
        benchmark={"warmup": args.warmup, "iterations": args.iterations},
        search=search_params,
        db_path="" if args.no_db else None,
    )

    state = {"batch": 0, "evals": 0, "best": float("inf"), "shown": float("inf"),
             "engine": "?"}

    def on_event(e: dict) -> None:
        if e["type"] == "baseline":
            state["engine"] = e["engine"]
            if e["engine"] == "simulated":
                console.print("\n[yellow bold]⚠ No CUDA GPU: using the deterministic "
                              "SIMULATED engine.[/yellow bold]")
                console.print("[yellow]  All numbers below are model estimates for "
                              "development, NOT hardware measurements.[/yellow]")
            console.print(f"\nBaseline (PyTorch): [bold]{e['torch_ms']:.4f} ms[/bold]")
            if e["naive_ms"]:
                console.print(f"Baseline (Triton naive): {e['naive_ms']:.4f} ms")
            console.print("\nSearching...\n")
        elif e["type"] == "iteration":
            state["evals"] = e["iteration"]
            if e["best_so_far_ms"]:
                state["best"] = e["best_so_far_ms"]
            if e["iteration"] % args.batch_size == 0:
                state["batch"] += 1
                marker = " [green]★[/green]" if state["best"] < state["shown"] else ""
                console.print(f"Generation {state['batch']:<4d} "
                              f"Best: {state['best']:.4f} ms{marker}")
                state["shown"] = state["best"]

    outcomes = run_experiment(cfg, callback=on_event)
    o = outcomes[0]

    if o.best_result is None:
        console.print("[red]No successful candidate found.[/red]")
        return 1

    sim = " [yellow](simulated)[/yellow]" if o.engine == "simulated" else ""
    console.print("\n[bold]Best candidate:[/bold]")
    for k, v in sorted(o.best_candidate.config.items()):
        console.print(f"  {k} = {v}")
    console.print(f"\nRuntime:{sim}\n  [bold]{o.best_latency_ms:.4f} ms[/bold]")
    console.print(f"\nSpeedup vs PyTorch:{sim}\n  [bold]{o.speedup:.2f}×[/bold]")

    cr = o.best_result
    mode = {"device": "verified on device", "emulated": "verified via CPU emulation",
            "none": "not verified"}[cr.correctness_mode]
    verdict = "[green]PASS[/green]" if cr.correct else "[red]FAIL[/red]"
    console.print(f"\nCorrectness:\n  {verdict} ({mode})")
    console.print(f"\nCandidates evaluated:\n  {o.candidates_evaluated:,}")
    console.print(f"\nCompilation success:\n  {o.compile_success_rate:.1%}")
    console.print(f"\nSearch time:\n  {o.duration_s:.1f}s")
    if not args.no_db:
        console.print(f"\nRun ID (dashboard):\n  {o.run_id}")

    source = render_source(args.task, o.best_candidate.config)
    if args.save_kernel:
        Path(args.save_kernel).write_text(source)
        console.print(f"\nBest kernel source written to [bold]{args.save_kernel}[/bold]")
    if args.show_source:
        console.print("\n[bold]Best kernel source:[/bold]")
        console.print(source)
    elif not args.save_kernel:
        console.print("\n(use --show-source or --save-kernel PATH to inspect the "
                      "generated kernel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

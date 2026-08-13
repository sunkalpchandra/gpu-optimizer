# Roadmap / Status

This file is the working plan. It is read both by humans and by the scheduled cloud agent
that continues development. Keep it honest: check items only when implemented **and tested**.

## Ground rules for any agent working on this repo

- Never fabricate benchmark numbers. The cloud/dev environments have no CUDA GPU: use the
  simulated engine (results are labeled `simulated`) or CPU-path unit tests.
- Run `python -m pytest` before pushing. Report failures honestly.
- Production quality: type hints, docstrings on key components, no placeholder/TODO core logic.
- Commit granularly with clear messages. Cloud agents: push to a branch
  `routine/YYYY-MM-DD` and open a PR to `main` — do not push to `main` directly.

## Stages

- [x] **Stage 1 — Benchmarking core**: hardware detection, Triton kernels (matmul, vecadd,
      reduction, softmax, layernorm, fused elementwise, attention), reference implementations,
      dtype-aware correctness checks, benchmark harness (warmup, CUDA sync, percentiles),
      SQLite persistence, simulated engine + CPU algorithm emulation for non-CUDA dev.
- [x] **Stage 2 — Parameter search**: structured candidate space, kernel → compile →
      benchmark → score loop, random + grid baselines, YAML experiment configs.
- [x] **Stage 3 — Evolutionary optimizer**: tournament selection, uniform crossover,
      step/reset mutation, immigrants, elitism-by-pool.
- [x] **Stage 4 — Performance model**: deep-ensemble surrogate (log-runtime / log-memory /
      compile-prob + epistemic uncertainty) over graph+hardware+config encoders;
      UCB and Thompson acquisition (`bayesian` searcher).
- [x] **Stage 5 — RL policy**: PPO/GAE over the structured transformation catalog;
      transformer actor-critic with masked discrete head + Gaussian continuous head;
      online transformation episodes trained during search.
- [x] **Stage 6 — Hybrid optimizer**: reserved RL episode budget + surrogate-UCB-ranked
      GA/immigrant pool; all learners share every real measurement.
- [x] **Stage 7 — Generalization experiments**: shape interpolation/extrapolation, workload
      transfer, hardware transfer (top-K protocol), search-efficiency comparison;
      honest simulated-labeled reports in `reports/`.
- [x] **Stage 8 — Dashboard**: FastAPI backend (live-pollable runs, search tree, kernel
      source, GPU catalog, background optimize) + React/TS/Tailwind frontend (overview,
      live run view, SVG search tree, kernel viewer, GPU metrics, reports) — `npm run build`
      clean, served from `frontend/dist` by the API server.
- [x] **Finalization**: `optimize.py` demo CLI, full README with Mermaid diagrams,
      integration test pass (71 passed, 14 GPU-gated skips on non-CUDA machines).

## Done since v0.1

- [x] Lint policy (ruff E/F/W/I/UP/B/RUF) enforced in CI; full-codebase cleanup.
- [x] Surrogate warm-start: checkpoints persist weights **and** training corpus;
      `--warm`/`--surrogate` flags; `scripts/train_surrogate.py` with held-out gating.
- [x] PPO policy persistence via `Searcher.finalize()`; `--warm`/`--policy` flags.
- [x] Dashboard code splitting (660 kB → 220 kB main chunk) + static demo mode;
      deployed to GitHub Pages from a recorded snapshot with honest banners:
      https://sunkalpchandra.github.io/gpu-optimizer/
- [x] CI (ruff + pytest + frontend build) and Pages deploy workflows.

## Open follow-ups

- [ ] **Real-GPU validation pass**: run `tests/test_triton_gpu.py` and a full
      `optimize.py --task matmul --size 4096` on a CUDA machine; fix any Triton API
      mismatches (kernels were written for Triton ≥2.1 but have never executed on hardware);
      record real example results in README (replacing/alongside simulated tables).
- [ ] Websocket push for the live run view instead of 1.5 s polling.
- [ ] CUDA C backend: extend `compiler/cuda/` beyond source rendering (e.g.
      `torch.utils.cpp_extension.load_inline` path for vecadd/reduction) on GPU machines.
- [ ] Multi-shape joint runs: one search that optimizes several shapes per task with
      shared surrogate/policy (infrastructure supports it; needs an orchestrator).
- [ ] Refresh the Pages demo snapshot (`scripts/export_demo_snapshot.py`) whenever
      richer runs (especially real-GPU ones) land in the results DB.

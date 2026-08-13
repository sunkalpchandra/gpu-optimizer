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

- [ ] **Stage 1 — Benchmarking core**: hardware detection, Triton kernels (matmul, vecadd,
      reduction, softmax, layernorm, fused elementwise, attention), reference implementations,
      dtype-aware correctness checks, benchmark harness (warmup, CUDA sync, percentiles),
      SQLite persistence, simulated engine for non-CUDA dev.
- [ ] **Stage 2 — Parameter search**: structured candidate space, kernel → compile →
      benchmark → score pipeline, random + grid search baselines.
- [ ] **Stage 3 — Evolutionary optimizer**: tournament selection, mutation, crossover, elitism.
- [ ] **Stage 4 — Performance model**: neural surrogate (runtime / memory / compile-prob +
      uncertainty via deep ensemble), trained from the benchmark DB; UCB/Thompson acquisition.
- [ ] **Stage 5 — RL policy**: PPO over the structured transformation space; program graph
      encoder + hardware encoder + transformer policy with discrete+continuous heads.
- [ ] **Stage 6 — Hybrid optimizer**: RL proposals + surrogate ranking + evolutionary
      population + real benchmarking, as the flagship search mode.
- [ ] **Stage 7 — Generalization experiments**: shape interpolation/extrapolation, workload
      transfer, (hardware transfer where multiple GPUs exist), with honest reports.
- [ ] **Stage 8 — Dashboard**: FastAPI backend + React/TS/Tailwind frontend (overview, live
      run view, search tree, kernel viewer, performance graph, GPU metrics).
- [ ] **Finalization**: `optimize.py` demo CLI, full README with Mermaid diagrams,
      integration test pass.

## Open follow-ups

(none yet)

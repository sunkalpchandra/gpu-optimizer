# gpu-optimizer

An autonomous GPU kernel optimization agent: reinforcement learning + evolutionary search +
a learned performance model, searching the configuration space of real Triton kernels and
learning from measured GPU benchmarks.

> **Status: under active construction.** This README is a stub; the full architecture
> documentation lands with the final stage. See [STATUS.md](STATUS.md) for the live roadmap.

## Core loop

```text
Tensor Program → Graph IR → Candidate Generator → RL / Search Policy
    → Kernel Transformation → Triton Code → Compile → GPU Benchmark
    → Reward → Replay Buffer → Policy Improvement → Better Candidates
```

## Honesty policy

- Benchmarks on real CUDA hardware are the ground truth.
- On machines without CUDA (e.g. macOS dev machines), a deterministic **simulated**
  benchmark engine is available for development only. Every simulated number is labeled
  `simulated` in the database, the API, and the UI. Simulated results are never presented
  as real measurements.

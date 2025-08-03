# Performance Upgrade Roadmap

This document tracks the ongoing work to speed-up **girth** and then integrate those gains into **loop_modulus** while keeping both GitHub repositories clean and easy to collaborate on.

---
## 1  Groundwork (both repos)

- [~] Standardise tooling (pytest, coverage, black/ruff, pre-commit).
- [~] Add GitHub Actions workflow to run lint & tests.
- [ ] Protect `master` with branch protection rules on GitHub (manual step).
- [ ] Freeze current public API of `girth`; record baseline test cases.

## 2  Deep-dive & optimise `girth`

> Target version: **girth v0.2.0** – same API, faster implementation

### 2.1  Profiling & baseline
- [~] Run existing `profile_algorithm.py` / `benchmark.py` on several graph sizes (baseline profiling in progress).
- [ ] Store cProfile + flamegraphs in `profiling_results/`.
- [ ] Commit a markdown table of wall-clock results (`docs/benchmark_baseline.md`).

### 2.2  Quick-win clean-ups
- [ ] Remove repeated NetworkX look-ups in inner loops.
- [ ] Replace per-edge `dict` access with cached locals / arrays where indices are dense.

### 2.3  Algorithmic upgrades (one PR each)
- [ ] **PR #1 – Refactor hot loops** (micro-optimisations, early exits).
- [ ] **PR #2 – Better heap** (switch to `heapdict` or optional Fibonacci heap).
- [ ] **PR #3 – Optional compiled speed-ups** (Cython/C++ extension) ‑ stretch goal.

### 2.4  Packaging & versioning
- [ ] Add `pyproject.toml` / `setup.cfg` so `pip install git+…` works.
- [ ] Adopt semantic versioning and tag `v0.2.0` after performance PRs merge.

---
## 3  Integrate into `loop_modulus`

- [ ] Pin `girth` dependency in `loop_modulus/requirements.txt` to `girth@v0.2.0`.
- [ ] Add small unit test that computes a modulus instance and asserts numeric output.
- [ ] Profile full pipeline again; decide next optimisation targets (batching, pruning…).

---
## 4  Release process

1. Merge feature branch via PR, update CHANGELOG, bump version.
2. `git tag -a vX.Y.Z -m "…"` and push tag.
3. Update downstream repo (`loop_modulus`) when a new `girth` version is published.

---
## 5  Status (checkbox legend)

- [ ] Pending / not started
- [x] Completed
- [~] In progress

Maintain this file after each significant change so everyone stays in sync.

---
## Phase-3 Optimisation Tasks (2024-08-03)

| ID | Task | Status | Notes |
|----|------|--------|-------|
| 1 | Adaptive Seed Count | ☐ Pending | Dynamic seeds = max(1, ⌊k·γ⌋); stop when BMSSP finds no new shorter cycle. |
| 2 | ρ-Weighted Graph Pruning | ☐ Pending | Build subgraph with ρ > ε (ε ≈ 1e-6) before each BMSSP call to shrink search space. |
| 3 | Active-Set QP / Constraint Dropping | ✅ Completed | Implemented: constraints with dual ≤ τ are removed after each OSQP solve; warm-starts adjusted. |

### Verification & Benchmark Protocol
1. Large-graph benchmark: run `girth/benchmark.py --run benchmarks` (Agg backend) and log times in `docs/benchmark_large_graphs.md`.
2. Correctness check: compute loop modulus on the Cholera graph; expected value 98–101.  Assert within this range in CI.

Update status and notes for each task as soon as code is merged or results measured.


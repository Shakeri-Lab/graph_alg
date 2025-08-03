# graph_alg

This repository is a **meta-package** that houses two tightly–coupled research codebases as *sub-directories* (or Git sub-modules, if you checked the repo out with `--recursive`).

* `girth/` –  state-of-the-art algorithms for **minimum-weight / girth / shortest-cycle** problems in weighted graphs.  Latest version (Phase-3) includes:
  * Bounded Multi-Source Shortest Path (BMSSP) implementation based on Duan et al. \cite{duan2025breaking}
  * Cython-compiled relaxation kernel + block-based priority queue
  * Euler / Binary / Optimal LCA back-ends
  * Optional graph-pruning, composite-distance cuts, and more

* `loop_modulus/` – iterative **loop-modulus** optimisation which now delegates all cycle searches to the upgraded `girth` library.  Phase-3 adds:
  * ρ-weighted graph pruning between iterations
  * Active–set QP with constraint dropping and warm starts
  * Adaptive seed selection and BMSSP integration

The parent `graph_alg` repo merely pins the exact versions of both components so you can reproduce the paper’s results with

```bash
# clone + nested repos
git clone --recursive https://github.com/Shakeri-Lab/graph_alg.git

# or, if you already cloned without --recursive
git submodule update --init --recursive
```

Current pointers (check via `git submodule status`):

| Module | Branch | Commit | Remote |
|--------|--------|--------|--------|
| girth  | `main` | `<hash>` | https://github.com/Shakeri-Lab/girth |
| loop_modulus | `main` | `<hash>` | https://github.com/Shakeri-Lab/loop_modulus |

> **Note**  The hashes are updated each time Phase-level milestones are merged.  If you develop inside one module remember to push the module *first*, then update the pointer in `graph_alg` and push this repo next.

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # ruff, black, pytest, cython, ...
pip install -e girth[cython]           # build the Cython kernel in-place
pip install -e loop_modulus
```

Running the test-suite:

```bash
pytest girth -q
pytest loop_modulus -q
```

Benchmarks and figures referenced in the manuscript live under `docs/`.

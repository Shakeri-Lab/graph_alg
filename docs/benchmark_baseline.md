# Baseline Benchmark Results

Initial performance of `sota_shortest_cycle` **before any optimisations**.
All runs executed on the same Linux 4.18 machine (Intel(R) Xeon(R) …, Python 3.12, NetworkX 3.5).

| Graph Type | Size (nodes) | Mean time (s) |
|------------|--------------|---------------|
| Grid       | 10×10 (100)  | 0.013 |
| Grid       | 20×20 (400)  | 0.044 |
| Grid       | 30×30 (900)  | 0.108 |
| Spatial    | 50           | 0.027 |
| Spatial    | 100          | 0.125 |
| Spatial    | 150          | 0.312 |

Detailed cProfile statistics are stored in `girth/profiling_results/`:

- `grid_40_profile.prof`, `grid_40_stats.txt`
- `grid_40_optimized.prof`, `grid_40_optimized_stats.txt`
- `grid_40_traditional.prof`, `grid_40_traditional_stats.txt`

These serve as the reference numbers for upcoming optimisation work.

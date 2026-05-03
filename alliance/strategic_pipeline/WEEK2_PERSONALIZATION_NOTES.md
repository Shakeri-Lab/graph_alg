# Week 2A — Personalization / Null-Baseline Diagnostic

**Status:** The Week-1 durable-rent score is **structurally non-personalizing
at population scale.** Only `brokerage_L2(focal, candidate)` produces
focal-specific top-1 picks; every other ranker we tested collapses to one
universal top candidate across all 7,626 focal firms.

## Scale of the diagnostic

- **18,064,303 (focal, candidate) rows** spanning 7,626 Compustat-matched focal
  firms × 2,369 L2-active candidates × 2017.
- Slurm: two batches (job 12257558 4000-task + 12262469 3626-task) + two
  small `--mem=32G` retries. 0 unrecovered failures.
- Per-focal scoring produced one parquet each in
  `outputs/strategic/aggregate/week2_personalization/`; stitched to
  `outputs/strategic/aggregate/week2_personalization_rows_2017.parquet`
  (~580 MB).

## Rankers compared

For each (focal, candidate) we computed:

| Ranker | Description | Focal-specific? |
|---|---|---|
| `global_degree_all`   | Candidate degree across all layers, 5-yr window | No |
| `global_degree_l2`    | Candidate L2 degree, 5-yr window                 | No |
| `n_current_ties`      | Distinct partners in 5-yr window                  | No |
| `w_tenure_only`       | sigmoid(z) shrunk by κ=5                          | No |
| `brokerage_only`      | brokerage_L2(focal, candidate) only               | **Yes** |
| `brokerage_x_tenure`  | brokerage_L2 × w_tenure_smooth                    | Mixed |
| `raw_score`           | Full Week-1 durable-rent score                    | Mixed |
| `blend_α` (α∈[0,1])   | α·rank(raw) + (1−α)·rank(residual)                | Mixed |
| `resid_rank`          | rank-pct of (raw − fitted) per focal              | Mixed |

The residualizer fits OLS of within-focal raw_score rank-percentile on
candidate-global features:
$\log(1+d_\text{all})$, $\log(1+d_{L_2})$, $\log(1+n_\text{current})$,
$\log(1+n_\text{hist})$, plus per-(SIC, type) group-mean offsets.

## Headline result: N_eff top-1 across 7,626 focals

| Ranker                 | N_eff |   Top candidate |   Top share |
|------------------------|------:|----------------:|------------:|
| `brokerage_only`       | **1,811.9** | (no single dominant — top firms each appear in ~10 focals) | 0.13 % |
| `global_degree_l2`     |  1.001 | (single candidate) | 99.95 % |
| `global_degree_all`    |  1.001 | (single candidate) | 99.96 % |
| `n_current_ties`       |  1.001 | (single candidate) | 99.96 % |
| `w_tenure_only`        |  1.001 | MIT             | 99.96 % |
| `brokerage_x_tenure`   |  1.001 | (single candidate) | 99.96 % |
| `raw_score`            | **1.000** | **Stanford University** | **100.0 %** |
| `blend_025` … `blend_100` | 1.000 | Stanford       | 100.0 % |
| `blend_000` (= residual) | 1.001 | (single candidate) | 99.96 % |
| `resid_rank`           |  1.001 | (single candidate) | 99.96 % |

`brokerage_only` is the **only** ranker producing genuinely focal-specific
top picks. Every other ranker — including the residualized version, and
including all blends except $\alpha = 0$ — converges on essentially one
universal candidate.

## Why residualization did not help

The residualization regresses `rank(raw_score)` on candidate-global
features and takes the residual. But when the raw score is itself
dominated by candidate-side features (because `brokerage_L2` saturates at
1.0 for ~99% of candidates), the residual is also candidate-side. The
within-focal rank of a candidate-side residual is identical across
focals — so residualization cannot undo the collapse.

Personalization requires either (a) using a focal-specific signal
directly (`brokerage_only`), or (b) introducing **new** focal–candidate
interaction features that the current pipeline does not produce
(industry adjacency, geographic proximity, prior co-investor links,
two-hop alliance distance to the focal, baseline financial-similarity).

## Type-stratified collapse

Restricting to single candidate types produces the same collapse, just
with a different universal winner per type:

| Candidate type        | N_eff (raw_score) | Notes |
|-----------------------|------------------:|-------|
| `university_research` | 1.000 | Stanford for all 7,626 |
| `hospital_medical`    | 1.000 | Single hospital winner |
| `private_other`       | 1.000 | Single private winner |
| `public_compustat`    | 2.000 | Two-firm tie |
| `sovereign_state`     | 2.000 | Two-state tie |

The Stanford-collapse hypothesized in WEEK1_NOTES.md is therefore not
a "university-bridge" phenomenon — every candidate-type stratum has its
own universal dominant pick. The pathology is structural to the
multiplicative-feature score, not to any one candidate.

## Decision rule for Week 2B

The user's decision rule (raw vs. residualized vs. blended) reduces to a
single answer here: only `brokerage_only` survives the personalization
diagnostic. All multiplicative variants of the durable-rent score
collapse to one candidate.

For the Week-2B persistence and sales backtests, **use
`brokerage_only` as the primary ranker** with `w_tenure_smooth`,
`g(R&D)`, and `w_redundancy` retained as **post-rank scalars** (lift /
discount factors applied to a brokerage-ranked top-K, not as
multiplicative components in the rank itself).

In notation:

$$
\text{rank}_{fc} = \operatorname{rank}_{c \in \mathcal C_f}
\bigl( \text{brokerage}_{L_2}(f, c) \bigr)
$$

then the *value* attached to a chosen candidate is

$$
v_{fc} = \text{brokerage}_{L_2}(f, c) \cdot
w_{\text{tenure}}(c) \cdot
g(\text{R\&D}_f) \cdot
\exp(-\rho \cdot \text{DepRisk}(c)),
$$

reported alongside the rank but not driving it. Equivalently: the
recommender becomes "rank by structural opportunity, then score the
top-K by relational capability and dependency exposure."

This preserves the Hankel-DMD interpretation (the durable-rent components
still describe value composition) without the population-scale collapse.

## What Week 2B should do next

1. **Persistence prediction.** For realized 2011 ties, predict
   $\Pr(T_{fc} \ge 3)$ using brokerage-ranked top-20 + $w_\text{tenure}$
   features; compare AUC against degree-only and current-ties-only
   baselines.
2. **Sales enrichment.** For 2011-2013 realized partners, regress
   $\Delta_h \log\text{Sales}_{f, t+h}$ on indicators for "realized
   partner is brokerage-top-20" and "realized partner is in top-K by
   value $v_{fc}$"; $h \in \{2, 4\}$.
3. **Dependency tradeoff.** Sweep $\rho \in \{0, 0.5, 1.0, 1.5, 2.0,
   3.0\}$; report top-k structural exposure vs durable value retained.
4. **Personalization upgrade (Week 2C).** Add focal–candidate interaction
   features and re-run this diagnostic. Candidates: candidate's adjacency
   to focal in the L2 5-yr graph (two-hop indicator), candidate's
   nation-share match, candidate's prior co-investor share with focal.
   Without these, no scoring scheme over the current feature set can
   produce N_eff > 1812 for raw_score.

## Artifacts

- `outputs/strategic/aggregate/week2_personalization_rows_2017.parquet` (18M rows)
- `outputs/strategic/aggregate/week2_top1_concentration_by_variant.csv`
- `outputs/strategic/aggregate/week2_overlap_matrix_by_variant.csv`
- `outputs/strategic/aggregate/week2_type_stratified_neff.csv`
- `outputs/strategic/aggregate/week2_personalization_summary.csv`
- `outputs/strategic/aggregate/candidate_features_2017.parquet` (2,369 rows)
- `outputs/strategic/figures/week2_top1_concentration.png`
- `outputs/strategic/figures/week2_baseline_overlap_heatmap.png`
- `outputs/strategic/figures/week2_type_stratified_neff.png`
- `strategic_pipeline/aggregate/week2_personalization.py` (pre-compute + per-focal scorer)
- `strategic_pipeline/aggregate/week2_aggregate.py` (stitch + residualize + diagnostics)
- `strategic_pipeline/slurm/run_week2_personalization_array.slurm`

## What we are NOT doing

- **No regeneration of the 7,626 alignment_commercialization.md reports** —
  the multiplicative score they currently use is now known to collapse;
  regenerating would lock in that collapse.
- **No commit yet** — branch `week2-personalization-baselines` is
  uncommitted pending review.
- **No Week-2B backtest** until the personalization decision is locked in.

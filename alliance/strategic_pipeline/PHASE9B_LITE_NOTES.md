# Phase 9B-lite — Tie-Robust Validation

**Status:** Three crisp findings.
- **Coverage gap is the dominant constraint.** Across 6 backtest years
  (T = 2009–2014, 1,638 directed realized rows), only **7.3%** are
  reachable by the recommender (Compustat focal AND candidate had
  prior L2 activity).  72% have a non-Compustat focal, 12.6% are
  genuine new L2 entrants, and cross-layer-conversion candidates are
  vanishingly rare (0.2%).
- **Within the saturated brokerage frontier, degree-style annotations
  identify realized partners (mean P ≈ 0.82, CI excludes 0.5);
  durable-rent annotations point in the WRONG direction**
  (`annotated_value_synthetic` mean P = 0.15, CI [0.12, 0.19]).
  Realized partners are systematically less persistent, less
  systemic, and lower dependency-risk than other candidates inside
  their tie block.
- **Sales association at focal-year level is null at this sample
  size** (N = 35–43 per regression cell; all 95% CIs span 0).

## A. Coverage breakdown

For each year T ∈ {2009, …, 2014}, every realized new L2 dyad
(focal, candidate, T) is classified into one of seven taxonomy
buckets.  Pooled counts:

| Coverage class                          | Total | % of 1,638 |
|-----------------------------------------|------:|-----------:|
| **focal_not_compustat**                 | 1,183 |     72.2 % |
| candidate_genuine_new_L2_entrant        |   206 |     12.6 % |
| **candidate_in_pool** (← reachable)     |   120 |      7.3 % |
| candidate_prior_L2_excluded_by_pool     |    62 |      3.8 % |
| candidate_prior_SDC_no_L2               |    63 |      3.8 % |
| candidate_prior_nonL2_same_focal        |     4 |      0.2 % |
| candidate_unmatched                     |     0 |      0.0 % |

The recommender, by current design, is a tool for choosing among
already-established players.  In a market where 72% of realized new
L2 ties involve a non-Compustat focal and another 12.6% involve a
genuine new L2 entrant, it cannot be a discovery engine — and the
0.2% cross-layer-conversion rate kills the alternative
"recommend conversions from L1/L3/L4" hypothesis empirically.

Per-year breakdown is in
[`outputs/strategic/aggregate/phase9b_lite_coverage_by_year.csv`](../outputs/strategic/aggregate/phase9b_lite_coverage_by_year.csv);
the figure is [`fig phase9b_lite_coverage_flow.png`](../outputs/strategic/figures/phase9b_lite_coverage_flow.png).

## B. Within-saturated-block annotation rankings

For each in-pool realized dyad, compute the realized partner's
within-tie-block percentile under each annotation:

$$
P^{a}_{fc^\star t}
= \frac{\#\{c \in \mathcal B : a_{fct} < a_{fc^\star t}\}
        + 0.5 \cdot \#\{c \in \mathcal B : a_{fct} = a_{fc^\star t}\}}
       {|\mathcal B_{f,t}(c^\star)|}
$$

Random ordering inside the block ⇒ $\mathbb{E}[P^a] = 0.5$.
Aggregated across N = 119 in-pool dyads (1 dropped for block_size < 2),
with 95% CI from a focal-clustered bootstrap (1,000 reps):

| Annotation                       |   N | Mean P | 95 % CI         | Reading |
|----------------------------------|----:|-------:|-----------------|---------|
| **n_hist_ties**                  | 119 | **0.820** | [0.781, 0.857] | ✅ degree-style identifies realized partners |
| **n_current_ties**               | 119 | **0.820** | [0.772, 0.861] | ✅ same |
| **candidate_degree_all**         | 119 | **0.819** | [0.772, 0.861] | ✅ same |
| candidate_degree_l2              | 119 | 0.745 | [0.693, 0.793] | ✅ moderate degree-style |
| w_tenure_smooth                  | 119 | **0.221** | [0.178, 0.274] | ❌ realized partners are LESS tenured |
| w_redundancy                     | 119 | **0.216** | [0.174, 0.257] | ❌ realized partners have HIGHER dependency exposure |
| **annotated_value_synthetic**    | 119 | **0.154** | [0.119, 0.192] | ❌ multiplicative durable-rent ANTI-correlates |
| dep_risk                         |  81 | **0.084** | [0.072, 0.096] | ❌ realized partners are LESS systemic |

Per the user's decision rule:
> $\bar P^a > 0.55$ is weak evidence; $\bar P^a > 0.60$ with CI excluding $0.5$ is real evidence.

**Three annotations clear the strong-evidence bar in the positive
direction** — the realized partner is in the *top 18%* of
degree/portfolio-depth within its saturated brokerage block:
`n_hist_ties`, `n_current_ties`, `candidate_degree_all`.

**The recommender's own durable-rent annotations clear the bar in the
NEGATIVE direction** — the realized partner is in the *bottom 22%*
of `w_tenure_smooth`, *bottom 22%* of `w_redundancy`, *bottom 8%*
of dependency rank (i.e., realized partners are decidedly
non-systemic), and *bottom 15%* of the multiplicative
`annotated_value_synthetic`.  These signals are not just useless;
they actively misdirect.

The Hankel-DMD persistence-vs-acquisition asymmetry says that
*sustained* L2 brokerage (≥ 4 yr) is the state where the sales
premium realizes — not that *more-tenured candidates* should be the
ones we choose for new ties.  The empirical pattern here is that
firms select fresh, low-history, low-systemic-risk partners as new
allies; the recommender's durable-rent score points the other way.

The annotation-percentile figure is
[`fig phase9b_lite_annotation_percentiles.png`](../outputs/strategic/figures/phase9b_lite_annotation_percentiles.png).

## C. Sales-association regression (focal-year level)

Collapse to focal-year: for each (focal, T) with at least one
in-pool realized partner, take the maximum within-block percentile
under each annotation.  Then OLS with year fixed effects + Compustat
controls (log_sales_t, log_assets_t, rd_intensity_t,
n_realized_inpool):

$$
\Delta_h \log\text{Sales}_{f,t+h}
  = \alpha_T + \beta_h \cdot \max_c P^a_{fc t} + \Gamma X_{f,t} + \varepsilon
$$

| Annotation              | h | N focal-yrs | β     | 95% CI            | CI excl. 0 |
|-------------------------|--:|------------:|------:|-------------------|-----------|
| candidate_degree_l2     | 4 |          35 | +0.82 | [−0.18, +1.66]    | NO        |
| dep_risk                | 4 |          22 | +7.54 | [−6.22, +12.37]   | NO        |
| w_redundancy            | 4 |          35 | +0.54 | [−0.31, +1.63]    | NO        |
| n_hist_ties             | 4 |          35 | +0.01 | [−1.23, +0.69]    | NO        |
| candidate_degree_all    | 4 |          35 | +0.16 | [−1.50, +0.80]    | NO        |
| n_current_ties          | 4 |          35 | +0.16 | [−1.49, +0.79]    | NO        |
| w_tenure_smooth         | 4 |          35 | +0.08 | [−0.61, +1.02]    | NO        |
| candidate_degree_l2     | 2 |          43 | +0.09 | [−0.44, +0.54]    | NO        |
| n_hist_ties             | 2 |          43 | −0.35 | [−0.89, +0.08]    | NO        |
| n_current_ties          | 2 |          43 | −0.20 | [−0.81, +0.25]    | NO        |
| w_redundancy            | 2 |          43 | +0.16 | [−0.32, +0.59]    | NO        |
| dep_risk                | 2 |          32 | −0.39 | [−4.92, +2.41]    | NO        |
| w_tenure_smooth         | 2 |          43 | +0.11 | [−0.35, +0.56]    | NO        |
| candidate_degree_all    | 2 |          43 | −0.20 | [−0.81, +0.25]    | NO        |

**No annotation has a sales β whose CI excludes zero.**  At
N = 35–43 per cell, with year FE eating most of the cross-section
variance and noisy 4-year sales deltas, the test has too little
power to detect anything.  The point estimates are split between
positive (degree, w_tenure) and negative (n_hist_ties, dep_risk)
without consistency.

The sales-association forest is
[`fig phase9b_lite_sales_beta_forest.png`](../outputs/strategic/figures/phase9b_lite_sales_beta_forest.png).

## Decision rules → next moves

| Finding                                              | Decision                                                                                  |
|------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Reach rate 7.3% pooled across 2009-2014              | The recommender is an established-player chooser, not a discovery engine.  Document it.   |
| Cross-layer conversion rate 0.2%                     | Drop the "convert prior non-L2 ties" hypothesis as a Phase 10 candidate.                   |
| Degree annotations P ≈ 0.82, CI excludes 0.5          | Within the saturated frontier, **candidate degree** is a real validated signal.            |
| Durable-rent annotations P < 0.25 (anti-correlated)  | The recommender's value/risk annotations point AWAY from realized choices.  STOP using them as priority signals; keep as descriptive labels only.        |
| Sales β CIs all span 0 at N = 35–43                  | No outcome validation possible at this sample size; do not claim sales lift.               |

The next research problem is **how do we predict new L2 entrants
and identify the established firms most likely to be chosen by them**,
not "how do we tune the current candidate ranker."  The current
ranker's saturated-block ordering is determined by candidate degree,
and degree is what realized choices follow.

## Operational implication for the recommender

The recommender's per-firm `alignment_commercialization.md` should:

1. Continue to rank by `brokerage_L2` (the only focal-specific signal),
   accepting that the saturated tie-block ordering is informative
   only via candidate degree.
2. Display the `annotated_value` column as a **descriptive label**, not
   a recommendation priority.  A "warning" line should note that, in
   the in-pool 2009-2014 backtest, realized partners systematically
   came from the *low* end of `annotated_value` within their block.
3. Add an explicit "coverage" statement: this recommender is for
   firms with prior L2 activity; ~78% of realized new L2 ties involve
   focals or candidates outside its addressable universe.

## Artifacts

- `outputs/strategic/aggregate/phase9b_lite_realized_panel.parquet`
  (1,638 directed rows, 819 unique dyads)
- `outputs/strategic/aggregate/phase9b_lite_coverage_by_year.csv`
- `outputs/strategic/aggregate/phase9b_lite_block_annotation_ranks.csv`
  (119 in-pool dyads × 8 annotations)
- `outputs/strategic/aggregate/phase9b_lite_block_annotation_summary.csv`
- `outputs/strategic/aggregate/phase9b_lite_sales_regression.csv`
- `outputs/strategic/figures/phase9b_lite_coverage_flow.png`
- `outputs/strategic/figures/phase9b_lite_annotation_percentiles.png`
- `outputs/strategic/figures/phase9b_lite_sales_beta_forest.png`
- `strategic_pipeline/aggregate/phase9b_lite_realized_panel.py`
- `strategic_pipeline/aggregate/phase9b_lite_score_realized_focals.py`
  (manifest builder + Slurm task script)
- `strategic_pipeline/aggregate/phase9b_lite_tie_robust_validation.py`
- `strategic_pipeline/slurm/run_phase9b_lite_focal_array.slurm`

Per-focal scoring parquets at score years 2008, 2009, 2011, 2012, 2013
(98 in-pool focals; ~1.5 GB total when including the 7,626 already
present at SY=2010) are intentionally NOT committed; regenerable from
the Slurm array.

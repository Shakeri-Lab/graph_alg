# Week 2B — Out-of-Time Backtest at SCORE_YEAR=2010, T_REALIZED=2011

**Status:** Brokerage_only achieves **93.75% hit rate at K=5** on the
in-pool subsample (15/16 realized 2011 partners caught in the top-5 by
brokerage alone).  Multiplicative variants (raw_score,
durable_value, w_tenure_only) are at **0% at K=5** and only 18.75% at
K=20.  Simple degree baselines do worse than brokerage_only at every
K.  The Week-2A decision rule is empirically validated on the
in-pool subsample, with the small-sample caveat documented below.

## Coverage gap (the dominant finding)

The realized 2011 panel contains 121 unique L2 dyads first appearing
at year 2011 (242 (focal, candidate) rows after materializing both
orderings).  Two filters reduce this to the joinable backtest sample:

| Filter | N rows | % of 242 |
|---|---:|---:|
| Realized 2011 L2 dyads (both orderings) | 242 | 100.0 % |
| Candidate is in 2010 L2 candidate pool | 54 | 22.3 % |
| Candidate in pool AND focal is Compustat-matched | **16** | **6.6 %** |

**78% of realized 2011 partners are GENUINE NEW ENTRANTS to L2** —
their first L2 appearance is at year 2011 itself, so they cannot be
in the recommender's 2010 candidate pool by construction.  The
recommender is therefore deaf to the majority of new alliance
formations because the brokerage-based candidate pool requires
prior L2 activity.

This is itself a substantive finding: the recommender's reach
is bounded by the structural constraint that candidates must have
non-empty L2 neighborhoods in the pre-rank window.  In a market where
~3/4 of new ties involve genuine new entrants, the recommender as
currently designed is a tool for choosing among the established
players, not for spotting new entrants.

## Hit rate (the central diagnostic)

For the 16 in-pool realized dyads, what fraction are caught by each
ranker's top-K?

| Ranker | K=5 | K=10 | K=20 | K=50 | K=100 |
|---|---:|---:|---:|---:|---:|
| **`brokerage_only`** | **0.9375** | **0.9375** | **0.9375** | **0.9375** | **0.9375** |
| `raw_score` (Week-1 full)        | 0.000  | 0.000  | 0.1875 | 0.1875 | 0.1875 |
| `durable_value` (Week-1 partial) | 0.000  | 0.000  | 0.1875 | 0.1875 | 0.1875 |
| `w_tenure_only`                  | 0.000  | 0.000  | 0.1875 | 0.1875 | 0.1875 |
| `n_current_ties`                 | 0.000  | 0.0625 | 0.1250 | 0.1875 | 0.3750 |
| `candidate_degree_l2`            | 0.0625 | 0.0625 | 0.0625 | 0.1875 | 0.3750 |
| `candidate_degree_all`           | 0.000  | 0.000  | 0.1250 | 0.1875 | 0.3750 |

**Brokerage_only is dominant at every K.** At K=5 it catches 15 of
16 realized partners (93.75%); the next-best ranker is
`candidate_degree_l2` at 1 of 16 (6.25%).  The four multiplicative
variants of the durable-rent score (`raw_score`, `durable_value`,
`w_tenure_only`, and the implicit Week-1 product) catch zero of 16
at K=5 because they collapse to one universal Stanford-style
candidate that almost never matches the realized partner.

**Reading.**  The Week-2A diagnostic predicted this exactly: brokerage
is the only focal-specific feature in the current pipeline, and the
multiplicative score destroys that focal-specificity by leaning on
candidate-side features that all peak at the same one-or-two
candidates.  The out-of-time backtest confirms the diagnostic in
the strongest possible way: brokerage_only is operationally
informative at K=5; the multiplicative score is operationally
worthless at K=5.

**Caveat (do not over-read N=16).**  Sixteen dyads is a directional
read, not a powered test.  We cannot rule out that the dominance
collapses with a larger sample, but the gap (15/16 vs 0/16 at K=5)
is sufficiently large that we expect it to hold qualitatively under
multi-year pooling (Phase 9B).

## Sales lift (preliminary, very small N)

For the realized in-pool dyads, mean Δ log Sales of the focal at
horizons h=2 (2013-2011) and h=4 (2015-2011), conditional on whether
the realized partner was in the ranker's top-K.  All cells have
N ∈ {1, 2, 3, 4, 5}; results are illustrative only.

Selected cells at K=20:

| Ranker | h | N in topK | N out topK | mean Δ in topK | mean Δ out topK | lift |
|---|---:|---:|---:|---:|---:|---:|
| `raw_score`         | 2 | 1 | 5 | −0.049 |  0.012 | **−0.061** |
| `raw_score`         | 4 | 1 | 1 | −0.590 |  0.325 | **−0.916** |
| `n_current_ties`    | 2 | 1 | 5 |  0.222 | −0.042 | **+0.264** |
| `n_current_ties`    | 4 | 1 | 1 |  0.325 | −0.590 | **+0.916** |
| `candidate_degree_l2`| 2 | 1 | 5 |  0.123 | −0.022 | **+0.145** |

The pattern (multiplicative score → negative lift, degree-style
ranker → positive lift) inverts the brokerage hit-rate finding and
deserves a multi-year look before any interpretation.  At N ≤ 5 per
cell, these are not powered comparisons.

## Persistence (degenerate at t=2011)

`sustained_persist = 1` if the dyad re-appears in any layer for ≥ 3
years over [2012, 2016].  At t=2011 the count is **0/16** — none of
the in-pool realized 2011 partners re-appear over the next 5 years.
This is consistent with the broader 0% sustained-share across all
121 realized dyads and reflects SDC's announcement-record semantics
(deals are not re-recorded just because the alliance remains active).
The persistence test cannot be run meaningfully on this slice.

## Implications

1. **Use `brokerage_only` as the operational ranker** (Week-2A's
   decision rule, now empirically validated at the K=5 level for the
   16 in-pool dyads).  Apply `w_tenure × g(R&D) × exp(-ρ·DepRisk)` as
   a value scalar attached to the brokerage-top-K, not as inputs to
   the rank.

2. **Be honest about the recommender's reach.**  Of new 2011 L2
   formations, 78% involve a partner the recommender by construction
   cannot rank.  Per-firm reports should label this gap: "this
   recommender ranks among firms with prior L2 activity in the 5-year
   window; ~78% of new alliance formations involve genuine new
   entrants we cannot score."

3. **Persistence is not measurable from announcement data alone.**
   The 0% sustained-share over [t+1, t+5] for 121 realized 2011 dyads
   is an artifact of SDC recording the START of each deal but not
   its duration.  This invalidates the persistence-vs-acquisition
   asymmetry as something we can test directly via tie re-appearance.
   A different operationalization (e.g., portfolio composition over
   the focal firm's stress reports) is required.

## Phase 9B (multi-year pooling) — recommended

The N=16 sample is enough to see the brokerage_only vs. multiplicative
gap, but not enough to power the sales-lift comparison or to
characterize the new-entrant share more precisely.  Phase 9B should
pool five backtest years (t = 2009, 2010, 2011, 2012, 2013) so the
combined in-pool sample reaches N ≈ 80–100 dyads:

- Pre-compute `candidate_features_<year>.parquet` for years 2008,
  2009, 2011, 2012 (~5 min each; 2010 and 2017 already done).
- Submit four 7,626-task Slurm arrays at SCORE_YEAR ∈ {2008, 2009,
  2011, 2012} writing to per-year subdirs.  ~12-20 cluster-hours total
  at recent drain rates.
- Build `week2b_realized_ties_<t>.parquet` for t ∈ {2009, 2010, 2011,
  2012, 2013}.
- Extend `week2b_backtest.py` to pool over (SCORE_YEAR, T) pairs and
  re-run hit-rate / sales-lift / persistence at the larger combined
  sample.

## Artifacts (Phase 9A only)

- `outputs/strategic/aggregate/week2b_realized_with_ranks_2011.parquet`
- `outputs/strategic/aggregate/week2b_hit_rate_by_ranker.csv`
- `outputs/strategic/aggregate/week2b_sales_lift_by_ranker.csv`
- `outputs/strategic/aggregate/week2b_backtest_summary.csv`
- `outputs/strategic/figures/week2b_hit_rate.png`
- `outputs/strategic/figures/week2b_sales_lift.png`
- `strategic_pipeline/aggregate/week2b_outcomes.py`
- `strategic_pipeline/aggregate/week2b_backtest.py`
- `strategic_pipeline/slurm/run_week2b_scoring_2010.slurm`

Per-focal scoring parquets at year=2010 (7,626 files, ~1.2 GB) are
intentionally NOT committed to git — regenerable from the Slurm array
on `candidate_features_2010.parquet` (which IS committed).

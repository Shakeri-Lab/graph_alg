# Week 2B — Out-of-Time Backtest at SCORE_YEAR=2010, T_REALIZED=2011

> **UPDATE — tie audit has reversed the Phase-9A headline.**  The "15/16 at K=5"
> result reported below was an artifact of deterministic dataframe-sort
> tie-breaking inside saturated brokerage blocks.  Under random tie-breaking,
> the K=5 hit rate is **0.16%**, statistically indistinguishable from
> picking K=5 candidates uniformly at random from a 2,896-firm pool.
> See the **Brokerage tie audit** section near the bottom of this file
> for the corrected reading.  The Week-2A diagnostic still stands
> (multiplicative scores collapse), but the Week-2B claim that
> brokerage_only's hit rate validates the decision rule does NOT.

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

---

# Brokerage tie audit — corrects the Phase-9A headline

The Phase-9A "15/16 at K=5" hit rate for `brokerage_only` was suspect on
its face: brokerage_L2 saturates at 1.0 for ~99% of candidates in
sparse-portfolio focals, so a 15/16 result probably reflects the
*deterministic* dataframe-sort tie-breaking inside saturated tie blocks
rather than a real signal.  We quantified this with
`week2b_brokerage_tie_audit.py`.

For each realized in-pool dyad we compute:

- $r_{\min}(f, c^\star) = 1 + \#\{c : \text{brokerage}(f,c) > \text{brokerage}(f,c^\star)\}$
- $r_{\max}(f, c^\star) = \#\{c : \text{brokerage}(f,c) \ge \text{brokerage}(f,c^\star)\}$
- tie-block size = $r_{\max} - r_{\min} + 1$.

Then three hit-rate variants per top-K:

- $\text{Hit}^{\text{optimistic}}@K = \mathbf{1}\{r_{\min} \le K\}$
  (best case: $c^\star$ is at the head of its tie block).
- $\text{Hit}^{\text{pessimistic}}@K = \mathbf{1}\{r_{\max} \le K\}$
  (worst case: $c^\star$ is at the tail).
- $\text{Hit}^{\text{random tie}}@K$ = expected hit under uniform
  within-tie ordering.

## What we found

| Metric                 | Value (16 in-pool dyads, 2010 → 2011) |
|------------------------|---------------------------------------|
| 15 of 16 dyads have    | $\text{brokerage}_{L_2} = 1.0$        |
| Median tie-block size  | **2,896** candidates                  |
| Mean tie-block size    | 2,711                                 |
| 16th dyad              | brokerage = 0.5, $r_{\min} = r_{\max} = 2{,}892/2{,}892$ (dead last) |

| K | Optimistic | Pessimistic | Random-tie |
|---:|---:|---:|---:|
|   5 | 0.9375 | 0.0000 | **0.00162** |
|  10 | 0.9375 | 0.0000 | **0.00324** |
|  20 | 0.9375 | 0.0000 | **0.00648** |
|  50 | 0.9375 | 0.0000 | **0.01621** |
| 100 | 0.9375 | 0.0000 | **0.03242** |

**Reading.**  Under random tie-breaking the brokerage_only hit rate at
K=5 is 0.162%.  The chance of picking the realized partner by
selecting K=5 candidates uniformly at random from a 2,896-firm pool
is $5/2896 \approx 0.173\%$.  These are statistically the same number.
**Brokerage_only has no measurable predictive power on this 16-dyad
sample once the tie ambiguity is accounted for.**

The "15/16 at K=5" Phase-9A number arose because the dataframe sort
happened to put the realized partner near the top of its saturated
tie block — an artifact of the row order in the per-focal scoring
parquets.  A different but equally legitimate sort order would have
produced "0/16 at K=5".  The optimistic and pessimistic bounds are
0.9375 and 0.0 at every K, with an ambiguity band of 0.94 — almost
the entire $[0, 1]$ interval.

## Implications for the recommender

1. **The brokerage saturation is the dominant problem**, more so than
   the choice of multiplicative versus brokerage-only score.  Within
   the saturated brokerage = 1.0 region — which contains essentially
   the entire candidate pool for almost every Compustat focal — the
   recommender has no informative ranking signal, regardless of which
   feature combination we use to break ties.

2. **The Week-2A decision rule still stands**: a multiplicative score
   that combines candidate-side features with brokerage collapses to
   one universal candidate (Stanford), so it is strictly worse than
   ranking by brokerage_L2 alone.  But "ranking by brokerage_L2 alone"
   is itself nearly information-free on the realized-tie test, because
   nearly every candidate has the maximum brokerage value.

3. **The recommender as currently designed is most useful as a
   *frontier annotator*, not a *ranker*.**  For a focal firm, the
   brokerage = 1.0 candidates are the "everyone in your potential
   neighborhood" set, and the value/risk annotations
   (`w_tenure_smooth`, `g(R&D)`, `w_redundancy`) are the only signal
   that distinguishes them.  Per-firm reports should be honest about
   this: the order of the top-K is partially arbitrary; the annotation
   columns are the substantive content.

4. **Sample-size caveat is no longer the binding constraint.** Even
   with 100× more dyads from multi-year pooling (Phase 9B), the
   brokerage saturation issue remains: most realized partners would
   still sit inside the saturated block.  Phase 9B should be
   redesigned to focus on the **annotation-conditional** hit rate
   (does `annotated_value` rank do better than `brokerage_L2` rank
   *within the saturated block*?) and on the **coverage breakdown**
   (what fraction of realized partners is even in the saturated
   block vs. genuinely new entrants), not on the brokerage ranking
   per se.

## Code reframe (committed alongside this update)

`alignment_recommender.py` was reframed to honor the audit finding:

- Sort key = `brokerage_L2` only.  No candidate-side tiebreaker.
  Earlier versions used `[score_durable_rent, durable_value,
  n_current_ties]` as the sort, which recreated Stanford-style
  collapse inside the saturated block.
- New column `annotated_value` = $\text{brokerage}_{L_2} \times
  w_{\text{tenure}} \times g(\text{R\&D}) \times w_{\text{redundancy}}$
  is reported but NOT used to reorder.
- `df.attrs["reranker"]` renamed to
  `"brokerage_frontier_with_annotations"` with a message that
  references this audit.
- `report_writer.py` narrative updated to describe the brokerage-frontier
  framing: "Candidates are ranked by focal-specific L2 brokerage
  opportunity. Durability, absorptive capacity, and systemic-dependency
  risk are reported as value/risk annotations on the brokerage frontier;
  they are NOT used to reorder the candidate list."

## Phase 9B redesign

The original Phase 9B plan ("pool 5 backtest years to bulk the
sample") is no longer the right next step.  The binding constraint is
brokerage saturation, not sample size.  A revised Phase 9B should
answer:

1. **Coverage breakdown**: of all realized 2011 L2 ties, what fraction
   falls into each of these categories?
   - Candidate has prior L2 activity → ranked by brokerage (16 of 242)
   - Candidate has prior non-L2 SDC activity → could be reachable with
     a multi-layer brokerage definition
   - Candidate is in firm-meta but no prior alliance activity →
     genuine new alliance-network entrant
   - Candidate is unmatched / identifier failure → data hygiene
2. **Within-saturated-block ranking**: does the `annotated_value`
   ordering beat random tie-breaking *among brokerage = 1.0 candidates*?
   This isolates the annotation signal from the saturation artifact.
3. **Continuous rank-percentile sales association**: regress
   $\Delta_h \log\text{Sales}$ on $\text{rank\_pct}^B$ (1 − r/N) using
   all in-pool realized dyads pooled across multiple t-years, with
   year fixed effects and Compustat controls.  This sidesteps the
   tiny top-K cell counts that made the Phase-9A sales-lift table
   uninterpretable.

These three questions can be answered without 4 more 7,626-task Slurm
arrays.  Question 1 and Question 2 use the existing per-focal parquets
at year=2010; Question 3 needs additional realized panels at t ∈
{2009, 2012, 2013} but only 2-3 additional per-focal-array runs (or
a multi-year refactor of the 2010 array).

## Audit artifacts

- `outputs/strategic/aggregate/week2b_brokerage_tie_audit.csv` (16 rows)
- `outputs/strategic/aggregate/week2b_tie_audit_hit_rates.csv` (5 rows
  × 6 columns)
- `strategic_pipeline/aggregate/week2b_brokerage_tie_audit.py`

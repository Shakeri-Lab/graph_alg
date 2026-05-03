# Week 2C — Focal-Conditional Interaction Features

**Status:** None of the cheap focal-conditional interaction features
beats the Week-2A `brokerage_only` baseline.  The recommender's
personalization ceiling under the current data is therefore
$N_{\mathrm{eff}}^{\text{top-1}} \approx 1{,}812$ across 7,626 focal
firms.  Week 2B can proceed with `brokerage_L2` as the rank-driver,
without holding for further feature engineering.

## What this run tested

We computed seven new $(f, c)$-conditional features on the same
18,064,303-row Week-2A scoring frame:

| Feature              | Definition |
|----------------------|------------|
| `n_shared_partners`  | $\lvert N(f) \cap N(c) \rvert$ in the 5-yr window, any layer |
| `jaccard_partners`   | $\lvert N(f) \cap N(c) \rvert / \lvert N(f) \cup N(c) \rvert$ |
| `share_focal_in_c`   | $\lvert N(f) \cap N(c) \rvert / \lvert N(f) \rvert$ |
| `same_sic2`          | $\mathbf 1\{\text{sic}_2(f) = \text{sic}_2(c)\}$ |
| `same_sic1`          | $\mathbf 1\{\text{sic}_1(f) = \text{sic}_1(c)\}$ |
| `sic2_distance`      | $\lvert \text{sic}_2(f) - \text{sic}_2(c) \rvert$ |
| `nation_match`       | $\mathbf 1\{\text{nation}(f) = \text{nation}(c)\}$ |

These are the cheapest focal-conditional signals available in the
existing data (no new model fitting, no new external data).  The
hypothesis was that any of these would let the recommender produce
focal-specific top picks beyond what `brokerage_L2` alone delivers.

## Headline result

| Ranker                              | $N_{\mathrm{eff}}^{\text{top-1}}$ | Top candidate                | Top share |
|-------------------------------------|-----------------------------------:|------------------------------|-----------:|
| **`brokerage_only` (Week-2A baseline)** | **1,811.9**                     | Acuity Brands Inc            |    0.13 % |
| `jaccard_only`                      | 1,734.1                            | New York Power Authority     |    0.20 % |
| `n_shared_partners_only`            | 1,630.3                            | General Electric Co          |    0.43 % |
| `share_focal_in_c_only`             | 1,630.3                            | General Electric Co          |    0.43 % |
| `nation_match_only`                 | 1,521.1                            | Twitter Inc                  |    0.16 % |
| `same_sic2_only`                    | 1,125.5                            | HSBC                         |    0.39 % |
| `broker_x_jaccard` (rank-sum)       | 1,732.2                            | New York Power Authority     |    0.20 % |
| `broker_x_shared` (rank-sum)        | 1,626.5                            | General Electric Co          |    0.43 % |
| `broker_x_nation` (rank-sum)        | 1,521.6                            | University of Guelph         |    0.16 % |
| `broker_x_sic2` (rank-sum)          | 1,125.5                            | HSBC                         |    0.39 % |
| `full_interaction` (rank-sum, all)  | 831.2                              | Everi Holdings Inc           |    0.84 % |
| `raw_score` (Week-2A full score)    | 1.000                              | Stanford University          |   100.0 % |

`brokerage_only` remains at the top.  Every other ranker ties or
underperforms.  The full rank-sum combining brokerage with three
interaction features actually **halves** the personalization to
$N_{\mathrm{eff}} \approx 831$.

## Why the interaction features didn't help

Each focal-conditional feature is *focal-conditional* in the sense that
its value depends on $f$, but at the *top of each focal's rank* it
collapses back into a candidate-global pattern:

- `n_shared_partners` and `share_focal_in_c` rank degree-heavy
  candidates universally high (GE wins).  Even though the value differs
  across focals, the *argmax* candidate is the same hub almost
  everywhere.
- `jaccard_partners` is similar, with one normalization layer.  The
  top is dominated by candidates with many partners overlapping a
  typical focal portfolio.
- `same_sic2` partitions candidates into a "match" set and a "no-match"
  set per focal, but within the match set the order is random — so
  whichever same-SIC candidate appears first wins every focal in that
  SIC.  HSBC dominates because it shares SIC with many financial
  focals.
- `nation_match` has the same structure.  The entire "matching"
  partition collapses to the first-listed nation-mate.

The rank-sum combinations carry this dilution into brokerage's
ranking and reduce it.  This is the same pathology Week 2A surfaced,
just with a slightly different mechanism: when a feature is binary or
ordinal-with-few-levels and many candidates share its top value, the
within-focal argmax is determined by tie-breaking, not by the feature.

## Implication for Week 2B

The Week-2A decision rule is reaffirmed and tightened:

1. **Rank by `brokerage_L2(f, c)`.** This is the personalization
   ceiling under currently available features.  Adding any of the
   tested interaction features either ties or decreases personalization.
2. **Do not multiply or rank-sum brokerage with these features.** Use
   them — if at all — as filters or as components of a *value scalar*
   attached to the brokerage-top-K, not as inputs to the rank itself.
3. **Recognize the ceiling.** $N_{\mathrm{eff}} \approx 1{,}812$ across
   7,626 focals means the most-frequent top-1 candidate appears in
   $\approx 0.13\%$ (about 10) of focal-firm reports.  This is a
   meaningful level of focal-specificity but is not "every firm gets
   its own bespoke partner."  The per-firm reports should be honest
   about this: roughly 1 in 750 focals will see Acuity Brands at #1,
   not because Acuity is uniquely fitted to that focal but because
   Acuity has the most focal-non-overlapping L2 portfolio in the
   2017 candidate pool.

To exceed $N_{\mathrm{eff}} = 1{,}812$ a future Week 2C+ would need
*new data* the current pipeline does not expose:

- A tie-formation hazard model conditional on $(f, c)$ covariates
  (would estimate $\Pr(\text{tie}|f, c)$ rather than relying on
  observed structural overlap).
- Cross-portfolio similarity of the focal and candidate's strategic
  position (e.g., earnings-call topic embeddings, patent-class
  similarity, technology-class distance).
- Geographic / supply-chain proximity beyond nation match.
- Prior corporate transactions (board interlocks, director overlap,
  M&A history with the same target).

These are real options for a Week 3 or later phase, but each requires
either new external data or substantial modeling work.

## Artifacts

- `outputs/strategic/aggregate/week2c_personalization_rows_2017.parquet`
  (~620 MB; same 18M rows + 7 interaction features)
- `outputs/strategic/aggregate/week2c_top1_concentration_by_variant.csv`
- `outputs/strategic/aggregate/week2c_overlap_matrix_by_variant.csv`
- `outputs/strategic/aggregate/week2c_interaction_summary.csv`
- `outputs/strategic/figures/week2c_top1_concentration.png`
- `outputs/strategic/figures/week2c_overlap_heatmap.png`
- `outputs/strategic/figures/week2c_brokerage_vs_interaction_lift.png`
- `strategic_pipeline/aggregate/week2c_interaction_features.py`
- `strategic_pipeline/aggregate/week2c_aggregate.py`

## What we are NOT doing

- **No further feature-engineering loops** chasing $N_{\mathrm{eff}}$
  improvements with the existing data.  The diagnostics show
  `brokerage_L2` is the ceiling under in-data features.
- **No regeneration of the 7,626 per-firm reports** until Week 2B
  closes the rank-vs-value-scalar separation.
- **No commit to main** — Week 2C lands on the same
  `week2-personalization-baselines` branch as a follow-on diagnostic.

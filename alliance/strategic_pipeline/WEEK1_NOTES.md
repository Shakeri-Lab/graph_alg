# Week 1 — Durable-Rent Reframe: Canary Findings

**Status:** Score functional, hyperparameters not frozen.  Full
regeneration of the 7,626 per-firm reports is **on hold pending the
Week 2 out-of-time backtest.**

## What the canary tested

21 focal firms (`intermediate/canary_cusips.csv`) span:

- **Six 2-digit SIC sectors**: 28 (chemicals/pharma), 73 (services
  /software), 67 (financials), 87 (engineering), 48 (communications),
  38 (instruments), plus 36 (electronics) and 59 (retail) via the
  hand-picked hubs.
- **Three portfolio-density tiers** within each SIC: high L2-degree,
  median L2-degree, low L2-degree (single tie).
- **Two known systemic hubs**: General Electric (369604) and
  Amazon.com (023135).
- **Both top-quartile and below-quartile R&D focal firms.**

For each firm the alignment recommender was run at year=2017 under the
Week-1 score:

```
durable_value(c)        = brokerage_L2(focal, c)  ×  w_tenure_smooth(c)
w_redundancy(c)         = exp(-1.5  ×  DepRisk(c))    if observed
                        = 1.0                          if unobserved
score_durable_rent(c)   = durable_value(c)  ×  w_redundancy(c)
g(R&D_focal)            = 1 + 0.5 · 1{focal ∈ top-quartile R&D}
```

with `w_tenure_smooth = 0.5 + n/(n+5) · (σ(z) − 0.5)` (κ=5
shrinkage), z capped at ±3, and the n<2-tie default of 0.5.

## Headline finding: rankings collapse to a population-wide list

Across 21 focal firms, the top-1 candidate is one of just two firms:

| Top-1 candidate          | n_ties | Times appears at #1 |
|--------------------------|-------:|--------------------:|
| Stanford University      |     28 |              16 / 21 |
| Office Depot Inc         |     11 |               5 / 21 |

The mechanism: brokerage_L2 saturates at 1.0 for ≥99% of candidates
(focal L2 portfolios are tiny, so almost every candidate's L2
neighborhood is non-overlapping).  After saturation, the only
differentiator is the candidate's own `n_ties × w_tenure_smooth × w_redundancy` — none
of which depend on the focal firm.  The recommender therefore
collapses toward a single global ranking modulated only by which
candidates happen to be in each focal's 5-yr L2 window.

**This is a real problem.**  A score that produces the same top-5 for
Microsoft, MaxCyte (a 200-employee biotech), and General Electric is
not a strategic recommender — it is a *population* ranking.  The
Week 2 backtest must address this directly.

## DepRisk semantics worked as intended

`dep_risk_observed` distinguishes "in systemic panel" from "not
ranked."  The observed share among top-20 picks varies meaningfully:

| share_dep_observed | n focal firms | example focals                     |
|-------------------:|--------------:|------------------------------------|
|           0.30–0.35 |             5 | Qualcomm, Linedata, AccuWeather, Aimia, Amazon |
|           0.10–0.15 |            16 | most others                                    |

Firms with higher observed-DepRisk shares are those whose L2
candidate pools intersect more with the systemic top-2,939
(typically firms in services / media / consumer sectors).  None of
the canary's top-1 picks are themselves observed-DepRisk hubs in this
sample, but the lookup is wired correctly (verified independently:
GE w_redundancy=0.22, Microsoft 0.27, IBM 0.49).

## Shrinkage behaved sensibly

The κ=5 small-n shrinkage suppressed the n=2 outlier that ranked #1
under the unshrunk version.  Via Licensing Corp (n=2, median tenure
1.5 yr, z=3 clipped) drops from w=0.95 → w=0.63 and falls out of
top-3 in every canary report.  Multi-tie portfolios (Stanford 28,
Johns Hopkins 12, Office Depot 11) now lead.

## R&D gate

Only **Amazon (023135)** triggers the R&D top-quartile gate among
the 21 canary firms (g_R&D = 1.5).  All other firms — including
Microsoft, Qualcomm, Johnson & Johnson, GE — fall outside top-quartile
*within their SIC* in 2017.  This is consistent with the paper's H2
finding (the L2 sales premium is concentrated in within-industry
R&D leaders, not absolute leaders).  But because g_R&D is per-focal
and constant within a single firm's report, it does **not** change
within-firm rankings — it only matters for cross-firm comparisons,
which the canary does not test.

## Open questions for the Week 2 backtest

1. **Does the score collapse persist out of time?**  If 1991–2010
   training produces "Stanford / Office Depot" top picks for almost
   every 2011 focal, the score is universally non-discriminating and
   needs a focal-conditional component (e.g., a focal-candidate
   interaction term, candidate's industry adjacency to focal,
   geographic proximity, prior tie strength).  The current score has
   only one focal-specific input (brokerage_L2), and that input
   saturates.

2. **Does w_tenure improve durable-tie prediction?**  Train on
   1991–2010 realized L2 ties, predict $\Pr(T \ge 3)$ on 2011 ties,
   evaluate on 2012–17 outcomes.  Compare AUC for: brokerage only;
   brokerage × w_tenure; brokerage × w_tenure × g(R&D); full score.

3. **Does ρ=1.5 reduce fragile dependency without killing value?**
   Sweep ρ ∈ {0, 0.5, 1.0, 1.5, 2.0, 3.0}; report top-k churn
   *and* mean dep_risk in the top-k for each ρ.  If the top-k just
   shuffles among unobserved-DepRisk candidates, ρ is doing nothing
   for typical focals.

4. **Does the score enrich for later sales response?**  For focal
   firm-years that form L2 ties, test whether high-scoring realized
   partners are associated with higher $\Delta_h \log \text{Sales}$
   at $h=2,4$.  This is the direct test of whether the durable-rent
   ranking recovers the DMD finding out of time.

## What we are NOT doing

- **No full regeneration of the 7,626 per-firm reports.**  Until the
  hyperparameters (κ, ρ, R&D threshold, n<2 default, missing-DepRisk
  semantics) are validated by Week 2, regenerating would create a
  large polished artifact around values we may need to change.
- **No causal claims** in any per-firm report.  The score is
  associational; it is a structurally defensible re-ordering of
  brokerage_L2 that incorporates behavioral signals, not a forecast
  of joint future value.

## Artifacts

- `outputs/strategic/aggregate/canary_summary.csv` — one row per
  canary firm with mean w_tenure, w_redundancy, score, and top-1
  candidate.
- `outputs/strategic/<cusip>/alignment_commercialization.md` — full
  per-firm reports for the 21 canary firms (with the new score, the
  shrinkage, and the dep_risk_observed flag).
- `outputs/strategic/<cusip>/fig_alignment_commercialization_frontier.png`
  — 2-axis (DepRisk, durable_value) scatter with quadrant labels
  and a separate "DepRisk unobserved" point class.
- `intermediate/canary_cusips.csv` — the canary firm list.
- `strategic_pipeline/slurm/run_canary_alignment.slurm` — the array
  script used (now defaults to `--account=cdt_computing`).

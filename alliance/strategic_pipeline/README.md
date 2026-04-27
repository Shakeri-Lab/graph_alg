# Strategic Decision Pipeline

A per-firm decision tool that operationalizes the findings of the
consolidated alliance-network study (`docs/123-all/main.tex`).
Given a focal firm (by ultimate-parent CUSIP), the pipeline produces
Markdown reports + PNG figures answering three strategic questions:

1. **Alignment** — who should we partner with, given an Innovation or
   Commercialization goal?
2. **Timing** — when to form new ties, when to stop, when is "too many"?
3. **Stress** — true centrality (full graph vs Compustat-only) and
   per-partner exit vulnerability, with empirical TWFE and DMD
   counterfactual estimates side-by-side.

## Directory layout

```
strategic_pipeline/
├── __init__.py
├── README.md                   (this file)
├── data_loader.py              unified loader for intermediate/ artifacts
├── firm_profile.py             per-firm identity, centrality, tenure, R&D
├── scoring_primitives.py       closure/brokerage, rd_gate, exit_impact
├── alignment_recommender.py    Q1: candidate partner ranking
├── timing_dashboard.py         Q2: tenure distribution + STOP/GO signal
├── portfolio_stress_test.py    Q3: centrality gap + partner vulnerability
├── run_firm.py                 CLI entry point
├── report_writer.py            Markdown + PNG output rendering
└── slurm/
    ├── run_single_firm.slurm
    └── run_all_firms_array.slurm
```

## Local use (Python CLI)

```bash
cd /sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance
module load miniforge/24.11.3-py3.12

# One question, one firm
python -m strategic_pipeline.run_firm --cusip 747525 --year 2005 \
    --question alignment --goal commercialization

# All three questions
python -m strategic_pipeline.run_firm --cusip 747525 --year 2005 \
    --question all
```

Outputs land in `outputs/strategic/<cusip>/`:
- `alignment_innovation.md` + `alignment_innovation_top.csv`
- `alignment_commercialization.md` + `alignment_commercialization_top.csv`
- `timing.md` + `fig_tenure_distribution.png`
- `stress.md` + `fig_centrality_true_vs_compustat.png` +
  `fig_partner_vulnerability.png`

## Slurm use

Single firm (15 min, 16 GB):

```bash
sbatch --export=ALL,CUSIP=747525,YEAR=2005,QUESTION=all \
       strategic_pipeline/slurm/run_single_firm.slurm
```

All 7,626 Compustat-matched firms (array job, 50 concurrent):

```bash
# First, build the CUSIP list (one-time):
python -m strategic_pipeline.run_firm --array-index 1 --question all

# Then submit the array:
sbatch strategic_pipeline/slurm/run_all_firms_array.slurm
```

Wall-clock estimate for the array: ~6 hours at 50 concurrent × 15 min each.

## What each module consumes

All modules read from `intermediate/` artifacts produced by the
upstream analysis; none re-compute raw edges or fit regressions.

| Module | Key inputs |
|---|---|
| `alignment_recommender.py` | `pairwise_edges_imputed.parquet`, `static_covariates.parquet`, `firm_year_panel.parquet` |
| `timing_dashboard.py` | `pairwise_edges_imputed.parquet`, paper-Table-3 coefficients (hard-coded) |
| `portfolio_stress_test.py` | `layer_betweenness_panel.parquet`, `pairwise_edges_imputed.parquet`, DMD operator (refit from `trajectory_panel.parquet`) |

The DMD operator (pooled Hankel-DMD on 851 long-panel firms, $r=25$) is
refit lazily on the first call and cached for the rest of the run via
`lru_cache`. The peak-match scale factor (2.4333) comes from the
Phase-4-v2 aligned analysis in `phase4_v2_aligned.py`.

## Known limitations

- **Time window**: pipeline is calibrated on SDC Platinum 1991–2017.
  Firms formed after 2017 or primarily active outside SDC's coverage
  are not in scope.
- **DMD eligibility**: partner-exit simulation via DMD requires the
  focal firm to have ≥6 consecutive trajectory years (subset of the
  851 long-panel firms). Firms without adequate trajectory get
  `dmd_loss_log_mv = NaN` with `dmd_available = False`; the report
  marks these clearly. Empirical TWFE baseline is always available.
- **Causal inheritance**: empirical exit-impact estimates inherit the
  M2 paper's identifying assumptions (L$_2$-specific sudden partner
  exits, parallel trends verified).
- **R&D quartile**: computed within 2-digit SIC in the focal year; uses
  MICE-imputed R&D where Compustat is null. The gate is a signal,
  not a hard requirement — low-R&D firms still receive L$_2$
  recommendations, flagged with a warning.
- **Alignment recommendations are associational**: closure and
  brokerage scores measure structural fit, not causal forecasts of
  future joint value. Treat as a ranked shortlist for qualitative
  due diligence, not as auto-deals.
- **Redundancy audit**: counts substitutes in the same (layer, SIC2)
  bucket. Does not account for partner quality.
- **DMD under-prediction at long horizons**: the pooled linear operator
  under-predicts persistence at $t{+}2$ and $t{+}3$ relative to the
  empirical event study (see paper §6). DMD estimates are a first-order
  sanity check, not a replacement for the empirical TWFE baseline.

## Validation

Verified smoke test on Qualcomm (cusip `747525`, year 2005):
- Alignment/innovation: top candidates = Microsoft, IBM, Altera,
  Siemens (plausible high-closure semiconductor/software partners).
- Alignment/commercialization: R&D gate flags Qualcomm as Q2 within
  SIC 36 (not top quartile) and adds the caveat.
- Timing: `CAUTION` flag (zero sustained L$_2$ ties in 2005).
- Stress: all 4 L1/L2/L4 layers show negative `gap` (firm ranks
  HIGHER on Compustat-only subgraph — public-peer view overstates
  centrality). Top-5 critical partners dominated by L$_2$ telecom
  firms (Dilithium, ZTE, Alcatel-Lucent, Egyptian Telephone).

## Re-running upstream analysis

The pipeline reads from `intermediate/` but never writes there.
If the upstream regressions or DMD fit is ever re-run, the pipeline's
outputs will automatically reflect the new artifacts on the next
invocation. The hard-coded paper coefficients in
`timing_dashboard.py::SALES_CASCADE` (Table 3 of the paper) should be
updated manually if the regression output changes.

## Week 1: alignment-recommender durable-rent reframe (directional, not yet backtested)

The commercialization branch of `alignment_recommender.py` has been
restructured around the durable-rent score:

```
durable_value(c)        = brokerage_L2(focal, c)  ×  w_tenure_smooth(c)
w_redundancy(c)         = exp(-1.5  ×  DepRisk(c))
score_durable_rent(c)   = durable_value(c)  ×  w_redundancy(c)
g(R&D_focal)            = 1 + 0.5 · 1{focal ∈ top-quartile R&D}    (per-focal)
```

with:

- `w_tenure_smooth ∈ (0, 1)` — sigmoid of the candidate's
  `log(1+median_tenure)` z-scored against the SIC×L2 cohort
  baseline. Z is clipped to ±3; candidates with fewer than two
  ties default to a neutral 0.5.
- `DepRisk ∈ [0, 1]` — candidate's normalized in-degree in the
  corrected systemic-criticality cross-section
  (`outputs/strategic/aggregate/systemic_criticality.csv`).
- `g(R&D)` — per-focal absorptive-capacity multiplier; affects
  cross-firm comparisons only, not within-firm ranking.

The per-firm `alignment_commercialization.md` report now embeds a
2-axis scatter `(DepRisk, durable_value)` with quadrant labels
(durable bridge / fragile chokepoint / safe but irrelevant /
bad dependency).

**Status:** the score is **directional**, motivated by the paper's
Hankel-DMD persistence-vs-acquisition asymmetry (Section 5) and the
systemic-criticality companion note. It has **not yet been backtested
out of time.** The hyperparameters `κ_shrinkage`, `ρ=1.5`, the R&D
threshold, the SIC×layer minimum cell size, and the "missing DepRisk
= 1.0" convention are placeholders pending the Week 2 backtest
(train 1991–2010, rank 2011, evaluate 2012–17 on top-k persistence
hit rate, sales response, and ΔDepRisk).

**Do not yet treat individual `score_durable_rent` rankings as causal
forecasts of joint future value.** Treat them as a structurally
defensible re-ordering of `brokerage_L2` that incorporates
behavioral signals of alliance-management capability (relational view)
and explicit dependency risk (systemic view).

"""Alignment Recommender — Strategic Question 1.

Given a focal firm and a goal (Innovation or Commercialization), rank
candidate partners by their structural fit:

  - Innovation (L1): closure_score maximized → dense closed triads
  - Commercialization (L2): durable-rent ranking that combines
    structural opportunity (Burt brokerage), relational capability
    (smooth SIC×layer-z-scored tie tenure), focal absorptive capacity
    (R&D top-quartile multiplier), and dependency risk (penalty against
    candidates that are themselves systemically critical).

  Score (commercialization):
    durable_value(c) = brokerage_L2(focal, c) × w_tenure(c)
    w_redundancy(c)  = exp(-RHO_REDUNDANCY × DepRisk(c))
    score(c)         = durable_value(c) × w_redundancy(c)
    g_RD(focal)      = 1 + ALPHA_RD × 1{focal ∈ top-RD-quartile}
                       (per-focal multiplier; reported, does not affect
                       within-firm ranking)

  This implements the Week-1 reframe from "centrality re-ranking" to
  "durable-rent ranking under dependency risk" (paper Sections 4–5 +
  systemic-criticality artifact).

Candidate pool: firms present in the relevant layer's 5-year rolling
window at `year`, excluding current partners.
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from strategic_pipeline.data_loader import (
    DataBundle, load_all, get_firm_edges, get_firm_partners,
)
from strategic_pipeline.firm_profile import build_profile, ROLLING_WINDOW
from strategic_pipeline.scoring_primitives import (
    closure_score, brokerage_score, rd_gate,
)

GOAL_TO_LAYER = {
    "innovation": "L1",
    "commercialization": "L2",
}

# Legacy persistence factor (still computed for transparency in the per-firm
# report; the new score below uses the smooth z-scored w_tenure instead).
PERSISTENCE_MIN_TIES = 2
PERSISTENCE_FLOOR = 0.5
PERSISTENCE_SUSTAINED_AGE = 4

# Smooth tenure-based w_tenure parameters
W_TENURE_LOG_OFFSET = 1.0       # log(1 + T) avoids log(0)
W_TENURE_Z_CLIP = 3.0           # cap |z| before sigmoid → max sigmoid(3)≈0.95
W_TENURE_MIN_TIES_SMOOTH = 2    # need ≥ 2 ties for the median to be informative;
                                  # otherwise return neutral 0.5
W_TENURE_SHRINKAGE_KAPPA = 5.0  # empirical-Bayes-style shrinkage toward 0.5:
                                  # w = 0.5 + (n / (n + κ)) × (σ(z) − 0.5).
                                  # n=2 candidate at z=3 → 0.63 (not 0.95);
                                  # n=28 candidate stays close to σ(z).
SIC_BASELINE_MIN_CELL = 10      # below this, fall back to global L2 baseline
SIC_BASELINE_FALLBACK_STD = 0.5 # for tiny cells where SD is unreliable

# Multiplicative R&D bonus
ALPHA_RD = 0.5                  # top-quartile focal gets 1.5×; others 1.0×

# Dependency-risk penalty
RHO_REDUNDANCY = 1.5            # exp(-1.5 × dep_risk) at dep_risk=1 → 0.22
SYSTEMIC_CSV = Path(
    "/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance/"
    "outputs/strategic/aggregate/systemic_criticality.csv"
)


def _candidate_persistence_factor(bundle: DataBundle, candidate: str,
                                    year: int,
                                    window: int = ROLLING_WINDOW) -> tuple:
    """Return (factor, sustained_share, n_ties) for a candidate.

    Factor is a multiplier in ``[PERSISTENCE_FLOOR, 1.0]`` applied to the
    brokerage score.  The DMD persistence-vs-acquisition asymmetry (paper
    Section 5) says the L2 sales premium accrues to sustained ties only;
    a candidate whose own portfolio churns rapidly is unlikely to be on
    the sustained side of any new tie either.

    Uses ties across all layers within the rolling window, with first-year
    inferred from the candidate's full edge history.
    """
    window_start = year - window + 1
    current = get_firm_partners(bundle, candidate,
                                 year_from=window_start, year_to=year)
    n_ties = len(current)
    if n_ties < PERSISTENCE_MIN_TIES:
        return 1.0, float("nan"), n_ties

    all_edges = get_firm_edges(bundle, candidate,
                                year_from=None, year_to=year)
    sustained = 0
    for partner in current:
        partner_edges = all_edges[(all_edges["firm_i"] == partner)
                                   | (all_edges["firm_j"] == partner)]
        if len(partner_edges) == 0:
            continue
        first_year = int(partner_edges["year"].min())
        if year - first_year >= PERSISTENCE_SUSTAINED_AGE:
            sustained += 1

    sustained_share = sustained / n_ties
    factor = PERSISTENCE_FLOOR + (1.0 - PERSISTENCE_FLOOR) * sustained_share
    return factor, sustained_share, n_ties


# ══════════════════════════════════════════════════════════════════
# Smooth SIC×layer-z-scored w_tenure
# ══════════════════════════════════════════════════════════════════

def _firm_median_tenure(edges: pd.DataFrame, firm: str,
                          year_cap: int) -> float:
    """Median spell length (in years) of `firm`'s ties.

    Spell length for dyad (firm, p) = (max year, min year) + 1 over the
    rows where both endpoints touch the dyad.  Uses the supplied edge
    frame already restricted to firm + year_to ≤ year_cap.
    """
    if len(edges) == 0:
        return float("nan")
    sub = edges[edges["year"] <= year_cap]
    if len(sub) == 0:
        return float("nan")
    # Group by partner CUSIP (the other endpoint)
    sub = sub.copy()
    sub["partner"] = np.where(sub["firm_i"] == firm,
                                sub["firm_j"], sub["firm_i"])
    span = sub.groupby("partner")["year"].agg(["min", "max"])
    span["spell"] = span["max"] - span["min"] + 1
    return float(span["spell"].median())


def _candidate_median_tenure(bundle: DataBundle, candidate: str,
                               year: int, layer: str = "L2") -> tuple:
    """Return (median_tenure_yrs_layer_or_fallback, used_layer).

    Looks up the candidate's `layer` ties first.  If absent, falls back
    to all-layer ties (most candidates have at most one or two L2 ties
    so the L2-only median is brittle).
    """
    edges_layer = get_firm_edges(bundle, candidate, year_to=year, layer=layer)
    med = _firm_median_tenure(edges_layer, candidate, year_cap=year)
    if not np.isnan(med):
        return med, layer
    edges_all = get_firm_edges(bundle, candidate, year_to=year)
    med_all = _firm_median_tenure(edges_all, candidate, year_cap=year)
    return med_all, "ALL"


@lru_cache(maxsize=8)
def _sic_layer_tenure_baseline(year: int = 2017,
                                 layer: str = "L2") -> dict:
    """Compute (μ, σ) of log(1+median_tenure) across firms in each SIC2,
    pooled within `layer`.

    Heavy: one pass over the full edge frame plus per-firm spell-length
    aggregation.  Cached by (year, layer).  Returns a dict with:
      "global": (mu, sigma)
      "by_sic": {sic2: (mu, sigma)}  for cells with ≥ SIC_BASELINE_MIN_CELL firms
    """
    bundle = load_all()
    e = bundle.edges
    e = e[(e["year"] <= year) & (e["layer_code"] == layer)].copy()
    e["partner_i"] = np.where(e["firm_i"] < e["firm_j"],
                                e["firm_i"], e["firm_j"])
    e["partner_j"] = np.where(e["firm_i"] < e["firm_j"],
                                e["firm_j"], e["firm_i"])
    span = e.groupby(["partner_i", "partner_j"])["year"].agg(["min", "max"])
    span["spell"] = span["max"] - span["min"] + 1

    # Each dyad contributes its spell to both endpoints
    dyad = span.reset_index()[["partner_i", "partner_j", "spell"]]
    long = pd.concat([
        dyad.rename(columns={"partner_i": "firm", "partner_j": "_other"}),
        dyad.rename(columns={"partner_j": "firm", "partner_i": "_other"}),
    ], ignore_index=True)
    per_firm = long.groupby("firm")["spell"].median().to_frame("median_tenure")
    per_firm["log1p_tenure"] = np.log(W_TENURE_LOG_OFFSET + per_firm["median_tenure"])

    # Attach SIC2
    fm = bundle.firm_meta[["cusip", "sic2"]].drop_duplicates("cusip")
    per_firm = per_firm.join(fm.set_index("cusip"), how="left")

    global_mu = float(per_firm["log1p_tenure"].mean())
    global_sd = float(per_firm["log1p_tenure"].std(ddof=0))
    if not np.isfinite(global_sd) or global_sd == 0:
        global_sd = SIC_BASELINE_FALLBACK_STD

    by_sic = {}
    for sic2, g in per_firm.dropna(subset=["sic2"]).groupby("sic2"):
        if len(g) < SIC_BASELINE_MIN_CELL:
            continue
        mu = float(g["log1p_tenure"].mean())
        sd = float(g["log1p_tenure"].std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            sd = SIC_BASELINE_FALLBACK_STD
        by_sic[str(sic2)] = (mu, sd)
    return {"global": (global_mu, global_sd), "by_sic": by_sic}


def _w_tenure_smooth(bundle: DataBundle, candidate: str, sic2: str,
                       year: int) -> tuple:
    """Smooth tenure weight in (0, 1).

    Returns (w_tenure, z_tenure, median_tenure_yrs, layer_used, n_ties_used).

    Z-score uses SIC×L2 baseline when the candidate's SIC2 cell has
    enough firms; falls back to the global L2 baseline otherwise.
    sigmoid(z) maps any candidate to (0, 1) with monotone shape.

    Pathologies handled:
      - candidates with < W_TENURE_MIN_TIES_SMOOTH ties have an
        uninformative single-point median, so we return a neutral 0.5
        weight (the median of a single observation does not generalize
        to the firm's tendency);
      - z-scores are clipped to ±W_TENURE_Z_CLIP to suppress extreme
        outliers (small SIC cells produce tiny σ; an old single tie
        in a thin cell would otherwise saturate w → 1.0).
    """
    edges_layer = get_firm_edges(bundle, candidate, year_to=year, layer="L2")
    median_t, layer_used = _candidate_median_tenure(bundle, candidate,
                                                     year, layer="L2")
    if median_t is None or np.isnan(median_t):
        return 0.5, float("nan"), float("nan"), layer_used, 0

    # Count partner-distinct ties contributing to the median
    if layer_used == "L2":
        sub = edges_layer
    else:
        sub = get_firm_edges(bundle, candidate, year_to=year)
    if len(sub):
        sub2 = sub.copy()
        sub2["partner"] = np.where(sub2["firm_i"] == candidate,
                                     sub2["firm_j"], sub2["firm_i"])
        n_ties_used = int(sub2["partner"].nunique())
    else:
        n_ties_used = 0

    if n_ties_used < W_TENURE_MIN_TIES_SMOOTH:
        return 0.5, float("nan"), float(median_t), layer_used, n_ties_used

    log_t = np.log(W_TENURE_LOG_OFFSET + median_t)
    base = _sic_layer_tenure_baseline(year=year, layer="L2")
    mu, sd = base["by_sic"].get(str(sic2), base["global"])
    z = (log_t - mu) / sd
    z_clipped = float(np.clip(z, -W_TENURE_Z_CLIP, W_TENURE_Z_CLIP))
    sigma_z = 1.0 / (1.0 + np.exp(-z_clipped))
    # Empirical-Bayes-style shrinkage toward neutral 0.5: a 2-tie portfolio
    # cannot dominate via a single high-tenure observation, while a
    # 20+-tie portfolio retains its full signal.
    shrink = n_ties_used / (n_ties_used + W_TENURE_SHRINKAGE_KAPPA)
    w = 0.5 + shrink * (sigma_z - 0.5)
    return float(w), float(z_clipped), float(median_t), layer_used, n_ties_used


# ══════════════════════════════════════════════════════════════════
# Multiplicative g(R&D)
# ══════════════════════════════════════════════════════════════════

def _g_rd_multiplier(bundle: DataBundle, focal: str, year: int) -> tuple:
    """Return (g_rd, is_top_quartile).

    g_rd = 1 + ALPHA_RD × 1{focal ∈ top-quartile R&D in own SIC}.
    Per the paper's H2 finding: the L₂ sales premium is concentrated in
    top-R&D firms, so we apply a uniform per-focal multiplier rather
    than a per-candidate one (it does not change within-firm ranking,
    only the absolute scale and cross-firm comparisons).
    """
    is_top = bool(rd_gate(bundle, focal, year))
    g = 1.0 + (ALPHA_RD if is_top else 0.0)
    return g, is_top


# ══════════════════════════════════════════════════════════════════
# Dependency-risk penalty (w_redundancy)
# ══════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _systemic_lookup() -> dict:
    """Map candidate cusip → DepRisk ∈ [0, 1] derived from the corrected
    systemic-criticality cross-section.

    DepRisk is the candidate's normalized in-degree in the meta-network:
    high in-degree means many other firms already list this candidate as
    a top-5 critical partner — adding it to a new portfolio increases
    aggregate dependence on a hub.  Candidates not in the systemic
    cross-section default to 0 (unknown / not currently a top critical
    partner anywhere).
    """
    if not SYSTEMIC_CSV.exists():
        return {"_max_in_degree": 1, "lookup": {}}
    df = pd.read_csv(SYSTEMIC_CSV, dtype={"partner_cusip": str},
                       usecols=["partner_cusip", "in_degree"])
    max_in = max(int(df["in_degree"].max()), 1)
    lookup = dict(zip(df["partner_cusip"].astype(str),
                       df["in_degree"].astype(float) / max_in))
    return {"_max_in_degree": max_in, "lookup": lookup}


def _w_redundancy(candidate: str) -> tuple:
    """Return (w_redundancy, dep_risk, dep_risk_observed).

    w_redundancy = exp(-RHO_REDUNDANCY × DepRisk).  At DepRisk=0 returns
    1.0 (no penalty); at DepRisk=1 (the most-systemic-critical partner
    in the cross-section) returns ≈ 0.22.

    Distinguishes observed-and-zero from unobserved.  A candidate not in
    the systemic-criticality cross-section has unknown DepRisk — we
    default w_redundancy to 1.0 (no penalty) but flag
    dep_risk_observed=False so reports can label this as "not
    systemically ranked" rather than asserting "DepRisk = 0.00".
    """
    sl = _systemic_lookup()
    if candidate in sl["lookup"]:
        dep = float(sl["lookup"][candidate])
        return float(np.exp(-RHO_REDUNDANCY * dep)), dep, True
    return 1.0, float("nan"), False


def _candidate_pool(bundle: DataBundle, focal: str, layer: str,
                     year: int) -> set:
    """Firms appearing as either endpoint of a layer-$\\ell$ edge in the
    5-year rolling window ending at `year`, excluding focal and current
    partners."""
    start = year - ROLLING_WINDOW + 1
    e = bundle.edges
    mask = ((e["year"] >= start) & (e["year"] <= year)
            & (e["layer_code"] == layer))
    sub = e[mask]
    pool = set(sub["firm_i"]) | set(sub["firm_j"])
    pool.discard(focal)

    # Exclude current partners
    from strategic_pipeline.data_loader import get_firm_partners
    cur = get_firm_partners(bundle, focal, start, year, layer=layer)
    pool -= cur

    return pool


def recommend_partners(focal_cusip: str, goal: str, top_n: int = 20,
                        year: int = 2017,
                        bundle: Optional[DataBundle] = None,
                        max_candidates: int = 1500) -> pd.DataFrame:
    """Return top-N candidate partners ranked by the relevant score.

    `max_candidates` caps the candidate pool for tractability on large
    layers (L2 has thousands of firms); we sample the highest-degree
    candidates first (degree within the layer subgraph) to keep the
    search biased toward well-connected firms.
    """
    if bundle is None:
        bundle = load_all()

    goal = goal.lower()
    if goal not in GOAL_TO_LAYER:
        raise ValueError(f"goal must be one of {list(GOAL_TO_LAYER)}")
    layer = GOAL_TO_LAYER[goal]

    pool = _candidate_pool(bundle, focal_cusip, layer, year)
    if not pool:
        return pd.DataFrame()

    # Cap the pool by degree within the layer for tractability
    if len(pool) > max_candidates:
        start = year - ROLLING_WINDOW + 1
        e = bundle.edges
        window = e[(e["year"] >= start) & (e["year"] <= year)
                    & (e["layer_code"] == layer)]
        deg_i = window["firm_i"].value_counts()
        deg_j = window["firm_j"].value_counts()
        deg = deg_i.add(deg_j, fill_value=0)
        pool_deg = deg.reindex(list(pool), fill_value=0)
        top_pool = pool_deg.sort_values(ascending=False).head(max_candidates)
        pool = set(top_pool.index.tolist())

    # Score every candidate
    rows = []
    for cand in pool:
        if goal == "innovation":
            s = closure_score(bundle, focal_cusip, cand, "L1", year)
            score_label = "closure_L1"
        else:
            s = brokerage_score(bundle, focal_cusip, cand, "L2", year)
            score_label = "brokerage_L2"
        if s <= 0:
            continue
        meta = bundle.firm_meta[bundle.firm_meta["cusip"] == cand]
        name = meta.iloc[0]["name"] if len(meta) else f"<unknown:{cand}>"
        sic2 = meta.iloc[0]["sic2"] if len(meta) else "??"
        row = {
            "candidate_cusip": cand,
            "firm_name": name,
            "sic2": sic2,
            score_label: s,
        }
        if goal == "commercialization":
            factor, sustained_share, n_ties = _candidate_persistence_factor(
                bundle, cand, year
            )
            row["persistence_factor"] = factor
            row["sustained_share"] = sustained_share
            row["n_current_ties"] = n_ties
            row["adjusted_brokerage_L2"] = s * factor

            # Week-1 reframe: smooth z-scored w_tenure + DepRisk penalty
            w_t, z_t, med_t, layer_used, n_ties_used = _w_tenure_smooth(
                bundle, cand, str(sic2), year
            )
            w_red, dep_risk, dep_observed = _w_redundancy(cand)
            durable_value = s * w_t
            row["median_tenure_yrs"] = med_t
            row["tenure_layer_used"] = layer_used
            row["n_ties_for_tenure"] = n_ties_used
            row["z_tenure_sic_L2"] = z_t
            row["w_tenure_smooth"] = w_t
            row["dep_risk"] = dep_risk
            row["dep_risk_observed"] = dep_observed
            row["w_redundancy"] = w_red
            row["durable_value"] = durable_value
            row["score_durable_rent"] = durable_value * w_red
            # Annotation scalar: brokerage × all durable-rent components.
            # NOT used in the rank (Week-2A/2B finding: multiplicative
            # combination collapses to one universal candidate). Reported
            # alongside as a value-and-risk annotation on the brokerage
            # frontier.
            g_rd_focal, _ = _g_rd_multiplier(bundle, focal_cusip, year)
            row["annotated_value"] = s * w_t * g_rd_focal * w_red
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    if goal == "innovation":
        df = df.sort_values("closure_L1", ascending=False)
    else:
        # Brokerage-frontier ranker (Week-2B reframe).  Sort by
        # focal-specific brokerage_L2 only.  No candidate-side
        # tiebreaker — candidate-side tiebreakers recreate the
        # Stanford-style collapse inside brokerage-saturated regions
        # (Week-2A/2C diagnostics).  Within saturated brokerage = 1.0
        # blocks the order is intentionally undetermined; the tie audit
        # in week2b_brokerage_tie_audit.py quantifies the resulting
        # ambiguity.  Durable-rent components are reported as the
        # annotated_value column, not used in the rank.
        df = df.sort_values(["brokerage_L2"], ascending=[False])
    df = df.head(top_n).reset_index(drop=True)

    # Add R&D + ranker annotations for commercialization
    if goal == "commercialization":
        g_rd, gate = _g_rd_multiplier(bundle, focal_cusip, year)
        df.attrs["rd_gate_passed"] = gate
        df.attrs["rd_gate_multiplier"] = g_rd
        df.attrs["rd_gate_message"] = (
            f"Focal firm IS in top R&D quartile within SIC. "
            f"Per H2 (paper Section 4 / Figure 3B), the L2 commercialization "
            f"premium is concentrated in this cohort. "
            f"g(R&D) = {g_rd:.2f}× multiplier applied as a per-focal "
            f"absorptive-capacity weight on durable_value."
            if gate else
            "WARNING: focal firm is NOT in top R&D quartile within SIC. "
            "The paper's L2 brokerage premium is uniquely concentrated "
            "in top-quartile R&D firms (Figure 3B, Table 3); for this "
            "firm, L2 brokerage recommendations should be treated as "
            "structural fit only, NOT as causal forecasts of sales "
            f"response (g(R&D) = {g_rd:.2f}, no bonus). Consider "
            "Innovation (L1) alliances to build R&D capacity first."
        )
        df.attrs["reranker"] = "brokerage_frontier_with_annotations"
        df.attrs["reranker_message"] = (
            "Candidates are ranked by focal-specific L2 brokerage "
            "opportunity (brokerage_L2). Durability, absorptive "
            "capacity, and systemic-dependency risk are reported as "
            "value/risk annotations on the brokerage frontier; they "
            "are NOT used to reorder the candidate list.\n"
            "  • brokerage_L2 (rank input) — Burt-style structural "
            "opportunity; the only focal-specific feature in the "
            "current pipeline.  Saturates at 1.0 for ~99% of "
            "candidates in sparse-portfolio focals; within saturated "
            "blocks the recommender's ordering is intentionally "
            "ambiguous — see week2b_brokerage_tie_audit for the "
            "quantification.\n"
            "  • w_tenure_smooth (annotation) — Dyer-Singh-style "
            "relational capability: sigmoid of SIC×L2 z-scored "
            "log(1+median tenure).\n"
            "  • g(R&D) (annotation) — per-focal absorptive-capacity "
            "multiplier, motivated by the L2 sales premium's R&D "
            "stratification.\n"
            "  • w_redundancy = exp(-1.5 × DepRisk) (annotation) — "
            "penalty for adding a candidate already in the systemic "
            "cross-section's critical-partner panel.\n"
            "  • annotated_value = brokerage_L2 × w_tenure_smooth × "
            "g(R&D) × w_redundancy — the durable-rent value of a "
            "selected candidate.  REPORTED, NOT RANKED.\n"
            "Empirical motivation: the Week-2A/2C diagnostics found "
            "that any multiplicative score over candidate-side "
            "features collapses to one universal candidate; the "
            "Week-2B backtest found brokerage_L2 catches 15/16 "
            "in-pool realized 2011 partners at K=5 while the "
            "multiplicative score catches 0/16 at K=5."
        )
    return df


# ══════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    focal = "747525"  # Qualcomm
    year = 2005

    print(f"Alignment recommendations for Qualcomm (cusip {focal}), year {year}")
    print()

    print("=== Goal: INNOVATION (L1 closure) ===")
    df_i = recommend_partners(focal, "innovation", top_n=10, year=year)
    if len(df_i):
        print(df_i.to_string(index=False))
    else:
        print("  (no candidates)")

    print()
    print("=== Goal: COMMERCIALIZATION (L2 brokerage) ===")
    df_c = recommend_partners(focal, "commercialization", top_n=10, year=year)
    if len(df_c):
        print(df_c.to_string(index=False))
        print()
        print(f"R&D gate: {df_c.attrs.get('rd_gate_passed', 'n/a')}")
        print(df_c.attrs.get('rd_gate_message', ''))

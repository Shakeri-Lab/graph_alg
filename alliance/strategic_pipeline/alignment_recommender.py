"""Alignment Recommender — Strategic Question 1.

Given a focal firm and a goal (Innovation or Commercialization), rank
candidate partners by their structural fit:

  - Innovation (L1): closure_score maximized → dense closed triads
  - Commercialization (L2): brokerage_score maximized → non-redundant
    market access. Gated by R&D top-quartile status (paper's H2 finding).

Candidate pool: firms present in the relevant layer's 5-year rolling
window at `year`, excluding current partners.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd

from strategic_pipeline.data_loader import DataBundle, load_all
from strategic_pipeline.firm_profile import build_profile, ROLLING_WINDOW
from strategic_pipeline.scoring_primitives import (
    closure_score, brokerage_score, rd_gate,
)

GOAL_TO_LAYER = {
    "innovation": "L1",
    "commercialization": "L2",
}


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
        rows.append({
            "candidate_cusip": cand,
            "firm_name": name,
            "sic2": sic2,
            score_label: s,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    score_col = f"closure_L1" if goal == "innovation" else "brokerage_L2"
    df = df.sort_values(score_col, ascending=False).head(top_n).reset_index(drop=True)

    # Add R&D gate annotation for commercialization
    if goal == "commercialization":
        gate = rd_gate(bundle, focal_cusip, year)
        df.attrs["rd_gate_passed"] = gate
        df.attrs["rd_gate_message"] = (
            "Focal firm IS in top R&D quartile within SIC — L2 premium "
            "expected per paper's H2 / R&D-conditioning finding."
            if gate else
            "WARNING: focal firm is NOT in top R&D quartile within SIC. "
            "The paper's L2 brokerage premium is uniquely concentrated "
            "in top-quartile R&D firms (Figure 3B, Table 3); for this "
            "firm, L2 brokerage recommendations should be treated as "
            "associational rather than causal. Consider Innovation (L1) "
            "alliances to build R&D capacity first."
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

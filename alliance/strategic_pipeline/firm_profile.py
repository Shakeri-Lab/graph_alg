"""Per-firm summary profile.

Given a cusip + reference year, produce a structured FirmProfile
with identity, current financials, layer-specific centrality,
current portfolio composition by layer, and tenure distribution.
Consumed by all three decision modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from strategic_pipeline.data_loader import (
    DataBundle, load_all,
    get_firm_edges, get_firm_partners,
    get_firm_centrality, get_firm_financials, get_firm_static,
    consecutive_years, dmd_eligible,
)

LAYERS = ["L1", "L2", "L3", "L4"]
LAYER_NAMES = {
    "L1": "Innovation",
    "L2": "Commercialization",
    "L3": "Distribution",
    "L4": "Operations",
}
ROLLING_WINDOW = 5


@dataclass
class FirmProfile:
    cusip: str
    name: str
    sic2: str
    nation: str
    year: int
    # Eligibility
    dmd_eligible: bool
    consec_years: list
    has_compustat: bool
    # Financials at `year`
    financials: dict
    baseline_log_assets: Optional[float]
    baseline_rd: Optional[float]
    # Centrality per layer at `year`
    centrality: dict               # {L1_btw, L2_btw, L3_btw, L4_btw}
    # Portfolio composition: current partners in the 5-year window
    partners_by_layer: dict        # {L1: set, L2: set, ...}
    n_partners_by_layer: dict      # counts
    # Tenure distribution: for each layer, count partners by tenure bucket
    tenure_by_layer: dict          # {L1: {"new": n, "mid": n, "sustained": n}}
    # R&D quartile (within SIC-year), relevant for L2 recommendations
    rd_quartile: Optional[int]     # 1..4 or None
    rd_quartile_top: bool          # True iff in top quartile within SIC

    def summary_line(self) -> str:
        return (f"{self.name} ({self.cusip}) | SIC {self.sic2} | "
                f"{self.nation} | Compustat={self.has_compustat} | "
                f"DMD-eligible={self.dmd_eligible} | "
                f"Partners L1/L2/L3/L4 = "
                f"{self.n_partners_by_layer.get('L1', 0)}/"
                f"{self.n_partners_by_layer.get('L2', 0)}/"
                f"{self.n_partners_by_layer.get('L3', 0)}/"
                f"{self.n_partners_by_layer.get('L4', 0)}")


# ══════════════════════════════════════════════════════════════════
# Tenure classification
# ══════════════════════════════════════════════════════════════════

def _dyad_first_year(edges_sub: pd.DataFrame, focal: str,
                      partner: str) -> int:
    """Earliest year the (focal, partner) dyad appears in edges_sub."""
    mask = (((edges_sub["firm_i"] == focal) & (edges_sub["firm_j"] == partner))
            | ((edges_sub["firm_i"] == partner) & (edges_sub["firm_j"] == focal)))
    return int(edges_sub[mask]["year"].min())


def compute_tenure_distribution(bundle: DataBundle, cusip: str,
                                  layer: str, year: int,
                                  window: int = ROLLING_WINDOW) -> dict:
    """For partners of `cusip` in layer `layer` within the 5-yr window
    ending at `year`, classify by tenure-so-far:

      - "new":        dyad-first-year ≥ year - 1   (<2 years)
      - "mid":        year - 4 ≤ dyad-first-year ≤ year - 2
      - "sustained":  dyad-first-year ≤ year - 4    (≥4 years)

    Requires full edge history back to inception for correct
    first-year inference.
    """
    # Window for "current partner" set: 5 years ending at `year`
    window_start = year - window + 1
    current_partners = get_firm_partners(bundle, cusip,
                                          year_from=window_start,
                                          year_to=year,
                                          layer=layer)
    # Full edge history to infer first-year of each dyad
    all_edges = get_firm_edges(bundle, cusip, year_from=None, year_to=year,
                                layer=layer)

    buckets = {"new": 0, "mid": 0, "sustained": 0}
    for p in current_partners:
        fy = _dyad_first_year(all_edges, cusip, p)
        age = year - fy
        if age < 2:
            buckets["new"] += 1
        elif age <= 4:
            buckets["mid"] += 1
        else:
            buckets["sustained"] += 1
    return buckets


# ══════════════════════════════════════════════════════════════════
# R&D quartile within SIC-year
# ══════════════════════════════════════════════════════════════════

def compute_rd_quartile(bundle: DataBundle, cusip: str,
                         year: int) -> tuple:
    """Return (quartile, is_top) for focal's R&D intensity relative to
    its 2-digit SIC peers in the same year.  If R&D is missing, returns
    (None, False).
    """
    sic2 = None
    sub = bundle.static_cov[bundle.static_cov["ult_parent_cusip"] == cusip]
    if len(sub):
        sic2 = sub.iloc[0].get("sic2")
    if sic2 is None:
        return (None, False)

    # Peer set: firms with same sic2 in firm_year panel at `year`
    fy = bundle.firm_year
    peers = fy[(fy["year"] == year)
               & (fy["rd_intensity"].notna())]
    # Attach sic2 from static_cov
    sic_map = dict(zip(bundle.static_cov["ult_parent_cusip"],
                        bundle.static_cov["sic2"]))
    peers = peers.assign(sic2=peers["ult_parent_cusip"].map(sic_map))
    peers = peers[peers["sic2"] == sic2]

    if len(peers) < 4:
        # Too few peers to define quartile; fallback to global
        peers = fy[(fy["year"] == year) & (fy["rd_intensity"].notna())]

    focal_row = peers[peers["ult_parent_cusip"] == cusip]
    if len(focal_row) == 0 or pd.isna(focal_row.iloc[0]["rd_intensity"]):
        return (None, False)

    focal_rd = float(focal_row.iloc[0]["rd_intensity"])
    quartile = int(np.digitize(focal_rd,
                                np.quantile(peers["rd_intensity"].values,
                                             [0.25, 0.5, 0.75])) + 1)
    quartile = min(quartile, 4)
    is_top = (quartile == 4)
    return (quartile, is_top)


# ══════════════════════════════════════════════════════════════════
# Main profile builder
# ══════════════════════════════════════════════════════════════════

def build_profile(cusip: str, year: int = 2017,
                   bundle: Optional[DataBundle] = None) -> FirmProfile:
    if bundle is None:
        bundle = load_all()

    static = get_firm_static(bundle, cusip)
    name = bundle.firm_name(cusip)
    sic2 = static.get("sic2", "??")
    nation = static.get("nation", "??")
    financials = get_firm_financials(bundle, cusip, year)
    has_compustat = cusip in bundle.compustat_firms
    centrality = get_firm_centrality(bundle, cusip, year)

    window_start = year - ROLLING_WINDOW + 1
    partners_by_layer = {}
    n_partners_by_layer = {}
    tenure_by_layer = {}
    for L in LAYERS:
        p = get_firm_partners(bundle, cusip, window_start, year, layer=L)
        partners_by_layer[L] = p
        n_partners_by_layer[L] = len(p)
        tenure_by_layer[L] = compute_tenure_distribution(bundle, cusip,
                                                           L, year)

    quartile, is_top = compute_rd_quartile(bundle, cusip, year)
    consec = consecutive_years(bundle, cusip)

    return FirmProfile(
        cusip=cusip,
        name=name,
        sic2=sic2,
        nation=nation,
        year=year,
        dmd_eligible=dmd_eligible(bundle, cusip),
        consec_years=consec,
        has_compustat=has_compustat,
        financials=financials,
        baseline_log_assets=static.get("baseline_log_assets"),
        baseline_rd=static.get("baseline_rd"),
        centrality=centrality,
        partners_by_layer=partners_by_layer,
        n_partners_by_layer=n_partners_by_layer,
        tenure_by_layer=tenure_by_layer,
        rd_quartile=quartile,
        rd_quartile_top=is_top,
    )


# ══════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = build_profile("747525", year=2005)
    print(p.summary_line())
    print(f"  consec_years span: {p.consec_years[0]}–{p.consec_years[-1]} "
          f"({len(p.consec_years)} yrs)")
    print(f"  centrality: {p.centrality}")
    print(f"  R&D quartile within SIC {p.sic2}: {p.rd_quartile}, "
          f"top-quartile={p.rd_quartile_top}")
    print(f"  tenure by layer:")
    for L, t in p.tenure_by_layer.items():
        total = sum(t.values())
        if total > 0:
            print(f"    {L} ({LAYER_NAMES[L]}): "
                  f"new={t['new']}, mid={t['mid']}, sustained={t['sustained']} "
                  f"(n={total})")

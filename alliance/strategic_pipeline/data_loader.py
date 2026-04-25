"""Unified data access for the strategic decision pipeline.

Single-responsibility: load the pre-computed intermediate artifacts
and provide cached accessors.  No computation, no transformation
beyond dtype coercion.  Downstream modules should import the
DataBundle and never touch parquet files directly.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Project imports ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import COL, DATA_PATH, INTERMEDIATE_DIR, OUTPUT_DIR
from strategic_pipeline.id_utils import normalize_cusip_columns, normalize_cusip_series


# ══════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════

EDGES_IMPUTED_PATH = INTERMEDIATE_DIR / "corrected" / "pairwise_edges_imputed.parquet"
EDGES_FALLBACK_PATH = INTERMEDIATE_DIR / "pairwise_edges.parquet"
LAYER_BTW_PATH = INTERMEDIATE_DIR / "layer_betweenness_panel.parquet"
FIRM_YEAR_PATH = INTERMEDIATE_DIR / "firm_year_panel.parquet"
ALLIANCE_IMPUTED_PATH = INTERMEDIATE_DIR / "alliance_data_imputed.parquet"
TRAJECTORY_PATH = INTERMEDIATE_DIR / "phase0" / "trajectory_panel.parquet"
STATIC_COV_PATH = INTERMEDIATE_DIR / "phase0" / "static_covariates.parquet"
LONG_PANEL_PATH = INTERMEDIATE_DIR / "phase0" / "long_panel_firms.csv"


# ══════════════════════════════════════════════════════════════════
# DataBundle
# ══════════════════════════════════════════════════════════════════

@dataclass
class DataBundle:
    """All artifacts needed by the strategic pipeline, loaded once."""
    edges: pd.DataFrame              # firm_i, firm_j, year, layer_code
    layer_btw: pd.DataFrame          # ult_parent_cusip, year, L1..L4_btw
    firm_year: pd.DataFrame          # financials
    trajectory: pd.DataFrame         # 10-d state + z-scored
    static_cov: pd.DataFrame         # SIC, baseline R&D, etc.
    firm_meta: pd.DataFrame          # cusip -> name, sic, nation (from raw)
    long_firms: set                  # 851 firms with ≥10 consec years

    # Cached helpers set lazily
    _cusip_to_name: dict = field(default_factory=dict)
    _compustat_firms: Optional[set] = None
    _rd_thresholds: Optional[pd.DataFrame] = None

    def firm_name(self, cusip: str) -> str:
        if not self._cusip_to_name:
            m = dict(zip(self.firm_meta["cusip"], self.firm_meta["name"]))
            self._cusip_to_name = m
        return self._cusip_to_name.get(cusip, f"<unknown:{cusip}>")

    @property
    def compustat_firms(self) -> set:
        if self._compustat_firms is None:
            has_comp = self.firm_year[self.firm_year["total_assets"].notna()]
            self._compustat_firms = set(has_comp["ult_parent_cusip"].unique())
        return self._compustat_firms

    def compustat_firm_count(self) -> int:
        return len(self.compustat_firms)


# ══════════════════════════════════════════════════════════════════
# Loader
# ══════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_all() -> DataBundle:
    """Load the complete bundle.  Cached so multiple modules share one copy."""
    # Edges (imputed preferred, fallback if missing)
    edges_path = EDGES_IMPUTED_PATH if EDGES_IMPUTED_PATH.exists() else EDGES_FALLBACK_PATH
    edges = pd.read_parquet(edges_path)
    # Ensure consistent column set
    required = {"firm_i", "firm_j", "year", "layer_code"}
    missing = required - set(edges.columns)
    if missing:
        raise RuntimeError(f"edges missing columns: {missing}")
    normalize_cusip_columns(edges, ["firm_i", "firm_j"])

    layer_btw = pd.read_parquet(LAYER_BTW_PATH)
    firm_year = pd.read_parquet(FIRM_YEAR_PATH)
    trajectory = pd.read_parquet(TRAJECTORY_PATH)
    static_cov = pd.read_parquet(STATIC_COV_PATH)
    normalize_cusip_columns(layer_btw, ["ult_parent_cusip"])
    normalize_cusip_columns(firm_year, ["ult_parent_cusip"])
    normalize_cusip_columns(trajectory, ["ult_parent_cusip"])
    normalize_cusip_columns(static_cov, ["ult_parent_cusip"])

    # Firm metadata (name, sic, nation) from the imputed alliance file
    meta_cols = [
        COL["ult_parent_cusip"], COL["ult_parent_name"],
        COL["ult_parent_nation"], COL["ult_parent_sic"],
        COL["year"],
    ]
    meta_src = ALLIANCE_IMPUTED_PATH if ALLIANCE_IMPUTED_PATH.exists() else DATA_PATH
    meta_raw = pd.read_parquet(meta_src, columns=meta_cols)
    meta_raw.columns = ["cusip", "name", "nation", "sic", "year"]
    meta_raw["cusip"] = normalize_cusip_series(meta_raw["cusip"])
    firm_meta = (meta_raw.sort_values("year", ascending=False)
                 .drop_duplicates("cusip", keep="first"))
    firm_meta["sic2"] = firm_meta["sic"].astype(str).str[:2]

    # Long-panel roster
    if LONG_PANEL_PATH.exists():
        lp = pd.read_csv(LONG_PANEL_PATH, dtype={"ult_parent_cusip": str})
        lp["ult_parent_cusip"] = normalize_cusip_series(lp["ult_parent_cusip"])
        long_firms = set(lp["ult_parent_cusip"].dropna().astype(str).tolist())
    else:
        long_firms = set()

    return DataBundle(
        edges=edges,
        layer_btw=layer_btw,
        firm_year=firm_year,
        trajectory=trajectory,
        static_cov=static_cov,
        firm_meta=firm_meta,
        long_firms=long_firms,
    )


# ══════════════════════════════════════════════════════════════════
# Convenience lookups
# ══════════════════════════════════════════════════════════════════

def get_firm_edges(bundle: DataBundle, cusip: str,
                    year_from: Optional[int] = None,
                    year_to: Optional[int] = None,
                    layer: Optional[str] = None) -> pd.DataFrame:
    """All edges involving `cusip` (as firm_i or firm_j)."""
    e = bundle.edges
    mask = (e["firm_i"] == cusip) | (e["firm_j"] == cusip)
    if year_from is not None:
        mask &= e["year"] >= year_from
    if year_to is not None:
        mask &= e["year"] <= year_to
    if layer is not None:
        mask &= e["layer_code"] == layer
    return e[mask].copy()


def get_firm_partners(bundle: DataBundle, cusip: str,
                       year_from: Optional[int] = None,
                       year_to: Optional[int] = None,
                       layer: Optional[str] = None) -> set:
    """Unique partner CUSIPs for `cusip` in the given window/layer."""
    sub = get_firm_edges(bundle, cusip, year_from, year_to, layer)
    partners = set(sub["firm_i"]) | set(sub["firm_j"])
    partners.discard(cusip)
    return partners


def get_firm_centrality(bundle: DataBundle, cusip: str,
                         year: int) -> dict:
    """Return L1..L4 betweenness for (cusip, year). NaN → 0."""
    sub = bundle.layer_btw[
        (bundle.layer_btw["ult_parent_cusip"] == cusip)
        & (bundle.layer_btw["year"] == year)
    ]
    if len(sub) == 0:
        return {f"L{k}_btw": 0.0 for k in range(1, 5)}
    row = sub.iloc[0]
    out = {}
    for k in range(1, 5):
        v = row.get(f"L{k}_btw", 0.0)
        out[f"L{k}_btw"] = 0.0 if pd.isna(v) else float(v)
    return out


def get_firm_financials(bundle: DataBundle, cusip: str,
                         year: int) -> dict:
    """Compustat record for (cusip, year), NaNs where missing."""
    fy = bundle.firm_year
    sub = fy[(fy["ult_parent_cusip"] == cusip) & (fy["year"] == year)]
    if len(sub) == 0:
        return {}
    row = sub.iloc[0]
    keys = ["total_assets", "sales", "market_value", "net_income",
            "leverage", "rd_intensity", "log_assets"]
    return {k: (None if pd.isna(row.get(k, np.nan)) else float(row[k]))
            for k in keys if k in row.index}


def get_firm_static(bundle: DataBundle, cusip: str) -> dict:
    """Static covariate row."""
    sub = bundle.static_cov[bundle.static_cov["ult_parent_cusip"] == cusip]
    if len(sub) == 0:
        return {}
    row = sub.iloc[0].to_dict()
    return row


def consecutive_years(bundle: DataBundle, cusip: str) -> list:
    """List of consecutive-run years in the trajectory panel for `cusip`.

    Returns a list of ints (possibly empty).  Used to test Hankel-DMD
    eligibility (≥6 consecutive years).
    """
    sub = bundle.trajectory[
        bundle.trajectory["ult_parent_cusip"] == cusip
    ]["year"].values
    if len(sub) == 0:
        return []
    years = np.sort(np.unique(sub))
    # Largest consecutive run
    best_run, cur_run = [years[0]], [years[0]]
    for i in range(1, len(years)):
        if years[i] == years[i-1] + 1:
            cur_run.append(years[i])
        else:
            if len(cur_run) > len(best_run):
                best_run = cur_run
            cur_run = [years[i]]
    if len(cur_run) > len(best_run):
        best_run = cur_run
    return [int(y) for y in best_run]


def dmd_eligible(bundle: DataBundle, cusip: str, h: int = 5) -> bool:
    """True if firm has ≥ h+1 consecutive trajectory years."""
    run = consecutive_years(bundle, cusip)
    return len(run) >= h + 1


# ══════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading DataBundle ...")
    b = load_all()
    print(f"  edges:       {b.edges.shape}")
    print(f"  layer_btw:   {b.layer_btw.shape}")
    print(f"  firm_year:   {b.firm_year.shape}")
    print(f"  trajectory:  {b.trajectory.shape}")
    print(f"  static_cov:  {b.static_cov.shape}")
    print(f"  firm_meta:   {b.firm_meta.shape}")
    print(f"  long_firms:  {len(b.long_firms):,d}")
    print(f"  Compustat firms (total_assets non-null): "
          f"{b.compustat_firm_count():,d}")

    print("\n-- Qualcomm (cusip 747525) --")
    q = "747525"
    print(f"  name: {b.firm_name(q)}")
    print(f"  static: {get_firm_static(b, q)}")
    print(f"  consec_years (in trajectory): {consecutive_years(b, q)}")
    print(f"  dmd_eligible: {dmd_eligible(b, q)}")
    print(f"  centrality 2005: {get_firm_centrality(b, q, 2005)}")
    print(f"  financials 2005: {get_firm_financials(b, q, 2005)}")
    print(f"  L2 partners 2001-2005: "
          f"{len(get_firm_partners(b, q, 2001, 2005, layer='L2'))}")

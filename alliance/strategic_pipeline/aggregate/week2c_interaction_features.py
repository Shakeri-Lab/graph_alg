"""Week 2C — focal-conditional interaction features.

The Week-2A diagnostic showed every multiplicative variant of the
durable-rent score collapses to one universal top-1 candidate, because
all rank-driving features are candidate-side.  Real personalization
requires features that depend on the (focal, candidate) pair itself.

This module computes four such features and joins them onto the
existing 18M-row Week-2A parquet:

- ``n_shared_partners(f, c)``     — |N(f) ∩ N(c)| in the 5-yr window
- ``jaccard_partners(f, c)``      — |N(f) ∩ N(c)| / |N(f) ∪ N(c)|
- ``share_focal_in_c(f, c)``      — |N(f) ∩ N(c)| / |N(f)|
- ``nation_match(f, c)``          — 1 if shared nation
- ``same_sic2(f, c)``             — 1 if shared 2-digit SIC
- ``same_sic1(f, c)``             — 1 if shared 1-digit SIC
- ``sic2_distance(f, c)``         — |sic2(f) − sic2(c)| (nominal proxy)

These are the cheapest focal-conditional signals in the existing data
that don't require new model fitting.  If even these don't break the
collapse, the answer for Week 2B is to live with brokerage-only as the
ranker.

Output: ``outputs/strategic/aggregate/week2c_interaction_features_2017.parquet``
joined back to ``week2_personalization_rows_2017.parquet`` →
``outputs/strategic/aggregate/week2c_personalization_rows_2017.parquet``
(same 18M rows, with the 7 new columns added).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategic_pipeline.data_loader import DataBundle, load_all
from strategic_pipeline.firm_profile import ROLLING_WINDOW


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
AGG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "aggregate"
ROWS_IN = AGG_DIR / "week2_personalization_rows_2017.parquet"
ROWS_OUT = AGG_DIR / "week2c_personalization_rows_2017.parquet"


def build_partner_dict(bundle: DataBundle, year: int = 2017,
                         window: int = ROLLING_WINDOW) -> dict:
    """firm_cusip → frozenset of partner CUSIPs in the rolling window
    (any layer)."""
    e = bundle.edges
    start = year - window + 1
    sub = e[(e["year"] >= start) & (e["year"] <= year)].copy()
    long = pd.concat([
        sub[["firm_i", "firm_j"]].rename(columns={
            "firm_i": "firm", "firm_j": "partner"}),
        sub[["firm_j", "firm_i"]].rename(columns={
            "firm_j": "firm", "firm_i": "partner"}),
    ], ignore_index=True)
    long["firm"] = long["firm"].astype(str)
    long["partner"] = long["partner"].astype(str)
    long = long[long["firm"] != long["partner"]]
    grouped = long.groupby("firm")["partner"].agg(frozenset).to_dict()
    return grouped


def build_meta_dict(bundle: DataBundle) -> tuple:
    """Two dicts: cusip → sic2, cusip → nation."""
    fm = bundle.firm_meta[["cusip", "sic2", "nation"]] \
        .drop_duplicates("cusip", keep="first")
    fm["cusip"] = fm["cusip"].astype(str)
    fm["sic2"] = fm["sic2"].astype(str)
    sic_map = dict(zip(fm["cusip"], fm["sic2"]))
    nation_map = dict(zip(fm["cusip"], fm["nation"].fillna("")))
    return sic_map, nation_map


def add_interaction_features(rows: pd.DataFrame,
                                partners: dict,
                                sic_map: dict,
                                nation_map: dict) -> pd.DataFrame:
    """Add the seven interaction features to `rows` (long (f, c) frame).

    The hot loop is per-row Python because frozensets do not vectorize.
    For 18M rows × ~5-element intersections this finishes in 5-15 min.
    """
    print(f"  [w2c] computing interaction features on {len(rows):,} rows")

    f_ids = rows["focal_cusip"].astype(str).to_numpy()
    c_ids = rows["candidate_cusip"].astype(str).to_numpy()

    n_shared = np.empty(len(rows), dtype=np.int32)
    n_focal = np.empty(len(rows), dtype=np.int32)
    n_cand = np.empty(len(rows), dtype=np.int32)
    same_sic2 = np.empty(len(rows), dtype=np.int8)
    same_sic1 = np.empty(len(rows), dtype=np.int8)
    sic_dist = np.empty(len(rows), dtype=np.float32)
    nation_match = np.empty(len(rows), dtype=np.int8)

    EMPTY = frozenset()
    for i in range(len(rows)):
        f = f_ids[i]
        c = c_ids[i]
        nf = partners.get(f, EMPTY)
        nc = partners.get(c, EMPTY)
        nf_len = len(nf)
        nc_len = len(nc)
        if nf_len and nc_len:
            shared = len(nf & nc)
        else:
            shared = 0
        n_shared[i] = shared
        n_focal[i] = nf_len
        n_cand[i] = nc_len

        f_sic = sic_map.get(f, "??")
        c_sic = sic_map.get(c, "??")
        same_sic2[i] = 1 if f_sic == c_sic else 0
        same_sic1[i] = 1 if (f_sic and c_sic and f_sic[0] == c_sic[0]) else 0
        try:
            sic_dist[i] = abs(int(f_sic) - int(c_sic))
        except (ValueError, TypeError):
            sic_dist[i] = np.nan

        f_nat = nation_map.get(f, "")
        c_nat = nation_map.get(c, "")
        nation_match[i] = 1 if (f_nat and c_nat and f_nat == c_nat) else 0

        if i and i % 1_000_000 == 0:
            print(f"  [w2c] {i:,}/{len(rows):,} rows")

    out = rows.copy()
    out["n_shared_partners"] = n_shared
    out["n_focal_partners"] = n_focal
    out["n_candidate_partners"] = n_cand
    out["same_sic2"] = same_sic2
    out["same_sic1"] = same_sic1
    out["sic2_distance"] = sic_dist
    out["nation_match"] = nation_match

    union = n_focal + n_cand - n_shared
    with np.errstate(divide="ignore", invalid="ignore"):
        out["jaccard_partners"] = np.where(union > 0,
                                              n_shared / union, 0.0)
        out["share_focal_in_c"] = np.where(n_focal > 0,
                                              n_shared / n_focal, 0.0)
    return out


def main() -> None:
    print("[w2c] loading bundle + 18M-row personalization parquet")
    bundle = load_all()
    rows = pd.read_parquet(ROWS_IN)
    print(f"  rows: {len(rows):,}")

    print("[w2c] building partner dict (5-yr window, any layer)")
    partners = build_partner_dict(bundle)
    print(f"  partner_dict size: {len(partners):,}")

    print("[w2c] building SIC + nation lookups")
    sic_map, nation_map = build_meta_dict(bundle)

    print("[w2c] computing interaction features")
    out = add_interaction_features(rows, partners, sic_map, nation_map)

    print(f"[w2c] writing {ROWS_OUT}")
    out.to_parquet(ROWS_OUT, index=False)
    print("[w2c] done.")


if __name__ == "__main__":
    main()

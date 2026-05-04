"""Week 2A — personalization / null-baseline diagnostic.

Tests whether the durable-rent recommender produces focal-specific
rankings or merely rediscovers globally deep candidates (the
Stanford / Office Depot collapse the canary surfaced).

Workflow
--------
1. ``precompute_candidate_features(year)`` builds a per-candidate
   feature parquet over the L2-active universe at ``year``.  This
   captures everything that does NOT depend on the focal firm:
   degree, current-tie count, w_tenure_smooth, DepRisk, candidate
   type, etc.  Run once.

2. ``score_focal_full(focal_cusip, year, candidate_features)``
   computes brokerage_L2(f, c) for every candidate ``c`` in the
   focal's L2 candidate pool, joins the candidate features, and
   produces a long DataFrame for one focal.

3. The Slurm array `run_week2_personalization_array.slurm` calls (2)
   for each focal in `intermediate/compustat_firm_list.csv`, dumping
   one parquet per focal to
   `outputs/strategic/aggregate/week2_personalization/`.

4. ``aggregate(year)`` stitches the per-focal parquets, computes the
   residualized and blended scores, baseline ranks, top-k overlaps,
   and N_eff diagnostics. Produces:
     - week2_personalization_rows_2017.parquet          (long; per (f, c))
     - week2_personalization_summary.csv                (per ranker)
     - week2_top1_concentration_by_variant.csv
     - week2_overlap_matrix_by_variant.csv
     - week2_type_stratified_neff.csv
     - figures/week2_top1_concentration.png
     - figures/week2_baseline_overlap_heatmap.png
     - figures/week2_type_stratified_neff.png

5. Decision rule (in WEEK2_PERSONALIZATION_NOTES.md): pick raw,
   residualized, or blended score for downstream persistence + sales
   backtests.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from strategic_pipeline.data_loader import (
    DataBundle, load_all, get_firm_edges, get_firm_partners,
)
from strategic_pipeline.firm_profile import ROLLING_WINDOW
from strategic_pipeline.scoring_primitives import brokerage_score
from strategic_pipeline.alignment_recommender import (
    _w_tenure_smooth, _w_redundancy, _systemic_lookup,
    _candidate_pool, _g_rd_multiplier, GOAL_TO_LAYER,
)


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "strategic" / "aggregate"
PERSONAL_DIR = OUTPUT_DIR / "week2_personalization"
FIG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "figures"
PERSONAL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def personal_dir_for_year(year: int) -> Path:
    """Per-year subdir so backtest years (e.g., 2011) don't clobber the
    main 2017 frame. The 2017 directory keeps the legacy un-suffixed
    name for backward compatibility."""
    if year == 2017:
        return PERSONAL_DIR
    d = OUTPUT_DIR / f"week2_personalization_{year}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ──────────────────────────────────────────────────────────────────────
# Candidate-type classifier (light-touch, name- and SIC-based)
# ──────────────────────────────────────────────────────────────────────

UNIVERSITY_RE = re.compile(
    r"\b(UNIVERSITY|UNIV\.?|COLLEGE|INSTITUTE|INSTITUT|SCHOOL|"
    r"\bUNIV\b|FOUNDATION|RESEARCH CTR|RESEARCH CENTER|ACADEMIA|"
    r"POLYTECHNIC)\b",
    re.IGNORECASE,
)
HOSPITAL_RE = re.compile(
    r"\b(HOSPITAL|MEDICAL CENTER|MEDICAL CTR|CLINIC|HEALTH SYSTEM|"
    r"CANCER CENTER|CANCER CTR|HEALTH SCIENCES)\b",
    re.IGNORECASE,
)
SOVEREIGN_SIC = {"99"}  # 'unclassified' SDC bucket — sovereign + state entities


def classify_candidate_type(name: str, sic2: str,
                              is_compustat: bool) -> str:
    if isinstance(name, str):
        if UNIVERSITY_RE.search(name):
            return "university_research"
        if HOSPITAL_RE.search(name):
            return "hospital_medical"
    if str(sic2) in SOVEREIGN_SIC:
        return "sovereign_state"
    if is_compustat:
        return "public_compustat"
    return "private_other"


# ──────────────────────────────────────────────────────────────────────
# Phase 1: candidate-side feature pre-compute
# ──────────────────────────────────────────────────────────────────────

def precompute_candidate_features(year: int = 2017,
                                    window: int = ROLLING_WINDOW,
                                    layer: str = "L2",
                                    bundle: Optional[DataBundle] = None
                                    ) -> pd.DataFrame:
    """Build the candidate-side feature table.

    Saves to ``OUTPUT_DIR / candidate_features_<year>.parquet``.
    """
    if bundle is None:
        bundle = load_all()

    e = bundle.edges
    start = year - window + 1
    e_layer = e[(e["year"] >= start) & (e["year"] <= year)
                 & (e["layer_code"] == layer)]
    e_window = e[(e["year"] >= start) & (e["year"] <= year)]

    # Universe: any firm appearing in the layer's rolling-window edges
    candidates = sorted(
        set(e_layer["firm_i"].astype(str))
        | set(e_layer["firm_j"].astype(str))
    )

    # Layer-specific degree (in window)
    deg_l2_i = e_layer["firm_i"].value_counts()
    deg_l2_j = e_layer["firm_j"].value_counts()
    deg_l2 = deg_l2_i.add(deg_l2_j, fill_value=0)

    # All-layer degree (in window)
    deg_all_i = e_window["firm_i"].value_counts()
    deg_all_j = e_window["firm_j"].value_counts()
    deg_all = deg_all_i.add(deg_all_j, fill_value=0)

    # Current ties (any layer, in window) — distinct partners
    long_window = pd.concat([
        e_window[["firm_i", "firm_j"]].rename(columns={
            "firm_i": "firm", "firm_j": "partner"}),
        e_window[["firm_j", "firm_i"]].rename(columns={
            "firm_j": "firm", "firm_i": "partner"}),
    ], ignore_index=True)
    n_current = (long_window.groupby("firm")["partner"].nunique()
                 .rename("n_current_ties"))

    # All-time historical tie count
    e_all = e[e["year"] <= year]
    long_all = pd.concat([
        e_all[["firm_i", "firm_j"]].rename(columns={
            "firm_i": "firm", "firm_j": "partner"}),
        e_all[["firm_j", "firm_i"]].rename(columns={
            "firm_j": "firm", "firm_i": "partner"}),
    ], ignore_index=True)
    n_hist = (long_all.groupby("firm")["partner"].nunique()
              .rename("n_hist_ties"))

    # Firm meta
    fm = bundle.firm_meta[["cusip", "name", "sic2", "nation"]] \
        .drop_duplicates("cusip", keep="first") \
        .set_index("cusip")
    compustat_set = set(str(c) for c in bundle.compustat_firms)

    # Systemic in-degree → DepRisk
    sl = _systemic_lookup()
    dep_lookup = sl["lookup"]

    rows = []
    for c in candidates:
        c = str(c)
        meta = fm.loc[c] if c in fm.index else None
        name = meta["name"] if meta is not None else f"<unknown:{c}>"
        sic2 = str(meta["sic2"]) if meta is not None else "??"
        nation = meta["nation"] if meta is not None else None
        is_compustat = c in compustat_set
        ctype = classify_candidate_type(name, sic2, is_compustat)

        w_t, z_t, med_t, layer_used, n_ties_used = _w_tenure_smooth(
            bundle, c, sic2, year
        )
        w_red, dep, dep_obs = _w_redundancy(c)

        rows.append({
            "candidate_cusip": c,
            "candidate_name": name,
            "candidate_sic2": sic2,
            "candidate_nation": nation,
            "candidate_compustat": is_compustat,
            "candidate_type": ctype,
            "candidate_degree_all": int(deg_all.get(c, 0)),
            "candidate_degree_l2": int(deg_l2.get(c, 0)),
            "n_current_ties": int(n_current.get(c, 0)),
            "n_hist_ties": int(n_hist.get(c, 0)),
            "median_tenure_yrs": med_t,
            "tenure_layer_used": layer_used,
            "n_ties_for_tenure": n_ties_used,
            "z_tenure_sic_l2": z_t,
            "w_tenure_smooth": w_t,
            "dep_risk": dep,
            "dep_risk_observed": dep_obs,
            "w_redundancy": w_red,
        })
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / f"candidate_features_{year}.parquet"
    df.to_parquet(out, index=False)
    print(f"[week2] candidate_features_{year}: {len(df):,} rows → {out}")
    return df


# ──────────────────────────────────────────────────────────────────────
# Phase 2: per-focal scoring (full pool, not top-20)
# ──────────────────────────────────────────────────────────────────────

def score_focal_full(focal_cusip: str, year: int,
                       candidate_features: pd.DataFrame,
                       bundle: Optional[DataBundle] = None,
                       layer: str = "L2") -> pd.DataFrame:
    """Score every candidate in focal's L2 candidate pool.

    Returns a DataFrame with brokerage_L2 + candidate features + raw
    durable-rent score for this focal, restricted to candidates the
    focal does NOT already partner with and excluding focal itself.
    """
    if bundle is None:
        bundle = load_all()

    pool = _candidate_pool(bundle, focal_cusip, layer, year)
    if not pool:
        return pd.DataFrame()

    # Brokerage_L2 for each candidate (focal-specific)
    rows = []
    for c in pool:
        s = brokerage_score(bundle, focal_cusip, c, "L2", year)
        if s <= 0:
            continue
        rows.append({"candidate_cusip": str(c), "brokerage_l2": float(s)})
    if not rows:
        return pd.DataFrame()
    bro = pd.DataFrame(rows)

    # Join candidate features
    out = bro.merge(candidate_features, on="candidate_cusip", how="left")

    # g(R&D) is per-focal — record but does not affect within-focal rank
    g_rd, is_top_rd = _g_rd_multiplier(bundle, focal_cusip, year)

    out["focal_cusip"] = str(focal_cusip)
    out["focal_top_rd"] = is_top_rd
    out["g_rd"] = g_rd
    out["year"] = int(year)

    # Raw score (within-focal personalization will be measured against
    # baselines that strip the focal-side input)
    out["durable_value"] = (
        out["brokerage_l2"].fillna(0.0) * out["w_tenure_smooth"].fillna(0.5)
    )
    out["raw_score"] = out["durable_value"] * out["w_redundancy"].fillna(1.0)

    return out


def write_focal_parquet(focal_cusip: str, year: int,
                          candidate_features: pd.DataFrame,
                          bundle: DataBundle) -> Path:
    df = score_focal_full(focal_cusip, year, candidate_features, bundle)
    out_dir = personal_dir_for_year(year)
    out = out_dir / f"{focal_cusip}.parquet"
    if len(df):
        df.to_parquet(out, index=False)
    else:
        # Sentinel empty parquet so the array can detect "ran but empty"
        pd.DataFrame(columns=["focal_cusip", "candidate_cusip"]).to_parquet(
            out, index=False)
    return out


# ──────────────────────────────────────────────────────────────────────
# CLI for the Slurm array
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("precompute")
    p1.add_argument("--year", type=int, default=2017)

    p2 = sub.add_parser("score")
    p2.add_argument("--cusip", type=str, default=None)
    p2.add_argument("--array-index", type=int, default=None)
    p2.add_argument("--year", type=int, default=2017)

    args = ap.parse_args()

    if args.cmd == "precompute":
        precompute_candidate_features(year=args.year)
    elif args.cmd == "score":
        bundle = load_all()
        feats_path = OUTPUT_DIR / f"candidate_features_{args.year}.parquet"
        if not feats_path.exists():
            print(f"  [score] candidate features missing; run "
                  f"`week2_personalization precompute` first.")
            raise SystemExit(2)
        feats = pd.read_parquet(feats_path)

        if args.cusip is not None:
            cusip = args.cusip
        else:
            cusip_list = pd.read_csv(
                PROJECT_ROOT / "intermediate" / "compustat_firm_list.csv"
            )
            idx = int(args.array_index) - 1
            cusip = str(cusip_list.iloc[idx]["ult_parent_cusip"])
        path = write_focal_parquet(cusip, args.year, feats, bundle)
        print(f"  [score] {cusip} → {path}")

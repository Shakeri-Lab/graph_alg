"""Phase 9B-lite — multi-year realized-tie panel with coverage taxonomy.

For each backtest year T ∈ {2009, 2010, 2011, 2012, 2013, 2014}, builds
a panel of "new L2 dyads first appearing at year T" and classifies each
realized (focal, candidate) row into a coverage taxonomy:

  realized_new_L2
  ├── focal_not_compustat                    (focal not in Compustat list)
  └── focal_compustat
      ├── candidate_in_pool                  (in score-year L2 candidate pool)
      └── candidate_not_in_pool
          ├── candidate_genuine_new_L2_entrant      (no L2 edges before T)
          ├── candidate_prior_SDC_no_L2             (any-layer SDC presence
                                                     before T but no L2)
          ├── candidate_prior_nonL2_same_focal      (cross-layer convert:
                                                     prior non-L2 with this f)
          ├── candidate_prior_L2_excluded_by_pool   (had prior L2 but
                                                     window/rule excluded)
          └── candidate_unmatched                   (no firm_meta lookup)

Output:
  outputs/strategic/aggregate/phase9b_lite_realized_panel.parquet
  outputs/strategic/aggregate/phase9b_lite_coverage_by_year.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from strategic_pipeline.data_loader import DataBundle, load_all
from strategic_pipeline.firm_profile import ROLLING_WINDOW
from strategic_pipeline.aggregate.week2_personalization import OUTPUT_DIR


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
AGG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "aggregate"

T_YEARS = (2009, 2010, 2011, 2012, 2013, 2014)
PERSISTENCE_HORIZON = 5
PERSISTENCE_THRESHOLD = 3
SALES_HORIZONS = (2, 4)


# ──────────────────────────────────────────────────────────────────────
# Per-year realized panel + coverage classification
# ──────────────────────────────────────────────────────────────────────

def _canonicalize_dyad(df: pd.DataFrame, i_col: str = "firm_i",
                         j_col: str = "firm_j") -> pd.DataFrame:
    df = df.copy()
    df[i_col] = df[i_col].astype(str)
    df[j_col] = df[j_col].astype(str)
    a = np.where(df[i_col] < df[j_col], df[i_col], df[j_col])
    b = np.where(df[i_col] < df[j_col], df[j_col], df[i_col])
    df["_a"] = a
    df["_b"] = b
    return df


def realized_l2_dyads(bundle: DataBundle, t: int) -> pd.DataFrame:
    e = bundle.edges[bundle.edges["layer_code"] == "L2"]
    e = _canonicalize_dyad(e)
    span = e.groupby(["_a", "_b"])["year"].agg(["min", "max"]).reset_index()
    realized = span[span["min"] == t].copy()
    out_a = realized.rename(columns={"_a": "focal_cusip",
                                       "_b": "candidate_cusip"})[
        ["focal_cusip", "candidate_cusip"]]
    out_b = realized.rename(columns={"_b": "focal_cusip",
                                       "_a": "candidate_cusip"})[
        ["focal_cusip", "candidate_cusip"]]
    out = pd.concat([out_a, out_b], ignore_index=True)
    out["year"] = t
    return out


def add_persistence(df: pd.DataFrame, bundle: DataBundle,
                      horizon: int = PERSISTENCE_HORIZON) -> pd.DataFrame:
    e = _canonicalize_dyad(bundle.edges)
    rows = []
    for t in df["year"].unique():
        sub = df[df["year"] == t]
        e_post = e[(e["year"] > t) & (e["year"] <= t + horizon)]
        if len(e_post) == 0:
            sub = sub.copy()
            sub["T_fc"] = 0
            rows.append(sub)
            continue
        post_years = (e_post.groupby(["_a", "_b"])["year"].nunique()
                       .rename("T_fc_raw").reset_index())
        sub = sub.copy()
        a = np.where(sub["focal_cusip"] < sub["candidate_cusip"],
                      sub["focal_cusip"], sub["candidate_cusip"])
        b = np.where(sub["focal_cusip"] < sub["candidate_cusip"],
                      sub["candidate_cusip"], sub["focal_cusip"])
        sub["_a"] = a
        sub["_b"] = b
        merged = sub.merge(post_years, left_on=["_a", "_b"],
                            right_on=["_a", "_b"], how="left")
        merged["T_fc"] = merged["T_fc_raw"].fillna(0).astype(int)
        merged = merged.drop(columns=["T_fc_raw", "_a", "_b"])
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def add_focal_sales_delta(df: pd.DataFrame, bundle: DataBundle,
                            horizons=SALES_HORIZONS) -> pd.DataFrame:
    fy = bundle.firm_year.copy()
    fy["ult_parent_cusip"] = fy["ult_parent_cusip"].astype(str)
    sales_col = "sales" if "sales" in fy.columns else None
    if sales_col is None:
        for h in horizons:
            df[f"delta_log_sales_h{h}"] = np.nan
        df["log_sales_t"] = np.nan
        df["log_assets_t"] = np.nan
        df["rd_intensity_t"] = np.nan
        return df

    fy["log_sales"] = np.log(fy[sales_col].clip(lower=1.0))
    if "at" in fy.columns:
        fy["log_assets"] = np.log(fy["at"].clip(lower=1.0))
    else:
        fy["log_assets"] = np.nan
    if "rd_intensity" in fy.columns:
        fy["rd_intensity"] = fy["rd_intensity"]
    else:
        fy["rd_intensity"] = np.nan

    sales_pivot = fy.pivot_table(index="ult_parent_cusip", columns="year",
                                    values="log_sales", aggfunc="mean")
    assets_pivot = fy.pivot_table(index="ult_parent_cusip", columns="year",
                                     values="log_assets", aggfunc="mean")
    rd_pivot = fy.pivot_table(index="ult_parent_cusip", columns="year",
                                 values="rd_intensity", aggfunc="mean")

    out = df.copy()
    out["log_sales_t"] = np.nan
    out["log_assets_t"] = np.nan
    out["rd_intensity_t"] = np.nan
    for h in horizons:
        out[f"delta_log_sales_h{h}"] = np.nan
    for t in out["year"].unique():
        idx = out["year"] == t
        # log_sales at t
        if t in sales_pivot.columns:
            out.loc[idx, "log_sales_t"] = (
                out.loc[idx, "focal_cusip"].map(sales_pivot[t]))
        if t in assets_pivot.columns:
            out.loc[idx, "log_assets_t"] = (
                out.loc[idx, "focal_cusip"].map(assets_pivot[t]))
        if t in rd_pivot.columns:
            out.loc[idx, "rd_intensity_t"] = (
                out.loc[idx, "focal_cusip"].map(rd_pivot[t]))
        for h in horizons:
            target = t + h
            if target in sales_pivot.columns and t in sales_pivot.columns:
                delta = sales_pivot[target] - sales_pivot[t]
                out.loc[idx, f"delta_log_sales_h{h}"] = (
                    out.loc[idx, "focal_cusip"].map(delta))
    return out


# ──────────────────────────────────────────────────────────────────────
# Coverage taxonomy
# ──────────────────────────────────────────────────────────────────────

def classify_coverage(df: pd.DataFrame, bundle: DataBundle) -> pd.DataFrame:
    """Add columns:
      focal_in_compustat: bool
      candidate_first_l2_year: int or NaN
      candidate_first_any_year: int or NaN
      candidate_priorL2: bool      (had any L2 edge before t)
      candidate_priorAnySDC: bool  (had any SDC edge before t)
      candidate_in_pool: bool
      candidate_priorNonL2_with_focal: bool
      candidate_unmatched: bool
      coverage_class: str  (from the taxonomy)
    """
    e = bundle.edges.copy()
    e["firm_i"] = e["firm_i"].astype(str)
    e["firm_j"] = e["firm_j"].astype(str)
    fm_set = set(str(c) for c in bundle.firm_meta["cusip"].astype(str).unique())
    compustat_set = set(str(c) for c in bundle.compustat_firms)

    # Per-firm first-year-in-each-layer (any-layer + L2)
    long = pd.concat([
        e[["firm_i", "year", "layer_code"]].rename(columns={"firm_i": "firm"}),
        e[["firm_j", "year", "layer_code"]].rename(columns={"firm_j": "firm"}),
    ], ignore_index=True)
    long["firm"] = long["firm"].astype(str)
    first_any = long.groupby("firm")["year"].min().to_dict()
    first_l2 = (long[long["layer_code"] == "L2"]
                .groupby("firm")["year"].min().to_dict())

    # Per-(focal, candidate) prior-non-L2 detection
    e_nonl2 = e[e["layer_code"] != "L2"].copy()
    e_nonl2 = _canonicalize_dyad(e_nonl2)
    nonl2_dyad_first = e_nonl2.groupby(["_a", "_b"])["year"].min().to_dict()

    out = df.copy()
    out["focal_cusip"] = out["focal_cusip"].astype(str)
    out["candidate_cusip"] = out["candidate_cusip"].astype(str)

    out["focal_in_compustat"] = out["focal_cusip"].isin(compustat_set)
    out["candidate_first_l2_year"] = out["candidate_cusip"].map(first_l2)
    out["candidate_first_any_year"] = out["candidate_cusip"].map(first_any)
    out["candidate_unmatched"] = ~out["candidate_cusip"].isin(fm_set)

    out["candidate_priorL2"] = (
        out["candidate_first_l2_year"].notna()
        & (out["candidate_first_l2_year"] < out["year"])
    )
    out["candidate_priorAnySDC"] = (
        out["candidate_first_any_year"].notna()
        & (out["candidate_first_any_year"] < out["year"])
    )

    # Cross-layer conversion: focal-candidate had a non-L2 edge before t
    a_arr = np.where(out["focal_cusip"] < out["candidate_cusip"],
                       out["focal_cusip"], out["candidate_cusip"])
    b_arr = np.where(out["focal_cusip"] < out["candidate_cusip"],
                       out["candidate_cusip"], out["focal_cusip"])
    nonl2_first = pd.Series(
        [nonl2_dyad_first.get((a, b), np.nan)
         for a, b in zip(a_arr, b_arr)],
        index=out.index,
    )
    out["candidate_priorNonL2_with_focal"] = (
        nonl2_first.notna() & (nonl2_first < out["year"])
    )

    # Candidate-pool membership: candidate had an L2 edge in
    # [t - ROLLING_WINDOW, t - 1].  This is the "score year" pool.
    e_l2 = e[e["layer_code"] == "L2"]
    rows = []
    for t in out["year"].unique():
        sub = out[out["year"] == t]
        score_year = t - 1
        start = score_year - ROLLING_WINDOW + 1
        window = e_l2[(e_l2["year"] >= start) & (e_l2["year"] <= score_year)]
        pool = set(window["firm_i"].astype(str)) | set(
            window["firm_j"].astype(str))
        sub = sub.copy()
        sub["candidate_in_pool"] = sub["candidate_cusip"].isin(pool)
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)

    # Final taxonomy
    def _classify(r):
        if not r["focal_in_compustat"]:
            return "focal_not_compustat"
        if r["candidate_in_pool"]:
            return "candidate_in_pool"
        if r["candidate_unmatched"]:
            return "candidate_unmatched"
        if r["candidate_priorL2"]:
            return "candidate_prior_L2_excluded_by_pool"
        if r["candidate_priorNonL2_with_focal"]:
            return "candidate_prior_nonL2_same_focal"
        if r["candidate_priorAnySDC"]:
            return "candidate_prior_SDC_no_L2"
        return "candidate_genuine_new_L2_entrant"

    out["coverage_class"] = out.apply(_classify, axis=1)
    return out


def coverage_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in sorted(panel["year"].unique()):
        sub = panel[panel["year"] == t]
        n = len(sub)
        n_unique = sub.drop_duplicates(["focal_cusip", "candidate_cusip"]).shape[0]
        for cls in sorted(panel["coverage_class"].unique()):
            n_cls = (sub["coverage_class"] == cls).sum()
            rows.append({
                "year": t,
                "coverage_class": cls,
                "n_rows": int(n_cls),
                "share_of_year": float(n_cls / n) if n else float("nan"),
            })
        rows.append({
            "year": t,
            "coverage_class": "TOTAL",
            "n_rows": int(n),
            "share_of_year": 1.0,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[p9b_panel] loading bundle")
    bundle = load_all()

    frames = []
    for t in T_YEARS:
        print(f"[p9b_panel] building realized L2 panel at t={t}")
        df = realized_l2_dyads(bundle, t)
        print(f"  rows: {len(df):,}")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    print(f"[p9b_panel] total rows across {len(T_YEARS)} years: "
          f"{len(panel):,}")

    print("[p9b_panel] adding persistence")
    panel = add_persistence(panel, bundle)
    panel["sustained_persist"] = (panel["T_fc"] >= PERSISTENCE_THRESHOLD).astype(int)

    print("[p9b_panel] adding focal sales deltas + controls")
    panel = add_focal_sales_delta(panel, bundle)

    print("[p9b_panel] classifying coverage")
    panel = classify_coverage(panel, bundle)

    out_path = AGG_DIR / "phase9b_lite_realized_panel.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"[p9b_panel] wrote {out_path}: {len(panel):,} rows")

    coverage = coverage_summary(panel)
    coverage.to_csv(AGG_DIR / "phase9b_lite_coverage_by_year.csv", index=False)
    print()
    print("=== coverage by year ===")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()

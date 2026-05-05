"""Phase 9B-lite — tie-robust validation of within-block annotations
and continuous sales association.

Three deliverables:

  A. Coverage breakdown (already produced by phase9b_lite_realized_panel
     and saved to phase9b_lite_coverage_by_year.csv).
  B. Within-saturated-block annotation percentiles.  For each in-pool
     realized dyad, compute the realized partner's tie-adjusted
     percentile within its brokerage tie block under each annotation
     feature.  Test mean percentile > 0.5 with bootstrap CI clustered
     by focal firm.
  C. Continuous sales regression at focal-year level.  Use the maximum
     within-focal block-percentile under each annotation as the regressor;
     OLS with year fixed effects + Compustat controls + bootstrap CI.

Outputs:
  outputs/strategic/aggregate/phase9b_lite_block_annotation_ranks.csv
  outputs/strategic/aggregate/phase9b_lite_block_annotation_summary.csv
  outputs/strategic/aggregate/phase9b_lite_sales_regression.csv
  outputs/strategic/figures/phase9b_lite_coverage_flow.png
  outputs/strategic/figures/phase9b_lite_annotation_percentiles.png
  outputs/strategic/figures/phase9b_lite_sales_beta_forest.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
AGG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "aggregate"
FIG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PATH = AGG_DIR / "phase9b_lite_realized_panel.parquet"
COVERAGE_PATH = AGG_DIR / "phase9b_lite_coverage_by_year.csv"

ANNOTATIONS = [
    "annotated_value_synthetic",  # brokerage_l2 × w_tenure × w_redundancy
                                    # (g_rd is per-focal, constant within block)
    "w_tenure_smooth",
    "w_redundancy",
    "dep_risk",          # higher = MORE risk; report sign aware
    "candidate_degree_l2",
    "candidate_degree_all",
    "n_current_ties",
    "n_hist_ties",
]

BOOTSTRAP_REPS = 1_000


# ──────────────────────────────────────────────────────────────────────
# B. Within-block annotation ranking
# ──────────────────────────────────────────────────────────────────────

def block_annotation_percentiles(panel: pd.DataFrame) -> pd.DataFrame:
    """For each in-pool realized dyad, compute the realized partner's
    within-tie-block percentile under each annotation feature.

    Within-block percentile under feature `a`:
      P^a = (#{c in B : a_c < a*} + 0.5 #{c in B : a_c == a*}) / |B|

    Random ordering inside the block → E[P^a] = 0.5.
    """
    in_pool = panel[panel["coverage_class"] == "candidate_in_pool"].copy()
    in_pool["score_year"] = in_pool["year"] - 1

    rows = []
    n_missing_score = 0
    for _, r in in_pool.iterrows():
        focal = str(r["focal_cusip"])
        cand = str(r["candidate_cusip"])
        sy = int(r["score_year"])
        score_dir = AGG_DIR / (
            "week2_personalization"
            if sy == 2017 else
            f"week2_personalization_{sy}"
        )
        path = score_dir / f"{focal}.parquet"
        if not path.exists():
            n_missing_score += 1
            continue
        df = pd.read_parquet(path)
        if not len(df):
            continue
        df["candidate_cusip"] = df["candidate_cusip"].astype(str)
        # Synthetic annotated_value: brokerage × w_tenure × w_redundancy.
        # g_rd is per-focal constant so it doesn't change within-block rank.
        if "brokerage_l2" in df.columns and \
           "w_tenure_smooth" in df.columns and \
           "w_redundancy" in df.columns:
            df["annotated_value_synthetic"] = (
                df["brokerage_l2"].astype(float).fillna(0.0)
                * df["w_tenure_smooth"].astype(float).fillna(0.5)
                * df["w_redundancy"].astype(float).fillna(1.0)
            )
        match = df[df["candidate_cusip"] == cand]
        if not len(match):
            continue
        b_star = float(match.iloc[0]["brokerage_l2"])
        block = df[df["brokerage_l2"].astype(float) == b_star].copy()
        block_size = len(block)
        if block_size < 2:
            continue
        row = {
            "focal_cusip": focal,
            "candidate_cusip": cand,
            "year": int(r["year"]),
            "score_year": sy,
            "brokerage_block_value": b_star,
            "block_size": int(block_size),
            "T_fc": int(r.get("T_fc", 0)),
            "delta_log_sales_h2": r.get("delta_log_sales_h2"),
            "delta_log_sales_h4": r.get("delta_log_sales_h4"),
            "log_sales_t": r.get("log_sales_t"),
            "log_assets_t": r.get("log_assets_t"),
            "rd_intensity_t": r.get("rd_intensity_t"),
        }
        for a in ANNOTATIONS:
            if a not in df.columns:
                row[f"P_{a}"] = float("nan")
                continue
            star_val = float(match.iloc[0][a]) if pd.notna(
                match.iloc[0][a]) else float("nan")
            block_a = block[a].astype(float).fillna(np.nan)
            if np.isnan(star_val):
                row[f"P_{a}"] = float("nan")
                continue
            n_lt = int((block_a < star_val).sum())
            n_eq = int((block_a == star_val).sum())
            row[f"P_{a}"] = (n_lt + 0.5 * n_eq) / block_size
        rows.append(row)
    if n_missing_score:
        print(f"  [w_block] {n_missing_score} dyads missing per-focal "
              f"scoring parquet (skipped)")
    return pd.DataFrame(rows)


def bootstrap_focal_clustered(values: np.ndarray, focals: np.ndarray,
                                 reps: int = BOOTSTRAP_REPS,
                                 seed: int = 17) -> tuple:
    """Cluster bootstrap by focal firm.  Returns (mean, ci_lo, ci_hi)."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    unique_focals = np.unique(focals)
    means = []
    for _ in range(reps):
        sample = rng.choice(unique_focals, size=len(unique_focals),
                              replace=True)
        idx = np.concatenate([
            np.flatnonzero(focals == f) for f in sample
        ])
        if len(idx) == 0:
            continue
        means.append(np.nanmean(values[idx]))
    means = np.array(means)
    return float(np.nanmean(values)), float(
        np.nanpercentile(means, 2.5)), float(
        np.nanpercentile(means, 97.5))


def block_annotation_summary(block_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    focals = block_df["focal_cusip"].to_numpy()
    for a in ANNOTATIONS:
        col = f"P_{a}"
        if col not in block_df.columns:
            continue
        valid = block_df[col].dropna().to_numpy()
        valid_focals = block_df.loc[block_df[col].notna(),
                                       "focal_cusip"].to_numpy()
        if len(valid) == 0:
            continue
        mean, ci_lo, ci_hi = bootstrap_focal_clustered(valid, valid_focals)
        rows.append({
            "annotation": a,
            "n": int(len(valid)),
            "mean_percentile": mean,
            "ci_lo_95": ci_lo,
            "ci_hi_95": ci_hi,
            "excludes_0.5": (ci_lo > 0.5) or (ci_hi < 0.5),
        })
    return pd.DataFrame(rows).sort_values("mean_percentile",
                                              ascending=False)


# ──────────────────────────────────────────────────────────────────────
# C. Continuous sales regression at focal-year
# ──────────────────────────────────────────────────────────────────────

def focal_year_sales_panel(block_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse block-level percentiles to focal-year level.

    For each (focal, year) with at least one in-pool realized partner,
    compute max within-focal block-percentile under each annotation, plus
    the focal-year sales delta and controls."""
    cols_keep = ["focal_cusip", "year", "delta_log_sales_h2",
                  "delta_log_sales_h4", "log_sales_t", "log_assets_t",
                  "rd_intensity_t"]
    annot_cols = [f"P_{a}" for a in ANNOTATIONS if f"P_{a}" in block_df.columns]
    rows = []
    for (focal, year), sub in block_df.groupby(["focal_cusip", "year"]):
        if len(sub) == 0:
            continue
        rec = {"focal_cusip": focal, "year": int(year),
               "n_realized_inpool": int(len(sub))}
        for c in ["delta_log_sales_h2", "delta_log_sales_h4",
                  "log_sales_t", "log_assets_t", "rd_intensity_t"]:
            v = sub[c].dropna()
            rec[c] = float(v.iloc[0]) if len(v) else float("nan")
        for a in annot_cols:
            v = sub[a].dropna()
            rec[f"max_{a}"] = float(v.max()) if len(v) else float("nan")
            rec[f"mean_{a}"] = float(v.mean()) if len(v) else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def ols_with_year_fe(y: np.ndarray, x: np.ndarray,
                       years: np.ndarray, controls: np.ndarray,
                       reps: int = BOOTSTRAP_REPS,
                       seed: int = 17) -> tuple:
    """OLS of y on x + year FE + controls.  Returns (beta, ci_lo, ci_hi).

    Year FE = one-hot dummy on `years` (drop one for identifiability).
    """
    n = len(y)
    if n < 5:
        return float("nan"), float("nan"), float("nan"), 0
    # Build design: intercept, x, year dummies, controls
    year_arr = years.astype(int)
    uniq_years = sorted(set(year_arr))
    if len(uniq_years) < 2:
        Xy = np.zeros((n, 0))
    else:
        Xy = np.zeros((n, len(uniq_years) - 1))
        for j, yr in enumerate(uniq_years[1:]):
            Xy[:, j] = (year_arr == yr).astype(float)
    X = np.column_stack([np.ones(n), x.astype(float), Xy,
                            controls.astype(float)])
    # Drop rows with any NaN
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]
    if len(y) < 5:
        return float("nan"), float("nan"), float("nan"), int(len(y))

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta = float(coef[1])

    # Bootstrap on rows
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), size=len(y))
        Xb, yb = X[idx], y[idx]
        try:
            cb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
            boot.append(float(cb[1]))
        except np.linalg.LinAlgError:
            continue
    if not boot:
        return beta, float("nan"), float("nan"), int(len(y))
    return beta, float(np.percentile(boot, 2.5)), float(
        np.percentile(boot, 97.5)), int(len(y))


def sales_regression(focal_year_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    annot_cols = [c for c in focal_year_panel.columns if c.startswith("max_P_")]
    for h in (2, 4):
        sales_col = f"delta_log_sales_h{h}"
        if sales_col not in focal_year_panel.columns:
            continue
        for col in annot_cols:
            df = focal_year_panel.dropna(subset=[col, sales_col]).copy()
            if len(df) < 5:
                continue
            controls = df[["log_sales_t", "log_assets_t", "rd_intensity_t",
                            "n_realized_inpool"]].fillna(0).to_numpy()
            beta, lo, hi, n_used = ols_with_year_fe(
                df[sales_col].to_numpy(),
                df[col].to_numpy(),
                df["year"].to_numpy(),
                controls,
            )
            rows.append({
                "annotation": col.replace("max_P_", ""),
                "horizon": h,
                "n_focal_years": int(n_used),
                "beta": beta,
                "ci_lo_95": lo,
                "ci_hi_95": hi,
                "sign": "+" if beta > 0 else ("-" if beta < 0 else "0"),
                "excludes_0": (lo > 0) or (hi < 0),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def fig_coverage_flow(coverage: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar: coverage class shares per year."""
    df = coverage[coverage["coverage_class"] != "TOTAL"].copy()
    pivot = df.pivot(index="year", columns="coverage_class",
                       values="n_rows").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", stacked=True, ax=ax,
                edgecolor="black", linewidth=0.4,
                colormap="Set2")
    ax.set_xlabel("Year of realized new L2 dyad (T)")
    ax.set_ylabel("Number of directed (focal, candidate) rows")
    ax.set_title("Phase 9B-lite — coverage taxonomy of realized new L2 dyads\n"
                  "by year (2009-2014)")
    ax.legend(loc="upper left", fontsize=8, ncol=2,
                bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_annotation_percentiles(summary: pd.DataFrame,
                                 out_path: Path) -> None:
    df = summary.sort_values("mean_percentile")
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(df))
    err = np.array([df["mean_percentile"] - df["ci_lo_95"],
                     df["ci_hi_95"] - df["mean_percentile"]])
    ax.errorbar(df["mean_percentile"], y_pos, xerr=err,
                  fmt="o", color="steelblue", ecolor="gray",
                  capsize=3, markersize=8)
    ax.axvline(0.5, color="firebrick", linestyle="--", linewidth=1.0,
                label="random ordering inside block (E=0.5)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["annotation"])
    ax.set_xlabel("Mean within-block percentile of realized partner "
                   "(95% CI, focal-clustered bootstrap)")
    ax.set_title("Phase 9B-lite — does any annotation rank realized\n"
                  "partners above random inside the saturated brokerage block?")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_sales_beta_forest(reg: pd.DataFrame, out_path: Path) -> None:
    if len(reg) == 0:
        return
    df = reg.sort_values(["horizon", "annotation"]).copy()
    df["label"] = df["annotation"] + " (h=" + df["horizon"].astype(str) + ")"
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(df) + 1.5))
    y_pos = np.arange(len(df))
    err = np.array([df["beta"] - df["ci_lo_95"],
                     df["ci_hi_95"] - df["beta"]])
    colors = ["forestgreen" if r else "gray" for r in df["excludes_0"]]
    ax.errorbar(df["beta"], y_pos, xerr=err, fmt="o",
                  ecolor="lightgray", capsize=3, markersize=7,
                  color="black")
    for i, c in enumerate(colors):
        ax.scatter([df["beta"].iloc[i]], [i], color=c, s=70, zorder=5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlabel("β (sales-delta on max within-focal block percentile, "
                   "year FE + controls; 95% bootstrap CI)")
    ax.set_title("Phase 9B-lite — sales-association forest\n"
                  "(green = CI excludes 0)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[p9b_lite] loading realized panel + coverage")
    panel = pd.read_parquet(PANEL_PATH)
    coverage = pd.read_csv(COVERAGE_PATH)

    print("[p9b_lite] B. building within-block annotation table")
    block_df = block_annotation_percentiles(panel)
    block_path = AGG_DIR / "phase9b_lite_block_annotation_ranks.csv"
    block_df.to_csv(block_path, index=False)
    print(f"  block-level rows: {len(block_df):,}  → {block_path}")

    print("[p9b_lite] B-summary: mean within-block percentile per annotation")
    summary = block_annotation_summary(block_df)
    summary_path = AGG_DIR / "phase9b_lite_block_annotation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    print("[p9b_lite] C. focal-year sales regression")
    focal_year = focal_year_sales_panel(block_df)
    print(f"  focal-year rows: {len(focal_year)}")
    reg = sales_regression(focal_year)
    reg_path = AGG_DIR / "phase9b_lite_sales_regression.csv"
    reg.to_csv(reg_path, index=False)
    if len(reg):
        print(reg.to_string(index=False))
    else:
        print("  (no sales-regression rows produced — N too small)")

    print("[p9b_lite] figures")
    fig_coverage_flow(coverage,
                        FIG_DIR / "phase9b_lite_coverage_flow.png")
    fig_annotation_percentiles(summary,
                                 FIG_DIR / "phase9b_lite_annotation_percentiles.png")
    fig_sales_beta_forest(reg,
                            FIG_DIR / "phase9b_lite_sales_beta_forest.png")
    print("[p9b_lite] done.")


if __name__ == "__main__":
    main()

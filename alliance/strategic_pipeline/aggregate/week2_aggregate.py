"""Week 2A aggregator: stitch per-focal parquets, compute baselines,
residualized + blended scores, and diagnostics.

Run after the per-focal Slurm array has finished writing
``outputs/strategic/aggregate/week2_personalization/<cusip>.parquet``
files.

Outputs (in ``outputs/strategic/aggregate/`` and
``outputs/strategic/figures/``):

  week2_personalization_rows_2017.parquet
  week2_personalization_summary.csv
  week2_top1_concentration_by_variant.csv
  week2_overlap_matrix_by_variant.csv
  week2_type_stratified_neff.csv
  figures/week2_top1_concentration.png
  figures/week2_baseline_overlap_heatmap.png
  figures/week2_type_stratified_neff.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
AGG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "aggregate"
PERSONAL_DIR = AGG_DIR / "week2_personalization"
FIG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2017
TOP_K = 10
RANKERS = [
    # Each tuple: (key, description, sort columns, ascending)
    ("global_degree_all", "candidate global degree (all layers)",
       ["candidate_degree_all"], False),
    ("global_degree_l2", "candidate L2 degree",
       ["candidate_degree_l2"], False),
    ("n_current_ties", "candidate current-tie count",
       ["n_current_ties"], False),
    ("w_tenure_only", "candidate w_tenure_smooth",
       ["w_tenure_smooth"], False),
    ("brokerage_only", "focal brokerage_L2 (focal-specific)",
       ["brokerage_l2"], False),
    ("brokerage_x_tenure", "brokerage_L2 × w_tenure",
       ["durable_value"], False),
    ("raw_score", "full durable-rent score",
       ["raw_score"], False),
]


# ──────────────────────────────────────────────────────────────────────
# Stitch per-focal parquets
# ──────────────────────────────────────────────────────────────────────

def stitch(verbose: bool = True) -> pd.DataFrame:
    files = sorted(PERSONAL_DIR.glob("*.parquet"))
    if verbose:
        print(f"[w2_agg] stitching {len(files):,} per-focal parquets")
    frames = []
    skipped = 0
    for p in files:
        df = pd.read_parquet(p)
        if len(df) == 0:
            skipped += 1
            continue
        frames.append(df)
    if verbose:
        print(f"  non-empty: {len(frames):,}  empty: {skipped:,}")
    out = pd.concat(frames, ignore_index=True)
    out_path = AGG_DIR / "week2_personalization_rows_2017.parquet"
    out.to_parquet(out_path, index=False)
    if verbose:
        print(f"  rows: {len(out):,} → {out_path}")
    return out


# ──────────────────────────────────────────────────────────────────────
# Residualized score
# ──────────────────────────────────────────────────────────────────────

def add_residualized(df: pd.DataFrame, sample_n: int = 500_000,
                       random_state: int = 17) -> pd.DataFrame:
    """Per-focal rank-normalize raw_score (target), regress on
    candidate-global numeric features, take the residual, then add
    candidate-type group-mean offsets.

    Memory-efficient: fits the regression on a sample (~500k rows) of
    just numeric features, then applies coefficients vectorized to all
    18M rows.  SIC×type effects are absorbed via per-(sic, type) mean
    offsets rather than dummy expansion.

    Returns df with new columns:
      raw_score_rank     — within-focal rank percentile [0,1]
      raw_score_pred     — fitted value from candidate-global model
      raw_score_resid    — actual − fitted    (personalization component)
      resid_rank         — within-focal rank percentile of the residual
    """
    print(f"  [resid] within-focal rank-normalize raw_score "
          f"({len(df):,} rows)")
    df = df.copy()
    df["raw_score_rank"] = (
        df.groupby("focal_cusip")["raw_score"].rank(pct=True,
                                                       method="average")
    )

    # Numeric design matrix (just 5 columns: const + 4 log-features)
    print("  [resid] building numeric design matrix")
    log_deg_all = np.log1p(df["candidate_degree_all"].fillna(0).to_numpy(
        dtype=np.float64))
    log_deg_l2 = np.log1p(df["candidate_degree_l2"].fillna(0).to_numpy(
        dtype=np.float64))
    log_n_curr = np.log1p(df["n_current_ties"].fillna(0).to_numpy(
        dtype=np.float64))
    log_n_hist = np.log1p(df["n_hist_ties"].fillna(0).to_numpy(
        dtype=np.float64))
    n = len(df)
    Xn = np.column_stack([np.ones(n, dtype=np.float64),
                            log_deg_all, log_deg_l2,
                            log_n_curr, log_n_hist])
    y = df["raw_score_rank"].fillna(0.5).to_numpy(dtype=np.float64)

    # Fit on a sample to avoid SVD on 18M × 5 (it would still be fast,
    # but the tradeoff is no real cost; the population is iid enough)
    if n > sample_n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_n, replace=False)
        X_fit, y_fit = Xn[idx], y[idx]
    else:
        X_fit, y_fit = Xn, y
    print(f"  [resid] OLS via normal equations on {len(X_fit):,} rows × "
          f"{X_fit.shape[1]} cols")
    XtX = X_fit.T @ X_fit
    Xty = X_fit.T @ y_fit
    coef = np.linalg.solve(XtX, Xty)

    print("  [resid] applying coefficients to full data")
    yhat = Xn @ coef

    # Group-mean offsets for SIC and type — these capture residual
    # candidate-global structure that the numeric features missed.
    print("  [resid] candidate-type group-mean offset")
    df["raw_score_resid_num"] = y - yhat
    type_off = (df.groupby("candidate_type")["raw_score_resid_num"].mean()
                .to_dict())
    sic_off = (df.groupby("candidate_sic2")["raw_score_resid_num"].mean()
                .to_dict())
    type_off_arr = df["candidate_type"].map(type_off).fillna(0.0).to_numpy()
    sic_off_arr = df["candidate_sic2"].map(sic_off).fillna(0.0).to_numpy()
    yhat_full = yhat + type_off_arr + sic_off_arr

    df["raw_score_pred"] = yhat_full
    df["raw_score_resid"] = y - yhat_full
    df.drop(columns=["raw_score_resid_num"], inplace=True)

    print("  [resid] within-focal rank of residual")
    df["resid_rank"] = (
        df.groupby("focal_cusip")["raw_score_resid"]
        .rank(pct=True, method="average")
    )
    return df


def add_blended(df: pd.DataFrame, alphas=(0.0, 0.25, 0.5, 0.75, 1.0)
                 ) -> pd.DataFrame:
    """Blend within-focal rank-normalized raw and residual scores.

    score_blend(α) = α · rank(raw) + (1 − α) · rank(resid)

    α=1 is the raw score; α=0 is the personalization-only score.
    """
    df = df.copy()
    raw_pct = df["raw_score_rank"]
    res_pct = df["resid_rank"]
    for a in alphas:
        col = f"blend_{int(a*100):03d}"
        df[col] = a * raw_pct + (1 - a) * res_pct
    return df


# ──────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────

def top1_per_focal(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    idx = df.groupby("focal_cusip")[score_col].idxmax()
    return df.loc[idx, ["focal_cusip", "candidate_cusip",
                          "candidate_name", "candidate_type", score_col]]


def neff(top1_df: pd.DataFrame) -> float:
    """Effective number of distinct top-1 candidates (inverse Herfindahl)."""
    counts = top1_df["candidate_cusip"].value_counts(normalize=True)
    return float(1.0 / (counts ** 2).sum()) if len(counts) else float("nan")


def topk(df: pd.DataFrame, score_col: str, k: int = TOP_K) -> dict:
    """Per-focal top-k candidate sets, keyed by focal."""
    out = {}
    for fc, sub in df.groupby("focal_cusip"):
        top = sub.nlargest(k, score_col)["candidate_cusip"].tolist()
        out[fc] = set(top)
    return out


def jaccard_matrix(df: pd.DataFrame, score_cols: list,
                    k: int = TOP_K) -> pd.DataFrame:
    sets_by_col = {c: topk(df, c, k=k) for c in score_cols}
    out = pd.DataFrame(index=score_cols, columns=score_cols, dtype=float)
    focals = list(next(iter(sets_by_col.values())).keys())
    for c1 in score_cols:
        for c2 in score_cols:
            if c1 == c2:
                out.loc[c1, c2] = 1.0
                continue
            jacs = []
            for fc in focals:
                a = sets_by_col[c1].get(fc, set())
                b = sets_by_col[c2].get(fc, set())
                if not (a or b):
                    continue
                jacs.append(len(a & b) / len(a | b))
            out.loc[c1, c2] = float(np.mean(jacs)) if jacs else float("nan")
    return out.astype(float)


def neff_table(df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    rows = []
    for c in score_cols:
        t1 = top1_per_focal(df, c)
        ne = neff(t1)
        top_share = float(t1["candidate_cusip"].value_counts(
            normalize=True).iloc[0]) if len(t1) else float("nan")
        rows.append({
            "ranker": c,
            "n_focals_with_top1": int(len(t1)),
            "n_eff_top1": ne,
            "top_candidate_share": top_share,
        })
    return pd.DataFrame(rows)


def neff_by_type(df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    rows = []
    for tp in sorted(df["candidate_type"].dropna().unique()):
        sub = df[df["candidate_type"] == tp]
        for c in score_cols:
            t1 = top1_per_focal(sub, c)
            ne = neff(t1)
            rows.append({
                "candidate_type": tp,
                "ranker": c,
                "n_focals_with_top1": int(len(t1)),
                "n_eff_top1": ne,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def fig_top1_concentration(neff_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df = neff_df.sort_values("n_eff_top1")
    ax.barh(df["ranker"], df["n_eff_top1"],
            color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Effective number of distinct top-1 candidates  (1 / Σ q²)")
    ax.set_title("Personalization diagnostic — top-1 concentration by ranker\n"
                 "Higher = more focal-specific; lower = global-popularity collapse")
    for i, v in enumerate(df["n_eff_top1"]):
        ax.text(v, i, f"  {v:.1f}", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_overlap_heatmap(jac: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(jac.to_numpy(dtype=float),
                   cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(jac.columns)))
    ax.set_xticklabels(jac.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(jac.index)))
    ax.set_yticklabels(jac.index, fontsize=8)
    for i in range(jac.shape[0]):
        for j in range(jac.shape[1]):
            v = jac.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=7, color="white" if v < 0.55 else "black")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Mean Jaccard@{TOP_K} across focal firms")
    ax.set_title(f"Top-{TOP_K} overlap between rankers")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_type_stratified(neff_type_df: pd.DataFrame, out_path: Path) -> None:
    pivot = neff_type_df.pivot(index="ranker", columns="candidate_type",
                                  values="n_eff_top1")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="barh", ax=ax, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("N_eff top-1")
    ax.set_title("Personalization stratified by candidate type")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cached = AGG_DIR / "week2_personalization_rows_2017.parquet"
    if cached.exists():
        print(f"[w2_agg] loading cached stitched parquet: {cached}")
        df = pd.read_parquet(cached)
        print(f"  rows: {len(df):,}")
    else:
        print("[w2_agg] stitching per-focal parquets")
        df = stitch()

    print("[w2_agg] residualizing raw_score against candidate-global features")
    df = add_residualized(df)
    df = add_blended(df)

    df.to_parquet(AGG_DIR / "week2_personalization_rows_2017.parquet",
                   index=False)

    score_cols = [r[0] for r in RANKERS] + [
        "blend_000", "blend_025", "blend_050", "blend_075", "blend_100",
        "resid_rank",
    ]

    # Build per-row score columns matching the RANKERS spec
    df["raw_score"] = df["raw_score"].fillna(0.0)
    df["durable_value"] = df["durable_value"].fillna(0.0)
    df["brokerage_l2"] = df["brokerage_l2"].fillna(0.0)
    df["w_tenure_smooth"] = df["w_tenure_smooth"].fillna(0.5)
    df["candidate_degree_all"] = df["candidate_degree_all"].fillna(0)
    df["candidate_degree_l2"] = df["candidate_degree_l2"].fillna(0)
    df["n_current_ties"] = df["n_current_ties"].fillna(0)
    # Map the RANKERS list keys to canonical score columns
    df["global_degree_all"] = df["candidate_degree_all"]
    df["global_degree_l2"] = df["candidate_degree_l2"]
    df["w_tenure_only"] = df["w_tenure_smooth"]
    df["brokerage_only"] = df["brokerage_l2"]
    df["brokerage_x_tenure"] = df["durable_value"]

    print("[w2_agg] N_eff top-1 per ranker")
    neff_df = neff_table(df, score_cols)
    neff_df.to_csv(AGG_DIR / "week2_top1_concentration_by_variant.csv",
                    index=False)
    print(neff_df.to_string(index=False))

    print("[w2_agg] top-K Jaccard between rankers")
    jac = jaccard_matrix(df, score_cols, k=TOP_K)
    jac.to_csv(AGG_DIR / "week2_overlap_matrix_by_variant.csv")

    print("[w2_agg] N_eff stratified by candidate type")
    neff_type_df = neff_by_type(df, ["raw_score", "resid_rank",
                                       "blend_050", "n_current_ties"])
    neff_type_df.to_csv(AGG_DIR / "week2_type_stratified_neff.csv",
                         index=False)

    # Summary one-row-per-ranker
    summary_rows = []
    for c in score_cols:
        t1 = top1_per_focal(df, c)
        if not len(t1):
            continue
        summary_rows.append({
            "ranker": c,
            "n_eff_top1": neff(t1),
            "top_candidate": t1["candidate_name"].mode().iloc[0]
                              if len(t1["candidate_name"].mode()) else "",
            "top_candidate_share":
                float(t1["candidate_cusip"].value_counts(normalize=True).iloc[0]),
        })
    pd.DataFrame(summary_rows).to_csv(
        AGG_DIR / "week2_personalization_summary.csv", index=False)

    print("[w2_agg] figures")
    fig_top1_concentration(neff_df, FIG_DIR / "week2_top1_concentration.png")
    fig_overlap_heatmap(jac, FIG_DIR / "week2_baseline_overlap_heatmap.png")
    fig_type_stratified(neff_type_df,
                          FIG_DIR / "week2_type_stratified_neff.png")

    print("[w2_agg] done.")


if __name__ == "__main__":
    main()

"""Week 2B — out-of-time backtest of the alignment recommender at t=2011.

For each realized 2011 L2 dyad (f, c, t=2011), we:

1. Look up f's full ranking of candidates at t=2011 from the per-focal
   parquets in ``outputs/strategic/aggregate/week2_personalization_2011/``.
2. Find the rank of the realized candidate c under each ranker
   (brokerage_only, raw_score, n_current_ties, candidate_degree_l2,
   etc.).
3. Compute hit-rate metrics: P(realized partner in top-K under
   ranker R) for K ∈ {5, 10, 20, 50, 100}.
4. For sales deltas (Δ log Sales at h=2, 4), test whether realized
   partners that are in the top-K under R have higher mean sales
   lift than realized partners outside top-K (a per-dyad
   conditional comparison) AND whether focal firms whose realized
   partners are in the top-K under R have higher focal sales lift
   than focals whose realized partners are not.
5. Persistence (T_fc ≥ 3): same hit-rate analysis stratified by
   sustained vs not.  At t=2011 the sustained-share is 0%, so this
   is a degenerate report; future runs at multiple t-years will
   bulk it up.

Outputs (in ``outputs/strategic/aggregate/``):
  week2b_backtest_summary.csv
  week2b_hit_rate_by_ranker.csv
  week2b_sales_lift_by_ranker.csv
  week2b_realized_with_ranks_2011.parquet
  figures/week2b_hit_rate.png
  figures/week2b_sales_lift.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
AGG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "aggregate"
# Rankings are produced at SCORE_YEAR (= T_REALIZED − 1) using only
# data from [SCORE_YEAR−4, SCORE_YEAR] = [2006, 2010] — no 2011 leakage.
T = 2011                                 # year of realized ties to evaluate
SCORE_YEAR = 2010                        # year whose rankings we test against
SCORING_DIR = AGG_DIR / f"week2_personalization_{SCORE_YEAR}"
FIG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TOP_K_VALUES = [5, 10, 20, 50, 100]

RANKERS = [
    ("brokerage_only", ["brokerage_l2"], [False]),
    ("raw_score", ["raw_score"], [False]),
    ("durable_value", ["durable_value"], [False]),
    ("w_tenure_only", ["w_tenure_smooth"], [False]),
    ("n_current_ties", ["n_current_ties"], [False]),
    ("candidate_degree_l2", ["candidate_degree_l2"], [False]),
    ("candidate_degree_all", ["candidate_degree_all"], [False]),
]


# ──────────────────────────────────────────────────────────────────────
# Load realized ties + per-focal rankings, join into one panel
# ──────────────────────────────────────────────────────────────────────

def load_realized() -> pd.DataFrame:
    df = pd.read_parquet(AGG_DIR / f"week2b_realized_ties_{T}.parquet")
    df["focal_cusip"] = df["focal_cusip"].astype(str)
    df["candidate_cusip"] = df["candidate_cusip"].astype(str)
    return df


def attach_ranks(realized: pd.DataFrame) -> pd.DataFrame:
    """For each realized (focal, candidate, t) row, find the rank of
    the realized candidate under every ranker in the focal's per-focal
    parquet."""
    rows = []
    missing_focals = 0
    missing_cands = 0
    for focal, sub in realized.groupby("focal_cusip"):
        path = SCORING_DIR / f"{focal}.parquet"
        if not path.exists():
            missing_focals += 1
            continue
        ranking = pd.read_parquet(path)
        if len(ranking) == 0:
            continue
        ranking["candidate_cusip"] = ranking["candidate_cusip"].astype(str)
        n_pool = len(ranking)
        for key, cols, asc in RANKERS:
            if cols[0] not in ranking.columns:
                continue
            ranking[f"_rank_{key}"] = ranking[cols[0]].rank(
                method="min", ascending=asc[0])
        for _, r in sub.iterrows():
            cand = r["candidate_cusip"]
            match = ranking[ranking["candidate_cusip"] == cand]
            if len(match) == 0:
                missing_cands += 1
                continue
            row = r.to_dict()
            row["pool_size"] = int(n_pool)
            for key, _, _ in RANKERS:
                col = f"_rank_{key}"
                if col in match.columns:
                    row[f"rank_{key}"] = float(match.iloc[0][col])
                    row[f"rank_pct_{key}"] = float(match.iloc[0][col]) / n_pool
            rows.append(row)
    print(f"  [w2b_backtest] missing focals: {missing_focals};  "
          f"missing candidates in pool: {missing_cands}")
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Hit-rate / sales-lift metrics
# ──────────────────────────────────────────────────────────────────────

def hit_rate_table(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, _, _ in RANKERS:
        col = f"rank_{key}"
        if col not in joined.columns:
            continue
        for k in TOP_K_VALUES:
            in_topk = (joined[col] <= k).astype(int)
            hr = in_topk.mean()
            rows.append({
                "ranker": key,
                "top_k": k,
                "hit_rate": hr,
                "n_realized": len(joined),
                "n_in_topk": int(in_topk.sum()),
            })
    return pd.DataFrame(rows)


def sales_lift_table(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, _, _ in RANKERS:
        col = f"rank_{key}"
        if col not in joined.columns:
            continue
        for k in TOP_K_VALUES:
            in_topk = (joined[col] <= k)
            for h in (2, 4):
                sales_col = f"delta_log_sales_h{h}"
                if sales_col not in joined.columns:
                    continue
                in_top = joined.loc[in_topk, sales_col].dropna()
                out_top = joined.loc[~in_topk, sales_col].dropna()
                if not (len(in_top) and len(out_top)):
                    continue
                rows.append({
                    "ranker": key,
                    "top_k": k,
                    "horizon": h,
                    "n_in_topk": int(len(in_top)),
                    "n_out_topk": int(len(out_top)),
                    "mean_sales_in_topk": float(in_top.mean()),
                    "mean_sales_out_topk": float(out_top.mean()),
                    "lift": float(in_top.mean() - out_top.mean()),
                })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def fig_hit_rate(hit_df: pd.DataFrame, out_path: Path) -> None:
    pivot = hit_df.pivot(index="top_k", columns="ranker", values="hit_rate")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Top-K cutoff")
    ax.set_ylabel(f"Hit rate (P(realized partner in top-K)) at t={T}")
    ax.set_title("Week-2B — out-of-time hit rate of realized 2011 L2 ties\n"
                  "by ranker × top-K cutoff")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_sales_lift(lift_df: pd.DataFrame, out_path: Path) -> None:
    if len(lift_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, h in zip(axes, (2, 4)):
        sub = lift_df[lift_df["horizon"] == h]
        if len(sub) == 0:
            continue
        pivot = sub.pivot(index="top_k", columns="ranker", values="lift")
        pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Top-K cutoff")
        ax.set_ylabel(f"Mean Δ log Sales lift  (in_topk − out_topk)")
        ax.set_title(f"horizon h = {h}")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(loc="best", fontsize=7, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Week-2B — sales-lift conditional on top-K membership at t={T}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[w2b_backtest] loading realized ties")
    realized = load_realized()
    print(f"  {len(realized)} (focal, candidate) rows; "
          f"{realized['focal_cusip'].nunique()} unique focals")

    print("[w2b_backtest] attaching ranks from per-focal parquets")
    joined = attach_ranks(realized)
    print(f"  joined: {len(joined)} rows")
    joined.to_parquet(AGG_DIR / f"week2b_realized_with_ranks_{T}.parquet",
                       index=False)

    print("[w2b_backtest] hit-rate table")
    hit_df = hit_rate_table(joined)
    hit_df.to_csv(AGG_DIR / "week2b_hit_rate_by_ranker.csv", index=False)
    print(hit_df.to_string(index=False))

    print("[w2b_backtest] sales-lift table")
    lift_df = sales_lift_table(joined)
    lift_df.to_csv(AGG_DIR / "week2b_sales_lift_by_ranker.csv", index=False)
    if len(lift_df):
        print(lift_df.to_string(index=False))

    print("[w2b_backtest] persistence summary")
    persist = joined.groupby("sustained_persist").size().rename(
        "n").reset_index()
    print(persist.to_string(index=False))

    print("[w2b_backtest] figures")
    fig_hit_rate(hit_df, FIG_DIR / "week2b_hit_rate.png")
    fig_sales_lift(lift_df, FIG_DIR / "week2b_sales_lift.png")

    # Top-line summary
    summary_rows = []
    for key, _, _ in RANKERS:
        col = f"rank_{key}"
        if col not in joined.columns:
            continue
        for k in TOP_K_VALUES:
            in_topk = (joined[col] <= k)
            summary_rows.append({
                "ranker": key,
                "top_k": k,
                "hit_rate": float(in_topk.mean()),
                "n_realized_in_topk": int(in_topk.sum()),
                "mean_rank": float(joined[col].mean()),
                "median_rank_pct": float(joined[f"rank_pct_{key}"].median()),
            })
    pd.DataFrame(summary_rows).to_csv(
        AGG_DIR / "week2b_backtest_summary.csv", index=False)

    print("[w2b_backtest] done.")


if __name__ == "__main__":
    main()

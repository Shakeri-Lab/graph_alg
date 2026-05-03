"""Week 2C aggregator: do focal-conditional interaction features
break the personalization collapse?

Compares Week-2A's brokerage-only N_eff (1,811.9 / 7,626) against
new rankers built from the interaction features (n_shared_partners,
jaccard_partners, share_focal_in_c, same_sic2, same_sic1, nation_match,
sic2_distance), alone and combined with brokerage_l2.

Outputs (in ``outputs/strategic/aggregate/`` and
``outputs/strategic/figures/``):
  week2c_top1_concentration_by_variant.csv
  week2c_overlap_matrix_by_variant.csv
  week2c_interaction_summary.csv
  figures/week2c_top1_concentration.png
  figures/week2c_overlap_heatmap.png
  figures/week2c_brokerage_vs_interaction_lift.png
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
FIG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "figures"
ROWS_IN = AGG_DIR / "week2c_personalization_rows_2017.parquet"

TOP_K = 10


# Each tuple: (key, description, sort columns, ascending list)
RANKERS = [
    # Week-2A baselines we want to compare against
    ("brokerage_only",
       "Week-2A brokerage_L2 only", ["brokerage_l2"], [False]),
    ("raw_score",
       "Week-2A raw durable-rent score", ["raw_score"], [False]),
    # New focal-conditional features standalone
    ("n_shared_partners_only",
       "shared partner count alone", ["n_shared_partners"], [False]),
    ("jaccard_only",
       "Jaccard partner overlap alone", ["jaccard_partners"], [False]),
    ("share_focal_in_c_only",
       "share of focal partners that are c partners",
       ["share_focal_in_c"], [False]),
    ("same_sic2_only",
       "same SIC2 indicator alone (with random within tie)",
       ["same_sic2"], [False]),
    ("nation_match_only",
       "nation match indicator alone", ["nation_match"], [False]),
    # Brokerage × interaction (additive in rank-percentiles)
    ("broker_x_jaccard",
       "rank(brokerage) + rank(jaccard)",
       ["_combined_brk_jaccard"], [False]),
    ("broker_x_shared",
       "rank(brokerage) + rank(n_shared_partners)",
       ["_combined_brk_shared"], [False]),
    ("broker_x_sic2",
       "rank(brokerage) + rank(same_sic2)",
       ["_combined_brk_sic2"], [False]),
    ("broker_x_nation",
       "rank(brokerage) + rank(nation_match)",
       ["_combined_brk_nation"], [False]),
    # Full new score: rank(brokerage) + rank(jaccard) + rank(same_sic2) + rank(nation_match)
    ("full_interaction",
       "rank-sum: brokerage + jaccard + same_sic2 + nation_match",
       ["_combined_full"], [False]),
]


def add_combined_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-focal rank-normalized combinations used as ranker inputs.

    Also aliases base columns to the ranker-key names used in RANKERS so
    the rest of the pipeline can index by ranker key uniformly.
    """
    print("  [w2c_agg] computing within-focal rank-percentiles for combos")
    out = df.copy()

    # Aliases: ranker key → underlying column
    out["brokerage_only"] = out["brokerage_l2"]
    out["n_shared_partners_only"] = out["n_shared_partners"]
    out["jaccard_only"] = out["jaccard_partners"]
    out["share_focal_in_c_only"] = out["share_focal_in_c"]
    out["same_sic2_only"] = out["same_sic2"].astype(float)
    out["nation_match_only"] = out["nation_match"].astype(float)

    pct_brk = out.groupby("focal_cusip")["brokerage_l2"].rank(pct=True,
                                                                  method="average")
    pct_jac = out.groupby("focal_cusip")["jaccard_partners"].rank(
        pct=True, method="average")
    pct_shared = out.groupby("focal_cusip")["n_shared_partners"].rank(
        pct=True, method="average")
    pct_sic2 = out.groupby("focal_cusip")["same_sic2"].rank(
        pct=True, method="average")
    pct_nation = out.groupby("focal_cusip")["nation_match"].rank(
        pct=True, method="average")

    out["_combined_brk_jaccard"] = pct_brk + pct_jac
    out["_combined_brk_shared"] = pct_brk + pct_shared
    out["_combined_brk_sic2"] = pct_brk + pct_sic2
    out["_combined_brk_nation"] = pct_brk + pct_nation
    out["_combined_full"] = pct_brk + pct_jac + pct_sic2 + pct_nation

    out["broker_x_jaccard"] = out["_combined_brk_jaccard"]
    out["broker_x_shared"] = out["_combined_brk_shared"]
    out["broker_x_sic2"] = out["_combined_brk_sic2"]
    out["broker_x_nation"] = out["_combined_brk_nation"]
    out["full_interaction"] = out["_combined_full"]
    return out


def top1_per_focal(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    idx = df.groupby("focal_cusip")[score_col].idxmax()
    return df.loc[idx, ["focal_cusip", "candidate_cusip",
                          "candidate_name", "candidate_type", score_col]]


def neff(top1_df: pd.DataFrame) -> float:
    counts = top1_df["candidate_cusip"].value_counts(normalize=True)
    return float(1.0 / (counts ** 2).sum()) if len(counts) else float("nan")


def topk_sets(df: pd.DataFrame, score_col: str, k: int = TOP_K) -> dict:
    out = {}
    for fc, sub in df.groupby("focal_cusip"):
        top = sub.nlargest(k, score_col)["candidate_cusip"].tolist()
        out[fc] = set(top)
    return out


def jaccard_matrix(df: pd.DataFrame, score_cols: list,
                    k: int = TOP_K) -> pd.DataFrame:
    sets_by_col = {c: topk_sets(df, c, k=k) for c in score_cols}
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


def neff_table(df: pd.DataFrame, rankers: list) -> pd.DataFrame:
    rows = []
    for key, desc, _, _ in rankers:
        t1 = top1_per_focal(df, key)
        ne = neff(t1)
        top_share = float(t1["candidate_cusip"].value_counts(
            normalize=True).iloc[0]) if len(t1) else float("nan")
        top_name = t1["candidate_name"].mode().iloc[0] \
            if len(t1["candidate_name"].mode()) else ""
        rows.append({
            "ranker": key,
            "description": desc,
            "n_focals": int(len(t1)),
            "n_eff_top1": ne,
            "top_candidate": top_name,
            "top_candidate_share": top_share,
        })
    return pd.DataFrame(rows)


def fig_top1_concentration(neff_df: pd.DataFrame, out_path: Path) -> None:
    df = neff_df.sort_values("n_eff_top1")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["lightcoral" if r in {"brokerage_only", "raw_score"}
              else "steelblue" for r in df["ranker"]]
    ax.barh(df["ranker"], df["n_eff_top1"], color=colors,
             edgecolor="black", linewidth=0.5)
    for i, v in enumerate(df["n_eff_top1"]):
        ax.text(v, i, f"  {v:,.1f}", va="center", fontsize=9)
    ax.axvline(7626, color="black", linestyle=":", linewidth=0.8,
                label="upper bound (= n_focals)")
    ax.set_xlabel("Effective number of distinct top-1 candidates "
                   "(1 / Σ q²)")
    ax.set_title("Week-2C — does adding focal-conditional interaction\n"
                  "features break the personalization collapse?")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_overlap(jac: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
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
    cbar.set_label(f"Mean Jaccard@{TOP_K} across 7,626 focals")
    ax.set_title(f"Week-2C — top-{TOP_K} overlap between rankers")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_brokerage_vs_interaction(neff_df: pd.DataFrame,
                                    out_path: Path) -> None:
    """Two-bar comparison: Week-2A brokerage_only vs the best Week-2C
    combined ranker."""
    base = neff_df[neff_df["ranker"] == "brokerage_only"]["n_eff_top1"].iloc[0]
    best_row = neff_df[~neff_df["ranker"].isin({"brokerage_only",
                                                    "raw_score"})] \
        .nlargest(1, "n_eff_top1").iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["Week-2A baseline\n(brokerage_only)",
              f"Week-2C best\n({best_row['ranker']})"],
            [base, best_row["n_eff_top1"]],
            color=["lightcoral", "steelblue"], edgecolor="black")
    for i, v in enumerate([base, best_row["n_eff_top1"]]):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=11,
                  weight="bold")
    ax.axhline(7626, color="black", linestyle=":", linewidth=0.8,
                label="theoretical upper bound (7,626)")
    ax.set_ylabel("N_eff top-1")
    ax.set_title(f"Lift from focal-conditional interaction features\n"
                  f"({best_row['n_eff_top1']/base:.2f}× over Week-2A baseline)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    print(f"[w2c_agg] loading {ROWS_IN}")
    df = pd.read_parquet(ROWS_IN)
    print(f"  rows: {len(df):,}")

    print("[w2c_agg] adding combined-rank columns")
    df = add_combined_ranks(df)

    score_cols = [r[0] for r in RANKERS]
    print("[w2c_agg] N_eff per ranker")
    nt = neff_table(df, RANKERS)
    nt.to_csv(AGG_DIR / "week2c_top1_concentration_by_variant.csv",
                index=False)
    print(nt.to_string(index=False))

    print("[w2c_agg] Jaccard@K matrix")
    jac = jaccard_matrix(df, score_cols, k=TOP_K)
    jac.to_csv(AGG_DIR / "week2c_overlap_matrix_by_variant.csv")

    nt.to_csv(AGG_DIR / "week2c_interaction_summary.csv", index=False)

    print("[w2c_agg] figures")
    fig_top1_concentration(nt, FIG_DIR / "week2c_top1_concentration.png")
    fig_overlap(jac, FIG_DIR / "week2c_overlap_heatmap.png")
    fig_brokerage_vs_interaction(
        nt, FIG_DIR / "week2c_brokerage_vs_interaction_lift.png")

    print("[w2c_agg] done.")


if __name__ == "__main__":
    main()

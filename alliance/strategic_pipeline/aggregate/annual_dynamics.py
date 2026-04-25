"""Annual dynamics of the systemic-criticality meta-network, 1995–2017.

The upgrade to ``build_systemic_criticality.py`` exposed an annual panel
(``systemic_criticality_annual.csv``, 100k rows) where each row is a
(year, partner) node with its aggregate predicted-loss and in-degree.
The prototype paper could only report the 2017 cross-section; this
module mines the panel for time-series findings:

  * Rank trajectories for persistent top firms
  * Top-20 turnover and persistence rate
  * Risers and fallers (largest rank improvements / declines)
  * Industry concentration evolution (Herfindahl-Hirschman on SIC-2
    shares of total predicted cost)
  * Compustat-matched vs. non-Compustat share over time
  * Aggregate system stress (sum of predicted log-MV cost by year)

Outputs
-------
CSV tables in ``outputs/strategic/aggregate/``:
    rank_trajectories.csv
    top20_persistence.csv
    risers_fallers.csv
    industry_concentration_annual.csv
    compustat_share_annual.csv
    aggregate_stress_annual.csv

Figures in ``outputs/strategic/aggregate/``:
    fig_d1_rank_heatmap.png          — rank trajectory for top-20 persistent firms
    fig_d2_concentration_over_time.png — HHI + top-3 share
    fig_d3_risers_fallers.png        — biggest rank gainers and losers
    fig_d4_compustat_share.png       — share of total cost on Compustat vs non-Compustat
    fig_d5_system_stress.png         — total predicted cost vs active-firm count
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
AGG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "aggregate"
ANNUAL_CSV = AGG_DIR / "systemic_criticality_annual.csv"

YEAR_MIN, YEAR_MAX = 1995, 2017
PERSISTENCE_MIN_YEARS = 9
RISER_FALLER_MIN_YEARS = 6


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def load_annual_panel() -> pd.DataFrame:
    df = pd.read_csv(ANNUAL_CSV, dtype={"partner_cusip": str, "sic2": str})
    df["year"] = df["year"].astype(int)
    for col in ("in_degree", "total_predicted_log_mv_cost",
                "mean_predicted_log_mv_cost"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    return df


# ──────────────────────────────────────────────────────────────────────
# Rank trajectories
# ──────────────────────────────────────────────────────────────────────

def compute_rank_trajectories(panel: pd.DataFrame,
                                min_years: int = PERSISTENCE_MIN_YEARS
                                ) -> pd.DataFrame:
    """Wide panel of rank by year for firms with many top-20 appearances."""
    top20 = panel[panel["rank"] <= 20]
    counts = (top20.groupby("partner_cusip")
              .agg(years_in_top20=("year", "nunique"),
                   name=("name", "first"),
                   sic2=("sic2", "first"),
                   is_compustat=("is_compustat", "first"))
              .reset_index())
    persistent = counts[counts["years_in_top20"] >= min_years].copy()
    persistent = persistent.sort_values("years_in_top20", ascending=False)

    wide = (panel[panel["partner_cusip"].isin(persistent["partner_cusip"])]
            .pivot_table(index="partner_cusip", columns="year",
                         values="rank", aggfunc="first"))
    wide = wide.reindex(persistent["partner_cusip"])
    wide.insert(0, "name", persistent.set_index("partner_cusip")["name"])
    wide.insert(1, "sic2", persistent.set_index("partner_cusip")["sic2"])
    wide.insert(2, "is_compustat",
                persistent.set_index("partner_cusip")["is_compustat"])
    wide.insert(3, "years_in_top20",
                persistent.set_index("partner_cusip")["years_in_top20"])
    return wide.reset_index()


# ──────────────────────────────────────────────────────────────────────
# Top-20 persistence
# ──────────────────────────────────────────────────────────────────────

def compute_top20_persistence(panel: pd.DataFrame) -> pd.DataFrame:
    """Fraction of year t top-20 that was in year t-1 top-20 (retention)."""
    rows = []
    years = sorted(panel["year"].unique())
    for t in years[1:]:
        t0 = t - 1
        a = set(panel.loc[(panel["year"] == t0) & (panel["rank"] <= 20),
                          "partner_cusip"])
        b = set(panel.loc[(panel["year"] == t) & (panel["rank"] <= 20),
                          "partner_cusip"])
        retained = len(a & b)
        rows.append({
            "year": t,
            "retained": retained,
            "new_entries": 20 - retained,
            "retention_rate": retained / 20.0 if len(b) else np.nan,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Risers and fallers
# ──────────────────────────────────────────────────────────────────────

RANK_RELEVANCE_THRESHOLD = 100  # firm must hit top-100 at least once


def compute_risers_fallers(panel: pd.DataFrame,
                             min_years: int = RISER_FALLER_MIN_YEARS,
                             top_n: int = 15) -> pd.DataFrame:
    """Firms with the largest rank improvement / decline.

    For each partner, compare the average rank in the first and last
    thirds of its observed years; require at least ``min_years`` of
    observations so the comparison is stable.  Lower rank = more
    systemic, so ``delta_rank = first_third_rank - last_third_rank``
    is positive for risers.

    Filters to firms that reached the top-``RANK_RELEVANCE_THRESHOLD``
    at least once, so the result is not dominated by fringe-to-fringe
    noise (the rank list extends to 6,000+ in peak alliance years).
    """
    by_firm = []
    for cusip, g in panel.groupby("partner_cusip"):
        if len(g) < min_years:
            continue
        if g["rank"].min() > RANK_RELEVANCE_THRESHOLD:
            continue
        g = g.sort_values("year")
        k = max(1, len(g) // 3)
        first = g.head(k)["rank"].mean()
        last = g.tail(k)["rank"].mean()
        by_firm.append({
            "partner_cusip": cusip,
            "name": g["name"].iloc[-1],
            "sic2": g["sic2"].iloc[-1],
            "is_compustat": g["is_compustat"].iloc[-1],
            "n_years": len(g),
            "first_period_mean_rank": first,
            "last_period_mean_rank": last,
            "delta_rank": first - last,
            "last_period_best_rank": g.tail(k)["rank"].min(),
        })
    out = pd.DataFrame(by_firm)
    out = out.sort_values("delta_rank", ascending=False).reset_index(drop=True)
    risers = out.head(top_n).copy()
    risers["category"] = "riser"
    fallers = out.tail(top_n).copy()
    fallers["category"] = "faller"
    return pd.concat([risers, fallers], ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# Industry concentration (Herfindahl on SIC-2 cost share)
# ──────────────────────────────────────────────────────────────────────

def compute_industry_concentration(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in panel.groupby("year"):
        g = g.dropna(subset=["sic2"]).copy()
        total = g["total_predicted_log_mv_cost"].sum()
        if total <= 0:
            continue
        shares = g.groupby("sic2")["total_predicted_log_mv_cost"].sum() / total
        hhi = (shares ** 2).sum()
        top3 = shares.nlargest(3).sum()
        top10 = shares.nlargest(10).sum()
        rows.append({
            "year": year,
            "n_active_sic2": (shares > 0).sum(),
            "hhi": hhi,
            "top3_share": top3,
            "top10_share": top10,
            "leading_sic2": shares.idxmax(),
            "leading_sic2_share": shares.max(),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Compustat vs non-Compustat share
# ──────────────────────────────────────────────────────────────────────

def compute_compustat_share(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in panel.groupby("year"):
        total = g["total_predicted_log_mv_cost"].sum()
        if total <= 0:
            continue
        compustat = g.loc[g["is_compustat"] == True,
                           "total_predicted_log_mv_cost"].sum()
        rows.append({
            "year": year,
            "compustat_cost": compustat,
            "non_compustat_cost": total - compustat,
            "compustat_share": compustat / total,
            "n_total_partners": len(g),
            "n_compustat_partners": int((g["is_compustat"] == True).sum()),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Aggregate system stress
# ──────────────────────────────────────────────────────────────────────

def compute_aggregate_stress(panel: pd.DataFrame) -> pd.DataFrame:
    agg = (panel.groupby("year")
           .agg(total_predicted_log_mv_cost=
                  ("total_predicted_log_mv_cost", "sum"),
                total_in_degree=("in_degree", "sum"),
                n_partners=("partner_cusip", "nunique"),
                mean_per_partner_cost=
                  ("total_predicted_log_mv_cost", "mean"))
           .reset_index())
    return agg.sort_values("year").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def plot_rank_heatmap(traj: pd.DataFrame, out: Path) -> None:
    years = list(range(YEAR_MIN, YEAR_MAX + 1))
    df = traj.copy()
    M = df[years].to_numpy(dtype=float)
    M_display = np.where(np.isnan(M), 25, M)  # absent years rendered as off-chart rank 25

    fig, ax = plt.subplots(figsize=(12, 0.38 * len(df) + 1.5))
    im = ax.imshow(M_display, aspect="auto", cmap="viridis_r",
                   vmin=1, vmax=20)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=60, fontsize=8)
    labels = [f"{n[:25]}  ({s})" for n, s in zip(df["name"], df["sic2"])]
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Year")
    ax.set_title("Systemic-criticality rank trajectory, top-20–persistent firms",
                 fontsize=11)
    # Annotate rank integers for ranks 1..10
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            r = M[i, j]
            if not np.isnan(r) and r <= 10:
                ax.text(j, i, f"{int(r)}", ha="center", va="center",
                        color="white" if r <= 5 else "black", fontsize=6)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Rank (1 = most systemic)")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_concentration(conc: pd.DataFrame, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(conc["year"], conc["hhi"], color="crimson",
             marker="o", linewidth=2, label="HHI on SIC-2 cost share")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Herfindahl index (0–1)", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(conc["year"], conc["top3_share"], color="steelblue",
             marker="s", linewidth=2, label="Top-3 SIC-2 share")
    ax2.plot(conc["year"], conc["top10_share"], color="steelblue",
             marker="^", linewidth=1.5, linestyle="--",
             label="Top-10 SIC-2 share")
    ax2.set_ylabel("Share of total predicted cost", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower left", fontsize=9)
    ax1.set_title("Industry concentration of systemic criticality, 1995–2017")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_risers_fallers(rf: pd.DataFrame, out: Path, top_n: int = 12) -> None:
    risers = rf[rf["category"] == "riser"].head(top_n).iloc[::-1]
    fallers = rf[rf["category"] == "faller"].tail(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=False)
    axes[0].barh(
        [f"{n[:22]} ({s})" for n, s in zip(risers["name"], risers["sic2"])],
        risers["delta_rank"], color="steelblue",
    )
    axes[0].set_xlabel("Rank gain (earlier mean – later mean)")
    axes[0].set_title("Top risers")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(
        [f"{n[:22]} ({s})" for n, s in zip(fallers["name"], fallers["sic2"])],
        fallers["delta_rank"], color="firebrick",
    )
    axes[1].set_xlabel("Rank change (negative = fell)")
    axes[1].set_title("Top fallers")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.suptitle("Rank gainers and losers in systemic criticality, 1995–2017")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_compustat_share(shr: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(shr["year"], 0, shr["compustat_share"],
                    color="steelblue", alpha=0.4,
                    label="Compustat-matched partner share")
    ax.plot(shr["year"], shr["compustat_share"],
            color="steelblue", marker="o", linewidth=2)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of total predicted log-MV cost")
    ax.set_ylim(0, 1)
    ax.set_title("Compustat-matched share of systemic criticality, 1995–2017")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_system_stress(agg: pd.DataFrame, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(agg["year"], agg["total_predicted_log_mv_cost"],
             color="darkorange", marker="o", linewidth=2,
             label="Total predicted log-MV cost")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total cost (log-MV units)", color="darkorange")
    ax1.tick_params(axis="y", labelcolor="darkorange")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(agg["year"], agg["n_partners"], color="teal",
             marker="s", linewidth=1.5, label="Active partners")
    ax2.set_ylabel("Active partners (distinct, per year)", color="teal")
    ax2.tick_params(axis="y", labelcolor="teal")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=9)
    ax1.set_title("Aggregate systemic stress and network size, 1995–2017")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[annual_dynamics] loading {ANNUAL_CSV}")
    panel = load_annual_panel()
    print(f"  rows={len(panel):,}  years={panel['year'].min()}"
          f"–{panel['year'].max()}  distinct partners="
          f"{panel['partner_cusip'].nunique():,}")

    print("[annual_dynamics] rank trajectories")
    traj = compute_rank_trajectories(panel)
    traj.to_csv(AGG_DIR / "rank_trajectories.csv", index=False)
    print(f"  {len(traj)} persistent firms (≥{PERSISTENCE_MIN_YEARS} yrs in top-20)")

    print("[annual_dynamics] top-20 persistence")
    persist = compute_top20_persistence(panel)
    persist.to_csv(AGG_DIR / "top20_persistence.csv", index=False)
    print(f"  mean retention rate "
          f"{persist['retention_rate'].mean():.2%}, "
          f"min={persist['retention_rate'].min():.2%} "
          f"(year {int(persist.loc[persist['retention_rate'].idxmin(), 'year'])})")

    print("[annual_dynamics] risers / fallers")
    rf = compute_risers_fallers(panel)
    rf.to_csv(AGG_DIR / "risers_fallers.csv", index=False)

    print("[annual_dynamics] industry concentration")
    conc = compute_industry_concentration(panel)
    conc.to_csv(AGG_DIR / "industry_concentration_annual.csv", index=False)
    print(f"  HHI {conc['hhi'].iloc[0]:.3f} ({YEAR_MIN}) → "
          f"{conc['hhi'].iloc[-1]:.3f} ({YEAR_MAX})")

    print("[annual_dynamics] compustat share")
    shr = compute_compustat_share(panel)
    shr.to_csv(AGG_DIR / "compustat_share_annual.csv", index=False)
    print(f"  Compustat cost share {shr['compustat_share'].iloc[0]:.1%} → "
          f"{shr['compustat_share'].iloc[-1]:.1%}")

    print("[annual_dynamics] aggregate stress")
    agg = compute_aggregate_stress(panel)
    agg.to_csv(AGG_DIR / "aggregate_stress_annual.csv", index=False)

    print("[annual_dynamics] figures")
    plot_rank_heatmap(traj, AGG_DIR / "fig_d1_rank_heatmap.png")
    plot_concentration(conc, AGG_DIR / "fig_d2_concentration_over_time.png")
    plot_risers_fallers(rf, AGG_DIR / "fig_d3_risers_fallers.png")
    plot_compustat_share(shr, AGG_DIR / "fig_d4_compustat_share.png")
    plot_system_stress(agg, AGG_DIR / "fig_d5_system_stress.png")

    print("[annual_dynamics] done.")


if __name__ == "__main__":
    main()

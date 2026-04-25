"""Portfolio Stress Test — Strategic Question 3.

Three sub-reports per firm:
  (a) True-centrality diagnostic: full-graph rank vs Compustat-only rank
      for the firm's L1–L4 betweenness. Large gap → firm's public-peer
      view misrepresents its true position.
  (b) Per-partner vulnerability (the hero product):
      For each current partner, compute empirical TWFE and DMD-simulated
      MV loss if that partner were to suddenly exit.
  (c) Redundancy audit: for critical partners (top-5 by empirical loss),
      count focal's other partners in the same layer-SIC block.
      Low count → high lock-in urgency for M&A or redundancy building.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from strategic_pipeline.data_loader import (
    DataBundle, load_all,
    get_firm_partners, get_firm_edges,
)
from strategic_pipeline.firm_profile import build_profile, LAYERS, LAYER_NAMES
from strategic_pipeline.scoring_primitives import (
    partner_exit_impact, _layer_mix_in_dyad,
)
from strategic_pipeline.id_utils import normalize_cusip

ROLLING_WINDOW = 5
AGG_PANEL_PATH = (
    "/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance/"
    "outputs/strategic/aggregate/critical_edges_panel.parquet"
)


@dataclass
class StressReport:
    cusip: str
    name: str
    year: int
    # (a) centrality diagnostic: per layer, full-graph rank vs Compustat-only rank
    centrality_diagnostic: pd.DataFrame   # cols: layer, full_rank, comp_rank, gap
    # (b) per-partner vulnerability
    partner_vulnerability: pd.DataFrame   # cols: partner_cusip, name, sic2,
                                          # dominant_layer, empirical_loss,
                                          # dmd_loss, rank
    critical_partners: pd.DataFrame       # top-5 by empirical loss
    # (c) redundancy audit (per critical partner)
    redundancy_audit: pd.DataFrame


# ══════════════════════════════════════════════════════════════════
# (a) True-centrality diagnostic
# ══════════════════════════════════════════════════════════════════

def _compustat_centrality_rank(bundle: DataBundle, cusip: str,
                                 layer: str, year: int) -> tuple:
    """Return (full_rank, compustat_rank) for focal's layer-L betweenness.

    Full rank uses the layer_betweenness_panel (computed on the full
    graph). Compustat-only rank is approximated by filtering the
    betweenness panel to firms with a Compustat match, then re-ranking.
    """
    cusip = normalize_cusip(cusip)
    # Full-graph rank
    lb = bundle.layer_btw
    sub = lb[lb["year"] == year].copy()
    col = f"{layer}_btw"
    if col not in sub.columns:
        return (None, None)
    sub[col] = sub[col].fillna(0.0)
    sub = sub[sub[col] > 0].sort_values(col, ascending=False)
    full_ranks = sub["ult_parent_cusip"].tolist()
    full_rank = (full_ranks.index(cusip) + 1) if cusip in full_ranks else None

    # Compustat-only rank: restrict to firms in Compustat-matched set
    comp = bundle.compustat_firms
    sub_c = sub[sub["ult_parent_cusip"].isin(comp)]
    comp_ranks = sub_c["ult_parent_cusip"].tolist()
    comp_rank = (comp_ranks.index(cusip) + 1) if cusip in comp_ranks else None

    return (full_rank, comp_rank)


def build_centrality_diagnostic(bundle: DataBundle, cusip: str,
                                  year: int) -> pd.DataFrame:
    cusip = normalize_cusip(cusip)
    rows = []
    for L in LAYERS:
        full, comp = _compustat_centrality_rank(bundle, cusip, L, year)
        rows.append({
            "layer": L,
            "layer_name": LAYER_NAMES[L],
            "full_rank": full,
            "compustat_only_rank": comp,
            "gap": ((comp - full) if (full is not None and comp is not None)
                    else None),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# (b) Per-partner vulnerability
# ══════════════════════════════════════════════════════════════════

def build_partner_vulnerability(bundle: DataBundle, cusip: str,
                                  year: int) -> pd.DataFrame:
    cusip = normalize_cusip(cusip)
    if Path(AGG_PANEL_PATH).exists():
        cols = [
            "year", "focal_cusip", "partner_cusip", "partner_name",
            "partner_sic2", "dominant_layer", "predicted_log_mv_loss",
            "predicted_dollar_loss", "rank", "tie_count", "tie_strength",
            "tenure", "substitute_count", "layer_mix_L1", "layer_mix_L2",
            "layer_mix_L3", "layer_mix_L4", "delta_L1_btw", "delta_L2_btw",
            "delta_L3_btw", "delta_L4_btw",
        ]
        panel = pd.read_parquet(AGG_PANEL_PATH, columns=cols)
        sub = panel[
            (panel["year"] == year) & (panel["focal_cusip"] == cusip)
        ].copy()
        if len(sub):
            sub = sub.rename(columns={
                "partner_name": "name",
                "partner_sic2": "sic2",
                "predicted_log_mv_loss": "empirical_loss_log_mv",
            })
            sub["dmd_loss_log_mv"] = np.nan
            sub["dmd_available"] = False
            sub = sub.sort_values("rank").reset_index(drop=True)
            return sub

    start = year - ROLLING_WINDOW + 1
    rows = []
    for L in LAYERS:
        partners = get_firm_partners(bundle, cusip, start, year, layer=L)
        for p in partners:
            mix = _layer_mix_in_dyad(bundle, cusip, p, year)
            if not mix:
                continue
            dom = max(mix.items(), key=lambda kv: kv[1])[0]
            impact = partner_exit_impact(bundle, cusip, p, year)
            meta = bundle.firm_meta[bundle.firm_meta["cusip"] == p]
            name = meta.iloc[0]["name"] if len(meta) else f"<unknown:{p}>"
            sic2 = meta.iloc[0]["sic2"] if len(meta) else "??"
            rows.append({
                "partner_cusip": p,
                "name": name,
                "sic2": sic2,
                "dominant_layer": dom,
                "empirical_loss_log_mv": impact["empirical_twfe_pct"],
                "dmd_loss_log_mv": impact["dmd_simulated_pct"],
                "dmd_available": impact["dmd_available"],
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["partner_cusip"])
    df["abs_empirical"] = df["empirical_loss_log_mv"].abs()
    df = df.sort_values("abs_empirical", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df = df.drop(columns=["abs_empirical"])
    return df


# ══════════════════════════════════════════════════════════════════
# (c) Redundancy audit
# ══════════════════════════════════════════════════════════════════

def build_redundancy_audit(bundle: DataBundle, cusip: str, year: int,
                            critical_df: pd.DataFrame) -> pd.DataFrame:
    """For each critical partner, count focal's other partners in the
    same (layer, SIC2) bucket. Low count → high lock-in risk."""
    cusip = normalize_cusip(cusip)
    start = year - ROLLING_WINDOW + 1
    rows = []
    for _, row in critical_df.iterrows():
        p = row["partner_cusip"]
        L = row["dominant_layer"]
        sic2 = row["sic2"]
        # Focal's other partners in layer L
        other = get_firm_partners(bundle, cusip, start, year, layer=L)
        other.discard(p)
        # Filter to those with same SIC2
        sic_map = dict(zip(bundle.firm_meta["cusip"], bundle.firm_meta["sic2"]))
        same_sic = [q for q in other if sic_map.get(q) == sic2]
        rows.append({
            "partner_cusip": p,
            "name": row["name"],
            "dominant_layer": L,
            "sic2": sic2,
            "substitutes_in_layer_sic": len(same_sic),
            "lock_in_urgency": ("HIGH" if len(same_sic) == 0
                                else "MEDIUM" if len(same_sic) <= 2
                                else "LOW"),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# Top-level orchestrator
# ══════════════════════════════════════════════════════════════════

def build_stress_report(focal_cusip: str, year: int = 2017,
                         top_n_critical: int = 5,
                         bundle: Optional[DataBundle] = None) -> StressReport:
    focal_cusip = normalize_cusip(focal_cusip)
    if bundle is None:
        bundle = load_all()

    name = bundle.firm_name(focal_cusip)
    cent = build_centrality_diagnostic(bundle, focal_cusip, year)
    vuln = build_partner_vulnerability(bundle, focal_cusip, year)
    critical = (vuln.head(top_n_critical).copy()
                if len(vuln) > 0 else pd.DataFrame())
    redund = (build_redundancy_audit(bundle, focal_cusip, year, critical)
              if len(critical) > 0 else pd.DataFrame())

    return StressReport(
        cusip=focal_cusip, name=name, year=year,
        centrality_diagnostic=cent,
        partner_vulnerability=vuln,
        critical_partners=critical,
        redundancy_audit=redund,
    )


# ══════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════

def plot_centrality_diagnostic(report: StressReport, out_path: str):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    cd = report.centrality_diagnostic
    x = np.arange(len(cd))
    full = cd["full_rank"].fillna(0).astype(int).values
    comp = cd["compustat_only_rank"].fillna(0).astype(int).values

    w = 0.38
    ax.bar(x - w/2, full, w, label="Full-graph rank", color="#2563EB")
    ax.bar(x + w/2, comp, w, label="Compustat-only rank", color="#E57200")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['layer']}\n{row['layer_name']}"
                         for _, row in cd.iterrows()],
                        rotation=20, ha="right")
    ax.set_ylabel("Rank (1 = highest betweenness)")
    ax.set_title(f"Centrality rank — {report.name} ({report.year})\n"
                 "Large gap = firm's public-peer view misrepresents true position")
    ax.legend()
    ax.invert_yaxis()  # rank 1 at top
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_partner_vulnerability(report: StressReport, out_path: str,
                                 top_n: int = 10):
    sns.set_theme(style="whitegrid")
    df = report.partner_vulnerability.head(top_n).copy()
    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df) + 1)))
    y = np.arange(len(df))
    w = 0.38
    emp = df["empirical_loss_log_mv"].values
    dmd_raw = df["dmd_loss_log_mv"].values
    dmd = np.where(pd.isna(dmd_raw.astype(float)) if dmd_raw.dtype == object
                    else np.isnan(dmd_raw.astype(float)),
                    0.0, dmd_raw.astype(float))
    has_dmd = df["dmd_available"].values

    ax.barh(y + w/2, emp, w, label="Empirical TWFE (M2 event study)",
            color="#E57200")
    # only plot DMD bars where available
    for i, (v, avail) in enumerate(zip(dmd, has_dmd)):
        if avail:
            ax.barh(y[i] - w/2, v, w,
                    color="#2563EB", label="DMD simulated" if i == 0 else None)

    labels = [f"{row['name'][:32]} ({row['dominant_layer']})"
              for _, row in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # rank 1 at top
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(r"Predicted $\log$(MV) response at $t+1$ if partner exits")
    ax.set_title(f"Partner-exit vulnerability — {report.name} ({report.year})\n"
                 "Orange = empirical (M2 event study); blue = DMD operator simulation")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    r = build_stress_report("747525", year=2005)
    print(f"Stress report for {r.name} ({r.cusip}), year {r.year}")
    print()
    print("=== (a) True-centrality diagnostic ===")
    print(r.centrality_diagnostic.to_string(index=False))
    print()
    print("=== (b) Partner vulnerability (top 10) ===")
    if len(r.partner_vulnerability):
        disp = r.partner_vulnerability.head(10)[[
            "rank", "name", "sic2", "dominant_layer",
            "empirical_loss_log_mv", "dmd_loss_log_mv",
        ]]
        print(disp.to_string(index=False))
    print()
    print("=== (c) Redundancy audit (critical partners) ===")
    if len(r.redundancy_audit):
        print(r.redundancy_audit.to_string(index=False))

    plot_centrality_diagnostic(r, "/tmp/test_cent.png")
    plot_partner_vulnerability(r, "/tmp/test_vuln.png")
    print("\n  Saved /tmp/test_cent.png, /tmp/test_vuln.png")

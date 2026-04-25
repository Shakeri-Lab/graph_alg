"""Visualizations for the systemic-criticality meta-network.

Fig M1: top-30 systemic-critical firms, bar chart (total empirical cost).
Fig M2: in_degree vs own firm market value (small-broker identification).
Fig M3: SIC-2 heatmap (top-20 industries by aggregate in-degree).
Fig M4: meta-network force-directed layout (top-50 nodes).

Run after build_systemic_criticality.py has written the CSVs.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from strategic_pipeline.data_loader import load_all

AGG_DIR = Path(
    "/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance/"
    "outputs/strategic/aggregate"
)

sns.set_theme(style="whitegrid", font_scale=1.1)


def fig_m1_top_firms(agg: pd.DataFrame):
    df = agg.head(30).copy()
    df["label"] = (df["name"].str[:38] + "  (" + df["sic2"].astype(str) + ")")

    fig, ax = plt.subplots(figsize=(9, 10))
    y = np.arange(len(df))
    ax.barh(y, df["total_empirical_cost"].values, color="#2563EB", alpha=0.85,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Total empirical exit cost across focal firms\n"
                  "(sum of $|\\log$(MV) loss$|$)")
    ax.set_title("Top 30 systemic-critical firms\n"
                 "(each firm is named by ≥1 focal firm as a top-5 critical partner)",
                 fontsize=12)
    # Annotate in_degree
    for i, row in df.iterrows():
        ax.text(row["total_empirical_cost"] * 1.01, i,
                f"  k={int(row['in_degree'])}",
                va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "fig_m1_top_firms.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_m1_top_firms.png")


def fig_m2_indegree_vs_mv(agg: pd.DataFrame, bundle):
    """Scatter: in_degree vs own firm market value.
    Firms in the upper-left (high in_degree, low MV) are small structural
    glue — they connect many larger firms without being large themselves."""
    # Join with firm_year to get market value at a recent year
    fy = bundle.firm_year
    mv_row = (fy[fy["market_value"].notna()]
              .sort_values("year", ascending=False)
              .drop_duplicates("ult_parent_cusip")
              [["ult_parent_cusip", "market_value"]])
    mv_row = mv_row.rename(columns={"ult_parent_cusip": "partner_cusip",
                                       "market_value": "mv_latest"})
    df = agg.merge(mv_row, on="partner_cusip", how="left")
    df = df[df["mv_latest"].notna() & (df["mv_latest"] > 0)]
    if len(df) == 0:
        print("  skip fig_m2: no MV data for any critical partners")
        return

    df = df.head(200)  # cap for readability

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["mv_latest"], df["in_degree"], s=30, alpha=0.5,
               color="#2563EB", edgecolor="none")
    # Highlight top 10
    top10 = df.head(10)
    ax.scatter(top10["mv_latest"], top10["in_degree"], s=80,
               color="#E57200", edgecolor="black", linewidth=0.5, zorder=3)
    for _, r in top10.iterrows():
        ax.annotate(r["name"][:22], (r["mv_latest"], r["in_degree"]),
                     fontsize=8, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Own firm latest market value (log scale, \\$M)")
    ax.set_ylabel("In-degree (# firms naming as critical)")
    ax.set_title("Systemic criticality vs firm scale\n"
                 "(upper-left region = small structural-glue firms)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "fig_m2_indegree_vs_mv.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_m2_indegree_vs_mv.png")


def fig_m3_sic_heatmap(agg: pd.DataFrame):
    sic_roll = (agg.groupby("sic2")
                .agg(total_in_degree=("in_degree", "sum"),
                      total_cost=("total_empirical_cost", "sum"),
                      n_firms=("partner_cusip", "count"),
                      mean_cost=("mean_empirical_cost", "mean"))
                .sort_values("total_in_degree", ascending=False)
                .head(20))
    sic_roll = sic_roll.reset_index()
    fig, ax = plt.subplots(figsize=(9, 8))
    # Normalize columns for comparable shading
    heat = sic_roll[["total_in_degree", "total_cost", "n_firms",
                      "mean_cost"]].copy()
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-10)
    sns.heatmap(heat_norm.values, annot=heat.values,
                fmt=".1f" if heat.values.dtype == float else "g",
                cmap="YlOrRd", ax=ax,
                xticklabels=["Total in-deg", "Total cost",
                              "#firms", "Mean cost"],
                yticklabels=sic_roll["sic2"].tolist())
    ax.set_title("Top-20 SIC-2 industries by aggregate systemic criticality",
                 fontsize=12)
    ax.set_ylabel("2-digit SIC")
    fig.tight_layout()
    fig.savefig(AGG_DIR / "fig_m3_sic_heatmap.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_m3_sic_heatmap.png")


def fig_m4_meta_network(agg: pd.DataFrame, edges: pd.DataFrame, top_k: int = 50):
    top_partners = agg.head(top_k)["partner_cusip"].tolist()
    sub = edges[edges["partner_cusip"].isin(top_partners)].copy()
    # Only keep focals that also appear in top_partners (internal view)
    # Otherwise include focals as distinct nodes
    # Here: include all focals as secondary nodes
    G = nx.DiGraph()
    for _, r in sub.iterrows():
        G.add_edge(r["focal_cusip"], r["partner_cusip"],
                   layer=r["dominant_layer"])
    # Node sizes: partners scaled by in_degree, focals fixed small
    focals = set(sub["focal_cusip"])
    partners = set(sub["partner_cusip"])
    name_lookup = dict(zip(agg["partner_cusip"], agg["name"]))
    sic_lookup = dict(zip(agg["partner_cusip"], agg["sic2"]))
    sizes = [300 + 30 * int(agg.loc[agg["partner_cusip"] == n, "in_degree"].iloc[0])
              if n in partners else 30
              for n in G.nodes()]
    # Colors: partners by SIC; focals gray
    sic_to_color = {}
    palette = sns.color_palette("tab20", 20)
    unique_sics = list({sic_lookup.get(n, "??") for n in G.nodes()
                         if n in partners})[:20]
    for i, s in enumerate(unique_sics):
        sic_to_color[s] = palette[i]
    colors = ["#888888" if n not in partners
               else sic_to_color.get(sic_lookup.get(n, "??"), "#2563EB")
               for n in G.nodes()]

    pos = nx.spring_layout(G, k=0.9, iterations=80, seed=42)

    fig, ax = plt.subplots(figsize=(14, 12))
    nx.draw_networkx_edges(G, pos, alpha=0.1, arrows=False, ax=ax,
                            edge_color="#888888", width=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors,
                            alpha=0.85, linewidths=0.3, edgecolors="black",
                            ax=ax)
    # Label only top-20 partners by in-degree
    top_20 = agg.head(20)["partner_cusip"].tolist()
    labels = {}
    for n in top_20:
        if n in G.nodes():
            nm = name_lookup.get(n, n)
            if not isinstance(nm, str):
                nm = str(n)
            labels[n] = nm[:18]
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    ax.set_title(f"Systemic-Criticality Meta-Network (top-{top_k} partners)\n"
                 "Node size = in-degree; color = 2-digit SIC; gray nodes = focal firms",
                 fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(AGG_DIR / "fig_m4_meta_network.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_m4_meta_network.png")


def write_master_summary(agg: pd.DataFrame, edges: pd.DataFrame):
    n_firms = edges["focal_cusip"].nunique()
    n_partners = len(agg)
    n_edges = len(edges)
    top20 = agg.head(20)
    audit_path = AGG_DIR / "identity_coverage_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        report_dirs = int(audit.loc[audit["year"].eq("all_report_dirs"), "total_report_dirs"].iloc[0])
        nonempty = int(audit.loc[audit["year"].eq("all_report_dirs"), "nonempty_partner_csv"].iloc[0])
    else:
        report_dirs = n_firms
        nonempty = n_firms
    sic_roll = (agg.groupby("sic2")
                .agg(total_in_degree=("in_degree", "sum"),
                      total_cost=("total_empirical_cost", "sum"),
                      n_firms=("partner_cusip", "count"))
                .sort_values("total_in_degree", ascending=False)
                .head(10))

    md = []
    md.append("# Systemic-Criticality Meta-Network — Master Summary\n")
    md.append(f"- **Report directories generated**: {report_dirs:,d}")
    md.append(f"- **Non-empty legacy 2017 vulnerability CSVs**: {nonempty:,d}")
    md.append(f"- **Active 2017 focal firms contributing edges**: {n_firms:,d}")
    md.append(f"- **Distinct critical partners**: {n_partners:,d}")
    md.append(f"- **Total edges (top-5 per focal)**: {n_edges:,d}")
    md.append(f"- **Duplicate normalized partner rows**: "
               f"{n_partners - agg['partner_cusip'].nunique():,d}")
    md.append(f"- **Compustat-matched among top-20**: "
               f"{int(top20['is_compustat'].sum())}/20\n")

    md.append("## Top 20 systemic-critical firms\n")
    disp = top20[["rank", "partner_cusip", "name", "sic2", "is_compustat",
                   "in_degree", "total_empirical_cost",
                   "mean_empirical_cost"]].copy()
    disp = disp.rename(columns={
        "total_empirical_cost": "total_predicted_log_mv_cost",
        "mean_empirical_cost": "mean_predicted_log_mv_cost",
    })
    md.append(disp.to_markdown(index=False))
    md.append("")

    md.append("## Industry concentration (top 10 SIC-2 by in-degree)\n")
    md.append(sic_roll.to_markdown())
    md.append("")

    md.append("## Figures\n")
    for fname, desc in [
        ("fig_m1_top_firms.png", "Top 30 firms by total exit cost."),
        ("fig_m2_indegree_vs_mv.png",
         "In-degree vs own firm market value; upper-left = small structural glue."),
        ("fig_m3_sic_heatmap.png", "Top-20 SIC-2 industries by aggregate metrics."),
        ("fig_m4_meta_network.png",
         "Meta-network force-directed layout (top-50 partners)."),
    ]:
        md.append(f"- ![{desc}]({fname})")
    md.append("")

    md.append("## Interpretation & caveats\n")
    md.append("- Each active focal firm contributes up to 5 directed edges "
               "(its top partners by partner-specific predicted loss); firms "
               "with fewer active partners contribute fewer than 5.")
    md.append("- **Predicted loss** combines layer-specific exit weights, dyad "
               "tenure, tie strength, redundancy, partner centrality, and "
               "first-order counterfactual betweenness exposure.")
    md.append("- Stacked-cohort estimates are the conservative causal reference "
               "where available; legacy TWFE estimates are retained in "
               "`estimator_robustness.csv` because they are not identical.")
    md.append("- Top systemic-critical firms are defined by *aggregate* "
               "cost; firms with low in-degree but large per-dyad cost can "
               "still rank high (e.g., a single very large partnership).")
    md.append("")

    out = AGG_DIR / "systemic_criticality.md"
    out.write_text("\n".join(md))
    print(f"  wrote {out}")


def main():
    agg_path = AGG_DIR / "systemic_criticality.csv"
    edges_path = AGG_DIR / "critical_edges.csv"
    if not agg_path.exists() or not edges_path.exists():
        print("Missing aggregate CSVs — run "
              "strategic_pipeline.aggregate.build_systemic_criticality first.")
        return

    agg = pd.read_csv(agg_path)
    edges = pd.read_csv(edges_path)
    bundle = load_all()

    fig_m1_top_firms(agg)
    fig_m2_indegree_vs_mv(agg, bundle)
    fig_m3_sic_heatmap(agg)
    fig_m4_meta_network(agg, edges, top_k=50)
    write_master_summary(agg, edges)


if __name__ == "__main__":
    main()

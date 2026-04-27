"""Render per-firm decision reports as Markdown + figures.

Each decision module (alignment, timing, stress) has a `write_*` function
here that takes its output dataclass/DataFrame and writes:
  - one Markdown file
  - zero or more PNGs

Output directory: outputs/strategic/<cusip>/
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategic_pipeline.timing_dashboard import (
    TimingReport, plot_tenure_distribution, LAYER_NAMES, LAYERS,
)
from strategic_pipeline.portfolio_stress_test import (
    StressReport, plot_centrality_diagnostic, plot_partner_vulnerability,
)


OUTPUTS_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance/outputs/strategic")


def _firm_dir(cusip: str) -> Path:
    d = OUTPUTS_ROOT / cusip
    d.mkdir(parents=True, exist_ok=True)
    return d


def _header(name: str, cusip: str, year: int, question: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (f"# {question} report — {name} ({cusip})\n\n"
            f"- **Year**: {year}\n"
            f"- **Generated**: {ts}\n"
            f"- **Pipeline**: strategic_pipeline v1\n\n"
            "---\n\n")


# ══════════════════════════════════════════════════════════════════
# Alignment report
# ══════════════════════════════════════════════════════════════════

def _plot_durable_value_scatter(df: pd.DataFrame, name: str, year: int,
                                  out_path: Path) -> None:
    """2D scatter of (DepRisk, durable_value) with quadrant labels.

    Quadrants:
      Top-left  (low risk,  high value): Durable bridge — preferred
      Top-right (high risk, high value): Fragile chokepoint — needs safeguards
      Bot-left  (low risk,  low value):  Safe but irrelevant
      Bot-right (high risk, low value):  Bad dependency — avoid

    Candidates with `dep_risk_observed=False` are rendered with an open
    marker on the y-axis at x=0 to show they are not in the systemic
    cross-section (DepRisk is unknown, not zero).
    """
    if not {"dep_risk", "durable_value", "firm_name"}.issubset(df.columns):
        return
    if len(df) == 0:
        return
    if "dep_risk_observed" in df.columns:
        observed = df["dep_risk_observed"].fillna(False).astype(bool)
    else:
        observed = pd.Series([True] * len(df), index=df.index)
    df_obs = df[observed]
    df_unobs = df[~observed]

    x_obs = df_obs["dep_risk"].astype(float)
    y_obs = df_obs["durable_value"].astype(float)
    x_med = float(x_obs.median()) if len(x_obs) and (x_obs > 0).any() else 0.05
    y_all = df["durable_value"].astype(float).fillna(0.0)
    y_med = float(y_all.median())

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    if len(df_obs):
        ax.scatter(x_obs, y_obs, s=70, c="steelblue", edgecolor="black",
                    linewidth=0.4, alpha=0.85,
                    label="DepRisk observed (in systemic panel)")
    if len(df_unobs):
        ax.scatter([0.0] * len(df_unobs),
                    df_unobs["durable_value"].astype(float),
                    s=70, c="white", edgecolor="gray", linewidth=0.8,
                    alpha=0.85, marker="o",
                    label="DepRisk unobserved (not systemically ranked)")
    # Annotate each point with the firm's short name
    for _, row in df.iterrows():
        nm = str(row["firm_name"])
        x_pos = float(row["dep_risk"]) if pd.notna(row["dep_risk"]) else 0.0
        y_pos = float(row["durable_value"]) if pd.notna(row["durable_value"]) else 0.0
        ax.annotate(nm[:24], (x_pos, y_pos), fontsize=7, alpha=0.75,
                     xytext=(3, 3), textcoords="offset points")

    # Quadrant lines and labels (use medians of THIS firm's top-N as cuts)
    ax.axvline(x_med, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(y_med, color="gray", linestyle=":", linewidth=0.8)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmin + (x_med - xmin) * 0.5, ymax - (ymax - y_med) * 0.05,
            "Durable bridges\n(preferred)",
            ha="center", va="top", fontsize=9, color="darkgreen", weight="bold")
    ax.text(xmax - (xmax - x_med) * 0.5, ymax - (ymax - y_med) * 0.05,
            "Fragile chokepoints\n(safeguard required)",
            ha="center", va="top", fontsize=9, color="darkorange",
            weight="bold")
    ax.text(xmin + (x_med - xmin) * 0.5, ymin + (y_med - ymin) * 0.05,
            "Safe but irrelevant",
            ha="center", va="bottom", fontsize=9, color="dimgray")
    ax.text(xmax - (xmax - x_med) * 0.5, ymin + (y_med - ymin) * 0.05,
            "Bad dependency (avoid)",
            ha="center", va="bottom", fontsize=9, color="firebrick",
            weight="bold")

    ax.set_xlabel("DepRisk (normalized systemic in-degree)  →")
    ax.set_ylabel("Durable value = brokerage_L2 × w_tenure_smooth")
    ax.set_title(f"Durable-value × dependency-risk frontier — {name} ({year})\n"
                 f"Each point: a top candidate. Cuts: this firm's medians.")
    ax.grid(True, alpha=0.3)
    if len(df_unobs):
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_alignment_report(cusip: str, name: str, year: int,
                             goal: str, df: pd.DataFrame) -> tuple:
    d = _firm_dir(cusip)
    md_path = d / f"alignment_{goal}.md"
    csv_path = d / f"alignment_{goal}_top.csv"
    fig_path = d / f"fig_alignment_{goal}_frontier.png"

    with open(md_path, "w") as f:
        f.write(_header(name, cusip, year, f"Alignment ({goal})"))

        if goal == "innovation":
            f.write("## Goal: Innovation alliances (L1)\n\n"
                    "Paper finding (Section 4): brokerage in $L_1$ yields "
                    "no detectable premium; tacit knowledge transfer "
                    "benefits from **closure**. Candidates below are "
                    "ranked by triadic closure score: fraction of your "
                    "existing L1 partners that are also partners of the "
                    "candidate.\n\n")
        else:
            gate = df.attrs.get("rd_gate_passed", None)
            f.write(
                "## Goal: Commercialization alliances (L2)\n\n"
                "### The strategic question\n\n"
                "Which downstream partners should you form a "
                "commercialization alliance with so that the "
                "partnership generates sales response, not just "
                "structural reach on paper?\n\n"
                "### What the data say\n\n"
                "- **Brokerage in L₂ pays.** L₂ brokerage is "
                "positively associated with future sales at "
                "$t{+}2$ ($p=0.045$) and $t{+}4$ ($p=0.031$) "
                "(paper Section 4, Table 3; two-way clustered SE).\n"
                "- **The premium is gated by R&D capability.** The "
                "effect concentrates in top-quartile R&D firms in "
                "the focal's 2-digit SIC (paper Figure 3B / H2).\n"
                "- **The premium is gated by tie persistence.** "
                "Hankel-DMD spectral analysis (paper Section 5) "
                "isolates a four-year sales cascade and shows "
                "newly-acquired L₂ brokerage produces *no* sales "
                "response. Value accrues to **sustained** ties "
                "(≥4 yr), not to acquisition.\n\n"
                "### How candidates are ranked (durable-rent score)\n\n"
                "$\\text{score\\_durable\\_rent}(c) "
                "= \\underbrace{"
                "\\text{brokerage}_{L_2}(\\text{focal}, c) \\times "
                "w_{\\text{tenure}}(c)}_{"
                "\\text{durable value}} \\;\\times\\; "
                "\\underbrace{\\exp\\!\\bigl(-\\rho \\cdot "
                "\\text{DepRisk}(c)\\bigr)}_{"
                "w_{\\text{redundancy}}(c)}$\n\n"
                "with a per-focal absorptive-capacity multiplier "
                "$g(\\text{R\\&D}_f) = "
                "1 + \\alpha \\cdot \\mathbf{1}\\{f \\in \\text{top-quartile R\\&D}\\}$.\n\n"
                "Component definitions:\n\n"
                "- **brokerage_L2** ∈ [0, 1] — Burt-style structural "
                "opportunity: fraction of candidate's L2 neighborhood "
                "*not* shared with the focal.\n"
                "- **w_tenure_smooth** ∈ (0, 1) — Dyer-Singh-style "
                "relational capability: $\\sigma(z)$ where $z$ is the "
                "candidate's $\\log(1 + \\text{median tenure})$ "
                "z-scored against the SIC×L2 cohort baseline. "
                "Candidates above the cohort median score $> 0.5$; "
                "below score $< 0.5$. Industry-normalized so a short "
                "spell in a fast-cycling sector does not look like "
                "churn.\n"
                "- **DepRisk** ∈ [0, 1] — candidate's normalized "
                "in-degree in the corrected systemic-criticality "
                "cross-section. High = many other firms already list "
                "this candidate as a top-5 critical partner.\n"
                "- **w_redundancy** = $\\exp(-1.5 \\cdot "
                "\\text{DepRisk})$ — penalty for adding a hub partner "
                "(creates portfolio fragility, paper §6 systemic "
                "report).\n"
                "- **durable_value** = brokerage_L2 × w_tenure_smooth "
                "— y-axis of the frontier scatter below.\n"
                "- **score_durable_rent** = durable_value × "
                "w_redundancy — the column the recommendation ranks on.\n\n"
            )
            f.write(f"**R&D gate** (top-quartile R\\&D intensity "
                    f"within your SIC): "
                    f"{'PASSED' if gate else 'NOT PASSED'}\n\n")
            f.write(f"> {df.attrs.get('rd_gate_message', '')}\n\n")
            if df.attrs.get("reranker_message"):
                f.write(f"> {df.attrs['reranker_message']}\n\n")

        f.write("## Top candidate partners\n\n")
        if len(df) == 0:
            f.write("_No qualifying candidates found._\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")

        if goal == "commercialization" and len(df):
            try:
                _plot_durable_value_scatter(df, name, year, fig_path)
                f.write(f"\n## Durable-value × dependency-risk frontier\n\n"
                        f"![Frontier]({fig_path.name})\n\n"
                        "**Quadrant reading** (cuts at this firm's medians):\n\n"
                        "- **Top-left — Durable bridges** (low risk, high "
                        "durable value). The preferred partner type: "
                        "structurally non-redundant, demonstrably persistent, "
                        "not yet a systemic hub.\n"
                        "- **Top-right — Fragile chokepoints** (high risk, "
                        "high durable value). Valuable but consider redundancy "
                        "safeguards (multi-source contracting, equity stake, "
                        "or M&A) before depending on a partner that many "
                        "other firms already depend on.\n"
                        "- **Bottom-left — Safe but irrelevant**. No "
                        "commercialization upside, no exposure created.\n"
                        "- **Bottom-right — Bad dependency**. Avoid unless "
                        "there is a separate strategic reason; the sales "
                        "premium is small and the systemic exposure is "
                        "large.\n\n")
            except Exception as exc:
                f.write(f"\n_(frontier scatter could not be rendered: "
                        f"{exc})_\n\n")

        f.write("\n---\n\n")
        if goal == "commercialization":
            f.write(
                "### Why the durable-rent score is more "
                "accurate than raw brokerage\n\n"
                "The previous version of this recommender ranked "
                "candidates on raw L₂ brokerage alone.  Two "
                "empirical and theoretical considerations make that "
                "ranking systematically biased:\n\n"
                "1. **Brokerage saturates for sparse focal "
                "portfolios.**  When the focal firm has few L₂ "
                "ties (true for most firms), almost every candidate "
                "achieves the maximum brokerage score of 1.0 "
                "(no overlap with the focal's L₂ neighborhood). "
                "The original ranking then surfaced whichever "
                "candidates the dataframe sort happened to put "
                "first — typically single-tie newcomers with no "
                "verifiable track record.\n"
                "2. **Single-tie newcomers are the wrong type.**  "
                "A candidate with one brand-new tie sits in the "
                "*acquisition* regime that the Hankel-DMD analysis "
                "shows produces no sales response.  A candidate "
                "with five sustained ties sits in the *persistence* "
                "regime where the L₂ premium realizes.  Ranking by "
                "raw brokerage alone systematically routes you "
                "toward partners least likely to generate value.\n\n"
                "The persistence re-ranker corrects both biases by "
                "down-weighting candidates whose own portfolios "
                "show high churn and by elevating candidates whose "
                "demonstrated tie maintenance signals the "
                "organizational capability to invest in joint "
                "value creation.\n\n"
                "### Management-science framing\n\n"
                "- **Brokerage vs. relational view, reconciled.**  "
                "Burt's structural-holes view says value comes "
                "from spanning disconnected clusters.  Dyer & "
                "Singh's relational view says value comes from "
                "partner-specific investments, governance, and "
                "knowledge-sharing routines built up over time.  "
                "The L₂ brokerage premium *only* materializes "
                "when both conditions hold: structural opportunity "
                "(brokerage) **and** relational capability "
                "(sustained ties).  The recommender now operationalizes "
                "both, in that order.\n"
                "- **Alliance capability as a dynamic capability.**  "
                "Firms that maintain alliances accumulate "
                "alliance-management routines, dedicated "
                "alliance functions, and partner-specific "
                "absorptive capacity (Anand & Khanna 2000; "
                "Kale, Dyer & Singh 2002).  A candidate's "
                "sustained-share is a behavioral signal of this "
                "capability — a Spence-style screening device "
                "that is hard to fake.\n"
                "- **Avoid the novelty trap.**  The temptation in "
                "alliance scouting is to chase fresh, unencumbered "
                "candidates who are 'available'.  The data say the "
                "L₂ commercialization premium goes to the "
                "*opposite* type: candidates whose calendars are "
                "already full of multi-year alliances are the ones "
                "who will make multi-year commitments to you.\n"
                "- **Redundancy as a feature, not a bug.**  In "
                "the structural-holes literature, partner-of-partners "
                "redundancy is treated as wasted bandwidth.  Under "
                "the corrected scorer it is partly *evidence of "
                "type*: a candidate whose neighborhood is densely "
                "co-active is more likely to have built the "
                "ecosystem-level routines that downstream "
                "commercialization requires.\n\n"
                "### How to read the table\n\n"
                "- `brokerage_L2` ∈ [0, 1]: classic Burt brokerage "
                "(higher = more non-redundant market access via "
                "this candidate).\n"
                "- `persistence_factor` ∈ [0.5, 1.0]: 1.0 = all "
                "current ties are sustained; 0.5 = none are.  "
                "Defaults to 1.0 for candidates with <2 current "
                "ties.\n"
                "- `sustained_share`: fraction of the candidate's "
                "current ties whose first year is ≥ 4 years before "
                "this report's year.\n"
                "- `n_current_ties`: candidate's tie count in the "
                "5-year window across all layers.\n"
                "- `adjusted_brokerage_L2`: the column the "
                "recommendation ranks on.\n\n"
                "### Limitations\n\n"
                "- Recommendations are **associational**.  Brokerage "
                "and persistence are structural and behavioral "
                "features, not causal forecasts of joint future "
                "value for a specific dyad.\n"
                "- The persistence factor uses ties across all four "
                "layers as a general 'maintain vs churn' proxy, "
                "not L₂-only — L₂ portfolios are too sparse for a "
                "layer-specific signal.\n"
                "- The brokerage premium is empirically conditioned "
                "on top-quartile R&D status (paper H2).  Firms "
                "outside that quartile should treat the ranking as "
                "structural fit only.\n"
            )
        else:
            f.write("### Interpretation\n\n")
            f.write(
                "- Scores in [0, 1]; higher = stronger structural "
                "fit for the goal.\n"
                "- Recommendations are **associational**. Closure "
                "scores measure structural match, not causal "
                "forecasts of joint future value.\n"
            )

    df.to_csv(csv_path, index=False)
    return (md_path, csv_path)


# ══════════════════════════════════════════════════════════════════
# Timing report
# ══════════════════════════════════════════════════════════════════

def write_timing_report(report: TimingReport) -> tuple:
    d = _firm_dir(report.cusip)
    md_path = d / "timing.md"
    fig_path = d / "fig_tenure_distribution.png"

    plot_tenure_distribution(report, str(fig_path))

    with open(md_path, "w") as f:
        f.write(_header(report.name, report.cusip, report.year, "Timing"))

        f.write("## Tenure distribution (current partners in 5-yr window)\n\n")
        f.write(f"![Tenure distribution]({fig_path.name})\n\n")
        f.write("| Layer | New (<2 yr) | Mid (2–4 yr) | Sustained (≥4 yr) | New/Sust ratio |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for L in LAYERS:
            t = report.tenure_by_layer[L]
            total = sum(t.values())
            if total == 0:
                continue
            ratio = report.new_sustained_ratio[L]
            ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
            f.write(f"| {L} ({LAYER_NAMES[L]}) | {t['new']} | "
                    f"{t['mid']} | {t['sustained']} | {ratio_str} |\n")
        f.write("\n")

        flag_color = {"STOP": "🛑", "CAUTION": "⚠️", "GO": "✅"}
        f.write(f"## Stop/Go signal: **{report.stop_go_flag}** "
                f"{flag_color.get(report.stop_go_flag, '')}\n\n")
        f.write(f"> {report.stop_go_reason}\n\n")

        f.write("## Predicted sales trajectory (empirical L₂ cascade)\n\n")
        f.write("From paper Table 3 (two-way clustered SE, firm + year FE):\n\n")
        f.write("| Horizon | Coefficient | $p$-value | Significant |\n")
        f.write("|---|---|---|---|\n")
        for row in report.predicted_sales_trajectory:
            mark = "**✓**" if row["significant"] else ""
            f.write(f"| $t{{+}}{row['horizon']}$ | {row['coef']} | "
                    f"{row['p_value']:.3f} | {mark} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("### Interpretation\n\n")
        f.write("- The sales response to $L_2$ brokerage is delayed: "
                "significant at $t{+}2$ and $t{+}4$, front-loaded on "
                "market value (not shown; see paper Table 3).\n")
        f.write("- **Persistence pays, acquisition does not** "
                "(paper Section 5). Newly-acquired $L_2$ brokerage "
                "does not produce a sales response; sustained brokerage "
                "(≥4 years) does.\n")
        f.write("- STOP/CAUTION flags are heuristic, not causal. They "
                "flag excessive churn in the L2 portfolio relative to "
                "sustained ties.\n")

    return (md_path, fig_path)


# ══════════════════════════════════════════════════════════════════
# Stress report
# ══════════════════════════════════════════════════════════════════

def write_stress_report(report: StressReport) -> tuple:
    d = _firm_dir(report.cusip)
    md_path = d / "stress.md"
    fig_cent = d / "fig_centrality_true_vs_compustat.png"
    fig_vuln = d / "fig_partner_vulnerability.png"
    pv_csv_path = d / "partner_vulnerability.csv"

    # Persist the full vulnerability table for downstream aggregation
    # (systemic-criticality meta-network); saved even for empty reports
    # so backfill logic can detect completion.
    if len(report.partner_vulnerability):
        report.partner_vulnerability.to_csv(pv_csv_path, index=False)
    else:
        # Empty-but-present sentinel so aggregation knows we ran
        pd.DataFrame(columns=[
            "partner_cusip", "name", "sic2", "dominant_layer",
            "empirical_loss_log_mv", "dmd_loss_log_mv",
            "dmd_available", "rank",
        ]).to_csv(pv_csv_path, index=False)

    plot_centrality_diagnostic(report, str(fig_cent))
    plot_partner_vulnerability(report, str(fig_vuln))

    with open(md_path, "w") as f:
        f.write(_header(report.name, report.cusip, report.year, "Stress test"))

        f.write("## (a) True-centrality diagnostic\n\n")
        f.write(f"![Centrality]({fig_cent.name})\n\n")
        f.write("Comparing full-graph rank to Compustat-only rank reveals "
                "how much the firm's position is misrepresented by "
                "subgraph extraction.  Negative `gap` = firm ranks HIGHER "
                "on Compustat-only than on the true full graph (implies "
                "overstated centrality in public-peer analysis).\n\n")
        f.write(report.centrality_diagnostic.to_markdown(index=False))
        f.write("\n\n")

        f.write("## (b) Partner-exit vulnerability\n\n")
        f.write(f"![Vulnerability]({fig_vuln.name})\n\n")
        f.write("Predicted $\\log$(Market Value) response at $t{+}1$ if "
                "each partner were to suddenly exit the network.  When the "
                "aggregate systemic panel is available, the score is "
                "partner-specific: layer exit weights are combined with dyad "
                "tenure, tie strength, redundancy, partner centrality, and "
                "first-order counterfactual betweenness exposure.  Legacy "
                "DMD estimates are shown only when the older per-firm scorer "
                "is used.\n\n")

        disp_cols = ["rank", "name", "sic2", "dominant_layer",
                     "empirical_loss_log_mv", "dmd_loss_log_mv",
                     "dmd_available"]
        if len(report.partner_vulnerability):
            f.write(report.partner_vulnerability[disp_cols].head(15)
                    .to_markdown(index=False))
            f.write("\n\n")

        f.write("## (c) Redundancy audit — critical partners\n\n")
        f.write("For each top-5 critical partner, `substitutes_in_layer_sic` "
                "counts your other partners in the same 2-digit SIC in the "
                "same layer.  Low count → difficult to replace → high "
                "lock-in urgency (candidate for M&A, equity stake, or "
                "contractual backup).\n\n")
        if len(report.redundancy_audit):
            f.write(report.redundancy_audit.to_markdown(index=False))
            f.write("\n")
        else:
            f.write("_No critical partners identified._\n")

        f.write("\n---\n\n")
        f.write("### Interpretation\n\n")
        f.write("- **Empirical estimates** inherit the M2 event-study "
                "identifying assumptions when the legacy scorer is used. "
                "The corrected systemic scorer uses stacked-cohort layer "
                "weights as the conservative reference and stores TWFE "
                "variants separately in the aggregate robustness files.\n")
        f.write("- **DMD estimates** are retained as a diagnostic for older "
                "reports; the corrected aggregate panel is the source of "
                "truth for systemic-criticality ranking.\n")
        f.write("- Redundancy audit is a first-order heuristic; it does "
                "not account for the quality of substitute partners.\n")

    return (md_path, fig_cent, fig_vuln)

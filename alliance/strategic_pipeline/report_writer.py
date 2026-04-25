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

def write_alignment_report(cusip: str, name: str, year: int,
                             goal: str, df: pd.DataFrame) -> tuple:
    d = _firm_dir(cusip)
    md_path = d / f"alignment_{goal}.md"
    csv_path = d / f"alignment_{goal}_top.csv"

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
            f.write("## Goal: Commercialization alliances (L2)\n\n"
                    "Paper finding (Section 4, Section 5): $L_2$ "
                    "brokerage is positively associated with future sales "
                    "at $t{+}2$ (p=0.045) and $t{+}4$ (p=0.031). The "
                    "effect is concentrated in top-quartile R\\&D firms. "
                    "Candidates are ranked by brokerage score: fraction "
                    "of the candidate's L2 neighborhood that is NOT "
                    "shared with you.\n\n")
            f.write(f"**R&D gate** (top-quartile R\\&D intensity within "
                    f"your SIC): "
                    f"{'PASSED' if gate else 'NOT PASSED'}\n\n")
            f.write(f"> {df.attrs.get('rd_gate_message', '')}\n\n")

        f.write("## Top candidate partners\n\n")
        if len(df) == 0:
            f.write("_No qualifying candidates found._\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")

        f.write("\n---\n\n")
        f.write("### Interpretation\n\n")
        f.write("- Scores in [0, 1]; higher = stronger structural fit "
                "for the goal.\n")
        f.write("- Recommendations are **associational**. Closure and "
                "brokerage scores measure structural match, not causal "
                "forecasts of joint future value.\n")
        f.write("- Paper Section 4: the $L_2$ brokerage premium applies "
                "under the persistence-not-acquisition mechanism — value "
                "realizes over 2–4 years of sustained partnership, not "
                "immediately.\n")

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

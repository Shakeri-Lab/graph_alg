"""Timing Dashboard — Strategic Question 2.

Reports:
  1. Tenure distribution per layer (new <2yr, mid 2-4yr, sustained ≥4yr).
  2. Predicted sales-response horizon from current L2 exposure, using
     the empirical cascade coefficients (Table 3 of paper).
  3. STOP/GO signal based on new/sustained ratio (bandwidth heuristic).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from strategic_pipeline.data_loader import DataBundle, load_all
from strategic_pipeline.firm_profile import (
    FirmProfile, build_profile, LAYERS, LAYER_NAMES,
)

# Empirical cascade coefficients from paper Table 3 (L2 → log_sales, 2-way
# clustered SE). Stars reflect p<0.05 significance.
SALES_CASCADE = {
    1: ("+4.79", 0.141, False),
    2: ("+4.84", 0.045, True),
    3: ("+4.35", 0.054, False),
    4: ("+4.57", 0.031, True),
}

# Bandwidth heuristic thresholds on new/sustained ratio
STOP_RATIO = 2.0    # above this: churning faster than relational capital matures
CAUTION_RATIO = 1.2
GO_RATIO = 1.0      # below this: stable, can absorb new ties


@dataclass
class TimingReport:
    cusip: str
    name: str
    year: int
    tenure_by_layer: dict        # {L: {"new", "mid", "sustained"}}
    new_sustained_ratio: dict    # {L: float (None if sustained=0)}
    stop_go_flag: str            # "STOP", "CAUTION", "GO"
    stop_go_reason: str
    predicted_sales_trajectory: list  # [(horizon, coef_str, p, significant)]


def _compute_new_sustained_ratio(tenure: dict) -> Optional[float]:
    new = tenure.get("new", 0)
    sustained = tenure.get("sustained", 0)
    if sustained == 0:
        return None  # firm has no sustained ties at all
    return new / sustained


def _stop_go(ratio_L2: Optional[float]) -> tuple:
    """Decide the stop/go flag based on L2 new/sustained ratio.
    L2 is the layer where the sales cascade lives per paper findings."""
    if ratio_L2 is None:
        return ("CAUTION",
                "Zero sustained L2 ties — portfolio is entirely nascent; "
                "no 4-year-old relational capital has formed. Hold on new "
                "L2 deal-making until existing ties mature past year 4, "
                "when the empirical cascade begins to produce sales.")
    if ratio_L2 >= STOP_RATIO:
        return ("STOP",
                f"L2 new/sustained ratio = {ratio_L2:.2f} ≥ {STOP_RATIO:.1f}. "
                "Firm is churning L2 ties faster than relational capital "
                "can mature to the 4-year horizon where the empirical "
                "sales premium activates. Focus on deepening existing "
                "2–3-year-old ties rather than forming new ones.")
    if ratio_L2 >= CAUTION_RATIO:
        return ("CAUTION",
                f"L2 new/sustained ratio = {ratio_L2:.2f} ∈ "
                f"[{CAUTION_RATIO:.1f}, {STOP_RATIO:.1f}). Portfolio is "
                "tilted toward nascent ties; monitor whether mid-tenure "
                "(2–4yr) ties are surviving into the sustained (≥4yr) "
                "bucket. If drop-off is high, consider a STOP.")
    return ("GO",
            f"L2 new/sustained ratio = {ratio_L2:.2f} < {GO_RATIO:.1f}. "
            "Portfolio has solid sustained-tie backbone; firm has "
            "bandwidth to absorb new L2 commercialization bridges.")


def build_timing_report(focal_cusip: str, year: int = 2017,
                         bundle: Optional[DataBundle] = None) -> TimingReport:
    if bundle is None:
        bundle = load_all()
    prof = build_profile(focal_cusip, year=year, bundle=bundle)

    ratios = {}
    for L in LAYERS:
        ratios[L] = _compute_new_sustained_ratio(prof.tenure_by_layer[L])

    flag, reason = _stop_go(ratios.get("L2"))

    trajectory = []
    for h in sorted(SALES_CASCADE):
        coef_str, p, sig = SALES_CASCADE[h]
        trajectory.append({
            "horizon": h,
            "coef": coef_str,
            "p_value": p,
            "significant": sig,
        })

    return TimingReport(
        cusip=focal_cusip,
        name=prof.name,
        year=year,
        tenure_by_layer=prof.tenure_by_layer,
        new_sustained_ratio=ratios,
        stop_go_flag=flag,
        stop_go_reason=reason,
        predicted_sales_trajectory=trajectory,
    )


def plot_tenure_distribution(report: TimingReport, out_path: str):
    """Stacked bar of tenure composition per layer."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    layers = LAYERS
    new_vals = [report.tenure_by_layer[L].get("new", 0) for L in layers]
    mid_vals = [report.tenure_by_layer[L].get("mid", 0) for L in layers]
    sust_vals = [report.tenure_by_layer[L].get("sustained", 0) for L in layers]

    x = np.arange(len(layers))
    ax.bar(x, new_vals, label="New (<2 yr)", color="#E57200")
    ax.bar(x, mid_vals, bottom=new_vals,
           label="Mid (2–4 yr)", color="#CCCCCC")
    ax.bar(x, sust_vals,
           bottom=[n + m for n, m in zip(new_vals, mid_vals)],
           label="Sustained (≥4 yr)", color="#2563EB")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}\n{LAYER_NAMES[L]}" for L in layers],
                        rotation=20, ha="right")
    ax.set_ylabel("Number of current partners")
    ax.set_title(f"Partner tenure distribution — {report.name} ({report.year})\n"
                 f"Sales cascade activates at ≥4 yr (blue = value-producing)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    r = build_timing_report("747525", year=2005)
    print(f"Timing report for {r.name} ({r.cusip}), year {r.year}")
    print()
    print("Tenure distribution (current partners in 5-yr window):")
    for L in LAYERS:
        t = r.tenure_by_layer[L]
        total = sum(t.values())
        if total == 0:
            continue
        ratio = r.new_sustained_ratio[L]
        ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a (sustained=0)"
        print(f"  {L} ({LAYER_NAMES[L]}): new={t['new']:2d}  mid={t['mid']:2d}  "
              f"sustained={t['sustained']:2d}  |  new/sustained = {ratio_str}")
    print()
    print(f"Stop/Go flag: {r.stop_go_flag}")
    print(f"  {r.stop_go_reason}")
    print()
    print("Predicted sales trajectory (empirical cascade, paper Table 3):")
    print(f"  {'Horizon':>8s}  {'Coef':>8s}  {'p':>7s}  Significant")
    for row in r.predicted_sales_trajectory:
        mark = "✓" if row['significant'] else " "
        print(f"  t+{row['horizon']:>5d}  {row['coef']:>8s}  "
              f"{row['p_value']:>7.3f}  {mark}")

    plot_tenure_distribution(r, "/tmp/test_timing.png")
    print("\n  Saved /tmp/test_timing.png")

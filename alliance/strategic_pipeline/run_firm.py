"""CLI entry point for the strategic decision pipeline.

Usage:
  python -m strategic_pipeline.run_firm --cusip 747525 --question alignment --goal commercialization
  python -m strategic_pipeline.run_firm --cusip 747525 --question timing
  python -m strategic_pipeline.run_firm --cusip 747525 --question stress
  python -m strategic_pipeline.run_firm --cusip 747525 --question all

  # Slurm array-index mode (reads row `SLURM_ARRAY_TASK_ID` from the CUSIP list):
  python -m strategic_pipeline.run_firm --array-index $SLURM_ARRAY_TASK_ID --question all
"""

from __future__ import annotations
import argparse
import os
import sys
import traceback
from pathlib import Path
import pandas as pd

from strategic_pipeline.data_loader import load_all
from strategic_pipeline.alignment_recommender import recommend_partners
from strategic_pipeline.timing_dashboard import build_timing_report
from strategic_pipeline.portfolio_stress_test import build_stress_report
from strategic_pipeline.report_writer import (
    write_alignment_report, write_timing_report, write_stress_report,
    OUTPUTS_ROOT,
)


CUSIP_LIST_PATH = Path(
    "/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance/intermediate/"
    "compustat_firm_list.csv"
)


def ensure_cusip_list(bundle) -> Path:
    """Build the CUSIP list for Slurm array jobs if it doesn't exist."""
    if CUSIP_LIST_PATH.exists():
        return CUSIP_LIST_PATH
    CUSIP_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    cusips = sorted(bundle.compustat_firms)
    pd.DataFrame({"ult_parent_cusip": cusips}).to_csv(
        CUSIP_LIST_PATH, index=False)
    print(f"  [init] wrote {CUSIP_LIST_PATH} with {len(cusips):,d} CUSIPs")
    return CUSIP_LIST_PATH


def _resolve_cusip(args, bundle) -> str:
    """Resolve --cusip or --array-index into a CUSIP string."""
    if args.cusip is not None:
        return args.cusip
    if args.array_index is None:
        raise SystemExit("error: must supply either --cusip or --array-index")
    path = ensure_cusip_list(bundle)
    df = pd.read_csv(path)
    idx = int(args.array_index) - 1  # Slurm array is 1-indexed
    if idx < 0 or idx >= len(df):
        raise SystemExit(f"array-index {args.array_index} out of bounds "
                          f"(list has {len(df)} rows)")
    return str(df.iloc[idx]["ult_parent_cusip"])


def run_alignment(cusip: str, goal: str, year: int, bundle):
    df = recommend_partners(cusip, goal, top_n=20, year=year, bundle=bundle)
    md, csv = write_alignment_report(cusip, bundle.firm_name(cusip),
                                       year, goal, df)
    print(f"  alignment ({goal}) → {md}")
    return md


def run_timing(cusip: str, year: int, bundle):
    rpt = build_timing_report(cusip, year=year, bundle=bundle)
    md, fig = write_timing_report(rpt)
    print(f"  timing → {md}")
    return md


def run_stress(cusip: str, year: int, bundle):
    rpt = build_stress_report(cusip, year=year, bundle=bundle)
    paths = write_stress_report(rpt)
    print(f"  stress → {paths[0]}")
    return paths[0]


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--cusip", type=str, default=None,
                     help="Ultimate-parent CUSIP (6 char)")
    ap.add_argument("--array-index", type=int, default=None,
                     help="Slurm array index (1-based); looks up CUSIP from "
                          "compustat_firm_list.csv")
    ap.add_argument("--question", required=True,
                     choices=["alignment", "timing", "stress", "all"])
    ap.add_argument("--goal",
                     choices=["innovation", "commercialization", "both"],
                     default="both",
                     help="For --question=alignment: which goal")
    ap.add_argument("--year", type=int, default=2017)
    args = ap.parse_args()

    # Default to the latest year with adequate coverage
    try:
        bundle = load_all()
    except Exception as e:
        print(f"FATAL: cannot load data bundle: {e}")
        raise

    cusip = _resolve_cusip(args, bundle)
    name = bundle.firm_name(cusip)
    print(f"\n==> Strategic pipeline for {name} ({cusip}), year {args.year}")

    q = args.question
    if q in ("alignment", "all"):
        goals = ["innovation", "commercialization"] if args.goal == "both" else [args.goal]
        for g in goals:
            try:
                run_alignment(cusip, g, args.year, bundle)
            except Exception as e:
                print(f"  [error] alignment/{g}: {e}")
                traceback.print_exc()

    if q in ("timing", "all"):
        try:
            run_timing(cusip, args.year, bundle)
        except Exception as e:
            print(f"  [error] timing: {e}")
            traceback.print_exc()

    if q in ("stress", "all"):
        try:
            run_stress(cusip, args.year, bundle)
        except Exception as e:
            print(f"  [error] stress: {e}")
            traceback.print_exc()

    print(f"\nOutputs → {OUTPUTS_ROOT}/{cusip}/")


if __name__ == "__main__":
    main()

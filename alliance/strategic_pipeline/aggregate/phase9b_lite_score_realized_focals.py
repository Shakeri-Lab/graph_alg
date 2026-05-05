"""Phase 9B-lite — manifest builder + per-task scorer for the Slurm array.

Two modes:

  build-manifest:
    Identify every (focal_cusip, score_year) pair that needs a per-focal
    scoring parquet for Phase 9B-lite (in-pool realized dyads).
    Pre-compute any missing candidate_features_<year>.parquet files
    (this is fast, single-pass, no Slurm needed).
    Write the manifest of pairs that DO NOT yet have a per-focal
    parquet on disk to:
      intermediate/phase9b_lite_focals_to_score.csv
        (columns: array_index, focal_cusip, score_year)

  score-task:
    Read the manifest, look up the row for SLURM_ARRAY_TASK_ID, and
    score that single (focal, score_year) pair via the existing
    write_focal_parquet helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from strategic_pipeline.data_loader import load_all
from strategic_pipeline.aggregate.week2_personalization import (
    OUTPUT_DIR, precompute_candidate_features, write_focal_parquet,
    personal_dir_for_year,
)


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
AGG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "aggregate"
PANEL = AGG_DIR / "phase9b_lite_realized_panel.parquet"
MANIFEST = PROJECT_ROOT / "intermediate" / "phase9b_lite_focals_to_score.csv"


def build_manifest() -> Path:
    print(f"[p9b_score] loading bundle + realized panel")
    bundle = load_all()
    panel = pd.read_parquet(PANEL)

    in_pool = panel[panel["coverage_class"] == "candidate_in_pool"].copy()
    in_pool["score_year"] = in_pool["year"] - 1
    pairs = (in_pool[["focal_cusip", "score_year"]]
             .drop_duplicates()
             .sort_values(["score_year", "focal_cusip"])
             .reset_index(drop=True))
    print(f"  {len(pairs)} unique (focal, score_year) pairs in scope")

    # Pre-compute any missing candidate-features parquets (cheap).
    score_years = sorted(pairs["score_year"].unique())
    for y in score_years:
        feats_path = AGG_DIR / f"candidate_features_{y}.parquet"
        if not feats_path.exists():
            print(f"[p9b_score] precomputing candidate_features_{y}")
            precompute_candidate_features(year=int(y), bundle=bundle)

    # Filter to pairs whose per-focal parquet does NOT yet exist
    todo = []
    for _, r in pairs.iterrows():
        focal = str(r["focal_cusip"])
        y = int(r["score_year"])
        out = personal_dir_for_year(y) / f"{focal}.parquet"
        if not out.exists():
            todo.append({"focal_cusip": focal, "score_year": y})
    todo_df = pd.DataFrame(todo)
    todo_df.insert(0, "array_index", range(1, len(todo_df) + 1))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    todo_df.to_csv(MANIFEST, index=False)
    print(f"[p9b_score] wrote {MANIFEST}: {len(todo_df)} pairs to score")
    print(f"  (skipped {len(pairs) - len(todo_df)} already-scored pairs)")
    return MANIFEST


def score_task(array_index: int) -> Path:
    if not MANIFEST.exists():
        raise SystemExit(
            f"[p9b_score] manifest missing at {MANIFEST}; "
            "run `build-manifest` first")
    manifest = pd.read_csv(MANIFEST,
                              dtype={"focal_cusip": str, "score_year": int})
    row = manifest[manifest["array_index"] == array_index]
    if not len(row):
        raise SystemExit(
            f"[p9b_score] array_index {array_index} not in manifest")
    focal = str(row.iloc[0]["focal_cusip"])
    y = int(row.iloc[0]["score_year"])

    feats_path = AGG_DIR / f"candidate_features_{y}.parquet"
    feats = pd.read_parquet(feats_path)
    bundle = load_all()
    out = write_focal_parquet(focal, y, feats, bundle)
    print(f"[p9b_score] task {array_index}: focal={focal} year={y} → {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-manifest")
    p2 = sub.add_parser("score-task")
    p2.add_argument("--array-index", type=int, required=True)
    args = ap.parse_args()
    if args.cmd == "build-manifest":
        build_manifest()
    elif args.cmd == "score-task":
        score_task(args.array_index)


if __name__ == "__main__":
    main()

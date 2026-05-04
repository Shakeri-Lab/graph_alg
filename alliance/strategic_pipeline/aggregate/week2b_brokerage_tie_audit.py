"""Week 2B follow-on — brokerage tie audit.

The Phase-9A backtest reported that brokerage_only catches 15/16 in-pool
realized 2011 partners at K=5.  But brokerage_L2 saturates at 1.0 for
~99% of candidates, which means the realized partner's "rank under
brokerage_only" is ambiguous within saturated tie-blocks: it depends on
how the dataframe sort breaks ties.

This module quantifies the ambiguity.  For each realized partner c* and
focal f, compute the brokerage-rank range:

  r_min(f, c*, t) = 1 + #{c : brokerage(f, c, t) > brokerage(f, c*, t)}
  r_max(f, c*, t) =     #{c : brokerage(f, c, t) >= brokerage(f, c*, t)}

Then report three hit-rate variants at each top-K:

  Hit_optimistic@K  = 1{r_min <= K}     (best-case: c* is at the start of
                                          its tie block)
  Hit_pessimistic@K = 1{r_max <= K}     (worst-case: c* is at the end)
  Hit_random@K      = expected hit under uniform random tie-breaking
                      = 1                      if r_max <= K
                      = 0                      if r_min > K
                      = (K - r_min + 1)        otherwise
                        / (r_max - r_min + 1)

If the gap between optimistic and pessimistic hit rate is small at K=5,
the (15/16) Phase-9A headline is robust.  If it is large, the headline
was driven by tie-block lottery and should be reported with an interval.

Outputs:
  outputs/strategic/aggregate/week2b_brokerage_tie_audit.csv
  outputs/strategic/aggregate/week2b_tie_audit_hit_rates.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
AGG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "aggregate"

T = 2011
SCORE_YEAR = 2010
SCORING_DIR = AGG_DIR / f"week2_personalization_{SCORE_YEAR}"
TOP_K_VALUES = [5, 10, 20, 50, 100]


def main() -> None:
    print(f"[w2b_tie_audit] loading realized panel for t={T}")
    realized = pd.read_parquet(AGG_DIR / f"week2b_realized_ties_{T}.parquet")
    realized["focal_cusip"] = realized["focal_cusip"].astype(str)
    realized["candidate_cusip"] = realized["candidate_cusip"].astype(str)
    print(f"  realized rows: {len(realized)}")

    rows = []
    for _, r in realized.iterrows():
        focal = r["focal_cusip"]
        cand = r["candidate_cusip"]
        path = SCORING_DIR / f"{focal}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if len(df) == 0:
            continue
        df["candidate_cusip"] = df["candidate_cusip"].astype(str)
        match = df[df["candidate_cusip"] == cand]
        if not len(match):
            continue
        c_score = float(match.iloc[0]["brokerage_l2"])
        scores = df["brokerage_l2"].astype(float).to_numpy()
        n_strict_better = int((scores > c_score).sum())
        n_at_least = int((scores >= c_score).sum())
        rows.append({
            "focal_cusip": focal,
            "candidate_cusip": cand,
            "candidate_score": c_score,
            "pool_size": int(len(df)),
            "r_min": n_strict_better + 1,
            "r_max": n_at_least,
            "tie_block_size": n_at_least - n_strict_better,
        })

    audit = pd.DataFrame(rows)
    audit_path = AGG_DIR / "week2b_brokerage_tie_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"  joined: {len(audit)} in-pool realized rows")
    if len(audit) == 0:
        print("[w2b_tie_audit] no rows joinable; abort")
        return

    print()
    print("=== per-realized-partner audit ===")
    print(audit.to_string(index=False))

    # Hit-rate variants per K
    out = []
    for k in TOP_K_VALUES:
        opt = (audit["r_min"] <= k).astype(int)
        pes = (audit["r_max"] <= k).astype(int)
        # random tie: expected hit under uniform within-tie ordering
        rnd = np.where(
            audit["r_max"] <= k, 1.0,
            np.where(audit["r_min"] > k, 0.0,
                       (k - audit["r_min"] + 1) /
                       (audit["r_max"] - audit["r_min"] + 1))
        )
        out.append({
            "top_k": k,
            "n_realized": len(audit),
            "hit_optimistic": float(opt.mean()),
            "hit_pessimistic": float(pes.mean()),
            "hit_random_tie": float(np.mean(rnd)),
            "ambiguity_band": float(opt.mean() - pes.mean()),
        })
    hit = pd.DataFrame(out)
    hit_path = AGG_DIR / "week2b_tie_audit_hit_rates.csv"
    hit.to_csv(hit_path, index=False)
    print()
    print("=== hit-rate variants ===")
    print(hit.to_string(index=False))

    # Distribution of tie-block sizes
    print()
    print("=== tie-block size distribution ===")
    print(audit["tie_block_size"].describe().to_string())


if __name__ == "__main__":
    main()

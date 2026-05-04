"""Week 2B — outcome-panel construction for the 2011 backtest.

For each Compustat-matched focal firm at year t = 2011:

1. **Realized new L2 ties**: dyads (f, c) that appear in the L2 layer at
   year t but did NOT appear in [t-4, t-1].  These are the "realized
   choices" the recommender's top-K is being evaluated against.

2. **Tie persistence**: for each realized dyad, count how many of the
   subsequent years t+1, t+2, ..., t+5 the dyad re-appears (any
   layer).  Y_persist = 1 if that count is at least 3 (the 3-year
   sustained-tie threshold from the Hankel-DMD analysis).

3. **Sales delta**: for each focal that formed at least one realized
   L2 tie at t, compute Δ log Sales at horizons h ∈ {2, 4} as
   log_sales(t+h) − log_sales(t).

Outputs (in ``outputs/strategic/aggregate/``):

  week2b_realized_ties_2011.parquet     — one row per (f, c, t=2011)
                                           realized L2 dyad with
                                           persistence + sales outcomes
  week2b_focal_outcomes_2011.parquet    — one row per focal firm with
                                           sales deltas
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategic_pipeline.data_loader import DataBundle, load_all
from strategic_pipeline.firm_profile import ROLLING_WINDOW


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
AGG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "aggregate"

T_BACKTEST = 2011
WINDOW = ROLLING_WINDOW          # 5-yr lookback for "current" partner set
PERSISTENCE_HORIZON = 5          # check t+1..t+5
PERSISTENCE_THRESHOLD = 3        # T_fc >= 3 = "sustained"
SALES_HORIZONS = (2, 4)


def realized_l2_ties(bundle: DataBundle, t: int = T_BACKTEST) -> pd.DataFrame:
    """All L2 dyads first appearing at year t (i.e., not in [t-4, t-1]).

    Returns a DataFrame with columns: focal_cusip, candidate_cusip, year=t.
    The (focal, candidate) ordering is symmetric — we duplicate each dyad
    with both orderings so a join from the recommender's per-focal
    parquets matches.
    """
    e = bundle.edges
    e = e[e["layer_code"] == "L2"].copy()
    e["firm_i"] = e["firm_i"].astype(str)
    e["firm_j"] = e["firm_j"].astype(str)

    # Canonicalize dyads
    a = np.where(e["firm_i"] < e["firm_j"], e["firm_i"], e["firm_j"])
    b = np.where(e["firm_i"] < e["firm_j"], e["firm_j"], e["firm_i"])
    e["a"] = a
    e["b"] = b
    span = e.groupby(["a", "b"])["year"].agg(["min", "max"]).reset_index()

    # Realized at t = appears at t and a-min == t (first appearance)
    realized = span[span["min"] == t].copy()
    print(f"  [w2b_outcomes] realized L2 dyads at t={t}: {len(realized):,}")

    # Materialize as both (focal=a, cand=b) and (focal=b, cand=a)
    forward = realized.rename(columns={"a": "focal_cusip",
                                         "b": "candidate_cusip"})[
        ["focal_cusip", "candidate_cusip"]]
    forward["year"] = t
    backward = realized.rename(columns={"b": "focal_cusip",
                                          "a": "candidate_cusip"})[
        ["focal_cusip", "candidate_cusip"]]
    backward["year"] = t
    out = pd.concat([forward, backward], ignore_index=True)
    return out


def add_persistence(realized: pd.DataFrame, bundle: DataBundle,
                      t: int = T_BACKTEST,
                      horizon: int = PERSISTENCE_HORIZON) -> pd.DataFrame:
    """For each realized dyad, count how many of years t+1..t+horizon
    the dyad re-appears in any layer.  Returns realized with a new
    column ``T_fc`` (int in [0, horizon])."""
    e = bundle.edges.copy()
    e["firm_i"] = e["firm_i"].astype(str)
    e["firm_j"] = e["firm_j"].astype(str)
    e_post = e[(e["year"] > t) & (e["year"] <= t + horizon)].copy()
    a = np.where(e_post["firm_i"] < e_post["firm_j"],
                  e_post["firm_i"], e_post["firm_j"])
    b = np.where(e_post["firm_i"] < e_post["firm_j"],
                  e_post["firm_j"], e_post["firm_i"])
    e_post["a"] = a
    e_post["b"] = b
    post_years = (e_post.groupby(["a", "b"])["year"]
                    .nunique()
                    .rename("T_fc_raw")
                    .reset_index())

    df = realized.copy()
    a2 = np.where(df["focal_cusip"] < df["candidate_cusip"],
                   df["focal_cusip"], df["candidate_cusip"])
    b2 = np.where(df["focal_cusip"] < df["candidate_cusip"],
                   df["candidate_cusip"], df["focal_cusip"])
    df["_a"] = a2
    df["_b"] = b2
    out = df.merge(post_years, left_on=["_a", "_b"],
                    right_on=["a", "b"], how="left").drop(
        columns=["_a", "_b", "a", "b"])
    out["T_fc"] = out["T_fc_raw"].fillna(0).astype(int)
    out["sustained_persist"] = (out["T_fc"] >= PERSISTENCE_THRESHOLD).astype(int)
    out.drop(columns=["T_fc_raw"], inplace=True)
    return out


def add_focal_sales_delta(realized: pd.DataFrame, bundle: DataBundle,
                            t: int = T_BACKTEST,
                            horizons=SALES_HORIZONS) -> pd.DataFrame:
    """Attach Δ log Sales at horizons relative to focal's t-year sales."""
    fy = bundle.firm_year.copy()
    fy["ult_parent_cusip"] = fy["ult_parent_cusip"].astype(str)
    sales_col = "sales" if "sales" in fy.columns else \
        ("net_sales" if "net_sales" in fy.columns else None)
    if sales_col is None:
        # Fall back: any column matching 'sale'
        cands = [c for c in fy.columns if "sale" in c.lower()]
        if not cands:
            print("  [w2b_outcomes] WARNING: no sales column in firm_year")
            realized["delta_log_sales_h2"] = float("nan")
            realized["delta_log_sales_h4"] = float("nan")
            return realized
        sales_col = cands[0]
    print(f"  [w2b_outcomes] using sales column: {sales_col}")

    fy["log_sales"] = np.log(fy[sales_col].clip(lower=1.0))
    pivot = (fy.pivot_table(index="ult_parent_cusip", columns="year",
                              values="log_sales", aggfunc="mean"))
    if t not in pivot.columns:
        print(f"  [w2b_outcomes] WARNING: year {t} missing in sales pivot")
        for h in horizons:
            realized[f"delta_log_sales_h{h}"] = float("nan")
        return realized

    out = realized.copy()
    for h in horizons:
        target = t + h
        if target in pivot.columns:
            delta = (pivot[target] - pivot[t]).rename(
                f"delta_log_sales_h{h}")
            out = out.merge(delta, left_on="focal_cusip", right_index=True,
                              how="left")
        else:
            out[f"delta_log_sales_h{h}"] = float("nan")
            print(f"  [w2b_outcomes] year {target} missing; h={h} → NaN")
    return out


def main() -> None:
    bundle = load_all()
    print(f"[w2b_outcomes] building realized-tie panel at t={T_BACKTEST}")
    realized = realized_l2_ties(bundle, t=T_BACKTEST)
    realized = add_persistence(realized, bundle, t=T_BACKTEST)
    realized = add_focal_sales_delta(realized, bundle, t=T_BACKTEST)
    out_path = AGG_DIR / f"week2b_realized_ties_{T_BACKTEST}.parquet"
    realized.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}: {len(realized):,} (focal, candidate) rows")
    print(f"  sustained-share: {realized['sustained_persist'].mean():.2%}")
    if realized["delta_log_sales_h2"].notna().any():
        print(f"  mean Δ log Sales h2: "
              f"{realized['delta_log_sales_h2'].mean():.3f}")
    if realized["delta_log_sales_h4"].notna().any():
        print(f"  mean Δ log Sales h4: "
              f"{realized['delta_log_sales_h4'].mean():.3f}")
    print("[w2b_outcomes] done.")


if __name__ == "__main__":
    main()

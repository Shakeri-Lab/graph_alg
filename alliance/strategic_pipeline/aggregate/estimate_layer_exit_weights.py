"""Estimate layer-specific partner-exit event-study weights.

The systemic-criticality scorer consumes ``layer_exit_weights.csv``.  The
primary coefficient is the stacked-cohort t+1 estimate when available; legacy
TWFE is retained for comparison because earlier manuscripts reported it.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np
import pandas as pd

from strategic_pipeline.data_loader import load_all
from strategic_pipeline.id_utils import normalize_cusip_columns


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
AGG_DIR = PROJECT_ROOT / "outputs" / "strategic" / "aggregate"
AGG_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = ["L1", "L2", "L3", "L4"]
EVENT_WINDOW = range(-3, 4)


@dataclass
class OLSResult:
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    nobs: int


def _safe_log(series: pd.Series) -> pd.Series:
    return np.log(series.clip(lower=0.01))


def build_financial_panel(bundle) -> pd.DataFrame:
    panel = bundle.firm_year[
        ["ult_parent_cusip", "year", "market_value", "total_assets"]
    ].copy()
    normalize_cusip_columns(panel, ["ult_parent_cusip"])
    panel = panel[panel["market_value"].notna() & panel["total_assets"].notna()].copy()
    panel["log_mv"] = _safe_log(panel["market_value"])
    panel["log_assets"] = _safe_log(panel["total_assets"])
    return panel.dropna(subset=["log_mv", "log_assets"])


def active_years_by_firm(edges: pd.DataFrame) -> dict[str, set[int]]:
    a = edges[["firm_i", "year"]].rename(columns={"firm_i": "cusip"})
    b = edges[["firm_j", "year"]].rename(columns={"firm_j": "cusip"})
    active = pd.concat([a, b], ignore_index=True).dropna()
    return active.groupby("cusip")["year"].apply(lambda s: set(map(int, s))).to_dict()


def identify_layer_shocks(edges: pd.DataFrame, layer: str, panel_firms: set[str]) -> pd.DataFrame:
    """One first partner-exit shock per focal firm for a given layer.

    Event year is the first absent year: partner active in ``event_year - 1``,
    absent in ``event_year``, and never returns in the sample.
    """
    active = active_years_by_firm(edges)
    last_year = {firm: max(years) for firm, years in active.items()}
    sudden_exit = {
        firm: last + 1
        for firm, last in last_year.items()
        if last < 2016 and (last - 1) in active.get(firm, set())
    }

    sub = edges[edges["layer_code"] == layer].copy()
    directed = pd.concat([
        sub[["firm_i", "firm_j", "year"]].rename(columns={"firm_i": "focal_cusip", "firm_j": "partner_cusip"}),
        sub[["firm_j", "firm_i", "year"]].rename(columns={"firm_j": "focal_cusip", "firm_i": "partner_cusip"}),
    ], ignore_index=True)
    directed = directed[directed["focal_cusip"].isin(panel_firms)].copy()
    directed["partner_exit_year"] = directed["partner_cusip"].map(sudden_exit)
    directed = directed.dropna(subset=["partner_exit_year"])
    directed["partner_exit_year"] = directed["partner_exit_year"].astype(int)
    directed = directed[directed["partner_exit_year"] >= directed["year"]]
    directed = directed[directed["partner_exit_year"].between(1995, 2016)]
    if directed.empty:
        return directed
    directed = directed.sort_values(["focal_cusip", "partner_exit_year", "year"])
    return directed.drop_duplicates("focal_cusip", keep="first")[
        ["focal_cusip", "partner_cusip", "partner_exit_year"]
    ]


def _event_cols() -> list[str]:
    cols = []
    for t in EVENT_WINDOW:
        if t == -1:
            continue
        cols.append(f"evt_m{abs(t)}" if t < 0 else f"evt_p{t}")
    return cols


def _add_event_dummies(df: pd.DataFrame) -> pd.DataFrame:
    for t in EVENT_WINDOW:
        if t == -1:
            continue
        col = f"evt_m{abs(t)}" if t < 0 else f"evt_p{t}"
        df[col] = (df["event_time"] == t).astype(float)
    return df


def fit_twfe_residualized(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    entity_col: str,
    time_col: str,
):
    """Fit y ~ X after absorbing entity and time fixed effects.

    This avoids a hard dependency on ``linearmodels`` in environments where it
    is unavailable.  Alternating demeaning converges quickly for the small set
    of variables used in the event-study specification.
    """
    work_cols = [y_col] + x_cols + [entity_col, time_col]
    work = df[work_cols].dropna().copy()
    values = work[[y_col] + x_cols].astype(float)
    grand = values.mean()
    resid = values - grand
    for _ in range(25):
        old = resid.to_numpy().copy()
        resid = resid - resid.groupby(work[entity_col]).transform("mean")
        resid = resid - resid.groupby(work[time_col]).transform("mean")
        if np.nanmax(np.abs(resid.to_numpy() - old)) < 1e-9:
            break
    y = resid[y_col].to_numpy(dtype=float)
    x = resid[x_cols].to_numpy(dtype=float)
    groups = work[entity_col].astype("category").cat.codes.to_numpy()

    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    u = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = np.zeros((len(x_cols), len(x_cols)))
    for g in np.unique(groups):
        idx = groups == g
        xg = x[idx]
        ug = u[idx][:, None]
        score = xg.T @ ug
        meat += score @ score.T
    n, k = x.shape
    g_count = max(len(np.unique(groups)), 1)
    correction = (g_count / max(g_count - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_stat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = np.array([erfc(abs(t) / sqrt(2.0)) for t in t_stat])
    return OLSResult(
        params=pd.Series(beta, index=x_cols),
        bse=pd.Series(se, index=x_cols),
        pvalues=pd.Series(pvals, index=x_cols),
        nobs=n,
    )


def estimate_twfe(panel: pd.DataFrame, shocks: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if shocks.empty:
        return {}, pd.DataFrame()
    lookup = shocks.set_index("focal_cusip")["partner_exit_year"].to_dict()
    treated = set(lookup)
    df = panel.copy()
    df["treated"] = df["ult_parent_cusip"].isin(treated)
    df["shock_year"] = df["ult_parent_cusip"].map(lookup)
    df["event_time"] = np.where(df["treated"], df["year"] - df["shock_year"], np.nan)
    df = _add_event_dummies(df)
    x_cols = _event_cols() + ["log_assets"]
    res = fit_twfe_residualized(df, "log_mv", x_cols, "ult_parent_cusip", "year")
    rows = []
    for t in EVENT_WINDOW:
        if t == -1:
            rows.append({"event_time": t, "coef": 0.0, "se": 0.0, "p": 1.0})
            continue
        col = f"evt_m{abs(t)}" if t < 0 else f"evt_p{t}"
        rows.append({
            "event_time": t,
            "coef": float(res.params.get(col, np.nan)),
            "se": float(res.bse.get(col, np.nan)),
            "p": float(res.pvalues.get(col, np.nan)),
        })
    summary = {
        "nobs": int(res.nobs),
        "treated_firms": len(treated),
        "t1_coef": rows[[r["event_time"] for r in rows].index(1)]["coef"],
        "t1_se": rows[[r["event_time"] for r in rows].index(1)]["se"],
        "t1_p": rows[[r["event_time"] for r in rows].index(1)]["p"],
    }
    return summary, pd.DataFrame(rows)


def estimate_stacked(panel: pd.DataFrame, shocks: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if shocks.empty:
        return {}, pd.DataFrame()
    shock_year = shocks.set_index("focal_cusip")["partner_exit_year"].to_dict()
    cohorts = sorted(shocks["partner_exit_year"].unique())
    frames = []
    all_firms = set(panel["ult_parent_cusip"])
    for cohort in cohorts:
        treated = set(shocks.loc[shocks["partner_exit_year"] == cohort, "focal_cusip"])
        clean_controls = {
            firm for firm in all_firms
            if firm not in shock_year or shock_year[firm] > cohort + 3
        }
        firms = treated | clean_controls
        sub = panel[
            panel["ult_parent_cusip"].isin(firms)
            & panel["year"].between(cohort - 3, cohort + 3)
        ].copy()
        if sub.empty:
            continue
        sub["cohort"] = cohort
        sub["treated"] = sub["ult_parent_cusip"].isin(treated)
        sub["event_time"] = np.where(sub["treated"], sub["year"] - cohort, np.nan)
        sub["stack_entity"] = sub["cohort"].astype(str) + "_" + sub["ult_parent_cusip"].astype(str)
        frames.append(sub)
    if not frames:
        return {}, pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = _add_event_dummies(df)
    x_cols = _event_cols() + ["log_assets"]
    res = fit_twfe_residualized(df, "log_mv", x_cols, "stack_entity", "year")
    rows = []
    for t in EVENT_WINDOW:
        if t == -1:
            rows.append({"event_time": t, "coef": 0.0, "se": 0.0, "p": 1.0})
            continue
        col = f"evt_m{abs(t)}" if t < 0 else f"evt_p{t}"
        rows.append({
            "event_time": t,
            "coef": float(res.params.get(col, np.nan)),
            "se": float(res.bse.get(col, np.nan)),
            "p": float(res.pvalues.get(col, np.nan)),
        })
    t1 = [r for r in rows if r["event_time"] == 1][0]
    summary = {
        "nobs": int(res.nobs),
        "treated_firms": shocks["focal_cusip"].nunique(),
        "cohorts": len(cohorts),
        "t1_coef": t1["coef"],
        "t1_se": t1["se"],
        "t1_p": t1["p"],
    }
    return summary, pd.DataFrame(rows)


def main() -> None:
    bundle = load_all()
    edges = bundle.edges[["firm_i", "firm_j", "year", "layer_code"]].copy()
    normalize_cusip_columns(edges, ["firm_i", "firm_j"])
    panel = build_financial_panel(bundle)
    panel_firms = set(panel["ult_parent_cusip"])

    weight_rows = []
    event_rows = []
    report = []
    for layer in LAYERS:
        print(f"Estimating {layer} partner-exit event studies ...")
        shocks = identify_layer_shocks(edges, layer, panel_firms)
        shocks.to_csv(AGG_DIR / f"layer_{layer}_exit_shocks.csv", index=False)
        twfe, twfe_events = estimate_twfe(panel, shocks)
        stacked, stacked_events = estimate_stacked(panel, shocks)
        primary = stacked.get("t1_coef", np.nan)
        source = "stacked_cohort_t1"
        if pd.isna(primary):
            primary = twfe.get("t1_coef", np.nan)
            source = "legacy_twfe_t1"
        if pd.isna(primary):
            primary = -0.02
            source = "fallback_no_estimate"
        weight_rows.append({
            "layer": layer,
            "primary_coef": primary,
            "primary_source": source,
            "twfe_t1_coef": twfe.get("t1_coef", np.nan),
            "twfe_t1_se": twfe.get("t1_se", np.nan),
            "twfe_t1_p": twfe.get("t1_p", np.nan),
            "twfe_nobs": twfe.get("nobs", np.nan),
            "twfe_treated_firms": twfe.get("treated_firms", np.nan),
            "stacked_t1_coef": stacked.get("t1_coef", np.nan),
            "stacked_t1_se": stacked.get("t1_se", np.nan),
            "stacked_t1_p": stacked.get("t1_p", np.nan),
            "stacked_nobs": stacked.get("nobs", np.nan),
            "stacked_treated_firms": stacked.get("treated_firms", np.nan),
            "stacked_cohorts": stacked.get("cohorts", np.nan),
        })
        for name, events in [("twfe", twfe_events), ("stacked", stacked_events)]:
            if len(events):
                events = events.copy()
                events["layer"] = layer
                events["estimator"] = name
                event_rows.append(events)
        report.append(f"{layer}: shocks={len(shocks):,d}; "
                      f"TWFE t+1={twfe.get('t1_coef', np.nan):+.4f} "
                      f"(p={twfe.get('t1_p', np.nan):.3f}); "
                      f"stacked t+1={stacked.get('t1_coef', np.nan):+.4f} "
                      f"(p={stacked.get('t1_p', np.nan):.3f})")

    weights = pd.DataFrame(weight_rows)
    weights.to_csv(AGG_DIR / "layer_exit_weights.csv", index=False)
    if event_rows:
        pd.concat(event_rows, ignore_index=True).to_csv(
            AGG_DIR / "layer_exit_event_studies.csv", index=False
        )
    (AGG_DIR / "layer_exit_weights_report.txt").write_text("\n".join(report))
    print(weights.to_string(index=False))


if __name__ == "__main__":
    main()

"""Build the systemic-criticality meta-network and robustness artifacts.

The original aggregate used top-5 rows from per-firm CSV reports.  This
version rebuilds the dependency panel directly from the typed network and
financial artifacts so that CUSIP identity, annual coverage, top-k
sensitivity, null models, backtests, and stress tests share one source of
truth.

Primary outputs:
  outputs/strategic/aggregate/critical_edges_panel.parquet
  outputs/strategic/aggregate/critical_edges.csv
  outputs/strategic/aggregate/systemic_criticality.csv
  outputs/strategic/aggregate/systemic_criticality_annual.csv
  outputs/strategic/aggregate/systemic_criticality_cumulative.csv
  outputs/strategic/aggregate/*_audit_or_experiment.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategic_pipeline.data_loader import DataBundle, load_all
from strategic_pipeline.id_utils import normalize_cusip, normalize_cusip_columns, normalize_cusip_series


PROJECT_ROOT = Path("/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/alliance")
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "strategic"
AGG_DIR = OUTPUTS_ROOT / "aggregate"
AGG_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(1995, 2018)
ROLLING_WINDOW = 5
TOP_K_PER_FIRM = 5
LAYERS = ["L1", "L2", "L3", "L4"]

# Used only when the layer-estimation artifact has not been generated yet.
FALLBACK_LAYER_WEIGHTS = {
    "L1": -0.020,
    "L2": -0.087,
    "L3": -0.030,
    "L4": -0.020,
}


def _safe_log(series: pd.Series) -> pd.Series:
    return np.log(series.clip(lower=0.01))


def load_layer_weights() -> pd.DataFrame:
    path = AGG_DIR / "layer_exit_weights.csv"
    if path.exists():
        weights = pd.read_csv(path)
        if {"layer", "primary_coef"}.issubset(weights.columns):
            weights["layer"] = weights["layer"].astype(str)
            return weights

    rows = []
    for layer, coef in FALLBACK_LAYER_WEIGHTS.items():
        rows.append({
            "layer": layer,
            "primary_coef": coef,
            "twfe_t1_coef": np.nan,
            "stacked_t1_coef": coef,
            "source": "fallback_until_layer_exit_weights_are_estimated",
        })
    return pd.DataFrame(rows)


def _firm_meta(bundle: DataBundle) -> pd.DataFrame:
    meta = bundle.firm_meta[["cusip", "name", "sic2", "nation"]].copy()
    meta["cusip"] = normalize_cusip_series(meta["cusip"])
    meta = meta.dropna(subset=["cusip"]).drop_duplicates("cusip", keep="first")
    return meta


def _financial_snapshot(bundle: DataBundle, year: int, prefix: str) -> pd.DataFrame:
    cols = [
        "ult_parent_cusip", "year", "market_value", "total_assets",
        "degree", "eigenvector", "sales",
    ]
    fy = bundle.firm_year.loc[bundle.firm_year["year"] == year, cols].copy()
    normalize_cusip_columns(fy, ["ult_parent_cusip"])
    fy = fy.drop_duplicates("ult_parent_cusip", keep="first")
    fy["log_market_value"] = _safe_log(fy["market_value"])
    fy = fy.rename(columns={"ult_parent_cusip": f"{prefix}_cusip"})
    rename = {
        c: f"{prefix}_{c}"
        for c in ["market_value", "total_assets", "degree", "eigenvector",
                  "sales", "log_market_value"]
        if c in fy.columns
    }
    return fy.rename(columns=rename).drop(columns=["year"])


def _layer_btw_snapshot(bundle: DataBundle, year: int, prefix: str) -> pd.DataFrame:
    lb = bundle.layer_btw.loc[bundle.layer_btw["year"] == year].copy()
    normalize_cusip_columns(lb, ["ult_parent_cusip"])
    lb = lb.drop_duplicates("ult_parent_cusip", keep="first")
    for layer in LAYERS:
        col = f"{layer}_btw"
        if col not in lb.columns:
            lb[col] = 0.0
        lb[col] = lb[col].fillna(0.0)
    lb = lb.rename(columns={"ult_parent_cusip": f"{prefix}_cusip"})
    lb = lb[[f"{prefix}_cusip"] + [f"{layer}_btw" for layer in LAYERS]]
    return lb.rename(columns={f"{layer}_btw": f"{prefix}_{layer}_btw"
                              for layer in LAYERS})


def _directed_edges_for_window(bundle: DataBundle, year: int) -> pd.DataFrame:
    start = year - ROLLING_WINDOW + 1
    e = bundle.edges[
        (bundle.edges["year"] >= start) & (bundle.edges["year"] <= year)
    ].copy()
    normalize_cusip_columns(e, ["firm_i", "firm_j"])
    e = e.dropna(subset=["firm_i", "firm_j", "layer_code"])
    e = e[e["firm_i"] != e["firm_j"]]
    if "weight_norm" not in e.columns:
        e["weight_norm"] = 1.0
    e["weight_norm"] = pd.to_numeric(e["weight_norm"], errors="coerce").fillna(1.0)

    keep = ["year", "firm_i", "firm_j", "layer_code", "weight_norm"]
    a = e[keep].rename(columns={"firm_i": "focal_cusip", "firm_j": "partner_cusip"})
    b = e[keep].rename(columns={"firm_j": "focal_cusip", "firm_i": "partner_cusip"})
    return pd.concat([a, b], ignore_index=True)


def _build_year_panel(bundle: DataBundle, year: int, weights: pd.DataFrame) -> pd.DataFrame:
    directed = _directed_edges_for_window(bundle, year)
    if directed.empty:
        return pd.DataFrame()

    keys = ["focal_cusip", "partner_cusip"]
    base = directed.groupby(keys).agg(
        tie_count=("year", "size"),
        tie_strength=("weight_norm", "sum"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    ).reset_index()
    base["year"] = year
    base["tenure"] = year - base["first_year"] + 1

    layer_strength = directed.pivot_table(
        index=keys, columns="layer_code", values="weight_norm",
        aggfunc="sum", fill_value=0.0,
    )
    layer_count = directed.pivot_table(
        index=keys, columns="layer_code", values="weight_norm",
        aggfunc="size", fill_value=0,
    )
    for layer in LAYERS:
        if layer not in layer_strength.columns:
            layer_strength[layer] = 0.0
        if layer not in layer_count.columns:
            layer_count[layer] = 0
    layer_strength = layer_strength[LAYERS].reset_index()
    layer_count = layer_count[LAYERS].reset_index()
    layer_strength = layer_strength.rename(columns={l: f"layer_strength_{l}" for l in LAYERS})
    layer_count = layer_count.rename(columns={l: f"layer_count_{l}" for l in LAYERS})
    df = base.merge(layer_strength, on=keys, how="left").merge(layer_count, on=keys, how="left")

    total_strength = df[[f"layer_strength_{l}" for l in LAYERS]].sum(axis=1).replace(0, np.nan)
    for layer in LAYERS:
        df[f"layer_mix_{layer}"] = (df[f"layer_strength_{layer}"] / total_strength).fillna(0.0)
    mix_cols = [f"layer_mix_{l}" for l in LAYERS]
    df["dominant_layer"] = df[mix_cols].idxmax(axis=1).str.replace("layer_mix_", "", regex=False)

    meta = _firm_meta(bundle)
    df = df.merge(
        meta.rename(columns={
            "cusip": "partner_cusip", "name": "partner_name",
            "sic2": "partner_sic2", "nation": "partner_nation",
        }),
        on="partner_cusip", how="left",
    )
    df = df.merge(
        meta.rename(columns={
            "cusip": "focal_cusip", "name": "focal_name",
            "sic2": "focal_sic2", "nation": "focal_nation",
        }),
        on="focal_cusip", how="left",
    )

    df = df.merge(_financial_snapshot(bundle, year, "focal"), on="focal_cusip", how="left")
    df = df.merge(_financial_snapshot(bundle, year, "partner"), on="partner_cusip", how="left")
    df = df.merge(_layer_btw_snapshot(bundle, year, "focal"), on="focal_cusip", how="left")
    df = df.merge(_layer_btw_snapshot(bundle, year, "partner"), on="partner_cusip", how="left")
    for layer in LAYERS:
        df[f"focal_{layer}_btw"] = df[f"focal_{layer}_btw"].fillna(0.0)
        df[f"partner_{layer}_btw"] = df[f"partner_{layer}_btw"].fillna(0.0)

    focal_layer_totals = (
        df.groupby("focal_cusip")[[f"layer_strength_{l}" for l in LAYERS]]
        .sum()
        .rename(columns={f"layer_strength_{l}": f"focal_total_strength_{l}" for l in LAYERS})
        .reset_index()
    )
    df = df.merge(focal_layer_totals, on="focal_cusip", how="left")
    for layer in LAYERS:
        denom = df[f"focal_total_strength_{layer}"].replace(0, np.nan)
        share = (df[f"layer_strength_{layer}"] / denom).fillna(0.0)
        df[f"delta_{layer}_btw"] = -df[f"focal_{layer}_btw"] * share
    df["delta_btw_abs"] = df[[f"delta_{l}_btw" for l in LAYERS]].abs().sum(axis=1)

    df["substitute_count"] = (
        df.groupby(["focal_cusip", "dominant_layer", "partner_sic2"])["partner_cusip"]
        .transform("nunique")
        .sub(1)
        .clip(lower=0)
        .fillna(0)
        .astype(int)
    )

    for layer in LAYERS:
        coef = float(weights.loc[weights["layer"] == layer, "primary_coef"].iloc[0]) \
            if (weights["layer"] == layer).any() else FALLBACK_LAYER_WEIGHTS[layer]
        twfe = weights.loc[weights["layer"] == layer, "twfe_t1_coef"]
        stacked = weights.loc[weights["layer"] == layer, "stacked_t1_coef"]
        df[f"coef_primary_{layer}"] = coef
        df[f"coef_twfe_{layer}"] = float(twfe.iloc[0]) if len(twfe) and pd.notna(twfe.iloc[0]) else coef
        df[f"coef_stacked_{layer}"] = float(stacked.iloc[0]) if len(stacked) and pd.notna(stacked.iloc[0]) else coef

    _score_year_panel(df)
    return df


def _weighted_base_loss(df: pd.DataFrame, coef_prefix: str) -> pd.Series:
    out = pd.Series(0.0, index=df.index)
    for layer in LAYERS:
        coef = df[f"{coef_prefix}_{layer}"].astype(float)
        out = out + df[f"layer_mix_{layer}"].astype(float) * (-coef.clip(upper=0.0))
    return out


def _score_year_panel(df: pd.DataFrame) -> None:
    tie_med = max(float(df["tie_strength"].median()), 1e-6)
    tie_factor = np.sqrt((df["tie_strength"] / tie_med).clip(lower=0.25, upper=4.0))
    tenure_factor = 1.0 + 0.08 * (df["tenure"].clip(lower=1, upper=5) - 1)
    redundancy_factor = 1.0 / (1.0 + 0.25 * df["substitute_count"].clip(lower=0))
    redundancy_factor = redundancy_factor.clip(lower=0.25, upper=1.0)

    partner_layer_btw = np.select(
        [df["dominant_layer"].eq(layer) for layer in LAYERS],
        [df[f"partner_{layer}_btw"] for layer in LAYERS],
        default=0.0,
    )
    df["partner_layer_btw"] = partner_layer_btw
    df["partner_layer_btw_pct"] = pd.Series(partner_layer_btw, index=df.index).rank(pct=True).fillna(0.0)

    denom = max(float(df["delta_btw_abs"].quantile(0.95)), 1e-9)
    delta_factor = 1.0 + 0.50 * (df["delta_btw_abs"] / denom).clip(lower=0.0, upper=1.0)
    partner_factor = 1.0 + 0.40 * df["partner_layer_btw_pct"]

    df["exposure_multiplier"] = (
        tie_factor * tenure_factor * redundancy_factor * delta_factor * partner_factor
    ).clip(lower=0.10, upper=4.00)

    df["base_layer_loss"] = _weighted_base_loss(df, "coef_primary")
    df["twfe_layer_loss"] = _weighted_base_loss(df, "coef_twfe")
    df["stacked_layer_loss"] = _weighted_base_loss(df, "coef_stacked")

    df["predicted_log_mv_loss"] = -(df["base_layer_loss"] * df["exposure_multiplier"])
    df["twfe_log_mv_loss"] = -(df["twfe_layer_loss"] * df["exposure_multiplier"])
    df["stacked_log_mv_loss"] = -(df["stacked_layer_loss"] * df["exposure_multiplier"])
    df["blended_stacked_twfe_log_mv_loss"] = 0.5 * (
        df["predicted_log_mv_loss"] + df["twfe_log_mv_loss"]
    )
    df["layer_only_log_mv_loss"] = -df["base_layer_loss"]
    df["no_redundancy_log_mv_loss"] = -(
        df["base_layer_loss"] *
        (df["exposure_multiplier"] / redundancy_factor.replace(0, np.nan)).fillna(df["exposure_multiplier"])
    ).clip(lower=0, upper=10)
    df["no_delta_log_mv_loss"] = -(
        df["base_layer_loss"] *
        (df["exposure_multiplier"] / delta_factor.replace(0, np.nan)).fillna(df["exposure_multiplier"])
    ).clip(lower=0, upper=10)
    degree_norm = np.sqrt((df["partner_degree"].fillna(0.0) + 1.0).rank(pct=True).fillna(0.0) + 0.25)
    df["raw_degree_log_mv_loss"] = -(df["base_layer_loss"] * degree_norm)

    mv = pd.to_numeric(df["focal_market_value"], errors="coerce")
    df["predicted_dollar_loss"] = mv * (1.0 - np.exp(df["predicted_log_mv_loss"]))
    df.loc[mv.isna(), "predicted_dollar_loss"] = np.nan


def build_critical_edges_panel(bundle: DataBundle, weights: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        print(f"  building systemic panel {year}")
        yr = _build_year_panel(bundle, year, weights)
        if yr.empty:
            continue
        active_compustat = set(bundle.compustat_firms)
        yr = yr[yr["focal_cusip"].isin(active_compustat)].copy()
        yr = yr.sort_values(
            ["focal_cusip", "predicted_log_mv_loss", "tie_strength"],
            ascending=[True, True, False],
        )
        yr["rank"] = yr.groupby("focal_cusip").cumcount() + 1
        frames.append(yr)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel["is_partner_compustat"] = panel["partner_cusip"].isin(bundle.compustat_firms)
    return panel


def aggregate_edges(edges: pd.DataFrame, value_col: str = "predicted_log_mv_loss") -> pd.DataFrame:
    top = edges[edges["rank"] <= TOP_K_PER_FIRM].copy()
    top["abs_loss"] = top[value_col].abs()
    agg = top.groupby("partner_cusip").agg(
        in_degree=("focal_cusip", "nunique"),
        edge_count=("focal_cusip", "size"),
        total_predicted_log_mv_cost=("abs_loss", "sum"),
        mean_predicted_log_mv_cost=("abs_loss", "mean"),
        total_predicted_dollar_cost=("predicted_dollar_loss", "sum"),
        mean_rank=("rank", "mean"),
        pct_partner_compustat=("is_partner_compustat", "mean"),
    ).reset_index()
    layer_mix = top.pivot_table(
        index="partner_cusip", columns="dominant_layer", values="focal_cusip",
        aggfunc="count", fill_value=0,
    )
    layer_mix = layer_mix.div(layer_mix.sum(axis=1), axis=0)
    for layer in LAYERS:
        agg[f"layer_frac_{layer}"] = agg["partner_cusip"].map(
            layer_mix[layer] if layer in layer_mix.columns else {}
        ).fillna(0.0)
    meta_cols = [
        "partner_cusip", "partner_name", "partner_sic2", "partner_nation",
        "partner_market_value", "partner_degree", "partner_eigenvector",
    ]
    meta = (
        edges.sort_values("year")
        .drop_duplicates("partner_cusip", keep="last")[meta_cols]
    )
    agg = agg.merge(meta, on="partner_cusip", how="left")
    agg["is_compustat"] = agg["pct_partner_compustat"] > 0
    agg = agg.sort_values(
        ["total_predicted_log_mv_cost", "in_degree"],
        ascending=[False, False],
    ).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)

    # Backward-compatible aliases for the existing plot/doc scripts.
    agg["name"] = agg["partner_name"]
    agg["sic2"] = agg["partner_sic2"]
    agg["nation"] = agg["partner_nation"]
    agg["total_empirical_cost"] = agg["total_predicted_log_mv_cost"]
    agg["mean_empirical_cost"] = agg["mean_predicted_log_mv_cost"]
    agg["pct_dmd_available"] = np.nan
    agg["pct_dmd_disagreement"] = np.nan
    return agg


def write_annual_and_cumulative(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = []
    for year, sub in panel.groupby("year"):
        agg = aggregate_edges(sub)
        agg["year"] = year
        annual.append(agg)
    annual_df = pd.concat(annual, ignore_index=True) if annual else pd.DataFrame()
    annual_df.to_csv(AGG_DIR / "systemic_criticality_annual.csv", index=False)

    top = panel[panel["rank"] <= TOP_K_PER_FIRM].copy()
    top["abs_loss"] = top["predicted_log_mv_loss"].abs()
    top["decay_weight"] = 0.85 ** (2017 - top["year"])
    top["decayed_cost"] = top["abs_loss"] * top["decay_weight"]
    cumulative = top.groupby("partner_cusip").agg(
        years_active=("year", "nunique"),
        cumulative_in_degree=("focal_cusip", "nunique"),
        total_cumulative_cost=("abs_loss", "sum"),
        total_decayed_recent_cost=("decayed_cost", "sum"),
        total_cumulative_dollar_cost=("predicted_dollar_loss", "sum"),
    ).reset_index()
    meta = (
        panel.sort_values("year")
        .drop_duplicates("partner_cusip", keep="last")
        [["partner_cusip", "partner_name", "partner_sic2", "partner_nation",
          "is_partner_compustat"]]
    )
    cumulative = cumulative.merge(meta, on="partner_cusip", how="left")
    cumulative = cumulative.sort_values(
        ["total_decayed_recent_cost", "total_cumulative_cost"],
        ascending=[False, False],
    ).reset_index(drop=True)
    cumulative["rank_recent_decayed"] = np.arange(1, len(cumulative) + 1)
    cumulative.to_csv(AGG_DIR / "systemic_criticality_cumulative.csv", index=False)
    return annual_df, cumulative


def write_identity_coverage_audit(panel: pd.DataFrame, bundle: DataBundle) -> pd.DataFrame:
    dirs = [
        d for d in OUTPUTS_ROOT.iterdir()
        if d.is_dir() and d.name not in ("logs", "aggregate")
    ]
    with_csv = 0
    nonempty = 0
    for d in dirs:
        csv = d / "partner_vulnerability.csv"
        if csv.exists():
            with_csv += 1
            try:
                if len(pd.read_csv(csv, dtype=str)) > 0:
                    nonempty += 1
            except Exception:
                pass
    rows = [{
        "year": "all_report_dirs",
        "total_report_dirs": len(dirs),
        "with_partner_csv": with_csv,
        "nonempty_partner_csv": nonempty,
        "normalized_duplicate_partner_rows_latest": np.nan,
        "active_compustat_focals": np.nan,
        "candidate_edges": np.nan,
    }]
    for year, sub in panel.groupby("year"):
        agg = aggregate_edges(sub)
        rows.append({
            "year": year,
            "total_report_dirs": len(dirs),
            "with_partner_csv": with_csv,
            "nonempty_partner_csv": nonempty if year == 2017 else np.nan,
            "normalized_duplicate_partner_rows_latest": (
                len(agg) - agg["partner_cusip"].nunique() if year == 2017 else np.nan
            ),
            "active_compustat_focals": sub["focal_cusip"].nunique(),
            "candidate_edges": len(sub),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(AGG_DIR / "identity_coverage_audit.csv", index=False)
    return audit


def write_topk_sensitivity(panel_latest: pd.DataFrame) -> pd.DataFrame:
    rankings = {}
    for k in [1, 3, 5, 10, 20, 10_000]:
        sub = panel_latest[panel_latest["rank"] <= k].copy()
        rankings[k] = aggregate_edges(sub.assign(rank=1))
    base = rankings[5].set_index("partner_cusip")["rank"]
    rows = []
    for k, agg in rankings.items():
        r = agg.set_index("partner_cusip")["rank"]
        common = base.index.intersection(r.index)
        rho = base.loc[common].corr(r.loc[common], method="spearman") if len(common) > 2 else np.nan
        overlap = len(set(base.head(20).index) & set(r.head(20).index))
        rows.append({
            "k": "all" if k == 10_000 else k,
            "edges": int((panel_latest["rank"] <= k).sum()) if k != 10_000 else len(panel_latest),
            "distinct_partners": len(agg),
            "spearman_vs_k5": rho,
            "top20_overlap_vs_k5": overlap,
            "top20_churn_vs_k5": 20 - overlap,
        })
    out = pd.DataFrame(rows)
    out.to_csv(AGG_DIR / "topk_sensitivity.csv", index=False)
    return out


def write_null_model(panel_latest: pd.DataFrame, b: int = 100, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    top = panel_latest[panel_latest["rank"] <= TOP_K_PER_FIRM].copy()
    top["abs_loss"] = top["predicted_log_mv_loss"].abs()
    obs_agg = aggregate_edges(panel_latest)
    obs_total = obs_agg["total_predicted_log_mv_cost"].sum()
    obs_top20_share = obs_agg.head(20)["total_predicted_log_mv_cost"].sum() / max(obs_total, 1e-12)
    obs_hhi = ((obs_agg["total_predicted_log_mv_cost"] / max(obs_total, 1e-12)) ** 2).sum()
    null_rows = []
    for i in range(b):
        sim = top.copy()
        sim_parts = []
        for _, g in sim.groupby("dominant_layer", dropna=False):
            shuffled = g["partner_cusip"].to_numpy().copy()
            rng.shuffle(shuffled)
            gg = g.copy()
            gg["partner_cusip"] = shuffled
            sim_parts.append(gg)
        sim = pd.concat(sim_parts, ignore_index=True)
        agg = sim.groupby("partner_cusip").agg(total=("abs_loss", "sum")).sort_values("total", ascending=False)
        total = agg["total"].sum()
        null_rows.append({
            "simulation": i,
            "top20_share": agg.head(20)["total"].sum() / max(total, 1e-12),
            "hhi": ((agg["total"] / max(total, 1e-12)) ** 2).sum(),
        })
    null = pd.DataFrame(null_rows)
    out = pd.DataFrame([{
        "observed_top20_share": obs_top20_share,
        "null_mean_top20_share": null["top20_share"].mean(),
        "null_p95_top20_share": null["top20_share"].quantile(0.95),
        "observed_hhi": obs_hhi,
        "null_mean_hhi": null["hhi"].mean(),
        "null_p95_hhi": null["hhi"].quantile(0.95),
        "n_simulations": b,
    }])
    out.to_csv(AGG_DIR / "null_model_concentration.csv", index=False)
    null.to_csv(AGG_DIR / "null_model_concentration_draws.csv", index=False)
    return out


def write_backtest(panel: pd.DataFrame, bundle: DataBundle) -> pd.DataFrame:
    e = bundle.edges.copy()
    normalize_cusip_columns(e, ["firm_i", "firm_j"])
    active = pd.concat([
        e[["firm_i", "year"]].rename(columns={"firm_i": "cusip"}),
        e[["firm_j", "year"]].rename(columns={"firm_j": "cusip"}),
    ]).dropna()
    last_year = active.groupby("cusip")["year"].max()

    fy = bundle.firm_year[["ult_parent_cusip", "year", "market_value"]].copy()
    normalize_cusip_columns(fy, ["ult_parent_cusip"])
    fy["log_mv"] = _safe_log(fy["market_value"])
    fy = fy.sort_values(["ult_parent_cusip", "year"])
    fy["log_mv_next"] = fy.groupby("ult_parent_cusip")["log_mv"].shift(-1)
    fy["delta_log_mv_next"] = fy["log_mv_next"] - fy["log_mv"]
    fy = fy[["ult_parent_cusip", "year", "delta_log_mv_next"]]

    test = panel[panel["year"] < 2017].copy()
    test["partner_last_year"] = test["partner_cusip"].map(last_year)
    test = test[test["partner_last_year"] == test["year"]]
    test = test.merge(fy, left_on=["focal_cusip", "year"], right_on=["ult_parent_cusip", "year"], how="left")
    test = test.dropna(subset=["delta_log_mv_next"])
    if test.empty:
        out = pd.DataFrame([{"n_exit_edges": 0}])
    else:
        test["rank_bucket"] = pd.cut(
            test["rank"], bins=[0, 5, 20, 10_000],
            labels=["top5", "rank6_20", "rank_gt20"], right=True,
        )
        out = test.groupby("rank_bucket", observed=True).agg(
            n_exit_edges=("partner_cusip", "size"),
            mean_next_log_mv_change=("delta_log_mv_next", "mean"),
            median_next_log_mv_change=("delta_log_mv_next", "median"),
            mean_predicted_loss=("predicted_log_mv_loss", "mean"),
        ).reset_index()
    out.to_csv(AGG_DIR / "backtest_results.csv", index=False)
    return out


def _rank_variant(panel_latest: pd.DataFrame, value_col: str) -> pd.DataFrame:
    tmp = panel_latest.copy()
    tmp["variant_loss"] = tmp[value_col]
    tmp = tmp.sort_values(["focal_cusip", "variant_loss"], ascending=[True, True])
    tmp["rank"] = tmp.groupby("focal_cusip").cumcount() + 1
    tmp["predicted_log_mv_loss"] = tmp["variant_loss"]
    return aggregate_edges(tmp)


def write_rank_robustness(panel_latest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = {
        "empirical_primary_partner_specific": "predicted_log_mv_loss",
        "empirical_legacy_twfe_weight": "twfe_log_mv_loss",
        "stacked_weight": "stacked_log_mv_loss",
        "blended_stacked_twfe": "blended_stacked_twfe_log_mv_loss",
        "layer_only_no_partner_features": "layer_only_log_mv_loss",
        "no_redundancy_discount": "no_redundancy_log_mv_loss",
        "no_counterfactual_delta": "no_delta_log_mv_loss",
        "raw_degree_proxy": "raw_degree_log_mv_loss",
    }
    primary = aggregate_edges(panel_latest).set_index("partner_cusip")["rank"]
    rows = []
    ablations = []
    for name, col in variants.items():
        agg = _rank_variant(panel_latest, col)
        rank = agg.set_index("partner_cusip")["rank"]
        common = primary.index.intersection(rank.index)
        rho = primary.loc[common].corr(rank.loc[common], method="spearman") if len(common) > 2 else np.nan
        overlap = len(set(primary.head(20).index) & set(rank.head(20).index))
        row = {
            "variant": name,
            "spearman_vs_primary": rho,
            "top20_overlap_vs_primary": overlap,
            "top20_churn_vs_primary": 20 - overlap,
            "distinct_partners": len(agg),
        }
        rows.append(row)
        if name not in (
            "empirical_primary_partner_specific",
            "empirical_legacy_twfe_weight",
            "stacked_weight",
            "blended_stacked_twfe",
        ):
            ablations.append(row)
    dmd = legacy_dmd_ranking()
    if len(dmd):
        rank = dmd.set_index("partner_cusip")["rank"]
        common = primary.index.intersection(rank.index)
        rho = primary.loc[common].corr(rank.loc[common], method="spearman") if len(common) > 2 else np.nan
        overlap = len(set(primary.head(20).index) & set(rank.head(20).index))
        rows.append({
            "variant": "dmd_only_legacy_available",
            "spearman_vs_primary": rho,
            "top20_overlap_vs_primary": overlap,
            "top20_churn_vs_primary": 20 - overlap,
            "distinct_partners": len(dmd),
        })
    robust = pd.DataFrame(rows)
    abl = pd.DataFrame(ablations)
    robust.to_csv(AGG_DIR / "estimator_robustness.csv", index=False)
    abl.to_csv(AGG_DIR / "ablation_results.csv", index=False)
    return robust, abl


def legacy_dmd_ranking() -> pd.DataFrame:
    """Aggregate available legacy per-firm DMD estimates for robustness.

    These estimates come from the older per-firm stress reports and are not
    used in the primary ranking.  They are retained only to satisfy the DMD
    robustness comparison and to quantify how different the DMD-only ranking
    is from the corrected systemic panel.
    """
    rows = []
    for d in OUTPUTS_ROOT.iterdir():
        if not d.is_dir() or d.name in ("logs", "aggregate"):
            continue
        focal = normalize_cusip(d.name)
        csv = d / "partner_vulnerability.csv"
        if focal is None or not csv.exists():
            continue
        try:
            df = pd.read_csv(csv, dtype=str)
        except Exception:
            continue
        if len(df) == 0 or "dmd_loss_log_mv" not in df.columns:
            continue
        df["partner_cusip"] = normalize_cusip_series(df["partner_cusip"])
        df["dmd_loss_log_mv"] = pd.to_numeric(df["dmd_loss_log_mv"], errors="coerce")
        df = df.dropna(subset=["partner_cusip", "dmd_loss_log_mv"])
        if df.empty:
            continue
        df["abs_loss"] = df["dmd_loss_log_mv"].abs()
        df = df.sort_values("abs_loss", ascending=False).head(TOP_K_PER_FIRM)
        for _, r in df.iterrows():
            rows.append({
                "focal_cusip": focal,
                "partner_cusip": r["partner_cusip"],
                "abs_loss": float(r["abs_loss"]),
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    agg = out.groupby("partner_cusip").agg(
        in_degree=("focal_cusip", "nunique"),
        total_cost=("abs_loss", "sum"),
    ).sort_values(["total_cost", "in_degree"], ascending=[False, False]).reset_index()
    agg["rank"] = np.arange(1, len(agg) + 1)
    agg.to_csv(AGG_DIR / "dmd_only_legacy_ranking.csv", index=False)
    return agg


def write_system_stress_test(panel_latest: pd.DataFrame, b: int = 200, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    top = panel_latest[panel_latest["rank"] <= TOP_K_PER_FIRM].copy()
    agg = aggregate_edges(panel_latest)
    partners = agg["partner_cusip"].dropna().unique()
    rows = []
    for n in [5, 10, 20, 50]:
        n_eff = min(n, len(partners))
        sets = {
            "systemic_top_cost": agg.head(n_eff)["partner_cusip"].tolist(),
            "high_in_degree": agg.sort_values("in_degree", ascending=False).head(n_eff)["partner_cusip"].tolist(),
            "high_market_value": agg.sort_values("partner_market_value", ascending=False).head(n_eff)["partner_cusip"].tolist(),
            "high_degree": agg.sort_values("partner_degree", ascending=False).head(n_eff)["partner_cusip"].tolist(),
            "high_eigenvector": agg.sort_values("partner_eigenvector", ascending=False).head(n_eff)["partner_cusip"].tolist(),
        }
        random_losses = []
        for _ in range(b):
            draw = rng.choice(partners, size=n_eff, replace=False)
            random_losses.append(top.loc[top["partner_cusip"].isin(draw), "predicted_dollar_loss"].sum())
        rand_mean = float(np.nanmean(random_losses))
        rand_p95 = float(np.nanpercentile(random_losses, 95))
        for label, selected in sets.items():
            loss = top.loc[top["partner_cusip"].isin(selected), "predicted_dollar_loss"].sum()
            rows.append({
                "n_removed": n_eff,
                "strategy": label,
                "total_predicted_dollar_loss": loss,
                "random_mean_loss": rand_mean,
                "random_p95_loss": rand_p95,
                "multiple_of_random_mean": loss / rand_mean if rand_mean else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(AGG_DIR / "system_stress_test.csv", index=False)
    return out


def write_master_summary(
    panel: pd.DataFrame,
    latest_edges: pd.DataFrame,
    agg: pd.DataFrame,
    audit: pd.DataFrame,
    topk: pd.DataFrame,
    null_model: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    report_row = audit[audit["year"].eq("all_report_dirs")].iloc[0]
    latest_focals = latest_edges["focal_cusip"].nunique()
    md = [
        "# Systemic-Criticality Meta-Network - Master Summary",
        "",
        f"- **Report directories generated**: {int(report_row['total_report_dirs']):,d}",
        f"- **Non-empty legacy 2017 vulnerability CSVs**: {int(report_row['nonempty_partner_csv']):,d}",
        f"- **Active 2017 Compustat focal firms in rebuilt panel**: {latest_focals:,d}",
        f"- **2017 candidate focal-partner edges**: {len(latest_edges):,d}",
        f"- **2017 top-{TOP_K_PER_FIRM} critical edges**: "
        f"{int((latest_edges['rank'] <= TOP_K_PER_FIRM).sum()):,d}",
        f"- **Distinct 2017 critical partners**: {len(agg):,d}",
        f"- **Duplicate normalized partner rows**: {len(agg) - agg['partner_cusip'].nunique():,d}",
        "",
        "## Top 20 Systemic-Critical Firms",
        "",
    ]
    cols = [
        "rank", "partner_cusip", "partner_name", "partner_sic2",
        "is_compustat", "in_degree", "total_predicted_log_mv_cost",
        "mean_predicted_log_mv_cost",
    ]
    md.append(agg[cols].head(20).to_markdown(index=False))
    md.extend([
        "",
        "## Top-k Sensitivity",
        "",
        topk.to_markdown(index=False),
        "",
        "## Null Model",
        "",
        null_model.to_markdown(index=False),
        "",
        "## Estimator And Ablation Robustness",
        "",
        robustness.to_markdown(index=False),
        "",
        "## Interpretation Caveats",
        "",
        "- `predicted_log_mv_loss` is partner-specific: it combines layer exit weights, dyad tenure, tie strength, redundancy, partner centrality, and first-order counterfactual betweenness exposure.",
        "- The `delta_L*_btw` fields are exposure proxies, not exact all-pairs betweenness recomputations for every possible partner deletion.",
        "- Stacked-cohort estimates are the conservative causal reference where available; legacy TWFE estimates are retained as a comparison.",
    ])
    (AGG_DIR / "systemic_criticality.md").write_text("\n".join(md))


def main() -> None:
    print("Loading DataBundle ...")
    bundle = load_all()
    weights = load_layer_weights()
    print("Layer weights:")
    print(weights.to_string(index=False))

    print("Building annual critical-edge panel ...")
    panel = build_critical_edges_panel(bundle, weights)
    if panel.empty:
        raise RuntimeError("No systemic-criticality panel rows were produced.")

    panel_path = AGG_DIR / "critical_edges_panel.parquet"
    panel.to_parquet(panel_path, index=False)
    print(f"  wrote {panel_path} ({len(panel):,d} rows)")

    latest = panel[panel["year"] == 2017].copy()
    latest_top = latest[latest["rank"] <= TOP_K_PER_FIRM].copy()
    latest_top["empirical_loss_log_mv"] = latest_top["predicted_log_mv_loss"]
    latest_top["dmd_loss_log_mv"] = np.nan
    latest_top["dmd_available"] = False
    latest_top.to_csv(AGG_DIR / "critical_edges.csv", index=False)
    print(f"  wrote critical_edges.csv ({len(latest_top):,d} rows)")

    agg = aggregate_edges(latest)
    agg.to_csv(AGG_DIR / "systemic_criticality.csv", index=False)
    print(f"  wrote systemic_criticality.csv ({len(agg):,d} partners)")

    annual, cumulative = write_annual_and_cumulative(panel)
    audit = write_identity_coverage_audit(panel, bundle)
    topk = write_topk_sensitivity(latest)
    null_model = write_null_model(latest)
    backtest = write_backtest(panel, bundle)
    robustness, _ = write_rank_robustness(latest)
    stress = write_system_stress_test(latest)
    write_master_summary(panel, latest, agg, audit, topk, null_model, robustness)

    print("\n=== Identity/Coverage Audit ===")
    print(audit.tail(5).to_string(index=False))
    print("\n=== Top 20 systemic-critical firms, 2017 ===")
    print(agg[[
        "rank", "partner_name", "partner_sic2", "is_compustat",
        "in_degree", "total_predicted_log_mv_cost",
    ]].head(20).to_string(index=False))
    print("\n=== Top-k sensitivity ===")
    print(topk.to_string(index=False))
    print("\n=== Backtest summary ===")
    print(backtest.to_string(index=False))
    print("\n=== Stress test summary ===")
    print(stress.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

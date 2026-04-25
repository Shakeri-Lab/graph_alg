"""Scoring primitives shared by all three decision modules.

Provides:
  - closure_score(focal, candidate, layer, year): triadic closure
  - brokerage_score(focal, candidate, layer, year): non-redundant reach
  - rd_gate(focal, year): R&D top-quartile-within-SIC boolean
  - partner_exit_impact(focal, partner, year): (empirical, DMD) estimates
"""

from __future__ import annotations
from functools import lru_cache
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from strategic_pipeline.data_loader import (
    DataBundle, load_all,
    get_firm_partners, dmd_eligible,
)
from strategic_pipeline.firm_profile import compute_rd_quartile
from strategic_pipeline.id_utils import normalize_cusip

# Empirical event-study magnitude: 10.6% MV drop at t+1 for L2 sudden exit
# (from Manuscript 2 TWFE event study, p=0.002)
EMPIRICAL_L2_T1_PCT = -0.106   # log-level change ≡ -10.6% market value

# For non-L2 layers: empirical event study was L2-specific.  We approximate
# by scaling by the ratio of within-firm L2 share to total share.  In
# absence of layer-specific event studies for L1/L3/L4, we report a
# baseline of 30% of the L2 magnitude for non-L2 ties as a conservative
# lower bound, tagged with a caveat in the report.
NON_L2_SCALING = 0.30

ROLLING_WINDOW = 5


# ══════════════════════════════════════════════════════════════════
# Closure / brokerage
# ══════════════════════════════════════════════════════════════════

def closure_score(bundle: DataBundle, focal_cusip: str,
                   candidate_cusip: str, layer: str, year: int) -> float:
    r"""Fraction of focal's current layer-$\ell$ partners that are
    ALSO layer-$\ell$ partners of the candidate, in the 5-yr window
    ending at `year`.  High -> dense closed triad (Ahuja closure).

    Returns 0 if focal has no layer-$\ell$ partners.
    """
    window_start = year - ROLLING_WINDOW + 1
    focal_partners = get_firm_partners(bundle, focal_cusip,
                                        window_start, year, layer=layer)
    focal_partners.discard(candidate_cusip)
    if len(focal_partners) == 0:
        return 0.0
    cand_partners = get_firm_partners(bundle, candidate_cusip,
                                       window_start, year, layer=layer)
    overlap = focal_partners & cand_partners
    return len(overlap) / len(focal_partners)


def brokerage_score(bundle: DataBundle, focal_cusip: str,
                     candidate_cusip: str, layer: str, year: int) -> float:
    r"""Fraction of candidate's layer-$\ell$ neighborhood that is NOT in
    focal's layer-$\ell$ neighborhood, in the 5-yr window.  High ->
    candidate bridges to disconnected clusters (Burt brokerage).

    Returns 0 if candidate has no partners in the layer.
    """
    window_start = year - ROLLING_WINDOW + 1
    cand_partners = get_firm_partners(bundle, candidate_cusip,
                                       window_start, year, layer=layer)
    cand_partners.discard(focal_cusip)
    if len(cand_partners) == 0:
        return 0.0
    focal_partners = get_firm_partners(bundle, focal_cusip,
                                        window_start, year, layer=layer)
    non_overlap = cand_partners - focal_partners
    return len(non_overlap) / len(cand_partners)


# ══════════════════════════════════════════════════════════════════
# R&D gate
# ══════════════════════════════════════════════════════════════════

def rd_gate(bundle: DataBundle, focal_cusip: str, year: int) -> bool:
    """True if focal is in top-quartile baseline_rd within its 2-digit
    SIC in `year`.  Used to qualify L2 brokerage recommendations."""
    _, is_top = compute_rd_quartile(bundle, focal_cusip, year)
    return is_top


# ══════════════════════════════════════════════════════════════════
# Partner-exit impact — empirical + DMD side-by-side
# ══════════════════════════════════════════════════════════════════

def _layer_mix_in_dyad(bundle: DataBundle, focal: str, partner: str,
                        year: int, window: int = ROLLING_WINDOW) -> dict:
    """Return the fraction of the dyad's edges in each layer within the
    window, used to weight the layer-specific exit cost."""
    focal = normalize_cusip(focal)
    partner = normalize_cusip(partner)
    window_start = year - window + 1
    e = bundle.edges
    mask = (
        (((e["firm_i"] == focal) & (e["firm_j"] == partner))
         | ((e["firm_i"] == partner) & (e["firm_j"] == focal)))
        & (e["year"] >= window_start) & (e["year"] <= year)
    )
    sub = e[mask]
    if len(sub) == 0:
        return {}
    counts = sub["layer_code"].value_counts(normalize=True).to_dict()
    return counts


def empirical_exit_impact(bundle: DataBundle, focal: str, partner: str,
                           year: int) -> float:
    """Blend layer-specific empirical TWFE event-study magnitudes by
    the dyad's layer composition.  Returns signed log-level change at
    t+1 (negative = MV decline)."""
    focal = normalize_cusip(focal)
    partner = normalize_cusip(partner)
    mix = _layer_mix_in_dyad(bundle, focal, partner, year)
    if not mix:
        return 0.0
    impact = 0.0
    for L, frac in mix.items():
        if L == "L2":
            impact += frac * EMPIRICAL_L2_T1_PCT
        elif L in ("L1", "L3", "L4"):
            impact += frac * (EMPIRICAL_L2_T1_PCT * NON_L2_SCALING)
    return impact


def dmd_exit_impact(bundle: DataBundle, focal: str, partner: str,
                     year: int) -> Optional[float]:
    """Simulate a partner-exit on the focal firm's dominant layer-
    centrality coordinate via the shared DMD operator; return the log-MV
    response at k=2 (which aligns to empirical t+1 per Phase 4 shift).

    Returns None if focal is not DMD-eligible (<6 consec years).
    """
    focal = normalize_cusip(focal)
    partner = normalize_cusip(partner)
    if not dmd_eligible(bundle, focal):
        return None

    # Determine dominant layer of the dyad
    mix = _layer_mix_in_dyad(bundle, focal, partner, year)
    if not mix:
        return 0.0
    dom_layer = max(mix.items(), key=lambda kv: kv[1])[0]
    if dom_layer not in ("L1", "L2", "L3", "L4"):
        return 0.0

    # Lazy-load DMD operator, shock magnitude, and peak-match scale
    U_r, A_tilde, shock_mag, scale = _get_or_fit_dmd_operator()
    if U_r is None:
        return None

    from phase2_dmd import FEAT_IDX
    from phase4_causal_validation import intervention_irf

    src_idx = FEAT_IDX[f"{dom_layer}_btw"]
    tgt_idx = FEAT_IDX["log_mv"]

    # Apply calibrated negative shock (partner exit reduces c^(L)_lag0)
    ir = intervention_irf(U_r, A_tilde, src_idx, shock_mag, tgt_idx, K=4)
    # k=2 corresponds to empirical t+1 (per Phase 4 v2 alignment)
    return float(np.real(ir[2]) * scale)


@lru_cache(maxsize=1)
def _get_or_fit_dmd_operator():
    """Fit the shared DMD operator once per session on the 851-firm
    long panel.  Module-level cache (lru_cache with no args).

    Returns (U_r, A_tilde, shock_mag, scale) where:
      - shock_mag is the calibrated z-scored L2-decline magnitude
      - scale is the Phase-4-v2 peak-match factor mapping z-units to
        log-level MV change (empirical peak / DMD median at k=2).
    """
    b = load_all()

    try:
        from phase2_dmd import build_snapshot_matrices, fit_dmd, FEAT_Z
        from phase4_causal_validation import calibrate_shock_magnitude
    except Exception as e:
        print(f"  [warn] DMD operator unavailable: {e}")
        return (None, None, None, None)

    traj = b.trajectory.copy()
    for c in FEAT_Z:
        traj[c] = traj[c].fillna(0.0)

    long_firms = set(b.long_firms)
    X, Y, _ = build_snapshot_matrices(traj, long_firms)
    U_r, s_r, A_tilde, eigvals, W, Phi = fit_dmd(X, Y, r=25)

    shock_mag = calibrate_shock_magnitude(traj, list(long_firms))
    # Phase-4-v2 peak-match calibration (empirical t+1 / DMD k=2 median)
    scale = 2.4333

    return (U_r, A_tilde, shock_mag, scale)


def partner_exit_impact(bundle: DataBundle, focal: str, partner: str,
                         year: int = 2017) -> dict:
    """Return dict with both empirical and DMD estimates of focal's
    log_MV response at t+1 to partner's exit."""
    return {
        "empirical_twfe_pct": empirical_exit_impact(bundle, focal, partner, year),
        "dmd_simulated_pct": dmd_exit_impact(bundle, focal, partner, year),
        "dmd_available": dmd_eligible(bundle, focal),
    }


# ══════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    b = load_all()
    focal = "747525"  # Qualcomm
    year = 2005
    partners_L2 = list(get_firm_partners(b, focal, 2001, year, layer="L2"))
    print(f"Qualcomm L2 partners (2001–2005): {len(partners_L2)}")
    for p in partners_L2[:3]:
        print(f"  partner {p} ({b.firm_name(p)})")
        print(f"    closure (L2): {closure_score(b, focal, p, 'L2', year):.3f}")
        print(f"    brokerage (L2): {brokerage_score(b, focal, p, 'L2', year):.3f}")
        impact = partner_exit_impact(b, focal, p, year)
        emp = impact['empirical_twfe_pct']
        dmd = impact['dmd_simulated_pct']
        print(f"    exit impact (log MV): empirical={emp:+.4f}, "
              f"dmd={'n/a' if dmd is None else f'{dmd:+.4f}'}")

    print(f"\n  R&D top-quartile gate: {rd_gate(b, focal, year)}")

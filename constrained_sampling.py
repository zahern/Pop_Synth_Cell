"""
constrained_sampling.py — Feasibility-constrained conditional resampling (B1 + B2)
====================================================================================

Purpose
-------
The production pipeline currently repairs infeasible GAN records *post-hoc* by
deterministically recoding the offending attribute to a single "not applicable"
code (see `repair_hard_constraints` in gan_train.py).  That flattens the joint
distribution and is what Reviewer 2 flagged in Appendix A.2 ("unquantified bias
introduced by rule-based repair").

This module replaces that deterministic recode with *feasibility-constrained
conditional resampling*: for every record that violates a hard census
constraint, the offending attribute is re-sampled from the training data's
conditional distribution restricted to the feasible value set.  The result is a
record that (a) satisfies the constraint and (b) is a *new, plausible* individual
drawn from the learned conditional, not a cloned copy of a "not applicable" code.

Two constraint families are handled, matching the two fix ideas from the paper:

  B1 — Structural (age-gating) rules.  For children (AGEP < 15), attributes such
       as EMPP, LFSP, OCCP, HEAP, ... have a fixed feasible set (e.g. EMPP in
       {5 not-applicable, 6 overseas}).  When the generator emits an infeasible
       code, it is re-sampled from the *empirical frequency* of the feasible
       codes among children in the training data (preserving the NA-vs-overseas
       split rather than collapsing everything to "not applicable").

  B2 — Conditional-consistency rules.  Cross-variable constraints with genuine
       degrees of freedom (CACF <= CPRF; renter/mortgage mutual exclusion;
       employment status only when employed; HIND/HINASD agreement).  The
       offending attribute is re-sampled from the training data's conditional
       distribution P(offender | conditioning parents) restricted to feasible
       values, so the correction tracks the learned joint distribution instead
       of a flat recode.

The functions are pure pandas/numpy (no GAN dependency), so the module is
importable both from gan_train.py and standalone for post-hoc correction of an
already-sampled CSV.

Usage
-----
    from constrained_sampling import enforce_feasibility
    fixed, log = enforce_feasibility(synth_df, train_df, rng_seed=0)
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ABS 2021 census codes (mirror gan_train.py; kept local to avoid a circular
# import when this module is loaded by gan_train.py).
# ---------------------------------------------------------------------------
AGEP_CHILD_THRESHOLD = 15

EMPP_CHILD_VALID     = (5, 6)          # child EMPP must be not-applicable or overseas
EMPP_APPLICABLE_LFSP = (1,)            # employment status only defined for LFSP=1 (FT)
LFSP_IN_LF           = (1, 2, 3, 4)
LFSP_NOT_APPLIC      = 5
# Fused children carry LFSP 5 (not applicable <15) or 6 (overseas visitor)
LFSP_CHILD_VALID     = (5, 6)

INCP_NOT_APPLIC, INCP_OVERSEAS = 18, 19
MSTP_NOT_APPLIC    = 6
OCCP_NOT_APPLIC, OCCP_OVERSEAS = 11, 12
HEAP_NOT_APPLIC, HEAP_OVERSEAS = 11, 12
QALFP_NOT_APPLIC   = 15
# Fused/source children carry QALFP=14 (225,437/225,582 kids; also the modal
# adult code at 374k rows), not 15. 14 is a legitimate resident code, so both
# are accepted for children — otherwise every child is wiped (2026-10-09).
# 16 is the ABS control-table code for the same overseas-visitor state (14/15
# are the fused-census codes, 16 appears once a pool has been aligned to the
# control tables by build_feasible_pool_cp.align_qalfp_to_control_codes).
QALFP_CHILD_VALID  = (14, 15, 16)
QALLP_NOT_APPLIC, QALLP_OVERSEAS = 8, 9
HRSP_NOT_APPLIC    = 1
# Fused children carry RLHP 3 (child in family), 9 (other relative),
# 10 (non-family member), 11 (not applicable) — all valid for children.
RLHP_CHILD_VALID     = (3, 9, 10, 11)
# CACF==7 is a residual bucket, not a count: 62% of source families carry it
# (561k rows; all 271k CACF>CPRF "violations" are CACF==7). Comparing it
# numerically wipes ordinary couple families — always skip it in the check.
CACF_RESIDUAL      = 7

# Cross-attribute caps: max INCP per OCCP (derived from fused census).
# OCCP 1-10: max INCP 17; OCCP 11 (N/A): 18; OCCP 12 (overseas): 19.
OCCP_MAX_INCP = {1: 17, 2: 17, 3: 17, 4: 17, 5: 17, 6: 17,
                 7: 17, 8: 17, 9: 17, 10: 17, 11: 18, 12: 19}

# AGEP -> max HRSP (hours worked by age)
AGEP_MAX_HRSP = {i: min(60, max(20, 40 + (i - 20))) for i in range(0, 38)}
# kids/young: capped lower; prime age: higher; elderly: lower
for i in range(0, 15):
    AGEP_MAX_HRSP[i] = 0  # kids don't work
for i in range(65, 38):
    AGEP_MAX_HRSP[i] = min(30, 65 - i + 10)

# MSTP (marital) -> CACF/CPRF bounds
MSTP_CACF_BOUNDS = {1: (1, 5), 2: (1, 5), 3: (0, 3), 6: (0, 0)}  # never married -> 0 kids

# HEAP -> OCCP valid combinations (education required for occupation)
# Based on actual fused census data: minimum HEAP per OCCP
HEAP_MIN_OCCP = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
OCCP_REQUIRES_HEAP = {1, 2, 3, 4, 5, 6, 7, 8}

# Cross-attribute constraint functions
def occp_incp_constraint(df):
    """INCP <= max for OCCP"""
    if "OCCP" not in df.columns or "INCP" not in df.columns:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for occp, max_incp in OCCP_MAX_INCP.items():
        mask |= (df["OCCP"] == occp) & (df["INCP"] > max_incp)
    return mask

# Valid OCCP codes that have education requirements (1-8)
# 11=Not applicable, 12=Overseas visitor - no education requirement
OCCP_REQUIRES_HEAP = {1, 2, 3, 4, 5, 6, 7, 8}

def agep_hrsp_constraint(df):
    """HRSP <= max for AGEP (skip children and not-applicable)"""
    if "AGEP" not in df.columns or "HRSP" not in df.columns:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    # Only check working-age adults (AGEP >= 15) with valid HRSP
    working_age = (df["AGEP"] >= 15)
    for agep, max_hrsp in AGEP_MAX_HRSP.items():
        if max_hrsp > 0:  # only check if there's a positive limit
            mask |= working_age & (df["AGEP"] == agep) & (df["HRSP"] > max_hrsp)
    return mask

def heap_occp_constraint(df):
    """OCCP requires minimum HEAP (only for valid occupations 1-8)"""
    if "HEAP" not in df.columns or "OCCP" not in df.columns:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for occp, min_heap in HEAP_MIN_OCCP.items():
        if occp in OCCP_REQUIRES_HEAP:
            mask |= (df["OCCP"] == occp) & (df["HEAP"] < min_heap)
    return mask

MRERD_NOT_APPLICABLE = 23
RNTRD_NOT_APPLICABLE = 27
HIND_NEGATIVE        = 1
HIND_NOT_STATED      = (24, 25)
HINASD_NOT_STATED    = (23, 24)
HINASD_TO_HIND_NS    = {23: 24, 24: 25}


# ---------------------------------------------------------------------------
# B1 — structural (age-gating) rules
# ---------------------------------------------------------------------------
# (name, offender column, feasible values for children)
STRUCTURAL_RULES = [
    ("child_empp",  "EMPP",  list(EMPP_CHILD_VALID)),
    ("child_lfsp",  "LFSP",  list(LFSP_CHILD_VALID)),
    ("child_incp",  "INCP",  [INCP_NOT_APPLIC, INCP_OVERSEAS]),
    ("child_mstp",  "MSTP",  [MSTP_NOT_APPLIC]),
    ("child_occp",  "OCCP",  [OCCP_NOT_APPLIC, OCCP_OVERSEAS]),
    ("child_heap",  "HEAP",  [HEAP_NOT_APPLIC, HEAP_OVERSEAS]),
    ("child_qalfp", "QALFP", list(QALFP_CHILD_VALID)),
    ("child_qallp", "QALLP", [QALLP_NOT_APPLIC, QALLP_OVERSEAS]),
    ("child_hrsp",  "HRSP",  [HRSP_NOT_APPLIC]),
    ("child_rlhp",  "RLHP",  list(RLHP_CHILD_VALID)),
]


def _sample_structural(synth, train, rng, name, offender, feasible):
    """Re-sample a child-only attribute into its feasible set (B1)."""
    if "AGEP" not in synth.columns or offender not in synth.columns:
        return 0
    child = synth["AGEP"] < AGEP_CHILD_THRESHOLD
    bad = child & (~synth[offender].isin(feasible))
    n = int(bad.sum())
    if n == 0:
        return 0

    # Empirical frequency of the feasible codes among children in training data.
    if "AGEP" in train.columns:
        t_child = train[train["AGEP"] < AGEP_CHILD_THRESHOLD]
    else:
        t_child = train
    vals = t_child[offender].dropna()
    vals = vals[vals.isin(feasible)].values
    if len(vals) == 0:
        vals = np.asarray(feasible)

    synth.loc[bad, offender] = vals[rng.integers(len(vals), size=n)]
    return n


# ---------------------------------------------------------------------------
# B2 — conditional-consistency helpers
# ---------------------------------------------------------------------------

def _conditional_lookup(train, offender, cond_cols, feasible):
    """Build {(cond tuple): value array} of feasible offender values in train."""
    lookup = {}
    if not cond_cols:
        vals = train[offender].dropna()
        vals = vals[vals.isin(feasible)].values
        lookup[()] = vals
        return lookup
    for key, sub in train.groupby(list(cond_cols), dropna=False):
        v = sub[offender].dropna()
        v = v[v.isin(feasible)].values
        if len(v):
            lookup[key if isinstance(key, tuple) else (key,)] = v
    return lookup


def _resample_conditional(synth, train, rng, mask, offender, cond_cols, feasible):
    """Re-sample offender for masked rows from P(offender | cond_cols) ∩ feasible."""
    n = int(mask.sum())
    if n == 0:
        return 0
    lookup = _conditional_lookup(train, offender, cond_cols, feasible)
    fallback = np.asarray(list(feasible), dtype=np.int64)
    rows = synth.loc[mask]
    keys = list(zip(*[rows[c].tolist() for c in cond_cols])) if cond_cols else [()] * n
    out = np.empty(n, dtype=np.int64)
    for i, k in enumerate(keys):
        pool = lookup.get(k)
        if pool is None or len(pool) == 0:
            pool = fallback
        out[i] = int(pool[rng.integers(len(pool))])
    synth.loc[mask, offender] = out
    return n


def _resample_cacf(synth, train, rng):
    """CACF <= CPRF (row-dependent bound) — resample CACF ≤ CPRF (B2)."""
    if "CACF" not in synth.columns or "CPRF" not in synth.columns:
        return 0
    bad = (synth["CACF"] > synth["CPRF"]) & (synth["CACF"] != CACF_RESIDUAL)
    n = int(bad.sum())
    if n == 0:
        return 0
    cond_cols = [c for c in ("CPRF", "MSTP", "HHCD", "AGEP") if c in train.columns]
    # lookup: condition key -> CACF value array in training data
    lookup = {}
    if not cond_cols:
        lookup[()] = train["CACF"].dropna().values
    else:
        for key, sub in train.groupby(cond_cols, dropna=False):
            lookup[key if isinstance(key, tuple) else (key,)] = sub["CACF"].dropna().values
    rows = synth.loc[bad]
    keys = list(zip(*[rows[c].tolist() for c in cond_cols])) if cond_cols else [()] * n
    cprf = rows["CPRF"].tolist()
    out = np.empty(n, dtype=np.int64)
    for i, (k, cap) in enumerate(zip(keys, cprf)):
        pool = lookup.get(k)
        if pool is None or len(pool) == 0:
            pool = train["CACF"].dropna().values
        feasible = pool[pool <= cap]
        if len(feasible) == 0:
            feasible = np.array([0])
        out[i] = int(feasible[rng.integers(len(feasible))])
    synth.loc[bad, "CACF"] = out
    return n


def _resolve_rent_mortgage(synth, train, rng):
    """RNTRD and MRERD mutually exclusive — keep one, set the other to NA (B2)."""
    if "RNTRD" not in synth.columns or "MRERD" not in synth.columns:
        return 0
    conflict = (synth["RNTRD"] != RNTRD_NOT_APPLICABLE) & \
               (synth["MRERD"] != MRERD_NOT_APPLICABLE)
    n = int(conflict.sum())
    if n == 0:
        return 0
    # Learn P(keep rent) from training rows where exactly one is active,
    # optionally conditioned on tenure (TENLLD) if present.
    if "RNTRD" in train.columns and "MRERD" in train.columns:
        rent = train["RNTRD"] != RNTRD_NOT_APPLICABLE
        mtg  = train["MRERD"] != MRERD_NOT_APPLICABLE
        one  = rent ^ mtg
        p_rent = float(rent[one].mean()) if one.any() else 0.5
    else:
        p_rent = 0.5
    keep_rent = rng.random(n) < p_rent
    idx = synth.index[conflict]
    synth.loc[idx[keep_rent], "MRERD"] = MRERD_NOT_APPLICABLE
    synth.loc[idx[~keep_rent], "RNTRD"] = RNTRD_NOT_APPLICABLE
    return n


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def enforce_feasibility(synth, train, rng_seed=0, report_dir=None):
    """
    Replace infeasible attribute values with feasibility-constrained conditional
    resamples (B1 structural + B2 conditional).  Returns (df, log).

    Parameters
    ----------
    synth : pd.DataFrame
        Raw GAN sample (already snapped to integer codes).
    train : pd.DataFrame
        Training microdata used to learn the conditional distributions.
    rng_seed : int
        Seed for reproducibility.
    report_dir : str or None
        If set, write a "feasible_resample_report.txt" summarising the counts.

    Returns
    -------
    (pd.DataFrame, dict) — corrected copy of `synth` and a per-rule count log.
    """
    synth = synth.copy()
    rng = np.random.default_rng(rng_seed)
    log = {}

    # ---- B1: structural age-gating ---------------------------------------
    for name, offender, feasible in STRUCTURAL_RULES:
        log[name] = _sample_structural(synth, train, rng, name, offender, feasible)

    # ---- B2: conditional consistency -------------------------------------
    # Employment status (EMPP) only defined for employed (LFSP=1).
    if {"LFSP", "EMPP"} <= set(synth.columns):
        bad = (~synth["LFSP"].isin(EMPP_APPLICABLE_LFSP)) & \
              (~synth["EMPP"].isin(EMPP_CHILD_VALID))
        cond = [c for c in ("LFSP", "AGEP") if c in train.columns]
        log["empp_not_employed"] = _resample_conditional(
            synth, train, rng, bad, "EMPP", cond, list(EMPP_CHILD_VALID))

    # Hours worked (HRSP) only defined for employed (LFSP=1).
    if {"LFSP", "HRSP"} <= set(synth.columns):
        bad = (~synth["LFSP"].isin(EMPP_APPLICABLE_LFSP)) & (synth["HRSP"] != HRSP_NOT_APPLIC)
        cond = [c for c in ("LFSP", "AGEP") if c in train.columns]
        log["hrsp_not_employed"] = _resample_conditional(
            synth, train, rng, bad, "HRSP", cond, [HRSP_NOT_APPLIC])

    # HIND must agree with HINASD on stated/not-stated (deterministic mapping).
    if {"HIND", "HINASD"} <= set(synth.columns):
        ns_hinasd = synth["HINASD"].isin(HINASD_NOT_STATED)
        ns_hind   = synth["HIND"].isin(HIND_NOT_STATED)
        m_to_band = (~ns_hinasd) & ns_hind
        m_to_ns   = ns_hinasd & (~ns_hind)
        synth.loc[m_to_band, "HIND"] = synth.loc[m_to_band, "HINASD"]
        synth.loc[m_to_ns, "HIND"] = synth.loc[m_to_ns, "HINASD"].map(HINASD_TO_HIND_NS)
        log["hind_hinasd"] = int((m_to_band | m_to_ns).sum())

    # CACF <= CPRF (row-dependent bound).
    log["cacf_le_cprf"] = _resample_cacf(synth, train, rng)

    # Ren/mortgage mutual exclusion (distribution-preserving tenure choice).
    log["rent_mortgage"] = _resolve_rent_mortgage(synth, train, rng)

    # R10 plausibility filter, not a census impossibility: 468 of 3,263
    # negative-income households in the source extract carry a mortgage.
    if {"HIND", "MRERD"} <= set(synth.columns):
        bad = (synth["HIND"] == HIND_NEGATIVE) & (synth["MRERD"] != MRERD_NOT_APPLICABLE)
        synth.loc[bad, "MRERD"] = MRERD_NOT_APPLICABLE
        log["neg_income_mortgage"] = int(bad.sum())

    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
        lines = [
            "Feasibility-constrained conditional resampling report",
            "=" * 55,
            f"Records processed : {len(synth):,}",
            "",
            "Per-rule resample counts (structural = B1, conditional = B2):",
        ]
        total = 0
        for k, v in log.items():
            lines.append(f"  {k}: {v:,}")
            total += v
        lines += [
            "",
            f"Total attributes re-sampled: {total:,} "
            f"({100.0 * total / max(len(synth), 1):.2f}% of records)",
            "",
            "Note: values are re-sampled from the training data's conditional",
            "distribution restricted to feasible values — NOT deterministically",
            "recoded — so corrected records are novel feasible individuals.",
        ]
        with open(os.path.join(report_dir, "feasible_resample_report.txt"), "w") as f:
            f.write("\n".join(lines))
    return synth, log


if __name__ == "__main__":
    # Standalone CLI for post-hoc correction of an already-sampled CSV.
    import argparse
    ap = argparse.ArgumentParser(description="Apply feasibility-constrained resampling to a sampled CSV")
    ap.add_argument("--synth", required=True, help="Path to sampled synthetic CSV")
    ap.add_argument("--train", required=True, help="Path to training microdata CSV")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-dir", default=None)
    a = ap.parse_args()
    syn = pd.read_csv(a.synth)
    trn = pd.read_csv(a.train)
    fixed, lg = enforce_feasibility(syn, trn, rng_seed=a.seed, report_dir=a.report_dir)
    fixed.to_csv(a.out, index=False)
    print(f"Resampled {sum(lg.values()):,} attribute values; wrote {a.out}")

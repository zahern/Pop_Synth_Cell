"""
build_feasible_pool_cp.py — constraint-programming-style feasible-population pool
==================================================================================

Alternative to the GAN (Stage 1).  Instead of learning a continuous latent
distribution and then repairing infeasible samples, this module builds the
candidate pool *directly* from the entire census microdata as a constraint
satisfaction / enumeration problem:

  1. Variables & domains   -- each attribute's feasible category set, taken from
                              the ABS coding (domains) present in the census.
  2. Constraints           -- the 11 census rules (child age-gating, employment
                              gating, family size CACF<=CPRF, tenure mutual
                              exclusion, income/mortgage) expressed as explicit
                              boolean predicates.
  3. Enumeration           -- the feasible cell space is the union of:
                              (a) every observed census record (feasible by
                                  construction), and
                              (b) structurally-feasible-but-unobserved cells
                                  ("sampling zeros") reconstructed by
                                  constraint-guided conditional resampling from
                                  the census's own conditional distributions.

The output is a *complete feasible pool* (person + household attribute vectors
with a weight column) that the SA stage can select from directly.  Because every
record is generated subject to the same constraints, no post-hoc repair is ever
needed -- which addresses the reviewer concern that the GAN pipeline "boils down
to a list of manual constraints": here the constraints are the method.

Key differences vs the GAN path
-------------------------------
  - Feasibility is guaranteed *by construction*, not by repair.
  - The pool is cell-complete over the modelled attributes (sampling zeros filled).
  - "Novelty" comes from constraint-consistent recombination of observed
    conditionals, not from a learned continuous distribution; there is therefore
    no out-of-sample learned-joint claim (the GAN's specific contribution).

Usage
-----
  python build_feasible_pool_cp.py --census LINKED_DATA_national.csv \
         --control-attrs sexp agep mstp heap occp incp lfsp empp hrsp qallp qalfp rlhp mtwp nprd vehrd mrerd rntrd strd hind bedrd hhcd \
         --min-count 5 --out outputs/feasible_pool.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config  # noqa: E402  (sibling module; path set above)

# Reuse the feasibility rules / codes and the conditional-resampling machinery.
try:
    from constrained_sampling import (
        enforce_feasibility, STRUCTURAL_RULES,
        AGEP_CHILD_THRESHOLD, EMPP_CHILD_VALID, EMPP_APPLICABLE_LFSP,
        MRERD_NOT_APPLICABLE, RNTRD_NOT_APPLICABLE, HIND_NEGATIVE,
        LFSP_CHILD_VALID, HRSP_NOT_APPLIC, MSTP_NOT_APPLIC,
        INCP_NOT_APPLIC, INCP_OVERSEAS, OCCP_NOT_APPLIC, OCCP_OVERSEAS,
        HEAP_NOT_APPLIC, HEAP_OVERSEAS, QALFP_CHILD_VALID, QALLP_NOT_APPLIC,
        QALLP_OVERSEAS, RLHP_CHILD_VALID, CACF_RESIDUAL,
        LFSP_CHILD_VALID, HRSP_NOT_APPLIC, MSTP_NOT_APPLIC,
        OCCP_MAX_INCP, AGEP_MAX_HRSP, HEAP_MIN_OCCP,
        occp_incp_constraint, agep_hrsp_constraint, heap_occp_constraint,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"constrained_sampling import failed: {exc}")


# ---------------------------------------------------------------------------
# Constraint model
# ---------------------------------------------------------------------------
# Each constraint is a predicate over a person/household row returning True when
# the row is INFEASIBLE (i.e. a violation).  This is the same rule set enforced
# by the GAN path, restated here as the CP constraint store.

def _child_mask(df):
    return df["AGEP"] < AGEP_CHILD_THRESHOLD


CONSTRAINTS = {
    "child_empp":   lambda d: _child_mask(d) & (~d["EMPP"].isin(EMPP_CHILD_VALID)),
    "child_lfsp":   lambda d: _child_mask(d) & (~d["LFSP"].isin(LFSP_CHILD_VALID)),
    "child_incp":   lambda d: _child_mask(d) & (~d["INCP"].isin([INCP_NOT_APPLIC, INCP_OVERSEAS])),
    "child_mstp":   lambda d: _child_mask(d) & (d["MSTP"] != MSTP_NOT_APPLIC),
    "child_occp":   lambda d: _child_mask(d) & (~d["OCCP"].isin([OCCP_NOT_APPLIC, OCCP_OVERSEAS])),
    "child_heap":   lambda d: _child_mask(d) & (~d["HEAP"].isin([HEAP_NOT_APPLIC, HEAP_OVERSEAS])),
    "child_qalfp":  lambda d: _child_mask(d) & (~d["QALFP"].isin(QALFP_CHILD_VALID)),
    "child_qallp":  lambda d: _child_mask(d) & (~d["QALLP"].isin([QALLP_NOT_APPLIC, QALLP_OVERSEAS])),
    "child_hrsp":   lambda d: _child_mask(d) & (d["HRSP"] != HRSP_NOT_APPLIC),
    "child_rlhp":   lambda d: _child_mask(d) & (~d["RLHP"].isin(RLHP_CHILD_VALID)),
    "empp_gate":    lambda d: (~d["LFSP"].isin(EMPP_APPLICABLE_LFSP)) & (~d["EMPP"].isin(EMPP_CHILD_VALID)),
    "hrsp_gate":    lambda d: (~d["LFSP"].isin(EMPP_APPLICABLE_LFSP)) & (d["HRSP"] != HRSP_NOT_APPLIC),
    "cacf_le_cprf": lambda d: ((d["CACF"] > d["CPRF"]) & (d["CACF"] != CACF_RESIDUAL)) if "CACF" in d.columns and "CPRF" in d.columns else pd.Series(False, index=d.index),
    "rent_x_mort":  lambda d: (d["RNTRD"] != RNTRD_NOT_APPLICABLE) & (d["MRERD"] != MRERD_NOT_APPLICABLE),
    "neg_inc_mort": lambda d: (d["HIND"] == HIND_NEGATIVE) & (d["MRERD"] != MRERD_NOT_APPLICABLE),
    # Cross-attribute caps
    "occp_incp":    lambda d: occp_incp_constraint(d) if "OCCP" in d.columns and "INCP" in d.columns else pd.Series(False, index=d.index),
    "agep_hrsp":    lambda d: agep_hrsp_constraint(d) if "AGEP" in d.columns and "HRSP" in d.columns else pd.Series(False, index=d.index),
    "heap_occp":    lambda d: heap_occp_constraint(d) if "HEAP" in d.columns and "OCCP" in d.columns else pd.Series(False, index=d.index),
}


# ---------------------------------------------------------------------------
# Control-table code alignment
# ---------------------------------------------------------------------------
# The fused census stores QALFP on a 15-code scale that is one category short
# of the ABS control tables above code 11: codes 1-11 are the same fields of
# study, but from 12 upward the census is one code behind, so children and
# other non-applicable persons carry 14 in the census where the ABS table says
# 15, and "not stated" is 13 where the table says 14.
#
# Measured against the national control marginals (L1 on proportions):
#   identity 0.997 | shift>=11 0.102 | shift>=12 0.053 | shift>=13 0.065
#   shift>=14 0.180 | merge 12+13 then shift 0.053
# Codes 1-11 already agree within sampling error (e.g. 3: 7.12% vs 7.28%).
# Leaving the pool on census coding makes the ABS cells 15/16 unfillable (49%
# of qalfp mass) while census 14 -- a code the table barely uses -- absorbs the
# whole margin: a 13x overshoot on the largest person cell of every zone.
#
# Applied *after* the structural rules and the support classification, both of
# which are written against census coding.
QALFP_SHIFT_FROM = 12


def align_qalfp_to_control_codes(pool, column="QALFP", shift_from=QALFP_SHIFT_FROM):
    """Recode QALFP from fused-census coding onto ABS control-table coding.

    Codes below `shift_from` are unchanged, codes at or above it move up by one.
    Returns (pool, n_recoded). Modifies `pool` in place and returns it.
    """
    col = next((c for c in pool.columns if str(c).lower() == column.lower()), None)
    if col is None:
        return pool, 0
    v = pd.to_numeric(pool[col], errors="coerce")
    n_moved = int((v >= shift_from).sum())
    out = v.where(v < shift_from, v + 1)
    if pd.api.types.is_integer_dtype(v.dtype) and not out.isna().any():
        out = out.astype("int64")
    pool[col] = out
    return pool, n_moved


def feasible_mask(df):
    """Return a boolean mask of rows that satisfy ALL constraints (feasible)."""
    ok = pd.Series(True, index=df.index)
    for name, pred in CONSTRAINTS.items():
        try:
            ok &= ~pred(df)
        except KeyError:
            continue  # column absent -> constraint not applicable
    return ok


# ---------------------------------------------------------------------------
# Data-derived feasibility: learn "what can exist" from the population
# ---------------------------------------------------------------------------
# The manual CONSTRAINTS above capture *structural* impossibility (age-gating,
# tenure mutual exclusion).  But they cannot say whether a rule-legal combination
# actually occurs in the population.  That is learned here, from the census
# itself, as the empirical co-occurrence structure: a combination the population
# never exhibits is treated as impossible even if no hand-written rule forbids it.

def resolve_columns(frame, attrs):
    """Map requested attribute names onto the frame's actual column names.

    The fused census carries upper-case ABS names (AGEP, HRSP, QALFP) while the
    builders, the control tables and pbs/prepare_full_354_pools.pbs all use the
    lower-case attribute names.  A plain `a in frame.columns` therefore matched
    nothing, which made the infill and existence layers silent no-ops and made
    identify_thin_cells call groupby([]) -- the "No group keys passed!" crash that
    killed the full-354 pool job on 2026-10-30.  Matching case-insensitively lets
    either spelling through.
    """
    lookup = {}
    for col in frame.columns:
        lookup.setdefault(str(col).lower(), col)
    return [lookup[str(a).lower()] for a in attrs if str(a).lower() in lookup]


def build_pairwise_support(census, attrs):
    """Learn the population's pairwise co-occurrence structure.

    For every pair of attributes, record the set of (value, value) combinations
    that actually occur in the census.  This is the data-derived feasibility
    criterion, complementing the structural rules.
    """
    import itertools
    attrs = resolve_columns(census, attrs)
    support = {}
    for a, b in itertools.combinations(attrs, 2):
        support[(a, b)] = set(zip(census[a], census[b]))
    return support


def plausibility_mask(df, support):
    """True where every pairwise attribute combination is observed in the census."""
    ok = pd.Series(True, index=df.index)
    for (a, b), allowed in support.items():
        if a not in df.columns or b not in df.columns:
            continue
        keys = pd.Series(zip(df[a], df[b]), index=df.index)
        ok &= keys.isin(allowed)
    return ok


def classify_existence(df, census, support, attrs):
    """Derive, from the population, whether each individual can exist.

    Returns a Series with one of:
      'observed'        -- exact attribute vector present in the census
      'sampling_zero'   -- every pair is supported but the full vector is unseen
                           (plausible-but-rare: the cell the method reconstructs)
      'structural_zero' -- some pair never co-occurs in the population
                           (implausible: the method must not emit it)
    """
    attrs = [a for a in resolve_columns(census, attrs) if a in df.columns]
    if not attrs:
        return pd.Series("observed", index=df.index)
    df_keys = df[attrs].astype(str).agg("|".join, axis=1)
    cen_keys = census[attrs].astype(str).agg("|".join, axis=1)
    observed = df_keys.isin(set(cen_keys))
    plausible = plausibility_mask(df, support)
    out = pd.Series("structural_zero", index=df.index)
    out[observed] = "observed"
    out[~observed & plausible] = "sampling_zero"
    return out


# ---------------------------------------------------------------------------
# Sampling-zero identification & infill
# ---------------------------------------------------------------------------

def identify_thin_cells(census, control_attrs, min_count=5):
    """Find control-table cells with <= min_count observed records (sampling zeros).

    Returns a DataFrame of unique thin cells (one row per cell) over control_attrs.
    """
    avail = resolve_columns(census, control_attrs)
    if not avail:
        # A caller may provide a control list for a census extract that does
        # not contain any of those fields. There are no identifiable thin
        # cells in that case; return an empty, schema-compatible result rather
        # than calling groupby([]).
        return pd.DataFrame(columns=["count"])
    cell_counts = census.groupby(avail).size().rename("count").reset_index()
    thin = cell_counts[cell_counts["count"] <= min_count].copy()
    return thin[avail + ["count"]]


def conditional_infill(census, thin_cells, target_per_cell=20, seed=0, relax_depth=1):
    """Reconstruct feasible individuals for thin cells by constraint-guided
    conditional resampling from the census's own conditional distributions.

    For each thin cell, we hold its `control_attrs` values fixed and sample the
    remaining attributes from the census rows that share that combination.  When
    too few such rows exist (the zero-cell itself), we relax one attribute at a
    time (drop the least-informative conditioning variable) until enough donor
    rows are available, then sample.  Every produced row is passed through
    enforce_feasibility so the result is feasible by construction.

    Returns a DataFrame of infilled records (same columns as census).
    """
    rng = np.random.default_rng(seed)
    avail = [c for c in thin_cells.columns if c in census.columns and c != "count"]
    rest = [c for c in census.columns if c not in avail]

    frames = []
    for _, cell in thin_cells.iterrows():
        fixed = {c: cell[c] for c in avail}
        need = max(0, int(target_per_cell - cell.get("count", 0)))
        if need <= 0:
            continue
        donors = _relaxed_donors(census, fixed, relax_depth)
        if donors.empty:
            continue
        draw = donors.sample(n=need, replace=True, random_state=seed)
        # Re-fix the conditioning attributes so the cell is preserved exactly.
        for c, v in fixed.items():
            draw[c] = v
        frames.append(draw)

    if not frames:
        return pd.DataFrame(columns=census.columns)
    infilled = pd.concat(frames, ignore_index=True)
    infilled, _ = enforce_feasibility(infilled, census, rng_seed=seed)
    return infilled


def _relaxed_donors(census, fixed, relax_depth):
    """Return census rows matching `fixed`, relaxing up to `relax_depth` keys."""
    keys = list(fixed.keys())
    for depth in range(relax_depth + 1):
        keep = keys[: len(keys) - depth]
        cond = pd.Series(True, index=census.index)
        for c in keep:
            cond &= census[c] == fixed[c]
        sub = census[cond]
        if len(sub) >= 10:
            return sub
    return census


# ---------------------------------------------------------------------------
# Control-category coverage
# ---------------------------------------------------------------------------
# A pool can be perfectly feasible and still be unable to represent the
# marginals: if no household in it carries a category, the MIP must either drop
# that cell (mip_fit.drop_unfillable) or miss it by the entire count. Measured
# on the Oct-2026 CP pool, 45% of the qalfp mass and 19% of the rlhp mass had no
# donor at all. Two causes: the census is a ~5% sample, so rare cells are absent
# outright, and coding differences remove whole blocks. Both are fixed here, so
# the pool is category-complete over the control tables before it is written.

# Census-coded source of a control-table category. Identity everywhere except
# QALFP, whose alignment is +1 above 12, so control 12 (Mixed Field Programmes)
# has no census ancestor at all.
_QALFP_CONTROL_TO_CENSUS = {13: 12, 14: 13, 15: 14, 16: 15}
_COVERAGE_HH_COL_CACHE = {}


def _census_category(attr, control_cat):
    """Census-coded value for a control-table category, or None if unmappable."""
    try:
        cat = int(control_cat)
    except (TypeError, ValueError):
        return None
    if str(attr).lower() != "qalfp":
        return cat
    if cat in _QALFP_CONTROL_TO_CENSUS:
        return _QALFP_CONTROL_TO_CENSUS[cat]
    return None if cat == 12 else cat      # 1-11 are shared with the census


def _column(df, attr):
    """Case-insensitive column lookup."""
    if attr in df.columns:
        return attr
    for c in df.columns:
        if str(c).lower() == str(attr).lower():
            return c
    return None


def control_coverage(pool, required):
    """Report, per control attribute, which required categories have no donor."""
    report = {}
    for attr, cats in required.items():
        col = _column(pool, attr)
        if col is None:
            report[attr] = {"expected": len(cats), "missing": sorted(cats),
                            "no_column": True}
            continue
        present = set(pd.to_numeric(pool[col], errors="coerce").dropna().astype(int))
        missing = sorted(int(c) for c in cats if int(c) not in present)
        report[attr] = {"expected": len(cats), "missing": missing,
                        "no_column": False}
    return report


def complete_control_coverage(pool, census, required, hh_attrs=(),
                              per_category=5, seed=0):
    """Add households so every control-table category has at least one donor.

    Real census households are preferred, because they are observed data and
    feasible by construction. When every household carrying a category is
    infeasible or oversized, the category is patched onto a feasible household
    (one member for a person attribute, every member for a household attribute)
    and kept only if the block still passes the rule store.

    Returns (pool, stats). Categories with no possible source are reported in
    stats['unsourceable'] rather than silently omitted.
    """
    stats = {"added_households": 0, "added_rows": 0, "copied": 0, "patched": 0,
             "coverage_gaps": [], "unsourceable": []}
    if not required:
        return pool, stats

    hh_col = next((c for c in census.columns if str(c).lower() == "abshid"), None)
    if hh_col is None:
        stats["unsourceable"].append({"reason": "census has no abshid"})
        return pool, stats

    rng = np.random.default_rng(seed)
    groups = census.groupby(hh_col, sort=False)
    hh_attrs = {str(a).lower() for a in hh_attrs}
    added, seen = [], set()
    existing_ids = set(pd.to_numeric(pool[hh_col], errors="coerce").dropna()
                       .astype("int64").tolist()) if hh_col in pool.columns else set()
    # Patched blocks are copies of one real household, so they must not share its
    # ABSHID: downstream groups by ABSHID, and several copies under one id would
    # merge into one oversized "household". Negative ids keep the column numeric.
    next_patch_id = -1

    for attr, cats in required.items():
        ccol = _column(census, attr)
        pcol = _column(pool, attr)
        if ccol is None or pcol is None:
            stats["unsourceable"].append({"attr": attr, "reason": "column absent"})
            continue
        # The pool is still in census coding here, so required control
        # categories are mapped before anything is compared.
        present = set(pd.to_numeric(pool[pcol], errors="coerce").dropna().astype(int))
        for cat in sorted(int(c) for c in cats):
            census_cat = _census_category(attr, cat)
            if census_cat is None:
                stats["unsourceable"].append(
                    {"attr": attr, "category": cat, "reason": "no census code"})
                continue
            if census_cat in present:
                continue
            vals = pd.to_numeric(census[ccol], errors="coerce")
            donor_ids = census.loc[vals == census_cat, hh_col].drop_duplicates()
            made = 0
            for hh_id in donor_ids.head(200):
                if made >= per_category:
                    break
                if int(hh_id) in existing_ids:
                    continue
                block = groups.get_group(hh_id)
                if len(block) > 12 or not bool(feasible_mask(block).all()):
                    continue
                key = (attr, cat, hh_id)
                if key in seen:
                    continue
                seen.add(key)
                added.append(block)
                made += 1
                stats["copied"] += 1
            if made:
                present.add(census_cat)
                continue

            # No usable observed household: patch a feasible one.
            sample = census.sample(n=min(len(census), 20000), random_state=seed)
            for hh_id in sample[hh_col].drop_duplicates().head(200):
                block = groups.get_group(hh_id).copy()
                if len(block) > 12 or not bool(feasible_mask(block).all()):
                    continue
                if str(attr).lower() in hh_attrs:
                    block[ccol] = census_cat
                else:
                    block.loc[block.index[0], ccol] = census_cat
                if not bool(feasible_mask(block).all()):
                    continue
                block[hh_col] = next_patch_id
                existing_ids.add(next_patch_id)
                next_patch_id -= 1
                added.append(block)
                made += 1
                stats["patched"] += 1
                break
            if made:
                present.add(census_cat)
            else:
                stats["coverage_gaps"].append({"attr": attr, "category": cat})

    if added:
        extra = pd.concat(added, ignore_index=True)
        extra["_source"] = "coverage_infill"
        stats["added_households"] = int(extra[hh_col].nunique())
        stats["added_rows"] = int(len(extra))
        pool = pd.concat([pool, extra], ignore_index=True)

    # Report in control-table terms: what is still missing once the pool is
    # aligned by the caller.
    remaining = {}
    for attr, cats in required.items():
        pcol = _column(pool, attr)
        if pcol is None:
            remaining[attr] = sorted(int(c) for c in cats)
            continue
        present = set(pd.to_numeric(pool[pcol], errors="coerce").dropna().astype(int))
        miss = [int(c) for c in cats
                if _census_category(attr, c) not in present]
        if miss:
            remaining[attr] = sorted(miss)
    stats["remaining_missing"] = remaining
    return pool, stats


def load_required_categories():
    """Control-table category universe and household-level attribute names.

    `config.SA_CONTROL_DIR` (or the SA_CONTROL_DIR environment variable) picks
    which control-table set is used. Returns (required, hh_attrs).
    """
    from sa_only_baseline import load_control_tables
    ind, hh = load_control_tables()
    # Band tables (age5p, hcfmf) are fitted through another pool column --
    # mip_fit._BAND_CELLS maps age5p onto single-year agep and hcfmf onto
    # families_in_household -- so their category labels are not pool values and
    # requiring them verbatim would report the whole table as missing.
    band_attrs = {"age5p", "hcfmf"}
    required, hh_attrs = {}, set()
    for tables, is_household in ((ind, False), (hh, True)):
        for table in tables.values():
            for key in table.get("keys", []):
                if key not in table["data"].columns or key in band_attrs:
                    continue
                vals = pd.to_numeric(table["data"][key], errors="coerce").dropna()
                if vals.empty:
                    continue
                required.setdefault(key, set()).update(
                    vals.astype(int).unique().tolist())
                if is_household:
                    hh_attrs.add(key)

    # Categories the census microdata cannot carry must not be demanded of the
    # pool (see config.QALFP_MERGE_FOR_COVERAGE): the donor folds them away, so
    # requiring them reports an unfillable cell on every build. Scoring keeps the
    # full category set -- this affects pool construction only.
    if getattr(config, "QALFP_MERGE_FOR_COVERAGE", False):
        for attr, mapping in getattr(config, "QALFP_MERGED_CATEGORIES", {}).items():
            if attr not in required:
                continue
            merged = set()
            for code in required[attr]:
                merged.add(int(mapping.get(code, code)))
            required[attr] = merged
    return required, hh_attrs


# ---------------------------------------------------------------------------
# Main pool builder
# ---------------------------------------------------------------------------

def build_feasible_pool(census, control_attrs, min_count=5, target_per_cell=20,
                        infill=True, seed=0, support_attrs=None, use_support=True,
                        max_infill_cells=60000, coverage=None,
                        coverage_per_category=5, hh_attrs=()):
    """Build the complete feasible candidate pool from the census.

    Two feasibility layers are applied:
      1. Structural rules (manual CONSTRAINTS) -- rule feasibility.
      2. Data-derived support (pairwise co-occurrence) -- population plausibility;
         records whose attribute combinations never occur in the census are
         dropped unless use_support=False.

    Returns (pool_df, report).
    """
    census = census.copy()
    ok = feasible_mask(census)
    n_bad = int((~ok).sum())
    if n_bad:
        census, _ = enforce_feasibility(census[ok].reset_index(drop=True), census, rng_seed=seed)

    support_attrs = support_attrs or control_attrs
    support = build_pairwise_support(census, support_attrs) if use_support else None

    report = {"n_census": len(census), "n_infeasible_in_census": n_bad}

    if infill and control_attrs:
        thin = identify_thin_cells(census, control_attrs, min_count=min_count)
        report["n_thin_cells"] = len(thin)
        # Use the groupby-based infill. The original per-cell implementation
        # built a 1.25M-element boolean mask per conditioning attribute for
        # every thin cell (~2.8e13 ops at 21 attributes) and retained one
        # DataFrame per cell until the final concat, which is what drove RSS
        # past 21 GB and never finished. See cp_fast_infill.py.
        from cp_fast_infill import conditional_infill as _fast_infill
        infilled = _fast_infill(census, thin,
                                target_per_cell=target_per_cell, seed=seed,
                                max_infill_cells=max_infill_cells)
        report["n_infilled"] = len(infilled)
        report["n_cells_infilled"] = infilled.attrs.get("n_cells_infilled", 0)
        report["n_cells_skipped_cap"] = infilled.attrs.get("n_cells_skipped_cap", 0)
        census["_source"] = "census"
        infilled["_source"] = "infilled"
        pool = pd.concat([census, infilled], ignore_index=True)
    else:
        report["n_thin_cells"] = 0
        report["n_infilled"] = 0
        census["_source"] = "census"
        pool = census

    # --- population-derived existence classification ---
    if support is not None:
        exists = classify_existence(pool, census, support, support_attrs)
        report["n_observed"] = int((exists == "observed").sum())
        report["n_sampling_zero"] = int((exists == "sampling_zero").sum())
        report["n_structural_zero"] = int((exists == "structural_zero").sum())
        pool = pool[exists != "structural_zero"].reset_index(drop=True)

    # Drop pathological household groups. The fused census carries sentinel
    # ABSHID values whose group holds tens of thousands of persons (largest:
    # 34,589 rows). As a single "household" it would inject 34k people into any
    # zone that selects it, and its attributes are not a real dwelling. The
    # legitimate maximum household size in the microdata is well under this cap.
    hh_col = next((c for c in pool.columns if str(c).lower() == "abshid"), None)
    max_hh = int(__import__("os").environ.get("CP_MAX_HH_SIZE", "12") or 12)
    if hh_col is not None:
        sizes = pool.groupby(hh_col).size()
        over = sizes[sizes > max_hh]
        if len(over):
            report["n_oversized_households_dropped"] = int(len(over))
            report["n_oversized_rows_dropped"] = int(over.sum())
            pool = pool[~pool[hh_col].isin(over.index)].reset_index(drop=True)

    # Structural feasibility audit of the final pool. Run on census coding, the
    # coding the rule store is written against.
    bad = feasible_mask(pool)
    report["n_infeasible_pool"] = int((~bad).sum())
    report["pool_size"] = len(pool)

    # Make the pool category-complete over the control tables before coding is
    # translated: this runs in census coding, which is what the census is in.
    if coverage:
        pool, cov = complete_control_coverage(
            pool, census, coverage, hh_attrs=hh_attrs,
            per_category=coverage_per_category, seed=seed)
        report["coverage_added_rows"] = cov["added_rows"]
        report["coverage_added_households"] = cov["added_households"]
        report["coverage_copied_households"] = cov["copied"]
        report["coverage_patched_households"] = cov["patched"]
        if cov["unsourceable"]:
            report["coverage_unsourceable"] = cov["unsourceable"]
        if cov["coverage_gaps"]:
            # No census household carries the category and patching one onto a
            # feasible household fails the rule store (e.g. HRSP status codes
            # above the age->hours cap): the cell needs a coding decision.
            report["coverage_unfillable"] = cov["coverage_gaps"]
        if cov.get("remaining_missing"):
            report["coverage_remaining_missing"] = cov["remaining_missing"]

    # Align QALFP with the ABS control coding the MIP is fitted against. This is
    # the last transformation: everything above reasons in census coding.
    pool, n_recoded = align_qalfp_to_control_codes(pool)
    report["qalfp_recoded_to_control_codes"] = n_recoded
    report["qalfp_shift_from"] = QALFP_SHIFT_FROM

    # --- Uniqueness flag: how many identical individuals exist? ---
    # Use the modeled attributes (those in control_attrs + rest) to identify
    # exact duplicates. A "unique" individual has no exact clone in the pool.
    attrs_for_uniqueness = [c for c in pool.columns if c not in ("_source", "_uniqueness")]
    if attrs_for_uniqueness:
        # Efficient: compute duplicate counts via groupby + transform (no merge fragmentation)
        dup_counts = pool.groupby(attrs_for_uniqueness, sort=False).size()
        # Map counts back using the index alignment (much faster than merge)
        pool["_dup_count"] = pool.index.map(
            dup_counts.to_dict()
        )
        # Handle the case where map returns NaN (shouldn't happen but safe)
        pool["_dup_count"] = pool["_dup_count"].fillna(1).astype(int)
        pool["_uniqueness"] = pool["_dup_count"].apply(
            lambda x: "unique" if x == 1 else f"duplicate_{int(x)}"
        )
        report["n_unique_individuals"] = int((pool["_dup_count"] == 1).sum())
        report["n_duplicate_individuals"] = int((pool["_dup_count"] > 1).sum())
        report["n_unique_groups"] = int(pool["_dup_count"].nunique())
    else:
        pool["_uniqueness"] = "unknown"

    return pool, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build a CP-style feasible pool from the census")
    ap.add_argument("--census", required=True, help="Fused census CSV (e.g. LINKED_DATA_national.csv)")
    ap.add_argument("--control-attrs", nargs="+", required=True,
                    help="Attribute names to hold fixed for zero-cell infill")
    ap.add_argument("--min-count", type=int, default=5,
                    help="Cells with <= this many records are treated as sampling zeros")
    ap.add_argument("--target-per-cell", type=int, default=20,
                    help="Infilled target count per thin cell")
    ap.add_argument("--no-infill", action="store_true", help="Skip infill (census only)")
    ap.add_argument("--max-infill-cells", type=int, default=60000,
                    help="Cap on how many thin cells get infilled. With 21 control "
                         "attributes the joint cell space is far larger than the "
                         "census, so most cells are empty; infilling all of them "
                         "is neither tractable nor useful. Largest-count cells "
                         "are kept when capped.")
    ap.add_argument("--coverage-tables", action="store_true",
                    help="Make the pool category-complete over the control tables: "
                         "add donors for every category the marginals require and "
                         "the pool/microdata currently cannot represent. Uses the "
                         "tables in config.SA_CONTROL_DIR.")
    ap.add_argument("--coverage-per-category", type=int, default=5,
                    help="Donor households to add per missing category")
    ap.add_argument("--strict-coverage", action="store_true",
                    help="Exit non-zero when any required category is still missing")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "feasible_pool.csv"))
    ap.add_argument("--report-json", default="",
                    help="Write the build report to this JSON path so the "
                         "submitting job can assert on the numbers instead of "
                         "parsing stdout.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    census = pd.read_csv(a.census)
    coverage, hh_attrs = (None, ())
    if a.coverage_tables:
        coverage, hh_attrs = load_required_categories()
        print(f"control-table coverage: {len(coverage)} attributes, "
              f"{sum(len(v) for v in coverage.values())} categories")

    pool, report = build_feasible_pool(
        census, a.control_attrs, min_count=a.min_count,
        target_per_cell=a.target_per_cell, infill=not a.no_infill, seed=a.seed,
        max_infill_cells=a.max_infill_cells, coverage=coverage,
        coverage_per_category=a.coverage_per_category, hh_attrs=hh_attrs)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pool.to_csv(a.out, index=False)

    print("CP feasible-pool build report")
    print("=" * 40)
    for k, v in report.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

    if a.report_json:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(a.report_json)), exist_ok=True)
        with open(a.report_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

    if coverage:
        gaps = control_coverage(pool, coverage)
        short = {a_: r_["missing"] for a_, r_ in gaps.items() if r_["missing"]}
        if short:
            print("WARNING: pool still cannot represent these control categories:")
            for attr, cats in short.items():
                print(f"  {attr}: {cats}")
            if a.strict_coverage:
                raise SystemExit(
                    f"strict coverage: {sum(len(v) for v in short.values())} "
                    f"category(ies) unfillable in {sorted(short)}")
        else:
            print(f"coverage: all {sum(len(v) for v in coverage.values())} "
                  f"control categories have donors")

    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()

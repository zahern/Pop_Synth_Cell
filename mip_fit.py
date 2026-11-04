"""
mip_fit.py — constraint-programming pool + Mixed-Integer Programming fitter
============================================================================

Zone fitting as a direct include/exclude decision over the CP feasible pool.

For each zone the decision variable is one binary per candidate household:

    x_h in {0,1}   -> is household h part of the synthetic population of zone z?
    d_{k,m} >= 0   -> slack (absolute deviation) for control-table cell (k,m)

    minimise    sum_{k,m} d_{k,m}
    subject to  | sum_h C_{h,k,m} x_h - T_{z,k,m} | <= d_{k,m}      (fit)
                d_{k,m} <= tol * T_{z,k,m}                          (tolerance)
                x_h in {0,1}                                         (in/out)

C_{h,k,m} is 1 when household h's value for a household-level attribute is m,
and the number of members of h falling in cell m for an individual-level
attribute -- so household atomicity is preserved automatically (a household is
wholly in or wholly out).

If the program is infeasible at the requested tolerance, the tolerance is
RELAXED (multiplied by --relax-factor) and re-solved, up to --max-relax rounds.
The reported tolerance is therefore the tightest one admitting a feasible
solution, which is a useful zone-level diagnostic in its own right.

Because the candidate set is pre-filtered to the zone's own SA4 slice (by REGUCP
concordance) and never falls back to the global pool, the problem stays sparse
and solvable: typically 10^4-10^5 binary variables.

Solver: HiGHS via scipy.optimize.milp.  If that is unavailable or times out, the
LP relaxation is solved and rounded (--allow-lp-fallback), which for these
near-integral assignment problems is usually exact.

Usage
-----
  python mip_fit.py --pool outputs/feasible_pool.csv \
         --zones 301011001 ... --tol 0.05 --relax-factor 2 --max-relax 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config

try:
    from sa_only_baseline import (
        load_control_tables, _is_household_table, _load_sa2_to_regucp,
        _normalize_zone_values,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"sa_only_baseline import failed: {exc}")

RESULTS_DIR = getattr(config, "CP_MIP_RESULTS_DIR",
                      os.path.join(config.OUTPUTS_DIR, "cp_mip_results"))
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Zone targets
# ---------------------------------------------------------------------------
# A handful of control tables (bedrd, hhcd, tend, lfsp) store category *labels*
# as text (e.g. 'None (includes studio apartments or bedsitters)') while the CP
# pool carries ABS integer codes. Blind int(label) raises ValueError and kills
# the whole job (Gadi popgen_cp_mip 180702265). Decode labels first; skip only
# genuinely unparseable rows with a warning.
# Codes come from the ABS "2021 Census 5% basic microdata" data item list.
# Keep in sync with benchmark_margins._LABEL_TO_CODE.
_LABEL_TO_CODE = {
    "bedrd": {
        "None (includes studio apartments or bedsitters)": 0, "One bedroom": 1,
        "Two bedrooms": 2, "Three bedrooms": 3, "Four bedrooms": 4,
        "Five bedrooms or more": 5, "Not stated": 6, "Not applicable": 7,
    },
    "hhcd": {
        "One family household": 1, "Two or more family household": 2,
        "Lone person household": 3, "Group household": 4, "Visitors only": 5,
        "Other non-classifiable": 6, "Not applicable": 7,
    },
    "tend": {
        "Owned outright": 1, "Owned with a mortgage": 2, "Rented": 3,
        "Other": 4, "Not stated": 5, "Not applicable": 6,
    },
    "lfsp": {
        "Employed": 1, "Unemployed": 2, "Not in the labour force": 3,
        "Not stated": 4, "Not applicable": 5, "Overseas visitor": 6,
    },
}


def _decode_category(attr, raw):
    """Map a control-table category value to the integer code used in the pool.

    Returns None when the value matches neither a plain code nor a known label.
    """
    s = str(raw).strip()
    try:
        return int(float(s))
    except (TypeError, ValueError):
        pass
    return _LABEL_TO_CODE.get(str(attr).lower(), {}).get(s)


# ---------------------------------------------------------------------------
# Band cells: control tables whose categories span SETS of pool codes.
# ---------------------------------------------------------------------------
# age5p bands (e.g. code 0 = ages 0-4) have no verbatim pool column; the pool
# carries single-year AGEP codes instead. hcfmf family-composition labels map
# onto the pool's families_in_household column. Without this, those cells get
# empty constraint rows, which force tolerance relaxation to >= 1.0 on every
# zone (slow + meaningless tolerances). Band definitions follow
# CensusRecordImporter age5p mappings and benchmark_margins._hcfmf_label.
_BAND_CELLS = {
    "age5p": ("agep", {
        0: frozenset({0, 1, 2, 3, 4}), 5: frozenset({5, 6, 7, 8, 9}),
        10: frozenset({10, 11, 12, 13, 14}),
        15: frozenset({15, 16, 17, 18, 19}),
        20: frozenset({20, 21, 22, 23, 24}),
        25: frozenset({25}), 26: frozenset({26}), 27: frozenset({27}),
        28: frozenset({28}), 29: frozenset({29}), 30: frozenset({30}),
        31: frozenset({31}), 32: frozenset({32}), 33: frozenset({33}),
        34: frozenset({34}), 35: frozenset({35}), 36: frozenset({36}),
        37: frozenset({37}),
    }),
    "hcfmf": ("families_in_household", {
        "One family household": frozenset({1}),
        "Two family household": frozenset({2}),
        "Three or more family household": frozenset({3, 4}),
    }),
}

# Pool columns that are constant within a household (household-level), beyond
# the keys coming from control tables.
_HOUSEHOLD_POOL_COLS = {"families_in_household"}


def _band_cell(attr, raw, code):
    """Return (pool_col, frozenset(codes)) for band-coded categories, else None."""
    spec = _BAND_CELLS.get(str(attr).lower())
    if spec is None:
        return None
    pool_col, bands = spec
    if code in bands:
        return pool_col, bands[code]
    s = str(raw).strip()
    if s in bands:
        return pool_col, bands[s]
    return None


def zone_targets(tables, zone):
    """Per-cell target counts for one zone: {(attr, category): count}."""
    targets = {}
    skipped = {}
    for key, entry in tables.items():
        df = entry.get("data")
        if df is None or "sa2" not in df.columns:
            continue
        keys = entry.get("keys", [])
        # Only single-attribute marginal tables define per-cell targets. A joint
        # table (e.g. sa2sexp_x_mstp with keys [sexp, mstp]) would be summed once
        # per second-attribute category and silently multiply the marginal
        # targets, so skip it here. Joints belong to the SA/IPU objective.
        if len(keys) != 1:
            continue
        attr = keys[0]
        if attr is None:
            continue
        zdf = df[_normalize_zone_values(df["sa2"]) == str(zone)]
        for _, row in zdf.iterrows():
            cat = _decode_category(attr, row[attr])
            try:
                cnt = int(float(row["count"]))
            except (TypeError, ValueError):
                cnt = None
            if cnt is None:
                skipped[key] = skipped.get(key, 0) + 1
                continue
            band = _band_cell(attr, row[attr], cat)
            if band is not None:
                pool_col, codes = band
                targets[(pool_col, codes)] = targets.get((pool_col, codes), 0) + cnt
                continue
            if cat is None:
                skipped[key] = skipped.get(key, 0) + 1
                continue
            targets[(attr, cat)] = targets.get((attr, cat), 0) + cnt
    if skipped:
        print(f"  zone_targets: skipped {sum(skipped.values())} "
              f"unparseable target rows: {skipped}")
    return targets


def _sa2_key(value):
    """Normalise an SA2 code from string, int, or float storage."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def zone_candidate_pool(pool, zone, sa2_to_regucp):
    """Pre-filter the CP pool to the households that can serve `zone`.

    Base records come from the zone's own SA4 microdata slice and never from a
    global store.  Records constructed for a specific SA2 by expansion are
    tagged in ``_target_sa2`` and are included only for that exact zone, so a
    constructed household is never shared with a sibling SA2 in the same SA4.
    The result is therefore a per-SA2 pool: the base slice plus that zone's own
    constructed additions.

    An empty result is a structural residual for the zone, not an invitation to
    borrow another SA2's or SA4's sample.
    """
    target_col = next((c for c in pool.columns
                       if str(c).lower() == "_target_sa2"), None)
    own = pool.iloc[0:0]
    tagged = pd.Series(False, index=pool.index)
    if target_col is not None:
        tagged = pool[target_col].notna()
        own = pool[pool[target_col].map(_sa2_key) == _sa2_key(zone)]

    if "regucp" not in pool.columns:
        if len(own):
            return own, f"constructed-only ({len(own)} records)"
        return pool.iloc[0:0], "empty (no regucp column)"

    reg = sa2_to_regucp.get(str(zone))
    if not reg:
        if len(own):
            return own, f"constructed-only ({len(own)} records)"
        return pool.iloc[0:0], "empty (zone not in concordance)"

    base = pool[(_normalize_zone_values(pool["regucp"]) == str(reg))
                & ~tagged]
    if len(own):
        sub = pd.concat([base, own], ignore_index=True)
        return sub, f"REGUCP={reg} + {len(own)} constructed"
    if len(base) == 0:
        return base, f"empty REGUCP={reg} (structural residual)"
    return base, f"REGUCP={reg}"


def cap_candidates(cand, person_target, max_candidates=150000,
                   cand_mult=10.0, cand_floor=20000, seed=0):
    """Bound the MIP candidate set so the program stays solvable.

    Keeps max(floor, mult * person_target / 2.6) households up to the absolute
    max_candidates ceiling, stratified by household size so the mix is
    preserved (a uniform sample would starve large households the zone may
    need).  Below the cap this is a no-op.  Seeded for reproducibility.

    Rationale: a 300-person zone needs ~130 households; choosing among 488k
    binaries adds nothing but memory (OOM) and branch-and-bound hours.  The
    cap is applied only within the zone's own SA4 slice; it is never used to
    make a global-pool fallback tractable, because there is no such fallback.
    """
    import math as _math
    hh_sizes = cand.groupby("abshid").size() if "abshid" in cand.columns \
        else pd.Series(1, index=np.arange(len(cand)))
    n_hh = int(hh_sizes.shape[0])
    need = person_target / 2.6 if person_target and person_target > 0 else 0
    cap = int(min(max_candidates, max(cand_floor, cand_mult * need)))
    if n_hh <= cap:
        return cand, f"kept all {n_hh:,}"
    rng = np.random.default_rng(seed)
    # Proportional allocation across household sizes (largest remainder).
    groups = {}
    for hh_id, size in hh_sizes.items():
        groups.setdefault(int(size), []).append(hh_id)
    sizes = sorted(groups)
    weights = np.array([len(groups[s]) for s in sizes], dtype=float)
    raw = weights / weights.sum() * cap
    take = np.floor(raw).astype(int)
    short = cap - int(take.sum())
    if short > 0:
        order = np.argsort(-(raw - take), kind="stable")[:short]
        take[order] += 1
    keep = []
    for s, t in zip(sizes, take):
        ids = groups[s]
        t = min(int(t), len(ids))
        if t <= 0:
            continue
        keep.extend(rng.choice(ids, size=t, replace=False).tolist())
    note = (f"capped {n_hh:,} -> {len(keep):,} households "
            f"(target ~{need:,.0f} persons)")
    if "abshid" in cand.columns:
        return cand[cand["abshid"].isin(set(keep))].copy(), note
    return cand.iloc[sorted(keep)].copy(), note


# ---------------------------------------------------------------------------
# Sparse contribution matrix: cells x candidate households
# ---------------------------------------------------------------------------

def build_constraint_matrix(pool, targets, ind_keys, hh_keys):
    """Build sparse C (n_cells x n_households) and the household id index.

    Household-level attributes contribute 1 (the household sits in that cell);
    individual-level attributes contribute the count of members in that cell.
    """
    import scipy.sparse as sp

    hh_ids = pool["abshid"].unique() if "abshid" in pool.columns else np.arange(len(pool))
    pos = {a: i for i, a in enumerate(hh_ids)}
    n_hh = len(hh_ids)

    hh_cols = _hh_pool_cols(pool, hh_keys)
    hh_first = None
    if hh_cols and "abshid" in pool.columns:
        hh_first = pool.groupby("abshid")[hh_cols].first()

    cells = [(attr, cat) for (attr, cat) in targets]
    rows, cols, data = [], [], []

    def _match(col, cat):
        if isinstance(cat, (set, frozenset, list, tuple)):
            return col.isin(list(cat))
        return col == cat

    for ci, (attr, cat) in enumerate(cells):
        if attr in hh_cols and hh_first is not None:
            vals = _match(hh_first[attr], cat).to_numpy(dtype=float)
            nz = np.nonzero(vals)[0]
            rows.extend([ci] * len(nz))
            cols.extend(nz.tolist())
            data.extend(vals[nz].tolist())
        elif attr in pool.columns:
            sub = pool[_match(pool[attr], cat)]
            if len(sub):
                counts = sub.groupby("abshid").size()
                idx = counts.index.map(pos)
                keep = idx.notna()
                rows.extend([ci] * int(keep.sum()))
                cols.extend(idx[keep].astype(int).tolist())
                data.extend(counts[keep].astype(float).tolist())

    C = sp.csr_matrix((data, (rows, cols)), shape=(len(cells), n_hh))
    return C, hh_ids, cells


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def _hh_pool_cols(pool, hh_keys):
    """Household-level pool columns (deduped, present in pool)."""
    return list(dict.fromkeys(
        [k for k in list(hh_keys) + sorted(_HOUSEHOLD_POOL_COLS)
         if k in pool.columns]))


def _tol_vector(cells, hh_attr_set, tol, hh_tol_mult, abs_floor=0.0):
    """Per-cell tolerance: household cells get tol*hh_tol_mult (relaxed fit).

    With abs_floor > 0 the allowed deviation is max(tol*T, abs_floor), i.e. a
    fixed absolute allowance per cell. Without it a 5% tolerance on a cell with
    target 2 persons means 'hit 2 exactly', which household atomicity makes
    infeasible for most zones -> every zone relaxed to the 1.6 ceiling
    (measured: 417/436 zones at tol=1.6, maxdev up to 6.3).
    """
    base = np.array([tol * hh_tol_mult if c[0] in hh_attr_set else tol
                     for c in cells], dtype=float)
    return base


def _abs_tol(cells, tol_vec, T, abs_floor):
    """Allowed absolute deviation per cell: max(tol*T, abs_floor)."""
    allowed = tol_vec * np.maximum(T, 1e-9)
    if abs_floor and abs_floor > 0:
        allowed = np.maximum(allowed, abs_floor)
    return allowed


def _assemble(C, T, tol_vec, abs_floor=0.0):
    """Sparse LP/MIP matrices for the tolerance-constrained program.

    tol_vec is per-cell (household cells relaxed via hh_tol_mult); the effective
    bound is max(tol_vec*T, abs_floor).
    """
    import scipy.sparse as sp
    n_cells, n_hh = C.shape
    n = n_hh + n_cells                      # [x (n_hh), d (n_cells)]
    Ct = C.tocsc()
    I = sp.identity(n_cells, format="csc")

    allowed = _abs_tol([None] * n_cells, tol_vec, T, abs_floor)

    # +Cx - d <= T      |   -Cx - d <= -T      |   d <= allowed
    A1 = sp.hstack([Ct, -I])
    A2 = sp.hstack([-Ct, -I])
    A3 = sp.hstack([sp.csr_matrix((n_cells, n_hh)), I])
    A = sp.vstack([A1, A2, A3], format="csc")
    b = np.concatenate([T, -T, allowed])
    return A, b, n_hh, n_cells


def _solve_mip(C, T, tol_vec, time_limit=None, abs_floor=0.0):
    """HiGHS MIP via scipy.optimize.milp. Returns x (0/1) or None."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    import scipy.sparse as sp

    A, b, n_hh, n_cells = _assemble(C, T, tol_vec, abs_floor=abs_floor)
    c_obj = np.zeros(n_hh + n_cells)
    c_obj[n_hh:] = 1.0                                  # minimise sum d
    integrality = np.zeros(n_hh + n_cells, dtype=int)
    integrality[:n_hh] = 1                              # x binary
    allowed = _abs_tol([None] * n_cells, tol_vec, T, abs_floor)
    bounds = Bounds(np.zeros(n_hh + n_cells),
                    np.concatenate([np.ones(n_hh), allowed]))
    cons = LinearConstraint(A, -np.inf, b)
    opts = {"time_limit": time_limit} if time_limit else {}
    try:
        res = milp(c=c_obj, integrality=integrality, bounds=bounds,
                   constraints=cons, options=opts)
    except TypeError:
        res = milp(c=c_obj, integrality=integrality, bounds=bounds,
                   constraints=cons)
    if res is None or not getattr(res, "success", False):
        return None
    return np.round(res.x[:n_hh]).astype(int)


def _cell_weights(cells, T, objective):
    """Objective weights per cell.

    "rel"   : 1/max(T,1) — each cell's *relative* deviation counts equally in
              raw sum. Large tables (sexp/agep with 10^4 persons) then get
              1/10^4 weight and are effectively ignored: measured 3,994 of
              18,621 persons selected for a zone.
    "table" : (default) same relative weight divided by the number of cells in
              that table, so every control table contributes equally instead of
              30 small tables outvoting the two that carry the population.
    "abs"   : 1 — absolute units; large cells dominate.
    """
    T = np.asarray(T, dtype=float)
    if objective == "abs":
        return np.ones_like(T)
    base = 1.0 / np.maximum(np.abs(T), 1.0)
    if objective == "rel":
        return base
    # table-normalised
    counts = {}
    for attr, _cat in cells:
        counts[attr] = counts.get(attr, 0) + 1
    per_cell = np.array([1.0 / max(counts.get(attr, 1), 1) for attr, _cat in cells])
    return base * per_cell


def _deviation_stats(C, T, x, hh_attr_set, cells):
    """Achieved deviation statistics for a solution.

    Returns dict with absolute and relative max/mean/sum deviations, plus the
    household-only subset (household cells are fitted more loosely by design).
    """
    got = np.asarray(C @ x).ravel()
    dev = np.abs(got - T)
    denom = np.maximum(np.abs(T), 1.0)
    rel = dev / denom
    is_hh = np.array([c[0] in hh_attr_set for c in cells], dtype=bool)
    out = {
        "abs_sum": float(dev.sum()),
        "rel_max": float(rel.max()) if rel.size else 0.0,
        "rel_mean": float(rel.mean()) if rel.size else 0.0,
        "rel_sum": float(rel.sum()),
    }
    if is_hh.any():
        out["rel_max_hh"] = float(rel[is_hh].max())
        out["rel_mean_hh"] = float(rel[is_hh].mean())
    if (~is_hh).any():
        out["rel_max_person"] = float(rel[~is_hh].max())
        out["rel_mean_person"] = float(rel[~is_hh].mean())
    return out


def _assemble_min(C, T, hh_persons=None, pop_target=None, pop_tol=0.0):
    """Deviation-minimising program: |Cx - T| <= d with d >= 0, no upper bound.

    Unlike the tolerance formulation there is no `d <= tol*T` row: the solver
    simply drives the deviations as small as household atomicity allows, so the
    reported numbers are *achieved* fit quality rather than an assumed ceiling.

    When hh_persons/pop_target are given, two extra rows anchor the selected
    population to the census total within +-pop_tol. Without it the relative
    objective happily drops ~80% of a zone's residents because the big
    person-count cells carry tiny 1/T weights (measured: 3,994 persons selected
    against an 18,621 census target).
    """
    import scipy.sparse as sp
    n_cells, n_hh = C.shape
    Ct = C.tocsc()
    I = sp.identity(n_cells, format="csc")
    A1 = sp.hstack([Ct, -I])
    A2 = sp.hstack([-Ct, -I])
    A = sp.vstack([A1, A2], format="csc")
    lb = np.full(2 * n_cells, -np.inf)
    ub = np.concatenate([np.asarray(T, dtype=float), -np.asarray(T, dtype=float)])

    if hh_persons is not None and pop_target is not None and pop_tol > 0:
        p = np.asarray(hh_persons, dtype=float)
        row = sp.csr_matrix(np.concatenate([p, np.zeros(n_cells)]).reshape(1, -1))
        band = abs(float(pop_target)) * float(pop_tol)
        # +/- anchor rows: p.x <= P+band and -p.x <= -(P-band)
        A = sp.vstack([A, row, -row], format="csc")
        lb = np.concatenate([lb, [-np.inf, -np.inf]])
        ub = np.concatenate([ub, [pop_target + band, -(pop_target - band)]])

    return A, lb, ub, n_hh, n_cells


def _solve_mip_min(C, T, weights, time_limit=None, hh_persons=None,
                   pop_target=None, pop_tol=0.0, rep_max=1,
                   max_rel_dev=None):
    """Minimise sum(weights * d) subject to |Cx - T| <= d (+population anchor).

    x_h is an integer *multiplicity* per candidate household, bounded by
    rep_max. The Census BASIC microdata is a ~5% sample, so a zone's true
    population is far larger than the number of candidate households present
    (measured: 3,448 candidates for a zone needing 9,354 households / 18,721
    persons). With x binary the model can never reach the targets; allowing
    replication lets one sampled household stand for several real ones while
    keeping every selected household internally consistent.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    A, lb, ub, n_hh, n_cells = _assemble_min(C, T, hh_persons, pop_target, pop_tol)
    c_obj = np.zeros(n_hh + n_cells)
    c_obj[n_hh:] = weights                                # minimise weighted d
    integrality = np.zeros(n_hh + n_cells, dtype=int)
    integrality[:n_hh] = 1                                # x integer multiplicity
    d_upper = np.full(n_cells, np.inf)
    if max_rel_dev is not None:
        d_upper = float(max_rel_dev) * np.maximum(np.abs(T), 1.0)
    bounds = Bounds(np.zeros(n_hh + n_cells),
                    np.concatenate([np.full(n_hh, float(max(rep_max, 1))),
                                    d_upper]))
    cons = LinearConstraint(A, lb, ub)
    opts = {}
    if time_limit:
        opts["time_limit"] = time_limit
    try:
        res = milp(c=c_obj, integrality=integrality, bounds=bounds,
                   constraints=cons, options=opts)
    except TypeError:
        res = milp(c=c_obj, integrality=integrality, bounds=bounds,
                   constraints=cons)
    if res is None or res.x is None:
        return None
    # Accept a feasible-but-not-proven-optimal incumbent (time limit).
    return np.round(res.x[:n_hh]).astype(int)


def _solve_mip_minimax(C, T, time_limit=None, hh_persons=None,
                       pop_target=None, pop_tol=0.0, rep_max=1):
    """Minimise the worst relative cell deviation.

    The auxiliary variable rho is constrained to be at least each cell's
    absolute deviation divided by max(target, 1).  This protects small but
    important cells from being traded away by a weighted-sum objective.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    import scipy.sparse as sp

    A, lb, ub, n_hh, n_cells = _assemble_min(
        C, T, hh_persons, pop_target, pop_tol)
    scales = np.maximum(np.abs(np.asarray(T, dtype=float)), 1.0)
    # d_c - scale_c * rho <= 0
    rho_row = sp.hstack([
        sp.csr_matrix((n_cells, n_hh)),
        sp.identity(n_cells, format="csr"),
        -sp.csr_matrix(scales.reshape(-1, 1)),
    ], format="csr")
    A = sp.hstack([
        A, sp.csr_matrix((A.shape[0], 1))
    ], format="csc")
    A = sp.vstack([A, rho_row], format="csc")
    lb = np.concatenate([lb, np.full(n_cells, -np.inf)])
    ub = np.concatenate([ub, np.zeros(n_cells)])

    c_obj = np.zeros(n_hh + n_cells + 1)
    c_obj[-1] = 1.0
    integrality = np.zeros(n_hh + n_cells + 1, dtype=int)
    integrality[:n_hh] = 1
    bounds = Bounds(
        np.zeros(n_hh + n_cells + 1),
        np.concatenate([
            np.full(n_hh, float(max(rep_max, 1))),
            np.full(n_cells, np.inf),
            [np.inf],
        ]),
    )
    opts = {"time_limit": time_limit} if time_limit else {}
    try:
        res = milp(
            c=c_obj,
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(A, lb, ub),
            options=opts,
        )
    except TypeError:
        res = milp(
            c=c_obj,
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(A, lb, ub),
        )
    if res is None or res.x is None:
        return None
    return np.round(res.x[:n_hh]).astype(int)


def _solve_lp_round_min(C, T, weights, hh_persons=None, pop_target=None,
                        pop_tol=0.0, rep_max=1):
    """LP relaxation + rounding for the minimisation program."""
    from scipy.optimize import linprog

    A, lb, ub, n_hh, n_cells = _assemble_min(C, T, hh_persons, pop_target, pop_tol)
    # linprog wants A_ub x <= b_ub and A_eq x == b_eq; fold the two-sided rows.
    A = A.tocsr()
    finite_ub = np.isfinite(ub)
    finite_lb = np.isfinite(lb)
    A_ub = A[finite_ub]
    b_ub = ub[finite_ub]
    if finite_lb.any():
        A_ub = __import__("scipy.sparse", fromlist=["vstack"]).vstack(
            [A_ub, -A[finite_lb]], format="csr")
        b_ub = np.concatenate([b_ub, -lb[finite_lb]])
    c_obj = np.zeros(n_hh + n_cells)
    c_obj[n_hh:] = weights
    bounds = [(0, max(rep_max, 1))] * n_hh + [(0, None)] * n_cells
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method="highs")
    except Exception:
        return None
    if not res.success:
        return None
    return np.round(res.x[:n_hh]).astype(int)


def _solve_lp_round(C, T, tol_vec, abs_floor=0.0):
    """LP relaxation + rounding fallback. Returns (x, exact) or None."""
    from scipy.optimize import linprog
    import scipy.sparse as sp

    A, b, n_hh, n_cells = _assemble(C, T, tol_vec, abs_floor=abs_floor)
    c_obj = np.zeros(n_hh + n_cells)
    c_obj[n_hh:] = 1.0
    bounds = [(0, 1)] * n_hh + [(0, None)] * n_cells
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    except Exception:
        return None
    if not res.success:
        return None
    x = np.round(res.x[:n_hh]).astype(int)
    return x


def _achieved_rel_dev(C, T, x):
    """Max relative deviation of the selected households vs targets."""
    got = (C @ x).A.ravel() if hasattr(C @ x, "A") else np.asarray(C @ x).ravel()
    denom = np.maximum(np.abs(T), 1.0)
    return float(np.max(np.abs(got - T) / denom))


def drop_inconsistent_households(cand, hh_keys):
    """Drop candidate households whose members disagree on household attrs.

    Household-level attributes (bedrooms, composition, vehicles, ...) must be
    identical for every member of a household; mixed rows are data errors and
    would make the MIP match a household against two cells at once. Returns
    (clean_df, n_dropped_households, n_dropped_rows).
    """
    if "abshid" not in cand.columns:
        return cand, 0, 0
    hh_cols = [c for c in _hh_pool_cols(cand, hh_keys) if c in cand.columns]
    if not hh_cols:
        return cand, 0, 0
    nun = cand.groupby("abshid")[hh_cols].nunique(dropna=False)
    bad_ids = set(nun.index[(nun > 1).any(axis=1)].tolist())
    if not bad_ids:
        return cand, 0, 0
    keep = ~cand["abshid"].isin(bad_ids)
    return (cand[keep].reset_index(drop=True), len(bad_ids),
            int((~keep).sum()))


def fit_zone(pool, targets, ind_keys, hh_keys, tol0=0.05,
             relax_factor=2.0, max_relax=5, time_limit=None,
             allow_lp_fallback=True, hh_tol_mult=2.0, abs_floor=2.0,
             fit_mode="minimize", objective="table", hh_weight=1.0, pop_tol=0.02,
             rep_max=1, solver="auto"):
    """Select households for one zone.

    fit_mode="minimize" (default)
        Solve a single program that *minimises* the deviations themselves:
            min  sum_c w_c * d_c   s.t.  |Cx - T| <= d,  x binary,  d >= 0
        There is no tolerance constraint, so the result is the best fit
        household atomicity allows. Reported metric = achieved max relative
        deviation. This replaces the old relax-until-feasible loop, which pinned
        417/436 zones at the 1.6 ceiling.

    fit_mode="tolerance"
        Legacy path: |Cx - T| <= d <= max(tol*T, abs_floor), relaxing tol.

    objective: "minimax" minimises the worst relative cell deviation,
               "table" uses table-normalised relative weights, "rel" uses
               equal relative weights, and "abs" uses absolute weights.
    hh_weight: extra multiplier on household-cell deviations.

    Returns (x, reported_metric, exact) or None. In minimize mode
    reported_metric is the achieved max relative deviation.
    """
    if not targets:
        return None
    C, hh_ids, cells = build_constraint_matrix(pool, targets, ind_keys, hh_keys)
    T = np.array([targets[c] for c in cells], dtype=float)
    if C.shape[1] == 0:
        return None
    hh_attr_set = set(_hh_pool_cols(pool, hh_keys))

    if fit_mode == "minimize":
        objective_name = "table" if objective == "minimax" else objective
        weights = _cell_weights(cells, T, objective_name)
        if hh_weight != 1.0:
            weights = weights * np.array(
                [hh_weight if c[0] in hh_attr_set else 1.0 for c in cells])

        # Population anchor: persons per household, aligned to hh_ids, versus
        # the census person total for the zone (largest single person table).
        hh_persons = None
        pop_target = None
        if pop_tol > 0 and "abshid" in pool.columns:
            sizes = pool.groupby("abshid").size()
            hh_persons = sizes.reindex(hh_ids).fillna(1.0).to_numpy(dtype=float)
            totals = {}
            for (attr, _cat), t in zip(cells, T):
                if attr in hh_attr_set:
                    continue
                totals[attr] = totals.get(attr, 0.0) + float(t)
            if totals:
                pop_target = max(totals.values())

        if objective == "minimax":
            x = None if solver == "lp" else _solve_mip_minimax(
                C, T, time_limit=time_limit, hh_persons=hh_persons,
                pop_target=pop_target, pop_tol=pop_tol, rep_max=rep_max)
        else:
            x = None if solver == "lp" else _solve_mip_min(
                C, T, weights, time_limit=time_limit, hh_persons=hh_persons,
                pop_target=pop_target, pop_tol=pop_tol, rep_max=rep_max)
        exact = x is not None
        if x is None and (allow_lp_fallback or solver == "lp"):
            x = _solve_lp_round_min(C, T, weights, hh_persons=hh_persons,
                                    pop_target=pop_target, pop_tol=pop_tol,
                                    rep_max=rep_max)
            exact = False
        if x is None or _all_zero(x):
            return None
        stats = _deviation_stats(C, T, x, hh_attr_set, cells)
        return x, stats["rel_max"], exact

    tol = tol0
    tol = max(tol, 1e-6)
    for _ in range(max_relax + 1):
        tol_vec = _tol_vector(cells, hh_attr_set, tol, hh_tol_mult)
        x = None if solver == "lp" else _solve_mip(
            C, T, tol_vec, time_limit=time_limit, abs_floor=abs_floor)
        # An all-empty selection is never a valid fit here (fit_zone is only
        # called with positive targets); reject it so the solver keeps
        # searching instead of silently emitting a 0-person output.
        if x is not None and not _all_zero(x):
            return x, tol, True
        if allow_lp_fallback:
            x = _solve_lp_round(C, T, tol_vec, abs_floor=abs_floor)
            if x is not None and not _all_zero(x):
                # verify the rounded solution really meets the allowance
                allowed = _abs_tol(cells, tol_vec, T, abs_floor)
                got = np.asarray(C @ x).ravel()
                if np.all(np.abs(got - T) <= allowed * 1.0001):
                    return x, tol, False
        tol *= relax_factor

    # Last resort: LP relaxation at a very loose tolerance.
    loose = max(tol, 1.0)
    x = _solve_lp_round(C, T, _tol_vector(cells, hh_attr_set, loose, hh_tol_mult),
                        abs_floor=abs_floor)
    if x is not None and not _all_zero(x):
        return x, loose, False
    return None


def drop_unfillable(cand, targets):
    """Split targets into fittable cells and cells no candidate can fill.

    Cells with a positive target but zero matching candidates (e.g. overseas /
    not-applicable codes absent from the resident pool) would otherwise force
    tolerance relaxation to >= 1.0 on every zone and waste solver rounds
    proving infeasibility. They are unfillable by construction, so they are
    dropped up front and reported (see the audit: ~6% of cells).

    Returns (kept, n_unfillable, n_trivial) where trivial cells have a
    non-positive target and no candidates (already satisfied).
    """
    kept = {}
    n_unfillable = 0
    n_trivial = 0
    for cell, t in targets.items():
        attr, cat = cell
        if attr not in cand.columns:
            n_unfillable += 1
            continue
        col = cand[attr]
        if isinstance(cat, (set, frozenset, list, tuple)):
            n = int(col.isin(list(cat)).sum())
        else:
            n = int((col == cat).sum())
        if n == 0:
            if t > 0:
                n_unfillable += 1
            else:
                n_trivial += 1
            continue
        kept[cell] = t
    return kept, n_unfillable, n_trivial


def drop_saturated_zero_cells(cand, targets):
    """Drop zero-target cells that candidate households inevitably carry.

    The MIP fits each cell to |Cx - T| <= tol*T.  For a cell with T == 0 this
    collapses to Cx == 0: any candidate household carrying that attribute
    combination makes selecting it infeasible.  Zones whose control tables mark
    combinations as absent, but whose REGUCP-wide candidate pool still carries
    them, therefore admit only the all-empty selection -- the solver returns 0
    households / 0 persons and cp_gap_report flags a header-only output.

    Such cells are unsatisfiable by construction (the pool cannot avoid them),
    so they are dropped from the hard constraint set.  The count is returned so
    the caller can log it; the resulting non-empty selection is what the SA /
    infill stage is expected to reconcile.
    """
    keep = {}
    n_dropped = 0
    for (attr, cat), t in targets.items():
        if t > 0:
            keep[(attr, cat)] = t
            continue
        col = cand[attr] if attr in cand.columns else None
        if col is None:
            # Attribute absent from the pool: Cx == 0 holds trivially, keep it.
            keep[(attr, cat)] = t
            continue
        if isinstance(cat, (set, frozenset, list, tuple)):
            n = int(col.isin(list(cat)).sum())
        else:
            n = int((col == cat).sum())
        if n > 0:
            n_dropped += 1
        else:
            keep[(attr, cat)] = t
    return keep, n_dropped


def _all_zero(x):
    """True when a solution vector selects no household at all."""
    return x is not None and not bool(np.any(np.asarray(x) > 0))


def materialize(pool, hh_ids, x, rng=None):
    """Emit person rows for the selected households, honouring multiplicities.

    x_h is the number of real households candidate h stands for. Each replica
    gets a fresh abshid so downstream counts see distinct households and the
    internal attribute consistency of the replica is preserved.
    """
    if len(hh_ids) == 0:
        return pool.iloc[0:0].copy()
    mult = {h: int(round(float(xi))) for h, xi in zip(hh_ids, x) if xi and xi >= 1}
    if not mult:
        return pool.iloc[0:0].copy()

    frames = []
    next_id = int(pd.to_numeric(pool["abshid"], errors="coerce").max() or 0) + 1
    for hh, k in mult.items():
        if k <= 0:
            continue
        block = pool[pool["abshid"] == hh]
        if block.empty:
            continue
        for _ in range(k):
            replica = block.copy()
            replica["abshid"] = next_id
            next_id += 1
            frames.append(replica)
    if not frames:
        return pool.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CP pool + MIP zone fitting (include/exclude)")
    ap.add_argument("--pool", required=True, help="CP feasible pool CSV")
    ap.add_argument("--zones", nargs="*", type=str)
    ap.add_argument("--fit-mode", choices=["minimize", "tolerance"],
                    default="minimize",
                    help="minimize (default): solve for the smallest achievable "
                         "deviations (no tolerance constraint); tolerance: legacy "
                         "relax-until-feasible behaviour")
    ap.add_argument("--objective", choices=["minimax", "table", "rel", "abs"],
                    default="minimax",
                    help="minimize-mode objective: minimax = minimise the worst "
                         "relative cell deviation; table = relative weight "
                         "normalised per control table (default, stops small "
                         "tables outvoting the population tables); rel = 1/max(T,1); "
                         "abs = unweighted")
    ap.add_argument("--pop-tol", type=float, default=0.02,
                    help="Anchor selected persons to the census zone total "
                         "within +-POP_TOL (0 disables)")
    ap.add_argument("--rep-max", type=int, default=40,
                    help="Max replicas per candidate household. The microdata is "
                         "a ~5%% census sample, so a sampled household must stand "
                         "for several real ones (default 40).")
    ap.add_argument("--hh-weight", type=float, default=1.0,
                    help="Extra objective weight on household-attribute cells")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="Initial tolerance (tolerance mode only)")
    ap.add_argument("--hh-tol-mult", type=float, default=2.0,
                    help="Household-attribute cells are fitted with tol*HH_TOL_MULT "
                         "(tolerance mode only)")
    ap.add_argument("--abs-floor", type=float, default=2.0,
                    help="Absolute deviation allowance per cell (tolerance mode only): "
                         "allowed = max(tol*T, ABS_FLOOR)")
    ap.add_argument("--relax-factor", type=float, default=2.0)
    ap.add_argument("--max-relax", type=int, default=5)
    ap.add_argument("--no-lp-fallback", action="store_true",
                    help="Fail instead of falling back to LP relaxation + rounding")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="Per-zone MIP time limit in seconds (HiGHS)")
    ap.add_argument("--max-candidates", type=int, default=150000,
                    help="Absolute ceiling on candidate households per zone "
                         "(default 150000; the Oct-2026 OOMs used 488k)")
    ap.add_argument("--cand-mult", type=float, default=10.0,
                    help="Target-scaled cap = mult * zone_persons / 2.6")
    ap.add_argument("--cand-floor", type=int, default=20000,
                    help="Minimum candidate households kept regardless of target")
    ap.add_argument("--csf-map", default=None, help="Path to csf_sa.csv (SA2->REGUCP)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver", choices=["auto", "mip", "lp"], default="auto",
                    help="auto (default) uses MIP with LP fallback; lp forces "
                         "the LP relaxation followed by integerisation")
    ap.add_argument("--results-dir", default=None,
                    help="Override output directory (production default unchanged)")
    a = ap.parse_args()
    global RESULTS_DIR
    if a.results_dir:
        RESULTS_DIR = os.path.abspath(a.results_dir)
        os.makedirs(RESULTS_DIR, exist_ok=True)

    from helper_functions import Helper
    pool = Helper().clean_my_datagan(a.pool)
    print(f"CP pool loaded: {len(pool):,} person rows, "
          f"{pool['abshid'].nunique():,} households" if "abshid" in pool.columns
          else f"CP pool loaded: {len(pool):,} rows")

    ind_tables, hh_tables = load_control_tables()
    ind_keys = sorted({k for t in ind_tables.values() for k in t.get("keys", [])
                       if k in pool.columns})
    hh_keys = sorted({k for t in hh_tables.values() for k in t.get("keys", [])
                      if k in pool.columns})
    print(f"  individual tables {len(ind_tables)} (keys {ind_keys})")
    print(f"  household  tables {len(hh_tables)} (keys {hh_keys})")

    sa2_to_regucp, _ = _load_sa2_to_regucp(a.csf_map)

    zones = a.zones or [str(z) for z in
                        next(iter(ind_tables.values()))["data"]["sa2"].dropna().unique().tolist()]

    rows = []
    for zone in zones:
        audit_base = {
            "zone": zone,
            "pool_stage": "household-first",
            "household_atomic": True,
            "integerised_output": True,
            "inconsistent_households_dropped": 0,
            "inconsistent_person_rows_dropped": 0,
            "unfillable_control_cells": 0,
            "saturated_zero_cells": 0,
        }
        tgt = {**zone_targets(hh_tables, zone), **zone_targets(ind_tables, zone)}
        if not tgt:
            print(f"Zone {zone}: no control targets — skipped")
            rows.append({**audit_base, "status": "failed",
                         "failure_reason": "no_control_targets",
                         "households_selected": 0,
                         "candidate_types_selected": 0, "persons": 0,
                         "solver": "LP relaxation + integerisation" if a.solver == "lp" else "MIP"})
            continue
        cand, how = zone_candidate_pool(pool, zone, sa2_to_regucp)
        if len(cand) == 0:
            # No fallback to the global pool: the zone's SA4 slice is the
            # candidate universe, and an empty slice is a structural residual.
            print(f"Zone {zone}: {how} — structural residual, zone skipped")
            rows.append({**audit_base, "status": "failed",
                         "failure_reason": "empty_zone_candidate_slice",
                         "candidate_source": how,
                         "households_selected": 0,
                         "candidate_types_selected": 0, "persons": 0,
                         "solver": "LP relaxation + integerisation" if a.solver == "lp" else "MIP"})
            continue
        cand, n_bad_hh, n_bad_rows = drop_inconsistent_households(
            cand, hh_keys)
        if n_bad_hh:
            print(f"Zone {zone}: dropped {n_bad_hh} inconsistent households "
                  f"({n_bad_rows} rows), candidates={how}")
        audit_base["inconsistent_households_dropped"] = n_bad_hh
        audit_base["inconsistent_person_rows_dropped"] = n_bad_rows
        # Bound the program: person target = largest person-table total.
        _person_totals = {}
        for (attr, _cat), cnt in tgt.items():
            if attr in ind_keys:
                _person_totals[attr] = _person_totals.get(attr, 0) + cnt
        _person_target = max(_person_totals.values()) if _person_totals else 0
        cand, _cap_note = cap_candidates(
            cand, _person_target, max_candidates=a.max_candidates,
            cand_mult=a.cand_mult, cand_floor=a.cand_floor, seed=a.seed)
        if "capped" in _cap_note:
            print(f"Zone {zone}: {_cap_note}")
            how = f"{how} [{_cap_note}]"
        tgt, n_unfill, n_triv = drop_unfillable(cand, tgt)
        if n_unfill or n_triv:
            print(f"Zone {zone}: dropped {n_unfill} unfillable + {n_triv} trivial "
                  f"cells, fitting {len(tgt)}")
        audit_base["unfillable_control_cells"] = n_unfill
        tgt, n_zero = drop_saturated_zero_cells(cand, tgt)
        if n_zero:
            print(f"Zone {zone}: dropped {n_zero} zero-target cells that the "
                  f"candidate pool cannot avoid (unsatisfiable hard-zero); "
                  f"fitting {len(tgt)}")
        audit_base["saturated_zero_cells"] = n_zero
        if not tgt:
            print(f"Zone {zone}: no fittable targets — skipped")
            rows.append({**audit_base, "status": "failed",
                         "failure_reason": "no_fittable_targets",
                         "households_selected": 0,
                         "candidate_types_selected": 0, "persons": 0,
                         "solver": "LP relaxation + integerisation" if a.solver == "lp" else "MIP"})
            continue
        res = fit_zone(cand, tgt, ind_keys, hh_keys, tol0=a.tol,
                       relax_factor=a.relax_factor, max_relax=a.max_relax,
                       time_limit=a.time_limit,
                       allow_lp_fallback=not a.no_lp_fallback,
                       hh_tol_mult=a.hh_tol_mult, abs_floor=a.abs_floor,
                       fit_mode=a.fit_mode, objective=a.objective,hh_weight=a.hh_weight, pop_tol=a.pop_tol,
                       rep_max=a.rep_max, solver=a.solver)
        if res is None:
            print(f"Zone {zone}: no selection found")
            rows.append({**audit_base, "status": "failed",
                         "failure_reason": "solver_returned_no_selection",
                         "households_selected": 0,
                         "candidate_types_selected": 0, "persons": 0,
                         "tolerance": np.nan,
                         "solver": "LP relaxation + integerisation" if a.solver == "lp" else "MIP",
                         "max_rel_dev": np.nan, "mean_rel_dev": np.nan,
                         "candidate_source": how})
            continue
        x, metric, exact = res
        C, hh_ids, cells = build_constraint_matrix(cand, tgt, ind_keys, hh_keys)
        T = np.array([tgt[c] for c in cells], dtype=float)
        hh_set = set(_hh_pool_cols(cand, hh_keys))
        stats = _deviation_stats(C, T, x, hh_set, cells)
        pop = materialize(cand, hh_ids, x, rng=np.random.default_rng(a.seed))
        # x[h] is the *multiplicity* of candidate h -- how many real households it
        # stands for -- and materialize() emits one replica per unit, each with a
        # fresh abshid.  Counting (x >= 1) therefore counts candidate *types*, not
        # households: Kilcoy (313031370) reported "201 households" next to 5,793
        # persons when its population actually held 2,183 households of 2.65
        # persons.  Report what the population contains and keep the type count
        # alongside it, since a type-level count reads as an impossible
        # persons-per-household ratio anywhere it is plotted next to persons.
        n_types = int((np.asarray(x) >= 1).sum())
        n_hh = int(pop["abshid"].nunique()) if "abshid" in pop.columns else n_types
        pop.to_csv(os.path.join(RESULTS_DIR, f"zone_{zone}_synthetic_population.csv"),
                   index=False)
        rows.append({**audit_base, "households_selected": n_hh,
                     "candidate_types_selected": n_types, "persons": len(pop),
                     "tolerance": metric,
                     "status": "success", "failure_reason": "",
                     "solver": ("MIP" if exact else "LP relaxation + integerisation")
                     if a.solver != "lp" else "LP relaxation + integerisation",
                     "max_rel_dev": round(stats["rel_max"], 4),
                     "mean_rel_dev": round(stats["rel_mean"], 4),
                     "max_rel_dev_hh": round(stats.get("rel_max_hh", np.nan), 4),
                     "max_rel_dev_person": round(stats.get("rel_max_person", np.nan), 4),
                     "candidate_source": how})
        print(f"Zone {zone}: {n_hh:,} households "
              f"({n_types:,} distinct candidate types) / {len(pop):,} persons | "
              f"maxdev={stats['rel_max']:.3f} meandev={stats['rel_mean']:.3f} "
              f"({a.fit_mode}/{a.objective}) {'MIP' if exact else 'LP'} | candidates={how}")

    if rows:
        out = pd.DataFrame(rows)
        # When several chunks run concurrently they all share RESULTS_DIR.  A
        # single shared cp_mip_summary.csv is then a race: each job overwrites
        # it and the per-chunk copy can pick up another chunk's rows.  Write
        # the chunk's own file directly instead (mirrors cp_pool_sa.py).
        _chunk = os.environ.get("CHUNK_ID", "").strip()
        _summary_name = (f"cp_mip_summary_chunk{_chunk}.csv" if _chunk
                         else "cp_mip_summary.csv")
        out.to_csv(os.path.join(RESULTS_DIR, _summary_name), index=False)
        out.loc[out["status"] != "success"].to_csv(
            os.path.join(RESULTS_DIR, f"zone_failure_manifest{'_chunk' + _chunk if _chunk else ''}.csv"),
            index=False
        )
        ok = out["households_selected"] > 0
        print(f"\nWrote {len(out)} zone rows ({int(ok.sum())} solved) to "
              f"{RESULTS_DIR}/{_summary_name}")
        if ok.any():
            print(f"  median reported metric: {out.loc[ok, 'tolerance'].median():.3f}")
            print(f"  median max rel dev    : {out.loc[ok, 'max_rel_dev'].median():.3f}")
            if "mean_rel_dev" in out.columns:
                print(f"  median mean rel dev   : {out.loc[ok, 'mean_rel_dev'].median():.3f}")


if __name__ == "__main__":
    main()

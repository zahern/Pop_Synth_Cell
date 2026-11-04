"""mip_pool_joint.py - joint-marginal feasible pool expansion.

The legacy ``mip_pool.py`` loop solves one small Hamming program per deficit
cell.  That answers "can a feasible row carry this cell?" but not "do the added
rows together represent the zone's marginal totals?".  This module keeps the
per-candidate repair (it is the only place a rule conflict is resolved) and
moves the decision to one selection program per zone:

    variables  x_j in {0,1}      candidate household selected
               e_c >= 0          coverage slack below nu_c
               d_{k,m} >= 0      joint marginal deviation

    min        lambda_0 sum e_c + sum w_{k,m} d_{k,m} + lambda_1 sum x_j
    s.t.       sum_j a_{j,c} x_j + e_c >= nu_c        zero cells
               |sum_j (a_{j,k,m} - q_{k,m} u_{j,k}) x_j| <= d_{k,m}
               sum_j x_j <= K_z, sum_j s_j x_j <= P_add
               sum_{j in shell h} x_j <= 1

with nu_c = ceil(E_{z,k,m} / rep_max) and K_z = sum_c nu_c: the number of
distinct carrier contributions is set by the cell target, not by a fixed budget.

Every deviation is computed from the same x for every control table, so the
selected agents are fitted to the zone's marginal composition jointly rather
than cell by cell.  Coverage is soft: an unsourceable cell is reported, not
allowed to make the program infeasible; the model must supply nu_c carriers per
zero cell but is not asked to close the full census shortfall at pool scale
(that is the fitter's job).

Candidate construction
----------------------
For each zero-capacity cell the module samples real census donor records that
carry the cell, pairs each donor member profile with a real local shell household
(same REGUCP, donor's RLHP role), and asks the legacy completion program for the
closest feasible profile.  The donor profile may come from the wider census
population, which is the point of expansion: it supplies a cell the SA4 sample
lacks.  The shell stays local, and every constructed record is tagged to the
target SA2 with a fresh identifier, so the SA2 pool grows without sharing the
record with a sibling SA2.  The target cell is a hard constraint in that repair,
so a candidate is emitted only if it actually covers the cell.  Whole real
households are copied verbatim for attributes outside the person model.  Every
candidate passes the shared CONSTRAINT_CHECKS.

The full constructed menu can be appended to the pool (default), not only the
coverage-minimal selection, so the SA2 has more candidates to fit from.

Usage
-----
  python mip_pool_joint.py --pool outputs/feasible_pool.csv \
      --census LINKED_DATA_clean.csv --rep-max 40 \
      --out outputs/feasible_pool_joint.csv

The per-zone budget is marginal-driven by default: each sample-zero cell with
target ``T`` requires ``ceil(T / rep_max)`` additional households, and the
zone's budget is that sum.  ``--max-transplant-cells``, ``--max-new-households``
and ``--max-new-persons`` are retained only as explicit legacy ceilings.
"""

import argparse
import json
import os
import sys
import time
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config
import mip_fit as mipf
import mip_pool as legacy
from sa_only_baseline import _load_sa2_to_regucp, load_control_tables


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _column(df, attr):
    """Case-insensitive column lookup."""
    if attr in df.columns:
        return attr
    for c in df.columns:
        if str(c).lower() == str(attr).lower():
            return c
    return None


def _as_code(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def block_covers(block, attr, code):
    """True when at least one row of a candidate household carries (attr, code)."""
    col = _column(block, attr)
    if col is None:
        return False
    vals = pd.to_numeric(block[col], errors="coerce")
    return bool((vals == int(code)).any())


_BLOCK_ID_COLUMNS = {
    "abshid", "absfid", "abspid", "persid", "person_id",
    "_source", "_target_sa2", "reindexed_slot",
}


def _block_signature(block):
    """Hashable attribute signature used to keep each SA2's added records unique."""
    columns = [c for c in block.columns
               if str(c).lower() not in _BLOCK_ID_COLUMNS]
    return tuple(tuple(str(v) for v in block[c].tolist()) for c in columns)


def _donor_profile(row):
    """The person-level donor profile used as the repair target."""
    out = {}
    for a in legacy.MODELED_PERSON:
        if a in row.index and pd.notna(row[a]):
            code = _as_code(row[a])
            if code is not None:
                out[a] = code
    return out


# ---------------------------------------------------------------------------
# Candidate menu
# ---------------------------------------------------------------------------

def _construct_source(block):
    """How a candidate household was built, from the block's own tag."""
    if "_source" in block.columns and len(block):
        tag = str(block["_source"].iloc[0])
        if tag == "mip-growth":
            return "growth"
        if tag == "mip-clone":
            return "clone"
    return "transplant"


def _candidate_from_donor(census, index, donor_profile, reg, attr, code,
                          global_dom, rlhp_dom, cuts, candidate_id,
                          shells_per_cell, size_preference=None,
                          shell_growth=False, family_clone=False,
                          donor_rows=None):
    """One donor/shell candidate, or None when the shell cannot host it."""
    role = _as_code(donor_profile.get("RLHP"))
    if role is None:
        return None
    required = {str(attr): int(code)}
    if shell_growth:
        # A grown member carries the required cell itself, so only a
        # dwelling-level requirement constrains the shell; the donor's role
        # need not be present, because the member is added rather than slotted
        # into an existing role.
        dwelling = str(attr).upper() in legacy.MODELED_DWELL
        grow_shells = legacy.choose_shells(
            index, reg, role if dwelling else None,
            max_shells=shells_per_cell,
            required=required if dwelling else None,
            size_preference=size_preference,
            any_of=legacy.FAMILY_HOST_ROLES, host_checks=True)
        grown = (legacy.transplant_growth(
            census, grow_shells, donor_rows, reg, global_dom, rlhp_dom,
            cuts, candidate_id, required=required) if grow_shells else None)
        if grown is not None:
            block, dev, shell, _family = grown
            if block_covers(block, attr, code):
                block["ABSHID"] = candidate_id
                return block, int(dev), shell
    if family_clone:
        # As for growth, a cloned carrier carries a person-level requirement
        # itself; a dwelling-level one must already be in the shell, because the
        # clone copies the shell's dwelling terms.
        dwelling = str(attr).upper() in legacy.MODELED_DWELL
        clone_shells = legacy.choose_shells(
            index, reg, None, max_shells=shells_per_cell,
            required=required if dwelling else None,
            size_preference=size_preference,
            any_of=legacy.FAMILY_HOST_ROLES, host_checks=True)
        cloned = (legacy.transplant_family_clone(
            census, clone_shells, donor_rows, reg, global_dom, rlhp_dom,
            cuts, candidate_id, required=required) if clone_shells else None)
        if cloned is not None:
            block, dev, shell, _family = cloned
            if block_covers(block, attr, code):
                block["ABSHID"] = candidate_id
                return block, int(dev), shell
    shells = legacy.choose_shells(index, reg, role,
                                  max_shells=shells_per_cell,
                                  required=required,
                                  size_preference=size_preference)
    if not shells:
        return None
    res = legacy.transplant_member(
        census, shells, index, donor_profile, reg, global_dom, rlhp_dom,
        cuts, candidate_id, required=required)
    if res is None:
        return None
    block, dev, shell = res
    if not block_covers(block, attr, code):
        return None
    block = block.drop(columns=["reindexed_slot"], errors="ignore")
    block["ABSHID"] = candidate_id
    return block, int(dev), shell


def build_candidate_menu(census, index, cells, reg, global_dom, rlhp_dom,
                         cuts, seed=0, donors_per_cell=4,
                         shells_per_cell=8, allow_global_donors=True,
                         cell_need=None, size_preference=None,
                         shell_growth=False, family_clone=False):
    """Feasible donor/shell candidates for a zone's zero-capacity cells.

    By default the shell is local (same REGUCP) but the donor member profile is
    sourced through the global donor ladder, so a cell the SA4 slice lacks can
    still be constructed from the wider population.  Every constructed record is
    tagged to this SA2 and receives a fresh identifier, so sibling SA2s do not
    share it.  ``allow_global_donors=False`` restricts the donor to the SA4
    slice and leaves unsourceable cells as structural residuals.

    ``cell_need`` maps each cell to its carrier requirement ``nu_c``.  At least
    that many donor variants (and shells) are constructed for the cell, so the
    selection program can reach the marginal-derived coverage rather than being
    capped by ``donors_per_cell``.

    Returns (blocks, meta, skipped).  ``cells`` is a list of (attr, code)
    tuples at person or household level; band cells have already been removed.
    """
    blocks, meta = [], []
    skipped = defaultdict(int)
    donor_cache = {}
    candidate_id = -1
    rng = np.random.default_rng(int(seed))
    padding = legacy._padding_codes()

    reg_col = _column(census, "regucp")
    local_frame = (census.iloc[0:0] if reg_col is None
                   else census[census[reg_col].astype(str) == str(reg)])

    for attr, code in cells:
        attr = str(attr).lower()
        try:
            code = int(code)
        except (TypeError, ValueError):
            skipped["non_integer_code"] += 1
            continue
        attr_col = _column(census, attr)
        if attr_col is None:
            skipped["attr_not_in_census"] += 1
            continue
        if (attr, code) in padding:
            skipped["padding_not_applicable"] += 1
            continue

        key = (str(reg), attr, code, bool(allow_global_donors))
        donors = donor_cache.get(key)
        shifted = False
        if donors is None:
            # Prefer the local SA4 donor; fall back to the wider census only
            # when the policy allows it.  The fallback is what lets the pool
            # reflect an SA2 target the SA4 sample cannot supply.
            donors = local_frame[local_frame[attr_col] == code]
            if donors.empty and allow_global_donors:
                donors = census[census[attr_col] == code]
            # Ordinal-adjacent fill: when a genuinely ordinal band (income,
            # hours worked, or qualification level) is carried by no record in
            # the local slice, borrow the nearest applicable band and shift the
            # value to the target.  Try local first, then global if allowed.
            if donors.empty and attr_col in legacy._ORDINAL_ATTRS:
                special = legacy._ORDINAL_SPECIAL.get(attr_col, set())
                if code not in special:
                    nb = legacy._nearest_ordinal_donors(
                        local_frame, attr_col, code, special)
                    if (nb is None or nb.empty) and allow_global_donors:
                        nb = legacy._nearest_ordinal_donors(
                            census, attr_col, code, special)
                    if nb is not None and not nb.empty:
                        donors = nb.copy()
                        donors[attr_col] = code
                        shifted = True
            donor_cache[key] = donors
        if shifted:
            skipped["ordinal_shift_cells"] += 1
        if donors.empty:
            skipped["no_sa4_donor" if not allow_global_donors
                    else "no_donor_anywhere"] += 1
            continue

        # Prefer donors whose household role actually exists in a local shell.
        # A donor with RLHP=11 (not applicable) has no family shell in many
        # regions; keeping only role-compatible donors lets the transplant
        # path run instead of falling back to a whole verbatim household.
        # The construction modes add the role instead of filling it, so the
        # filter is only meaningful for the slot path.
        if ("RLHP" in donors.columns and index.role_ids
                and not (shell_growth or family_clone)):
            reg_key = str(legacy._norm_scalar(reg))
            valid_roles = {r for (rr, r), ids in index.role_ids.items()
                           if rr == reg_key and ids}
            if valid_roles:
                role_ok = pd.to_numeric(donors["RLHP"], errors="coerce").isin(
                    valid_roles)
                filtered = donors.loc[role_ok]
                if not filtered.empty:
                    donors = filtered

        made = 0
        need_here = max(int((cell_need or {}).get((attr, code), 0)),
                        int(donors_per_cell))
        if len(donors):
            n_take = min(max(int(donors_per_cell), need_here), len(donors))
            salt = zlib.crc32(f"{reg}:{attr}:{code}".encode("utf-8"))
            sample = donors.sample(
                n=n_take,
                random_state=(int(seed) + int(salt)) % (2**31 - 1))
            for _, donor in sample.iterrows():
                profile = _donor_profile(donor)
                if "RLHP" not in profile:
                    skipped["no_rlhp"] += 1
                    continue
                res = _candidate_from_donor(
                    census, index, profile, reg, attr, code,
                    global_dom, rlhp_dom, cuts, candidate_id,
                    max(int(shells_per_cell), need_here),
                    size_preference=size_preference,
                    shell_growth=shell_growth, family_clone=family_clone,
                    donor_rows=donors)
                if res is None:
                    continue
                block, dev, shell = res
                source = _construct_source(block)
                block["_construct_source"] = source
                blocks.append(block)
                meta.append({"cell": f"{attr}:{code}", "shell": shell,
                             "source": source, "deviation": dev})
                candidate_id -= 1
                made += 1

        # Fill the remaining carrier requirement with whole real donor
        # households.  This path is essential for the "not applicable" /
        # "not stated" codes: they exist in the census but rarely admit the
        # person-level repair, and the earlier one-copy cap left nu_c unmet.
        if made < need_here:
            frames, candidate_id = legacy.verbatim_transplant(
                census, donors, reg, candidate_id,
                max(1, need_here - made), rng,
                size_preference=size_preference)
            for block in frames:
                block = block.drop(columns=["reindexed_slot"],
                                   errors="ignore").copy()
                if not block_covers(block, attr, code):
                    continue
                donor_shell = block["ABSHID"].iloc[0]
                block["_construct_source"] = "verbatim"
                blocks.append(block)
                meta.append({"cell": f"{attr}:{code}", "shell": donor_shell,
                             "source": "verbatim", "deviation": 0})
                made += 1

        if made == 0:
            skipped["unsourceable"] += 1

    return blocks, meta, dict(skipped)


def _candidate_pool(blocks):
    """Lower-case one concatenated candidate pool for the constraint matrix."""
    if not blocks:
        return None
    cand = pd.concat(blocks, ignore_index=True)
    if "ABSHID" not in cand.columns:
        cand["ABSHID"] = np.arange(len(cand))
    cand.columns = [str(c).lower() for c in cand.columns]
    if "abshid" not in cand.columns:
        cand["abshid"] = np.arange(len(cand))
    return cand


def _candidate_matrix(cand_lc, targets, ind_keys, hh_keys):
    C, hh_ids, cells = mipf.build_constraint_matrix(
        cand_lc, targets, ind_keys, hh_keys)
    sizes = (cand_lc.groupby("abshid").size()
             .reindex(hh_ids).fillna(1).to_numpy(dtype=float))
    return C, hh_ids, cells, sizes


# ---------------------------------------------------------------------------
# Joint selection MIP
# ---------------------------------------------------------------------------

def _table_shares(cells, T):
    counts = defaultdict(int)
    totals = defaultdict(float)
    for (attr, _cat), t in zip(cells, T):
        counts[attr] += 1
        totals[attr] += max(float(t), 0.0)
    q = np.zeros(len(cells), dtype=float)
    for i, (attr, _cat) in enumerate(cells):
        if totals[attr] > 0:
            q[i] = max(float(T[i]), 0.0) / totals[attr]
    return q, counts


def _selection_stats(x, A, B, w, zero_mask, sizes, zero_need=None):
    x = np.asarray(x, dtype=float)
    zero_mask = np.asarray(zero_mask, dtype=bool)
    sel = np.flatnonzero(x > 0)
    if len(sel):
        contribution = A[sel][:, zero_mask].sum(axis=0)
        covered = contribution > 0
    else:
        contribution = np.zeros(int(zero_mask.sum()), dtype=float)
        covered = np.zeros(int(zero_mask.sum()), dtype=bool)
    if zero_need is None:
        requirement = np.ones(int(zero_mask.sum()), dtype=float)
    else:
        requirement = np.asarray(zero_need, dtype=float)
    dev = np.abs(B.T.dot(x))
    return {
        "n_selected_households": int(len(sel)),
        "n_selected_persons": int(float(np.dot(np.asarray(sizes, dtype=float), x))),
        "n_zero_cells": int(zero_mask.sum()),
        "n_zero_cells_covered": int(covered.sum()),
        "n_zero_cells_uncovered": int((~covered).sum()),
        "n_zero_cells_at_requirement": int(
            (contribution >= requirement).sum()),
        "n_zero_cells_below_requirement": int(
            (contribution < requirement).sum()),
        "marginal_deviation_sum": float(np.dot(w, dev)),
        "marginal_deviation_mean": float(dev.mean()) if dev.size else 0.0,
        "marginal_deviation_max": float(dev.max()) if dev.size else 0.0,
    }


def _deviation_terms(A, T, cells, sizes, hh_attr_set):
    """Return the share-deviation matrix B and its table-normalised weights."""
    q, counts = _table_shares(cells, T)
    B = A.copy()
    is_hh = np.array([c[0] in hh_attr_set for c in cells], dtype=bool)
    sizes = np.asarray(sizes, dtype=float)
    for i in range(len(cells)):
        if is_hh[i]:
            B[:, i] -= q[i]
        else:
            B[:, i] -= q[i] * sizes

    # Table-normalised relative weights: every control table contributes
    # equally, and large person tables do not drown the smaller ones.
    w = np.array([1.0 / (counts[c[0]] * max(float(T[i]), 1.0))
                  for i, c in enumerate(cells)], dtype=float)
    w = w / max(float(w.max()), 1e-12)
    return B, w


def solve_selection(C, T, sizes, cells, zero_mask, shell_ids, hh_attr_set,
                    max_households=64, max_persons=256, time_limit=30.0,
                    zero_need=None, lambda_composition=1.0,
                    max_size7_share=None, max_mtwp17_share=None,
                    min_admitted_households=0):
    """Solve the joint-marginal selection program for one zone.

    Returns (x, stats).  x is a 0/1 vector aligned to C's columns, or None if
    the solver returned no point at all (the caller then uses the greedy
    fallback).

    ``zero_need`` gives, for each zero-capacity cell, the number of distinct
    carriers required so that replication at ``rep_max`` can reach the cell
    target: ``ceil(E / rep_max)``, written ``nu_c`` in the paper.  When
    omitted every zero cell needs one carrier (the legacy behaviour).
    """
    n_cand = C.shape[1]
    n_cells = C.shape[0]
    zero_mask = np.asarray(zero_mask, dtype=bool)
    n_zero = int(zero_mask.sum())
    if n_cand == 0 or n_cells == 0 or n_zero == 0:
        return np.zeros(n_cand, dtype=int), {"solver": "none"}

    zero_need = (np.ones(n_zero, dtype=float) if zero_need is None
                 else np.asarray(zero_need, dtype=float))
    if zero_need.shape[0] != n_zero:
        raise ValueError("zero_need must align with the zero-capacity cells")

    A = C.T.toarray().astype(float)
    B, w = _deviation_terms(A, T, cells, sizes, hh_attr_set)

    # The marginal term is bounded by the selected person count times the
    # number of cells, so 1e5 per unit of coverage makes meeting a zero cell's
    # carrier requirement worth more than any marginal improvement the added
    # set can buy.  Coverage is still soft so an unsourceable cell cannot make
    # the program infeasible, but the marginal-derived budget no longer binds
    # below what sourceable cells require.
    lam_cov = 1e5
    lam_card = 1e-2     # mild parsimony pressure
    n = n_cand + n_zero + n_cells
    c_obj = np.zeros(n)
    c_obj[:n_cand] = lam_card
    c_obj[n_cand:n_cand + n_zero] = lam_cov
    c_obj[n_cand + n_zero:] = w * float(max(0.0, lambda_composition))

    rows, lb, ub = [], [], []

    def add(row, lo, hi):
        rows.append(row)
        lb.append(lo)
        ub.append(hi)

    zero_idx = np.flatnonzero(zero_mask)
    for k, ci in enumerate(zero_idx):
        row = np.zeros(n)
        row[:n_cand] = -A[:, ci]
        row[n_cand + k] = -1.0
        # Requirement in carrier contributions: r_max copies of each selected
        # household must be able to reach the cell target.
        add(row, -np.inf, -float(zero_need[k]))

    for i in range(n_cells):
        row = np.zeros(n)
        row[:n_cand] = B[:, i]
        row[n_cand + n_zero + i] = -1.0
        add(row, -np.inf, 0.0)
        row = np.zeros(n)
        row[:n_cand] = -B[:, i]
        row[n_cand + n_zero + i] = -1.0
        add(row, -np.inf, 0.0)

    row = np.zeros(n)
    row[:n_cand] = 1.0
    add(row, float(max(0, int(min_admitted_households))),
        float(max(1, int(max_households))))
    row = np.zeros(n)
    row[:n_cand] = np.asarray(sizes, dtype=float)
    add(row, -np.inf, float(max(0, int(max_persons))))

    # Optional composition caps on the selected set.  These are hard linear
    # constraints, so use them with --coverage-only (the appended menu is
    # outside y and therefore cannot be capped here).
    def _cell_index(attr, code):
        for i, cell in enumerate(cells):
            if cell == (attr, code):
                return i
        return None

    if max_size7_share is not None:
        idx = _cell_index("nprd", 7)
        if idx is not None:
            row = np.zeros(n)
            row[:n_cand] = A[:, idx] - float(max_size7_share)
            add(row, -np.inf, 0.0)
    if max_mtwp17_share is not None:
        idx = _cell_index("mtwp", 17)
        if idx is not None:
            row = np.zeros(n)
            row[:n_cand] = (A[:, idx]
                            - float(max_mtwp17_share) * np.asarray(sizes))
            add(row, -np.inf, 0.0)

    shell_groups = defaultdict(list)
    for j, shell in enumerate(shell_ids):
        if shell is not None:
            shell_groups[shell].append(j)
    for idxs in shell_groups.values():
        if len(idxs) <= 1:
            continue
        row = np.zeros(n)
        row[idxs] = 1.0
        add(row, -np.inf, 1.0)

    A_mat = sp.csr_matrix(np.vstack(rows))
    constraints = LinearConstraint(A_mat, np.asarray(lb), np.asarray(ub))
    bounds = Bounds(
        np.zeros(n),
        np.concatenate([np.ones(n_cand),
                        np.full(n_zero + n_cells, np.inf)]))
    integrality = np.zeros(n, dtype=int)
    integrality[:n_cand] = 1
    options = {"time_limit": float(time_limit)} if time_limit else {}
    res = milp(c=c_obj, integrality=integrality, bounds=bounds,
               constraints=constraints, options=options)
    if res is None or res.x is None:
        return None, {"solver": "failed"}
    x = np.rint(np.asarray(res.x[:n_cand])).astype(int)
    stats = _selection_stats(x, A, B, w, zero_mask, sizes,
                             zero_need=zero_need)
    stats["solver"] = "optimal" if getattr(res, "success", False) else "incumbent"
    fun = getattr(res, "fun", None)
    stats["objective"] = float(fun) if fun is not None else float("nan")
    return x, stats


def greedy_selection(A, sizes, zero_mask, max_households=64,
                     max_persons=256, shell_ids=None, zero_need=None):
    """Coverage-first fallback when the MIP returns no point.

    This deliberately does not try to optimise the joint marginals: the
    fallback's job is to guarantee that the zero-cell coverage property is not
    lost when the solver fails, and the zone fit still has the final say.  It
    keeps the one-transplant-per-shell rule so the fallback cannot emit two
    variants of the same real dwelling, and it keeps adding distinct carriers
    until each zero cell reaches its marginal requirement.
    """
    n_cand = A.shape[0]
    zero_mask = np.asarray(zero_mask, dtype=bool)
    requirement = (np.ones(int(zero_mask.sum()), dtype=float)
                   if zero_need is None
                   else np.asarray(zero_need, dtype=float))
    selected = []
    selected_set = set()
    used_shells = set()
    covered = np.zeros(A.shape[1], dtype=float)
    persons = 0.0
    budget = max(0, int(max_households))
    while len(selected) < budget:
        best, best_gain, best_dev, best_shell = None, 0, np.inf, None
        pending = covered[zero_mask] < requirement
        for j in range(n_cand):
            if j in selected_set:
                continue
            shell = (shell_ids[j] if shell_ids is not None
                     and j < len(shell_ids) else None)
            if shell is not None and shell in used_shells:
                continue
            gain = int(np.sum((A[j, zero_mask] > 0) & pending))
            if gain <= 0:
                continue
            if persons + float(sizes[j]) > float(max_persons):
                continue
            dev = float(np.abs(A[j]).sum())
            if gain > best_gain or (gain == best_gain and dev < best_dev):
                best, best_gain, best_dev, best_shell = j, gain, dev, shell
        if best is None:
            break
        selected.append(best)
        selected_set.add(best)
        if best_shell is not None:
            used_shells.add(best_shell)
        covered[zero_mask] += (A[best, zero_mask] > 0)
        persons += float(sizes[best])
    x = np.zeros(n_cand, dtype=int)
    if selected:
        x[selected] = 1
    return x


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_joint(pool_path, census_path, rep_max, out_path,
              max_transplant_cells=None, shells_per_cell=8, zones_limit=None,
              max_per_cell=None, pool_nrows=None, census_nrows=None,
              report_path=None, max_new_households=None, max_new_persons=None,
              donors_per_cell=4, time_limit=30.0, seed=0, zones_file=None,
              allow_global_donors=True, append_menu_candidates=True,
              append_only=False, id_offset=0, include_partial_deficits=False,
              size_aware_shells=False, lambda_composition=1.0,
              max_size7_share=None, max_mtwp17_share=None,
              min_admitted_households=0, shell_growth=False,
              family_clone=False):
    """Joint-marginal expansion of a feasible pool.

    ``max_per_cell`` is accepted for CLI compatibility with the legacy driver;
    the joint model bounds the selected set per zone instead.  By default those
    bounds are derived from the zone's marginal gap: each sample-zero cell with
    target ``T`` needs ``ceil(T / rep_max)`` additional households, so the
    household budget is that sum over the zone and the person budget is that
    sum times the structural household-size cap.  Passing an explicit
    ``max_new_households`` or ``max_new_persons`` restores a legacy fixed cap.
    """
    if shell_growth and family_clone:
        raise ValueError("shell_growth and family_clone add different things "
                         "to a household; enable one at a time")
    t0 = time.time()
    report = {
        "mode": "joint",
        "rep_max": int(rep_max),
        "pool": pool_path,
        "census": census_path,
        "relationship_constraints": (
            "constraint_rules.CONSTRAINT_CHECKS plus observed-domain "
            "forbidden-pair cuts"
        ),
        "soft_dag_scoring": False,
        "donor_policy": ("global-donor-ladder" if allow_global_donors
                         else "SA4-strict"),
        "append_menu_candidates": bool(append_menu_candidates),
        "append_only": bool(append_only),
        "include_partial_deficits": bool(include_partial_deficits),
        "size_aware_shells": bool(size_aware_shells),
        "shell_growth": bool(shell_growth),
        "family_clone": bool(family_clone),
        "lambda_composition": float(lambda_composition),
        "max_size7_share": max_size7_share,
        "max_mtwp17_share": max_mtwp17_share,
        "min_admitted_households": int(min_admitted_households),
        "id_offset": int(id_offset),
        "budget": {
            "source": "ceil(gap/rep_max) per sample-zero cell",
            "max_new_households": max_new_households,
            "max_new_persons": max_new_persons,
            "max_transplant_cells": max_transplant_cells,
        },
        "donors_per_cell": int(donors_per_cell),
        "shells_per_cell": int(shells_per_cell),
    }

    pool_kw = {"low_memory": False}
    if pool_nrows:
        pool_kw["nrows"] = int(pool_nrows)
    census_kw = {"low_memory": False}
    if census_nrows:
        census_kw["nrows"] = int(census_nrows)
    pool = pd.read_csv(pool_path, **pool_kw)
    census = pd.read_csv(census_path, **census_kw)
    report["n_pool_rows"] = int(len(pool))
    report["n_census_rows"] = int(len(census))

    global_dom = legacy.observed_domains(census, legacy.MODELED)
    rlhp_dom = legacy.domains_given_rlhp(
        census, [a for a in legacy.MODELED_PERSON if a != "RLHP"])
    cuts = legacy.build_forbidden_pairs(global_dom, verbose=False)
    report["n_forbidden_pair_cuts"] = int(sum(len(v) for v in cuts.values()))

    ind_tables, hh_tables = load_control_tables()
    ind_keys = sorted({k for t in ind_tables.values()
                       for k in t.get("keys", [])})
    hh_keys = sorted({k for t in hh_tables.values()
                      for k in t.get("keys", [])})
    sa2_to_regucp, _csf_src = _load_sa2_to_regucp()
    zones = legacy._table_zones(ind_tables, hh_tables)
    if zones_file:
        with open(zones_file, encoding="utf-8") as fh:
            wanted = [line.strip() for line in fh
                      if line.strip() and not line.lstrip().startswith("#")]
        wanted_set = set(wanted)
        zones = [z for z in zones if str(z) in wanted_set]
        report["zones_file"] = zones_file
        report["n_zones_requested"] = len(wanted)
        report["zones_missing_from_tables"] = sorted(
            wanted_set - {str(z) for z in zones})
    if zones_limit:
        zones = list(zones)[:int(zones_limit)]

    pool_lc = pool.copy()
    pool_lc.columns = [str(c).lower() for c in pool_lc.columns]
    deficit_df, zones_total = legacy.audit_zones(
        pool_lc, zones, ind_tables, hh_tables, sa2_to_regucp,
        ind_keys, hh_keys, rep_max)
    report["n_zones_audited"] = len(zones)
    report["n_table_zones"] = int(zones_total)
    report["n_deficit_cells"] = int(len(deficit_df))
    print(f"[joint] zones={len(zones)} pool={len(pool)} "
          f"census={len(census)}", flush=True)
    if len(deficit_df):
        report["deficit_total_gap"] = float(
            (deficit_df["target"] - deficit_df["capacity"]).sum())

    zero_df = pd.DataFrame()
    if len(deficit_df):
        if include_partial_deficits:
            zero_df = deficit_df[
                (deficit_df["target"] > 0)
                & (deficit_df["capacity"] < deficit_df["target"])
                & (~deficit_df["banded"])].copy()
        else:
            zero_df = deficit_df[
                (deficit_df["target"] > 0)
                & (deficit_df["capacity"] <= 0)
                & (~deficit_df["banded"])].copy()
    n_zero = int(((deficit_df["capacity"] <= 0)
                  & (deficit_df["target"] > 0)
                  & (~deficit_df["banded"])).sum()) if len(deficit_df) else 0
    report["n_zero_cells"] = n_zero
    report["n_target_cells"] = int(len(zero_df))
    report["n_partial_target_cells"] = int(len(zero_df) - n_zero)
    report["n_partial_deficit_cells"] = int(
        len(deficit_df) - n_zero)

    index = legacy.ShellIndex(census)
    hh_col = _column(pool, "ABSHID")
    if hh_col is not None:
        max_id = pd.to_numeric(pool[hh_col], errors="coerce").max()
        next_id = max(1_000_000_000, int(max_id or 0) + 1) + int(id_offset)
    else:
        next_id = 1_000_000_000 + int(id_offset)

    hh_attr_set = ({str(k).lower() for k in hh_keys}
                   | {str(d).lower() for d in legacy.MODELED_DWELL}
                   | {str(c).lower() for c in mipf._HOUSEHOLD_POOL_COLS})
    all_blocks = []
    zone_reports = []
    skipped_all = defaultdict(int)
    processed_cells = 0

    if len(zero_df):
        for zone in sorted(zero_df["zone"].unique()):
            if (max_transplant_cells is not None
                    and processed_cells >= int(max_transplant_cells)):
                break
            reg = sa2_to_regucp.get(str(zone))
            if not reg:
                skipped_all["zone_not_in_concordance"] += 1
                continue
            cand, note = mipf.zone_candidate_pool(
                pool_lc, zone, sa2_to_regucp)
            zone_zero = zero_df[zero_df["zone"].astype(str) == str(zone)]
            if len(cand) == 0:
                # No fallback to the global pool: the zone's own SA4 slice is
                # the candidate universe, so an empty slice is a structural
                # residual and its zero cells are reported, not borrowed for.
                zone_reports.append({
                    "zone": str(zone),
                    "regucp": str(reg),
                    "n_zero_cells": int(len(zone_zero)),
                    "n_zero_cells_covered": 0,
                    "n_zero_cells_uncovered": int(len(zone_zero)),
                    "uncovered_cells": [
                        f"{a}:{c}" for a, c in
                        zip(zone_zero["attr"], zone_zero["code"])],
                    "status": "structural_residual",
                    "candidate_source": note,
                })
                processed_cells += len(zone_zero)
                continue
            targets = {}
            targets.update(mipf.zone_targets(hh_tables, zone))
            targets.update(mipf.zone_targets(ind_tables, zone))
            if not targets:
                skipped_all["no_zone_targets"] += 1
                continue
            size_preference = None
            if size_aware_shells:
                nprd = {int(c): float(v) for (a, c), v in targets.items()
                        if str(a).lower() == "nprd" and float(v) > 0}
                nprd_total = sum(nprd.values())
                if nprd_total > 0:
                    size_preference = {k: v / nprd_total
                                       for k, v in nprd.items()}

            C_pool, _hh_ids_pool, cells = mipf.build_constraint_matrix(
                cand, targets, ind_keys, hh_keys)
            T = np.array([targets[c] for c in cells], dtype=float)
            capacity = rep_max * np.asarray(C_pool.sum(axis=1)).ravel()
            if include_partial_deficits:
                zero_mask = (T > 0) & (capacity < T)
            else:
                zero_mask = (T > 0) & (capacity <= 0)
            zero_cells = [
                cells[i] for i in np.flatnonzero(zero_mask)
                if not isinstance(cells[i][1],
                                  (set, frozenset, list, tuple))]
            remaining = None
            if max_transplant_cells is not None:
                remaining = int(max_transplant_cells) - processed_cells
                if remaining <= 0:
                    break
                zero_cells = zero_cells[:remaining]
            if not zero_cells:
                continue
            zero_set = set(zero_cells)
            zero_mask = np.array([c in zero_set for c in cells], dtype=bool)

            # Marginal-driven budgets: every sample-zero cell needs
            # ceil(target / rep_max) additional households.  The household
            # budget is that sum over the zone; the person budget is the same
            # count times the structural household-size cap, so the budget
            # cannot bind below what coverage requires.
            cap_by_cell = {cells[i]: float(capacity[i])
                           for i in range(len(cells))}
            gap_by_cell = {c: max(0.0, float(targets[c]) - cap_by_cell[c])
                           for c in zero_cells}
            (zone_max_households, zone_max_persons,
             zone_need) = legacy.marginal_zone_budget(
                [gap_by_cell[c] for c in zero_cells], rep_max,
                max_households=max_new_households,
                max_persons=max_new_persons)
            if min_admitted_households > zone_max_households:
                zone_max_households = int(min_admitted_households)
                zone_max_persons = max(zone_max_persons,
                                       zone_max_households * 12)
            # Construct at least nu_c donor variants per target cell, not just
            # the fixed donors_per_cell menu, so the solver can reach the
            # marginal-derived carrier requirement.
            need_by_cell = {
                c: legacy.households_to_cover(gap_by_cell[c], rep_max)
                for c in zero_cells}

            blocks, meta, skipped = build_candidate_menu(
                census, index, zero_cells, reg, global_dom, rlhp_dom,
                cuts, seed=seed, donors_per_cell=donors_per_cell,
                shells_per_cell=shells_per_cell,
                allow_global_donors=allow_global_donors,
                cell_need=need_by_cell, size_preference=size_preference,
                shell_growth=shell_growth, family_clone=family_clone)
            for k, v in skipped.items():
                skipped_all[k] += int(v)

            zone_report = {
                "zone": str(zone),
                "regucp": str(reg),
                "n_zero_cells": int(len(zero_cells)),
                "n_candidates": int(len(blocks)),
                # How the menu was built; 'growth' appears only with
                # --shell-growth, so a run states which mechanism it used.
                "candidate_sources": (
                    pd.Series([m.get("source", "transplant") for m in meta]
                              ).value_counts().to_dict() if meta else {}),
                "budget_source": ("marginal"
                                  if max_new_households is None
                                  else "fixed"),
                "marginal_household_requirement": int(zone_need),
                "budget_households": int(zone_max_households),
                "budget_persons": int(zone_max_persons),
            }
            print(f"[{processed_cells} cells] zone {zone}: "
                  f"{len(zero_cells)} zero, {len(blocks)} candidates",
                  flush=True)
            if not blocks:
                zone_report["status"] = "no_candidate"
                zone_report["n_zero_cells_covered"] = 0
                zone_report["n_zero_cells_uncovered"] = len(zero_cells)
                zone_report["uncovered_cells"] = [
                    f"{a}:{c}" for a, c in zero_cells]
                zone_reports.append(zone_report)
                processed_cells += len(zero_cells)
                continue

            cand_lc = _candidate_pool(blocks)
            C_cand, hh_ids_cand, cells_cand, sizes = _candidate_matrix(
                cand_lc, targets, ind_keys, hh_keys)
            # Rebuild the target vector from cells_cand rather than reusing T:
            # the two matrices are built from the same targets dict today, but
            # tying T to cells_cand makes the alignment explicit and removes a
            # silent misalignment if build_constraint_matrix ever changes order.
            T_cand = np.array([targets[c] for c in cells_cand], dtype=float)
            cell_pos = {c: i for i, c in enumerate(cells)}
            zero_mask_cand = np.array(
                [zero_mask[cell_pos[c]] if c in cell_pos else False
                 for c in cells_cand], dtype=bool)
            # Per-cell carrier requirement: r_max replicas of each selected
            # household must be able to reach the cell target.
            zero_need_cand = np.array(
                [need_by_cell[c] for c in cells_cand if c in zero_set],
                dtype=float)
            if len(hh_ids_cand) == len(meta):
                shell_ids = [m.get("shell") for m in meta]
            else:
                shell_ids = [None] * len(hh_ids_cand)

            x, stats = solve_selection(
                C_cand, T_cand, sizes, cells_cand, zero_mask_cand, shell_ids,
                hh_attr_set, max_households=zone_max_households,
                max_persons=zone_max_persons, time_limit=time_limit,
                zero_need=zero_need_cand,
                lambda_composition=lambda_composition,
                max_size7_share=max_size7_share,
                max_mtwp17_share=max_mtwp17_share,
                min_admitted_households=min_admitted_households)
            if x is None:
                A = C_cand.T.toarray().astype(float)
                B, w = _deviation_terms(
                    A, T_cand, cells_cand, sizes, hh_attr_set)
                x = greedy_selection(
                    A, sizes, zero_mask_cand,
                    max_households=zone_max_households,
                    max_persons=zone_max_persons, shell_ids=shell_ids,
                    zero_need=zero_need_cand)
                stats = _selection_stats(
                    x, A, B, w, zero_mask_cand, sizes,
                    zero_need=zero_need_cand)
                stats["solver"] = "greedy"

            selected = np.flatnonzero(np.asarray(x) > 0)
            if len(selected) == 0:
                zone_report["status"] = "no_selection"
                zone_report.update(stats)
                zone_report["uncovered_cells"] = [
                    f"{a}:{c}" for a, c in zero_cells]
                zone_reports.append(zone_report)
                processed_cells += len(zero_cells)
                continue

            contribution = np.asarray(
                C_cand[zero_mask_cand][:, selected].sum(axis=1)
            ).ravel()
            covered_mask = contribution > 0
            below_mask = contribution < zero_need_cand
            uncovered_cells = [
                f"{cells_cand[i][0]}:{cells_cand[i][1]}"
                for i, ok in zip(np.flatnonzero(zero_mask_cand), covered_mask)
                if not ok]
            below_requirement_cells = [
                f"{cells_cand[i][0]}:{cells_cand[i][1]}"
                for i, short in zip(np.flatnonzero(zero_mask_cand), below_mask)
                if short]

            block_by_id = {}
            shell_by_id = {}
            for i, block in enumerate(blocks):
                cid = _as_code(pd.to_numeric(
                    block["ABSHID"], errors="coerce").iloc[0])
                if cid is not None:
                    block_by_id[cid] = block
                    shell_by_id[cid] = (meta[i].get("shell")
                                        if i < len(meta) else None)
            selected_blocks = []
            selected_menu_ids = set()
            used_shells = set()
            for j in selected:
                cid = _as_code(hh_ids_cand[j])
                selected_menu_ids.add(cid)
                block = block_by_id.get(cid)
                if block is None:
                    continue
                shell = shell_by_id.get(cid)
                if shell is not None:
                    if shell in used_shells:
                        continue
                    used_shells.add(shell)
                block = block.copy()
                block["ABSHID"] = next_id
                block["_source"] = "mip_joint"
                block["_target_sa2"] = str(zone)
                selected_blocks.append(block)
                next_id += 1

            # Pool growth: append the rest of the constructed menu for this
            # SA2 as additional options, not just the coverage-minimal set.
            # Every record is tagged to this SA2 and deduplicated by attribute
            # signature, so sibling SA2s never see or share it.  The
            # one-transplant-per-shell rule is retained for the appended menu.
            extra_blocks = []
            if append_menu_candidates:
                seen = {_block_signature(b) for b in selected_blocks}
                for block in blocks:
                    cid = _as_code(pd.to_numeric(
                        block["ABSHID"], errors="coerce").iloc[0])
                    if cid in selected_menu_ids:
                        continue
                    shell = shell_by_id.get(cid)
                    if shell is not None:
                        if shell in used_shells:
                            continue
                        used_shells.add(shell)
                    signature = _block_signature(block)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    extra = block.copy()
                    extra["ABSHID"] = next_id
                    extra["_source"] = "mip_joint_menu"
                    extra["_target_sa2"] = str(zone)
                    extra_blocks.append(extra)
                    next_id += 1

            all_blocks.extend(selected_blocks)
            all_blocks.extend(extra_blocks)
            zone_report.update(stats)
            zone_report["n_selected_households"] = len(selected_blocks)
            zone_report["n_selected_persons"] = int(
                sum(len(b) for b in selected_blocks))
            zone_report["n_menu_households"] = int(len(blocks))
            zone_report["n_appended_menu_households"] = int(len(extra_blocks))
            zone_report["uncovered_cells"] = uncovered_cells
            zone_report["below_requirement_cells"] = below_requirement_cells
            zone_report["status"] = "ok"
            zone_reports.append(zone_report)
            processed_cells += len(zero_cells)

    if append_only:
        out = (pd.concat(all_blocks, ignore_index=True) if all_blocks
               else pool.iloc[0:0].copy())
    else:
        out = pool.copy()
        if "_source" not in out.columns:
            out["_source"] = "census"
        if all_blocks:
            out = pd.concat([out] + all_blocks, ignore_index=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)

    report["n_zones_processed"] = int(len(zone_reports))
    report["n_transplant_households"] = int(len(all_blocks))
    report["n_transplant_rows"] = int(sum(len(b) for b in all_blocks))
    report["n_appended_menu_households"] = int(
        sum(r.get("n_appended_menu_households", 0) for r in zone_reports))
    report["n_selected_persons"] = int(
        sum(len(b) for b in all_blocks))
    report["n_zero_cells_covered"] = int(
        sum(r.get("n_zero_cells_covered", 0) for r in zone_reports))
    report["n_zero_cells_uncovered"] = int(
        sum(r.get("n_zero_cells_uncovered", 0) for r in zone_reports))
    report["n_zero_cells_at_requirement"] = int(
        sum(r.get("n_zero_cells_at_requirement", 0) for r in zone_reports))
    report["n_zero_cells_below_requirement"] = int(
        sum(r.get("n_zero_cells_below_requirement", 0)
            for r in zone_reports))
    report["n_ordinal_shift_cells"] = int(
        skipped_all.pop("ordinal_shift_cells", 0))
    report["skipped"] = dict(skipped_all)
    report["zone_reports"] = zone_reports
    report["out"] = out_path
    report["n_out_rows"] = int(len(out))
    report["runtime_s"] = round(time.time() - t0, 1)

    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    return report, out


def main():
    ap = argparse.ArgumentParser(
        description="Joint-marginal feasible pool expansion")
    ap.add_argument("--pool", default=os.path.join(
        HERE, "outputs", "feasible_pool.csv"))
    ap.add_argument("--census", default=config.CENSUS_MICRODATA)
    ap.add_argument("--rep-max", type=int, default=40)
    ap.add_argument("--max-transplant-cells", type=legacy._optional_limit,
                    default=None,
                    help="Legacy ceiling on processed zero cells; default all")
    ap.add_argument("--shells-per-cell", type=int, default=8)
    ap.add_argument("--max-new-households", type=legacy._optional_limit,
                    default=None,
                    help="Legacy fixed household budget; default is the "
                         "marginal-derived ceil(gap/rep_max) sum")
    ap.add_argument("--max-new-persons", type=legacy._optional_limit,
                    default=None,
                    help="Legacy fixed person budget; default is the "
                         "marginal-derived budget")
    ap.add_argument("--donors-per-cell", type=int, default=4)
    ap.add_argument("--append-only", action="store_true",
                    help="Write only newly constructed households, not the "
                         "full input pool (used by zone-sharded runs).")
    ap.add_argument("--id-offset", type=int, default=0,
                    help="Offset added to fresh household IDs so zone-sharded "
                         "outputs can be merged without collisions.")
    ap.add_argument("--donor-policy", choices=["global", "strict"],
                    default="global",
                    help="Donor sourcing for construction. 'global' (default) "
                         "imports donor profiles from the whole census while "
                         "keeping the shell local; 'strict' uses only the "
                         "zone's REGUCP slice.")
    ap.add_argument("--append-menu-candidates", dest="append_menu",
                    action="store_true", default=True,
                    help="Append the full constructed candidate menu for each "
                         "zone, not only the coverage-minimal selection, so "
                         "the SA2 pool has more options (default: on).")
    ap.add_argument("--coverage-only", dest="append_menu",
                    action="store_false",
                    help="Append only the households selected by the "
                         "coverage/marginal program.")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--zones-limit", type=int, default=None)
    ap.add_argument("--zones-file", default=None,
                    help="optional file of zone IDs to audit (one per line)")
    ap.add_argument("--pool-nrows", type=int, default=None)
    ap.add_argument("--census-nrows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        HERE, "outputs", "feasible_pool_joint.csv"))
    ap.add_argument("--report", default=os.path.join(
        HERE, "outputs", "mip_pool_joint_report.json"))
    a = ap.parse_args()
    run_joint(
        pool_path=a.pool, census_path=a.census, rep_max=a.rep_max,
        out_path=a.out, max_transplant_cells=a.max_transplant_cells,
        shells_per_cell=a.shells_per_cell, zones_limit=a.zones_limit,
        pool_nrows=a.pool_nrows, census_nrows=a.census_nrows,
        report_path=a.report, max_new_households=a.max_new_households,
        max_new_persons=a.max_new_persons,
        donors_per_cell=a.donors_per_cell, time_limit=a.time_limit,
        seed=a.seed, zones_file=a.zones_file,
        allow_global_donors=(a.donor_policy == "global"),
        append_menu_candidates=a.append_menu,
        append_only=a.append_only, id_offset=a.id_offset)


if __name__ == "__main__":
    main()

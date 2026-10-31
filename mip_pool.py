"""mip_pool.py - MIP-based feasible pool completion (HiGHS MILP).

The CLI defaults to the joint-marginal model in mip_pool_joint.py: one selection
program per zone fits the selected donor/shell candidates to all control-table
marginals together, with soft coverage of zero-capacity cells.  The per-cell
implementation below is retained as --mode per-cell so the earlier audit can
still be reproduced.

Complements build_feasible_pool_cp.py: where the CP pool builds novelty by
constraint-guided donor resampling, this module *solves* for the pool records
the census sample cannot supply locally, directly from the microdata.

Two layers mirror the two marginal structures in the pipeline:

1. Zone-level marginals (mip_fit):  min_w |Cx - T| over the SA2 control
   tables.  The pool is the candidate lattice that makes those problems
   solvable, so the pool's responsibility is *coverage capacity*:

       capacity(zone, cell) := rep_max * sum of candidate contributions

   Any cell whose capacity falls below the zone table target T is a deficit;
   the deficit set is the objective the pool MIP minimises.

2. Deficit-synthesis MIP (this module): for each uncovered (attr, code) cell
   in a zone's REGUCP, import the donor member profile observed elsewhere in
   the census (the population's own carrier of that category), fix the local
   shell's dwelling + RLHP role, and solve

       variables   y[a, c] in {0,1}                attribute indicators
       constraints sum_c y[a,c] == 1               one-hot per attribute
                   y[a1,c1] + y[a2,c2] <= 1        forbidden-pair cuts (exact
                                                   encoding of the hard rule
                                                   set: every rule decomposes
                                                   into unary/pairwise
                                                   predicates)
       objective   min sum_a (1 - y[a, target(a)])  minimum Hamming deviation
                                                   from the imported profile

   Household-level (dwelling/family) fields are FIXED to a real local shell
   household (same REGUCP), so every emitted household stays atomic and
   family-coherent, and member novelty is created by the solver rather than
   by post-hoc repair.  Every emitted household is validated against the
   shared CONSTRAINT_CHECKS; any failure is retried on another shell, never
   patched.

Emission contract
-----------------
Output = pool rows + transplant households appended.  Geometry caveat:
transplanted households carry the deficit zone's REGUCP (their candidate
assignment), which is a synthesis choice, not an observation; the report
records this.  ABSHID values are fresh numeric ids above the pool maximum.

Usage
-----
  python mip_pool.py --pool outputs/feasible_pool.csv \
      --census LINKED_DATA_clean.csv \
      --rep-max 40 --max-transplant-cells 4000 \
      --out outputs/feasible_pool_v5.csv
"""

import argparse
import inspect
import itertools
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse as _sp
from scipy.optimize import milp, LinearConstraint, Bounds

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config
import constraint_rules
from constrained_sampling import (
    occp_incp_constraint, agep_hrsp_constraint, heap_occp_constraint,
    MSTP_CACF_BOUNDS, RLHP_CHILD_VALID,
)
import mip_fit as mipf
from mip_fit import zone_targets, zone_candidate_pool
from sa_only_baseline import load_control_tables, _load_sa2_to_regucp

MAX_HH_SIZE = int(os.environ.get("CP_MAX_HH_SIZE", "12"))

# ---------------------------------------------------------------------------
# Modelled attribute space
# ---------------------------------------------------------------------------

# person-level fields solved by the deficit MIP (RLHP included: it selects
# the shell slot that receives the imported profile)
MODELED_PERSON = ["SEXP", "AGEP", "MSTP", "HEAP", "OCCP", "INCP", "LFSP",
                  "EMPP", "HRSP", "QALLP", "QALFP", "MTWP", "RLHP"]
# dwelling/family fields fixed by the local shell (every member row repeats
# them, so the shell passes them through unchanged)
MODELED_DWELL = ["MRERD", "RNTRD", "STRD", "NPRD", "VEHRD", "BEDRD", "HHCD",
                 "HIND", "HINASD"]
MODELED = MODELED_PERSON + MODELED_DWELL


def _lower(col):
    c = str(col).lower()
    return c if c in {a.lower() for a in MODELED} else c


# ---------------------------------------------------------------------------
# Cut library: exact pairwise encoding of the union rule set
# ---------------------------------------------------------------------------

_COL_TOKEN = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")


def _rule_cols(fn):
    """Columns a rule predicate reads.

    Taken from the code object's string constants, not from the source text.
    These rules are lambdas defined inside a dict literal, and
    inspect.getsource() returns only the first line of such a lambda, so source
    parsing saw 'LFSP' but never 'HRSP' in `hrsp_without_employment`.  Every
    multi-line rule was therefore read as single-column, matched no attribute
    pair in build_forbidden_pairs, and silently contributed no cuts — the MILP
    then emitted households that validate_household rejected.
    """
    consts = set()
    stack = [fn.__code__]
    while stack:
        code = stack.pop()
        for c in code.co_consts:
            if isinstance(c, str):
                consts.add(c)
            elif hasattr(c, "co_consts"):
                stack.append(c)
    return {c for c in consts if _COL_TOKEN.match(c)}


def _padding_codes():
    """Control-table codes that no census record can ever carry.

    normalize_control_tables.py pads household tables with "Not applicable"
    rows so their totals balance, and mip_fit maps that label to a code (BEDRD 7,
    HHCD 7, TEND 6, LFSP 5).  Those codes are structural, not observed: they are
    absent from the microdata by construction, so a capacity shortfall on them is
    padding showing through, not a gap the pool should be synthesised to fill.
    """
    out = set()
    for attr, mapping in getattr(mipf, "_LABEL_TO_CODE", {}).items():
        if "Not applicable" in mapping:
            out.add((str(attr).lower(), int(mapping["Not applicable"])))
    return out


def build_rule_sources():
    rules = dict(constraint_rules.CONSTRAINT_CHECKS)
    rules["ss_occp_incp"] = occp_incp_constraint
    rules["ss_agep_hrsp"] = agep_hrsp_constraint
    rules["ss_heap_occp"] = heap_occp_constraint
    return rules


def build_forbidden_pairs(domains, verbose=False):
    """Evaluate every rule over the domain mesh of each referenced attribute
    pair; record infeasible category combinations as MIP cuts.

    Only pairs both present in `domains` produce cuts; rules referencing
    derived (non-modelled) fields such as CACF/CPRF are skipped here and stay
    in the post-emission validation via CONSTRAINT_CHECKS.

    Returns {(a1, a2): [(c1, c2), ...]} with a1 < a2 alphabetically.
    """
    rules = build_rule_sources()
    rule_cols = {name: _rule_cols(fn) for name, fn in rules.items()}
    cuts = defaultdict(set)
    attrs = sorted(domains)
    for a1, a2 in itertools.combinations(attrs, 2):
        v1 = np.asarray(domains[a1], dtype=int)
        v2 = np.asarray(domains[a2], dtype=int)
        m1, m2 = np.meshgrid(v1, v2, indexing="ij")
        sub = pd.DataFrame({a1: m1.ravel(), a2: m2.ravel()})
        for name, fn in rules.items():
            cols = rule_cols[name]
            # rule must read exactly this pair (nothing more, nothing else)
            if not ({a1, a2}.issubset(cols) and cols.issubset({a1, a2})):
                continue
            viol = np.asarray(fn(sub), dtype=bool)
            if viol.any():
                for c1, c2 in zip(m1.ravel()[viol], m2.ravel()[viol]):
                    cuts[(a1, a2)].add((int(c1), int(c2)))
        if verbose and (a1, a2) in cuts:
            print(f"    cuts {a1}x{a2}: {len(cuts[(a1, a2)])}")
    return {k: sorted(v) for k, v in cuts.items()}


# ---------------------------------------------------------------------------
# Domains and conditional pruning
# ---------------------------------------------------------------------------

def observed_domains(census, attrs):
    return {a: sorted(int(x) for x in census[a].dropna().unique())
            for a in attrs if a in census.columns}


def domains_given_rlhp(census, attrs):
    """Per-attribute observed code sets conditioned on each RLHP value: the
    MIP analogue of the pool's empirial support layer.  Anchors the solver to
    payloads the census itself exhibits for the fixed household role."""
    out = {}
    for rlhp, sub in census.groupby("RLHP"):
        if pd.isna(rlhp):
            continue
        out[int(rlhp)] = {a: sorted(int(x) for x in sub[a].dropna().unique())
                          for a in attrs if a in sub.columns}
    return out


def prune_domains(free_attrs, global_dom, rlhp_dom, role, fixed, cuts):
    """Allowed codes per free attribute for one solve.

    Order: RLHP-conditional empirical domain, minus rule cuts against the
    fixed (dwelling/RLHP) attributes; falls back to the observed marginal
    domain when the conditional set is empty.
    """
    cond = rlhp_dom.get(int(role), {}) if role is not None else {}
    out = {}
    for a in free_attrs:
        codes = cond.get(a) or global_dom.get(a) or []
        if global_dom.get(a) and codes:
            codes = sorted(set(codes))
        bad = set()
        for (a1, a2), pairs in cuts.items():
            if a1 in fixed and a2 == a:
                for c1, c2 in pairs:
                    if int(fixed[a1]) == c1:
                        bad.add(int(c2))
            elif a2 in fixed and a1 == a:
                for c1, c2 in pairs:
                    if int(fixed[a2]) == c2:
                        bad.add(int(c1))
        if bad:
            pruned = [c for c in codes if int(c) not in bad]
            codes = pruned or codes      # never fully exclude an attribute
        out[a] = codes
    return out


# ---------------------------------------------------------------------------
# The deficit-synthesis MIP
# ---------------------------------------------------------------------------

def solve_nearest_profile(free_dom, fixed, target, cuts, time_limit=30.0,
                          required=None):
    """Minimum-Hamming-deviation MILP for one member slot.

    fixed   {attr: code}   dwelling + RLHP context (bounds, not variables)
    target  {attr: code}   imported donor profile to approximate
    free_dom {attr:[codes]} already domain-pruned
    required {attr: code}  categories the emitted row must carry (hard rows)

    Returns (solved_profile, deviation); deviation counts mismatched free
    categories vs `target`.  (None, None) when no feasible point exists
    (should not happen for real shells; the caller retries on other shells).
    """
    required = {str(a): int(c) for a, c in (required or {}).items()}
    for a, c in required.items():
        if a in fixed and int(fixed[a]) != c:
            return None, None
        if a not in fixed and a not in free_dom:
            return None, None
    free = [a for a in sorted(free_dom) if a not in fixed]
    col_of = {}
    cost = []
    for a in free:
        for c in free_dom[a]:
            col_of[(a, int(c))] = len(cost)
            cost.append(-1.0 if a in target and int(c) == int(target[a])
                        else 0.0)
    n = len(cost)
    if n == 0:
        return None, None

    rows, lb, ub = [], [], []
    for a in free:
        r = np.zeros(n)
        for c in free_dom[a]:
            r[col_of[(a, int(c))]] = 1.0
        rows.append(r)
        lb.append(1.0)
        ub.append(1.0)
    for a, c in required.items():
        if a not in free:
            continue
        key = (a, int(c))
        if key not in col_of:
            return None, None
        r = np.zeros(n)
        r[col_of[key]] = 1.0
        rows.append(r)
        lb.append(1.0)
        ub.append(1.0)
    for (a1, a2), pairs in cuts.items():
        if a1 in fixed or a2 in fixed:
            continue          # fixed-side cuts were absorbed by the pruning
        if a1 not in free or a2 not in free:
            continue
        for c1, c2 in pairs:
            k1, k2 = col_of.get((a1, int(c1))), col_of.get((a2, int(c2)))
            if k1 is None or k2 is None:
                continue
            r = np.zeros(n)
            r[k1] = 1.0
            r[k2] = 1.0
            rows.append(r)
            lb.append(-np.inf)
            ub.append(1.0)

    A = _sp.csr_matrix(np.asarray(rows)) if rows else None
    cons = LinearConstraint(A, np.asarray(lb), np.asarray(ub)) if rows else None
    res = milp(c=np.asarray(cost), integrality=np.ones(n),
               bounds=Bounds(np.zeros(n), np.ones(n)), constraints=cons,
               options={"time_limit": time_limit} if time_limit else {})
    if res is None or not res.success:
        return None, None
    y = np.rint(np.clip(np.asarray(res.x), 0, 1)).astype(int)
    profile = dict(fixed)
    for (a, c), k in col_of.items():
        if y[k]:
            profile[a] = c
    for a, c in required.items():
        if int(profile.get(a, -10**9)) != c:
            return None, None
    dev = sum(1 for a in free
              if a in target and profile.get(a) != int(target[a]))
    return profile, dev


# ---------------------------------------------------------------------------
# Household shell index (vectorised)
# ---------------------------------------------------------------------------

class ShellIndex:
    """Per-household metadata + positional row slices for the census."""

    def __init__(self, census):
        dw = [d for d in MODELED_DWELL if d in census.columns]
        g = census.groupby("ABSHID", sort=False)
        sizes = g.size()
        keep = sizes[(sizes >= 1) & (sizes <= MAX_HH_SIZE)].index
        keep_set = set(keep)
        self.positions = {h: np.asarray(ix, dtype=np.int64)
                          for h, ix in g.indices.items() if h in keep_set}
        agg = census.loc[census["ABSHID"].isin(keep)].groupby("ABSHID")
        self.meta = pd.DataFrame({
            "size": agg["ABSHID"].size(),
            "reg": _normalize(agg["REGUCP"].first()) if "REGUCP"
            in census.columns else pd.Series(dtype=object),
            **{d.lower(): agg[d].first() for d in dw},
        })
        self.n_dwell = len(dw)
        # Positional RLHP vector, so a shell can be checked for a given role
        # without materialising its rows.
        self.rlhp = (pd.to_numeric(census["RLHP"], errors="coerce").to_numpy()
                     if "RLHP" in census.columns else None)

    def has_role(self, rows, role):
        if self.rlhp is None or role is None:
            return True
        return bool((self.rlhp[rows] == role).any())

    def by_reg(self, reg):
        return self.meta.index[self.meta["reg"] == str(reg)]

    def dwell_of(self, hhid):
        row = self.meta.loc[hhid]
        return tuple(row[d.lower()] for d in MODELED_DWELL
                     if d.lower() in self.meta.columns)


def _norm_zone_series(series):
    return _normalize(series)


def _normalize(series):
    from sa_only_baseline import _normalize_zone_values
    return _normalize_zone_values(series)


def _norm_scalar(v):
    return str(_normalize(pd.Series([v])).iloc[0])


def choose_shells(index, reg, role, max_shells=8, required=None):
    """Candidate local shell households (household ids) for a transplant.

    Only households that already contain a member with the donor's RLHP role
    are eligible: the solved profile is written into that slot, so a shell
    without one can never host the transplant.  `required` additionally filters
    on dwelling terms the shell must already carry (the joint expansion uses
    this so a household-level deficit is never paired with a shell that cannot
    cover it).  Fields outside the shell metadata are left to the solver.
    """
    req = {str(a).lower(): int(c) for a, c in (required or {}).items()}
    cand = []
    for hhid in index.by_reg(reg):
        if req:
            meta = index.meta.loc[hhid]
            bad = False
            for a, c in req.items():
                if a in meta.index and pd.notna(meta.get(a)):
                    try:
                        if int(meta[a]) != c:
                            bad = True
                            break
                    except (TypeError, ValueError):
                        bad = True
                        break
            if bad:
                continue
        rows = index.positions.get(hhid)
        if rows is None or not index.has_role(rows, role):
            continue
        cand.append((hhid, rows))
        if len(cand) >= max_shells:
            break
    return cand


# ---------------------------------------------------------------------------
# Family recomputation + validation
# ---------------------------------------------------------------------------

def recompute_family_counts(block):
    """Keep CACF/CPRF and the derived household counters consistent with the
    solved member vectors (children = AGEP<15 within their FNOF family)."""
    if "FNOF" in block.columns:
        block["CPRF"] = block.groupby("FNOF")["FNOF"].transform("size")
        child = (block["AGEP"] < 15).astype(int)
        block["CACF"] = child.groupby(block["FNOF"]).transform("sum")
        if "persons_in_family" in block.columns:
            block["persons_in_family"] = block.groupby(
                "FNOF")["FNOF"].transform("size")
    if "persons_in_household" in block.columns:
        block["persons_in_household"] = len(block)
    if "families_in_household" in block.columns and "FMCF" in block.columns:
        block["families_in_household"] = block["FMCF"].iloc[0]
    return block


def validate_household(block):
    """The shared CONSTRAINT_CHECKS on one household.

    A household-level MSTP/CACF bound used to be applied here as well ("never
    married -> 0 children"), comparing the first parent's bound against every
    under-15 member of the dwelling.  It rejected 25% of untouched census
    households (760 of 3000 sampled): the census' MSTP=6 people are mostly young
    adults co-residing with their parents, so the count picked up their siblings
    as their children, and lone parents are legitimately never married.  Since a
    real household must be feasible by construction for the transplant paths to
    mean anything, the check is not applied; the shared rule set below is.
    """
    viol = []
    for name, fn in constraint_rules.CONSTRAINT_CHECKS.items():
        try:
            bad = np.asarray(fn(block), dtype=bool)
            if bad.any():
                viol.append(f"{name}:{int(bad.sum())}")
        except KeyError:
            continue
    return viol


# ---------------------------------------------------------------------------
# Transplant assembly
# ---------------------------------------------------------------------------

def transplant_member(census, shells, index, donor_profile, reg, global_dom,
                      rlhp_dom, cuts, next_id, required=None):
    """Solve one imported profile into a local shell household.

    shells: [(hhid, row_positions)] candidates, best first.
    required: {attr: code} categories the emitted household must carry.  The
    joint expansion uses this to guarantee that a candidate actually covers the
    zero cell it was generated for; the legacy per-cell path passes None.
    Returns (block, deviation, hhid) or None.
    """
    req = {str(a): int(c) for a, c in (required or {}).items()}
    role = int(donor_profile["RLHP"])
    fixed = {"RLHP": role}
    free_attrs = [a for a in MODELED_PERSON if a not in fixed]
    for hhid, rows in shells:
        block = census.iloc[rows].reset_index(drop=True).copy()
        dw_cols = [d for d in MODELED_DWELL if d in block.columns]
        for d in dw_cols:
            v = block[d].iloc[0]
            fixed[d] = int(v) if pd.notna(v) else -1
        blocked = False
        for a, c in req.items():
            if a in fixed and int(fixed[a]) != c:
                blocked = True
                break
            if a not in fixed and a not in free_attrs:
                blocked = True
                break
        if blocked:
            continue
        slot_rows = np.flatnonzero((block["RLHP"] == role).to_numpy())
        if slot_rows.size == 0:
            continue
        freedom = prune_domains(free_attrs, global_dom, rlhp_dom, role,
                                fixed, cuts)
        if any(not v for v in freedom.values()):
            continue
        for a, c in req.items():
            if a in freedom:
                freedom[a] = sorted(set(int(v) for v in freedom[a]) | {c})
        target = {a: int(donor_profile[a]) for a in donor_profile
                  if pd.notna(donor_profile[a])}
        profile, dev = solve_nearest_profile(freedom, fixed, target, cuts,
                                             required=req)
        if profile is None:
            continue
        slot = int(slot_rows[0])
        for a, code in profile.items():
            if a in block.columns:
                block.loc[slot, a] = code
        block = recompute_family_counts(block)
        if validate_household(block):
            continue
        covered = True
        for a, c in req.items():
            if a not in block.columns:
                covered = False
                break
            vals = pd.to_numeric(block[a], errors="coerce")
            if not bool((vals == c).any()):
                covered = False
                break
        if not covered:
            continue
        block["ABSHID"] = next_id
        if "REGUCP" in block.columns:
            block["REGUCP"] = reg
        block["_source"] = "mip"
        block["reindexed_slot"] = slot
        return block, dev, hhid
    return None


# ---------------------------------------------------------------------------
# Verbatim household transplant (for deficits outside the MIP model space)
# ---------------------------------------------------------------------------

def verbatim_transplant(census, donor_rows, reg, next_id, need, rng):
    """Copy real donor households carrying a scarce attribute profile into
    the deficit zone's candidate set (geography re-assigned, documented in
    the report).  No solver involvement: the block is copied verbatim, so it
    stays feasible by construction.

    donor_rows: frame of census rows with attr == code (already filtered); it is
    used only to find donor *household ids*.  The emitted block is the whole
    household taken from `census`, not the subset of rows that matched: for a
    person-level attribute the filtered frame holds one member out of several,
    and emitting that would produce a truncated household whose NPRD and family
    counts no longer describe its members.

    Returns (list of household blocks, next_id).
    """
    if donor_rows.empty or need <= 0:
        return [], next_id
    sizes = donor_rows.groupby("ABSHID").size()
    hh_ids = sizes[(sizes >= 1) & (sizes <= MAX_HH_SIZE)].index.to_numpy()
    if len(hh_ids) == 0:
        return [], next_id
    # Draw more candidates than needed so households rejected by
    # CONSTRAINT_CHECKS can be skipped without falling short of `need`.
    take = min(len(hh_ids), max(need * 4, need))
    picked = pd.Index(rng.choice(hh_ids, size=take, replace=False))
    frames = []
    for h, sub in census[census["ABSHID"].isin(picked)].groupby(
            "ABSHID", sort=False):
        if len(frames) >= need:
            break
        sub = sub.copy()
        sub["ABSHID"] = next_id
        sub["REGUCP"] = reg
        sub["_source"] = "mip"
        sub = recompute_family_counts(sub)
        # Copying a real household verbatim keeps it feasible with respect to
        # the fields it was selected for, but the microdata itself violates some
        # CONSTRAINT_CHECKS (e.g. overseas visitors, LFSP=6, carry hours).  The
        # emission contract is that every appended household passes the shared
        # checks, so those donors are skipped rather than emitted.
        if validate_household(sub):
            continue
        next_id += 1
        frames.append(sub)
    return frames, next_id


# ---------------------------------------------------------------------------
# Zone coverage audit
# ---------------------------------------------------------------------------

def audit_zones(pool_lc, zones, ind_tables, hh_tables, sa2_to_regucp,
                ind_keys, hh_keys, rep_max):
    """Per zone/cell attainable capacity under rep_max replication.

    Uses the exact same cell/match semantics as the production MIP build:
    zone_targets -> build_constraint_matrix (bands, household 'first' match
    and per-member counts included).
    """
    deficits = []
    zones_total = len(zones)
    for zi, zone in enumerate(zones, 1):
        tried = {}
        try:
            cand, note = zone_candidate_pool(pool_lc, zone, sa2_to_regucp)
        except Exception:
            continue
        if note.startswith("full-pool"):
            # the production fitter caps full-pool fallback to 20k-150k hh;
            # cap here as well so an unmapped zone cannot stall the audit
            cand_capped, _ = mipf.cap_candidates(cand, person_target=0)
            cand = cand_capped
        if cand is None or len(cand) == 0:
            continue
        targets = {}
        try:
            targets.update(zone_targets(hh_tables, zone))
            targets.update(zone_targets(ind_tables, zone))
        except Exception:
            continue
        if not targets:
            continue
        C, hh_ids, cells = mipf.build_constraint_matrix(
            cand, targets, ind_keys, hh_keys)
        sums = np.asarray(C.sum(axis=1)).ravel()
        rows = []
        for ci, cell in enumerate(cells):
            attr_l, cat = cell
            if isinstance(cat, (set, frozenset, list, tuple)):
                rows.append({"zone": str(zone), "attr": str(attr_l),
                             "code": "BAND:" + ",".join(sorted(str(c)
                                                               for c in cat)),
                             "target": float(targets[cell]),
                             "capacity": rep_max * float(sums[ci]),
                             "banded": True})
                continue
            rows.append({"zone": str(zone), "attr": str(attr_l),
                         "code": int(cat),
                         "target": float(targets[cell]),
                         "capacity": rep_max * float(sums[ci]),
                         "banded": False})
        deficits.extend(r for r in rows if r["capacity"] < r["target"])
    return pd.DataFrame(deficits, columns=["zone", "attr", "code", "target",
                                           "capacity", "banded"]), zones_total


# ---------------------------------------------------------------------------
# Ordinal-adjacent donor fill
# ---------------------------------------------------------------------------
# Some control cells are an *ordinal* band (hours worked, income, qualification
# level) whose exact value is absent from the donor pool but whose neighbours
# are present. A person observed at an adjacent band could plausibly have been
# recorded at the target band, so we borrow the nearest applicable band's donors
# and shift the ordinal value to the target. The special codes below are NOT part
# of the ordinal scale (not applicable / not stated / overseas) and are never
# fabricated this way; children and not-applicable persons are excluded from the
# donor set, and full individual/household feasibility is still enforced by the
# rule cuts in transplant_member (a shifted record that breaks a child or
# household rule is discarded, not emitted).
_ORDINAL_SPECIAL = {
    "HRSP":  {101, 103},   # Not stated, Overseas visitor
    "INCP":  {18, 19},     # Not applicable, Overseas visitor
    "QALLP": {8, 9},       # Not applicable, Overseas
}
_ORDINAL_ATTRS = set(_ORDINAL_SPECIAL)
_ORDINAL_CHILD_AGE_MAX = 14

# Attributes sourced from the WHOLE census rather than region-first. Expansion
# only has to make a zone's control cell selectable so the SA2 marginal can be
# matched -- it is not required to reproduce the region's composition. For the
# person-level ordinal attributes (income, hours worked, highest qualification)
# the band is realistic irrespective of region, and region-first sourcing merely
# starves the exact bands that are already sparse at the region level. Geographic
# and dwelling attributes stay region-first (housing tenure, dwelling type, rent
# and mortgage bands are regionally correlated, so the household shell is more
# realistic when drawn from the same region). Individual/household feasibility is
# enforced by the rule cuts either way.
_GLOBAL_DONOR_ATTRS = {"INCP", "HRSP", "QALLP"}


def _nearest_ordinal_donors(census, attr, code, special):
    """Donors from the nearest applicable ordinal band to `code`, excluding the
    special (non-ordinal) codes and children. Returns a frame or None.

    Adjacency is numeric on the pool's ordinal codes; the caller shifts the
    borrowed donors' value to the target band."""
    if attr not in census.columns:
        return None
    col = census[attr]
    applicable = sorted({int(c) for c in col.dropna().unique()
                         if int(c) not in special and int(c) != int(code)})
    if not applicable:
        return None
    age_ok = census["AGEP"] > _ORDINAL_CHILD_AGE_MAX if "AGEP" in census.columns else None
    for c in sorted(applicable, key=lambda v: (abs(v - int(code)), v)):
        mask = census[attr] == c
        if age_ok is not None:
            mask = mask & age_ok          # never shift a child into an ordinal band
        d = census[mask]
        if not d.empty:
            return d
    return None


def _source_donors(creg, census, attr, code, special, ordinal, report,
                   global_only=False):
    """Conditional donor ladder for a zero cell (attr, code).

    1. region-exact  : donors in the zone's REGUCP carrying the exact band;
    2. region-ordinal: nearest applicable band in the region, value shifted;
    3. global-exact  : donors anywhere carrying the band (preserves prior cover);
    4. global-ordinal: nearest applicable band anywhere, value shifted.

    Steps 2 and 4 are the conditional/ordinal fill: a band absent in the region
    (or absent everywhere) is reached by borrowing a real neighbour and shifting
    the ordinal value, with children and not-applicable codes excluded; full
    individual/household feasibility is then enforced by the caller's rule cuts.

    When `global_only` (see `_GLOBAL_DONOR_ATTRS`), the two region steps are
    skipped and the donor is drawn from the whole census: the attribute is not
    geographically determined, so the pool only needs the band to exist with a
    feasible profile for the SA2 marginal to be matchable. Returns a donor frame
    or None."""
    code = int(code)
    if not global_only:
        # 1. region exact
        if attr in creg.columns:
            d = creg[creg[attr] == code]
            if not d.empty:
                return d
        # 2. region ordinal shift
        if ordinal and code not in special and len(creg):
            d = _nearest_ordinal_donors(creg, attr, code, special)
            if d is not None and not d.empty:
                d = d.copy(); d[attr] = code
                report["n_conditional_zero_cells"] = report.get("n_conditional_zero_cells", 0) + 1
                report["n_ordinal_shift_cells"] = report.get("n_ordinal_shift_cells", 0) + 1
                return d
    # 3. global exact
    d = census[census[attr] == code]
    if not d.empty:
        if global_only:
            report["n_global_donor_cells"] = report.get("n_global_donor_cells", 0) + 1
        return d
    # 4. global ordinal shift
    if ordinal and code not in special:
        d = _nearest_ordinal_donors(census, attr, code, special)
        if d is not None and not d.empty:
            d = d.copy(); d[attr] = code
            report["n_ordinal_shift_cells"] = report.get("n_ordinal_shift_cells", 0) + 1
            if global_only:
                report["n_global_donor_cells"] = report.get("n_global_donor_cells", 0) + 1
            return d
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(pool_path, census_path, rep_max, out_path,
        max_transplant_cells=4000, shells_per_cell=8, zones_limit=None,
        max_per_cell=20, pool_nrows=None, census_nrows=None,
        report_path=None, seed=0):
    t0 = time.time()
    report = {"rep_max": rep_max, "pool": pool_path, "census": census_path,
              "n_conditional_zero_cells": 0, "n_ordinal_shift_cells": 0,
              "n_global_donor_cells": 0}

    read_kw = {"low_memory": False, "nrows": int(census_nrows)} \
        if census_nrows else {"low_memory": False}
    pool_kw = {**read_kw, "nrows": int(pool_nrows)} if pool_nrows else \
        dict(read_kw)
    pool = pd.read_csv(pool_path, **pool_kw)
    census = pd.read_csv(census_path, **read_kw)
    report["n_pool_rows"] = int(len(pool))
    report["n_census_rows"] = int(len(census))

    dwell = [d for d in MODELED_DWELL if d in census.columns]

    # cut library + domain structures (built once)
    global_dom = observed_domains(census, MODELED)
    rlhp_dom = domains_given_rlhp(census,
                                  [a for a in MODELED_PERSON if a != "RLHP"])
    cuts = build_forbidden_pairs(global_dom, verbose=False)
    report["n_forbidden_pair_cuts"] = int(sum(len(v) for v in cuts.values()))

    # --- coverage audit (lower-case working copy for the mip_fit helpers) --
    ind_tables, hh_tables = load_control_tables()
    ind_keys = sorted({k for t in ind_tables.values() for k in t.get("keys", [])})
    hh_keys = sorted({k for t in hh_tables.values() for k in t.get("keys", [])})
    sa2_to_regucp, _csf_src = _load_sa2_to_regucp()
    zones = _table_zones(ind_tables, hh_tables)
    if zones_limit:
        zones = list(zones)[:int(zones_limit)]
    pool_lc = pool.copy()
    pool_lc.columns = [str(c).lower() for c in pool_lc.columns]
    deficit_df, zones_total = audit_zones(pool_lc, zones, ind_tables,
                                          hh_tables, sa2_to_regucp,
                                          ind_keys, hh_keys, rep_max)
    report["n_zones_audited"] = len(zones) if zones_limit else zones_total
    report["n_table_zones"] = int(zones_total)
    report["n_deficit_cells"] = int(len(deficit_df))
    if len(deficit_df):
        report["deficit_by_attr"] = deficit_df["attr"].value_counts(
        ).to_dict()
        report["deficit_total_gap"] = float(
            (deficit_df["target"] - deficit_df["capacity"]).sum())

    # --- deficit -> transplants -------------------------------------------
    index = ShellIndex(census)
    next_id = max(1_000_000_000,
                  int(pd.to_numeric(pool["ABSHID"], errors="coerce").max()
                      or 0) + 1)
    donor_cache, cells_done = {}, 0
    reg_census_cache = {}
    blocks, devs, skipped = [], [], defaultdict(int)
    if len(deficit_df):
        deficit_df = deficit_df[~deficit_df["banded"]].copy()
        deficit_df["gap"] = deficit_df["target"] - deficit_df["capacity"]
        deficit_df = deficit_df.sort_values("gap", ascending=False)
        deficit_df["reg"] = deficit_df["zone"].map(sa2_to_regucp)
        deficit_df = deficit_df.dropna(subset=["reg"])
        # Control-table attrs arrive lower case while the census columns are
        # upper case (BEDRD, TEND, TYPP ...).  Resolve against the columns
        # actually present: matching on a fixed case left every non-modelled
        # attr (tend/typp/indp/stup/assnp/tenlld) with an empty donor frame,
        # which surfaced as a spurious "no_donor_anywhere".
        census_col = {str(c).lower(): str(c) for c in census.columns}
        deficit_df["attr_col"] = deficit_df["attr"].map(
            lambda a: census_col.get(str(a).lower()))

        padding = _padding_codes()
        is_pad = [(str(r["attr"]).lower(), int(r["code"])) in padding
                  for _, r in deficit_df.iterrows()]
        report["n_deficit_cells_padding"] = int(sum(is_pad))
        report["n_deficit_cells_coverable"] = int(len(is_pad) - sum(is_pad))

        rng = np.random.default_rng(seed)
        modeled_u = set(MODELED_PERSON)
        for _, row in deficit_df.head(int(max_transplant_cells)).iterrows():
            attr, code = row["attr_col"], int(row["code"])
            reg = _norm_scalar(row["reg"])
            gap_hh = int(np.ceil(max(0.0, float(row["gap"])) / rep_max))
            need = min(gap_hh, int(max_per_cell))
            if need <= 0:
                continue
            if not isinstance(attr, str) or attr not in census.columns:
                skipped["attr_not_in_census"] += 1
                continue
            # Source donors conditionally (region first), with an ordinal shift
            # from the nearest applicable band when a band is absent in-context.
            # Non-geographic person attributes are sourced globally -- the pool
            # only needs the band to be selectable to match the SA2 marginal.
            global_only = attr in _GLOBAL_DONOR_ATTRS
            reg_key = str(reg)
            if not global_only and reg_key not in reg_census_cache:
                reg_census_cache[reg_key] = (
                    census[census["REGUCP"].astype(str) == reg_key]
                    if "REGUCP" in census.columns else census.iloc[0:0])
            # For a global attr the region is irrelevant, so the donor frame is
            # shared across zones (cache keyed without the region).
            key = (attr, code, None if global_only else reg)
            if key not in donor_cache:
                creg = census.iloc[0:0] if global_only else reg_census_cache[reg_key]
                donor_cache[key] = _source_donors(
                    creg, census, attr, int(code),
                    _ORDINAL_SPECIAL.get(attr, set()),
                    attr in _ORDINAL_ATTRS, report, global_only=global_only)
            donors = donor_cache[key]
            if donors is None or donors.empty:
                if (str(row["attr"]).lower(), code) in padding:
                    skipped["padding_not_applicable"] += 1
                else:
                    skipped["no_donor_anywhere"] += 1
                continue
            if attr not in modeled_u:
                # non-modelled table attr: verbatim household transplant
                frames, next_id = verbatim_transplant(
                    census, donors, reg, next_id, need, rng)
                if not frames:
                    skipped["verbatim_no_household"] += 1
                    continue
                blocks.extend(frames)
                devs.append(0)     # verbatim copy introduces no deviation
                continue
            donor_profile = {a: donors[a].mode().iloc[0]
                             for a in MODELED_PERSON if a in donors.columns
                             and not donors[a].mode().empty}
            donor_profile = {a: v for a, v in donor_profile.items()
                             if pd.notna(v)}
            if "RLHP" not in donor_profile or pd.isna(
                    donor_profile["RLHP"]):
                skipped["no_rlhp"] += 1
                continue
            shells = choose_shells(index, reg, int(donor_profile["RLHP"]),
                                   max_shells=shells_per_cell)
            if not shells:
                skipped["no_local_shell"] += 1
                continue
            placed = 0
            for _ in range(need):
                res = transplant_member(census, shells, index,
                                        donor_profile, reg, global_dom,
                                        rlhp_dom, cuts, next_id)
                if res is None:
                    break   # same shells will fail again; stop retrying this cell
                block, dev, _ = res
                block = block.drop(columns=["reindexed_slot"])
                blocks.append(block)
                devs.append(int(dev))
                next_id += 1
                cells_done += 1
                placed += 1
            if placed < need:
                # No shell can host a solved profile for this cell: the role's
                # conditional domain can be empty once the rule cuts apply (e.g.
                # a role whose census members all violate a shared check).  Fall
                # back to copying real households that carry the category, the
                # same recourse the non-modelled attributes use, rather than
                # emitting nothing for a cell the pool genuinely lacks.
                frames, next_id = verbatim_transplant(
                    census, donors, reg, next_id, need - placed, rng)
                if frames:
                    blocks.extend(frames)
                    devs.extend([0] * len(frames))
                    cells_done += len(frames)
                else:
                    skipped["solve_or_validate_failed"] += 1
    report["n_transplant_households"] = len(blocks)
    report["n_transplant_rows"] = int(sum(len(b) for b in blocks))
    if devs:
        report["deviation_mean"] = float(np.mean(devs))
        report["deviation_max"] = int(max(devs))
    report["skipped"] = dict(skipped)

    # --- merge & write ----------------------------------------------------
    out = pool.copy()
    if "_source" not in out.columns:
        out["_source"] = "census"
    if blocks:
        out = pd.concat([out] + blocks, ignore_index=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    report["out"] = out_path
    report["n_out_rows"] = int(len(out))
    report["runtime_s"] = round(time.time() - t0, 1)

    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    return report, out


def _table_zones(ind_tables, hh_tables):
    zones = set()
    for tables in (ind_tables, hh_tables):
        for entry in tables.values():
            df = entry.get("data")
            if df is not None and "sa2" in df.columns:
                zn = _normalize(df["sa2"].dropna()).astype(str)
                zones.update(z for z in zn if z and z.lower() != "nan")
    return sorted(zones)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=os.path.join(
        HERE, "outputs", "feasible_pool.csv"))
    ap.add_argument("--census", default=config.CENSUS_MICRODATA)
    ap.add_argument("--rep-max", type=int, default=40)
    ap.add_argument("--max-transplant-cells", type=int, default=4000)
    ap.add_argument("--shells-per-cell", type=int, default=8)
    ap.add_argument("--max-per-cell", type=int, default=20)
    ap.add_argument("--zones-limit", type=int, default=None,
                    help="limit audit to the first N table zones (smoke)")
    ap.add_argument("--zones-file", default=None,
                    help="optional file of zone IDs to audit (one per line)")
    ap.add_argument("--pool-nrows", type=int, default=None)
    ap.add_argument("--census-nrows", type=int, default=None)
    ap.add_argument("--mode", choices=["joint", "per-cell"], default="joint",
                    help="joint (default): one joint-marginal selection per "
                         "zone; per-cell: legacy independent deficit program")
    ap.add_argument("--max-new-households", type=int, default=64,
                    help="joint mode: maximum selected households per zone")
    ap.add_argument("--max-new-persons", type=int, default=256,
                    help="joint mode: maximum selected persons per zone")
    ap.add_argument("--donors-per-cell", type=int, default=4,
                    help="joint mode: donor records sampled per zero cell")
    ap.add_argument("--joint-time-limit", type=float, default=30.0,
                    help="joint mode: HiGHS time limit per zone, seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        HERE, "outputs", "feasible_pool_v5.csv"))
    ap.add_argument("--report", default=os.path.join(
        HERE, "outputs", "mip_pool_report.json"))
    a = ap.parse_args()
    if a.mode == "joint":
        from mip_pool_joint import run_joint
        run_joint(
            pool_path=a.pool, census_path=a.census, rep_max=a.rep_max,
            out_path=a.out, max_transplant_cells=a.max_transplant_cells,
            shells_per_cell=a.shells_per_cell, zones_limit=a.zones_limit,            max_per_cell=a.max_per_cell, pool_nrows=a.pool_nrows,
            census_nrows=a.census_nrows, report_path=a.report,
            max_new_households=a.max_new_households,
            max_new_persons=a.max_new_persons,
            donors_per_cell=a.donors_per_cell, time_limit=a.joint_time_limit,
            seed=a.seed, zones_file=a.zones_file)
        return
    run(pool_path=a.pool, census_path=a.census, rep_max=a.rep_max,
        out_path=a.out, max_transplant_cells=a.max_transplant_cells,
        shells_per_cell=a.shells_per_cell, zones_limit=a.zones_limit,
        max_per_cell=a.max_per_cell, pool_nrows=a.pool_nrows,
        census_nrows=a.census_nrows, report_path=a.report, seed=a.seed)


if __name__ == "__main__":
    main()

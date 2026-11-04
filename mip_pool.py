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
      --rep-max 40 --out outputs/feasible_pool_v5.csv

Expansion is marginal-driven: the per-cell requirement is
``ceil(gap / rep_max)`` and the per-zone budget is that sum, so there is no
fixed transplant-cell or per-cell ceiling unless one is passed explicitly.
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
        # Households that already contain each role, per REGUCP.  This avoids
        # scanning every shell in the region for every donor/cell pair when a
        # role is required.
        self.role_ids = {}
        if ("RLHP" in census.columns and "REGUCP" in census.columns
                and "ABSHID" in census.columns):
            tmp = census[["ABSHID", "RLHP", "REGUCP"]].dropna()
            for (reg, role), grp in tmp.groupby(["REGUCP", "RLHP"], sort=False):
                try:
                    key = (str(_norm_scalar(reg)), int(role))
                except (TypeError, ValueError):
                    continue
                self.role_ids.setdefault(key, set()).update(
                    grp["ABSHID"].unique())
            valid = set(self.meta.index)
            self.role_ids = {k: v & valid
                             for k, v in self.role_ids.items()}

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

    @property
    def host_ids(self):
        """Households a grown member can join, computed once per census.

        A host must be a family household (HHCD 1 or 2) whose size code already
        describes its members; growth rewrites that code, and rewriting one that
        was already wrong would present the mismatch as audited.
        """
        if not hasattr(self, "_host_ids"):
            ok = pd.Series(True, index=self.meta.index)
            if "hhcd" in self.meta.columns:
                ok &= pd.to_numeric(self.meta["hhcd"],
                                    errors="coerce").isin([1, 2])
            if "nprd" in self.meta.columns:
                sizes = pd.to_numeric(self.meta["size"], errors="coerce")
                nprd = pd.to_numeric(self.meta["nprd"], errors="coerce")
                ok &= nprd == sizes.clip(upper=NPRD_MAX_CODE)
            self._host_ids = set(self.meta.index[ok.fillna(False)])
        return self._host_ids


def _norm_zone_series(series):
    return _normalize(series)


def _normalize(series):
    from sa_only_baseline import _normalize_zone_values
    return _normalize_zone_values(series)


def _norm_scalar(v):
    return str(_normalize(pd.Series([v])).iloc[0])


_SHELL_ORDER_CACHE = {}


def _ordered_reg_ids(index, reg, size_preference, role=None, roles=None):
    """Household ids in a region, ordered by target-size preference.

    Sorting the region once per zone (rather than once per candidate) matters:
    the expansion calls choose_shells for every donor/cell pair.

    ``roles`` restricts the pool to households holding any of those roles, which
    is how growth asks for a family rather than for one particular role.
    """
    if roles:
        reg_key = str(_norm_scalar(reg))
        base = sorted({hid for r in roles
                       for hid in index.role_ids.get((reg_key, int(r)), set())},
                      key=str)
    elif role is not None:
        base = list(index.role_ids.get(
            (str(_norm_scalar(reg)), int(role)), []))
        # A donor role such as 11 (not applicable) may have no shell at all.
        # In that case the shell slot supplies the role, so fall back to every
        # shell in the region rather than forcing a verbatim donor household.
        if not base:
            base = list(index.by_reg(reg))
    else:
        base = list(index.by_reg(reg))
    if not size_preference:
        return base
    key = (str(reg), None if role is None else int(role),
           tuple(int(r) for r in roles) if roles else None, tuple(sorted(
               (int(k), round(float(v), 8))
               for k, v in size_preference.items())))
    cached = _SHELL_ORDER_CACHE.get(key)
    if cached is not None:
        return cached

    def _weight(hhid):
        try:
            size = int(index.meta.loc[hhid, "size"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        return float(size_preference.get(size, 0.0))

    ids = base
    ids.sort(key=lambda h: (-_weight(h), str(h)))
    _SHELL_ORDER_CACHE[key] = ids
    return ids


def choose_shells(index, reg, role, max_shells=8, required=None,
                  size_preference=None, any_of=None, host_checks=False):
    """Candidate local shell households (household ids) for a transplant.

    Only households that already contain a member with the donor's RLHP role
    are eligible: the solved profile is written into that slot, so a shell
    without one can never host the transplant.  `required` additionally filters
    on dwelling terms the shell must already carry (the joint expansion uses
    this so a household-level deficit is never paired with a shell that cannot
    cover it).  Fields outside the shell metadata are left to the solver.

    ``any_of`` replaces the single-role filter with a set: growth needs a
    household that contains a family, not one that already holds the donor's
    role, and it must not be offered the non-private dwellings that open a
    region (RLHP 9/11, no family to join).  ``host_checks`` additionally keeps
    only households a grown member could actually join -- a family household
    (HHCD 1/2) whose size code already matches its member count -- so a small
    ``max_shells`` is spent on usable candidates rather than consumed by
    households the solver would reject.

    ``size_preference`` is an optional ``{household_size: weight}`` map, normally
    the zone's target ``nprd`` shares.  Eligible shells are tried in descending
    preference, so a size the zone barely needs (for example a size-7 household
    when the target has one) is used only when no better-sized shell can host
    the donor role and required cell.
    """
    req = {str(a).lower(): int(c) for a, c in (required or {}).items()}
    roles = tuple(int(r) for r in any_of) if any_of else None
    reg_ids = _ordered_reg_ids(index, reg, size_preference, role=role,
                               roles=roles)
    cand = []
    if host_checks:
        reg_ids = [h for h in reg_ids if h in index.host_ids]
    for hhid in reg_ids:
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
        if rows is None:
            continue
        if roles:
            if not any(index.has_role(rows, r) for r in roles):
                continue
        elif not index.has_role(rows, role):
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
    """The shared rule sets on one household.

    CONSTRAINT_CHECKS is applied row-wise and HOUSEHOLD_CHECKS once over the
    members together, which is where the size code (NPRD) is checked against
    the number of persons present.

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
    for name, fn in constraint_rules.HOUSEHOLD_CHECKS.items():
        try:
            bad = np.asarray(fn(block), dtype=bool)
            if bad.any():
                viol.append(f"household {name}:{int(bad.sum())}")
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
    try:
        preferred_role = int(donor_profile.get("RLHP"))
    except (TypeError, ValueError):
        preferred_role = None
    free_attrs = [a for a in MODELED_PERSON if a != "RLHP"]
    for hhid, rows in shells:
        block = census.iloc[rows].reset_index(drop=True).copy()
        roles = pd.to_numeric(block["RLHP"], errors="coerce").dropna()
        roles = [int(v) for v in roles.unique()]
        if not roles:
            continue
        role = preferred_role if preferred_role in roles else roles[0]
        fixed = {"RLHP": role}
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
# Opt-in shell growth: insert one donor member into an existing family
# ---------------------------------------------------------------------------

# RLHP roles a grown household can admit.  2 (lone parent) and 7 (non-family
# member) are absent on purpose: in the ABS family coding each of them starts a
# new family record, which changes the household's family count and with it
# HHCD, so that construction is a family clone rather than a growth.
GROWTH_ROLES = (1, 3, 4, 5, 6)

# Construction modes (shell growth, family clone) both need a shell household
# that already contains a family, so shell selection is restricted to
# households holding a parent or partner role.  Without this the region's
# non-private and non-classifiable dwellings (RLHP 9/11) fill the candidate
# list and every attempt is refused.
FAMILY_HOST_ROLES = (1, 2)

# Age window each role admits, in census AGEP coding (single years to 24, then
# five-year bands: 25-29 -> 25 ... 85+ -> 37).  A member outside its role's
# window is not a valid holder of that role, so the domain is cut before the
# solve instead of being checked afterwards.
GROWTH_ROLE_AGE = {1: (15, 37), 3: (0, 14), 4: (15, 24), 5: (15, 37),
                   6: (15, 37)}

# The fused microdata codes household size as 1-5 exact and 6 for "six or
# more"; 7 is "not applicable" (no usual resident size) and appears only in
# visitors-only / non-classifiable dwellings, which these construction paths
# never rewrite.  MAX_HH_SIZE (12) is the structural ceiling on members, not on
# the code.
NPRD_MAX_CODE = 6

# Fresh person ids for grown members: above any census id, and unique because
# the household id is embedded and a household gains at most MAX_HH_SIZE rows.
GROWN_PERSON_ID_BASE = 9_000_000_000

# Household- and family-level fields the census repeats on every member row.  A
# grown member copies these from its shell so it shares the dwelling and family
# context; every column not listed is person-level and keeps the donor's own
# value.  ABSFID/FNOF are then overwritten with the family the member joins.
HOUSEHOLD_FIELDS = MODELED_DWELL + [
    "TEND", "TENLLD", "FMCF", "FRLF", "SSCF", "SPLF", "CPAF", "CPAD",
    "CPLTHRD", "CPRF", "CACF", "ABSFID", "FNOF", "DWTD", "HIED",
    "INGDWTD", "LLDD", "MV1D", "MV5D", "FINASF", "AREAENUM", "REGUCP",
    "REGU1P", "REGU5P", "STATE",
]

# Columns restored to a numeric dtype after a row is appended: pandas stores a
# mixed row as object, and recompute_family_counts compares AGEP numerically.
_NUMERIC_AFTER_APPEND = ["AGEP", "NPRD", "FNOF", "ABSFID", "RLHP", "ABSPID",
                         "CACF", "CPRF", "persons_in_family",
                         "persons_in_household", "families_in_household"]


def nprd_of(size):
    """NPRD code for a member count: 1-5 exact, 6 = six or more."""
    return int(min(max(int(size), 1), NPRD_MAX_CODE))


def family_column(block):
    """Column that identifies one family inside a household, or None.

    FNOF is required because recompute_family_counts groups on it: without it
    the family counters could not be rewritten from the members present, so the
    household is not grown.
    """
    if "FNOF" not in block.columns:
        return None
    if not pd.to_numeric(block["FNOF"], errors="coerce").notna().all():
        return None
    return "FNOF"


def growth_families(block, role, key="FNOF"):
    """Families in `block` that a member holding `role` can join.

    A partner may only complete a couple that already has exactly one partner,
    and a child or other relative may only join a family that already has a
    parent.  A second lone parent is never admitted: it starts a new family.
    Families are returned smallest first, so the member joins the family with
    the most room, with ties broken by family code.
    """
    out = []
    if role not in GROWTH_ROLES:
        return out          # a lone parent or non-family member adds a family
    for value, part in block.groupby(key, sort=True):
        roles = pd.to_numeric(part["RLHP"], errors="coerce")
        partners = int((roles == 1).sum())
        lone = int((roles == 2).sum())
        if role == 1:
            if partners == 1 and lone == 0:
                out.append((value, part))
        elif partners + lone >= 1:
            out.append((value, part))
    out.sort(key=lambda kv: (len(kv[1]), str(kv[0])))
    return out


def _grown_row(donor, block, profile, role, part, next_id):
    """The new member row: donor person fields inside the shell's household.

    The donor row supplies every person-level field, including the ones the MIP
    does not model (industry, study, relationship), so the added person is a
    real census person rather than a copy of a sibling.  Household fields come
    from the shell and the family key from the family the member joins, which
    is what keeps the grown household internally consistent.
    """
    row = donor.to_dict()
    for col in HOUSEHOLD_FIELDS:
        if col in block.columns:
            row[col] = block[col].iloc[0]
    for attr, code in profile.items():
        if attr in row or attr in block.columns:
            row[attr] = code
    row["RLHP"] = int(role)
    # The new member belongs to the shell's household while the block is being
    # validated; the fresh id is written across the whole block afterwards.
    # Letting the donor's own ABSHID through (or assigning the new id only here)
    # splits the household in two for any check that groups by ABSHID, and the
    # size code is one of those.
    if "ABSHID" in block.columns:
        row["ABSHID"] = block["ABSHID"].iloc[0]
    if "ABSPID" in block.columns:
        # The base is above every census person id (all below 1.3M) and the
        # household part makes it unique; the joint menu runs on negative
        # household ids, so the magnitude is what identifies the household.
        row["ABSPID"] = (GROWN_PERSON_ID_BASE
                         + abs(int(next_id)) * 100 + (len(block) + 1))
    for col in ("ABSFID", "FNOF"):
        if col in block.columns and col in part.columns:
            row[col] = part[col].iloc[0]
    grown = pd.concat([block, pd.DataFrame([row])], ignore_index=True)
    for col in _NUMERIC_AFTER_APPEND:
        if col in grown.columns:
            grown[col] = pd.to_numeric(grown[col], errors="coerce")
    return _ensure_counters(grown)


def _ensure_counters(block):
    """Household/family counters a grown block must carry, derived from rows.

    The census repeats no ``persons_in_*`` fields, while the constructed pool
    carries all three counters; a block built from census rows would otherwise
    append them empty and the merged pool would lose them for exactly the
    households growth touched.  They are written here from the members present,
    so the block is consistent whichever frame it is merged into.
    """
    block["persons_in_household"] = len(block)
    if "FNOF" in block.columns:
        block["persons_in_family"] = block.groupby(
            "FNOF")["FNOF"].transform("size")
        if "FMCF" in block.columns:
            block["families_in_household"] = int(pd.to_numeric(
                block["FMCF"], errors="coerce").iloc[0])
        else:
            block["families_in_household"] = int(block["FNOF"].nunique())
    return block


def growth_violations(block, role, key="FNOF"):
    """Post-growth checks the shared rule set cannot make.

    CONSTRAINT_CHECKS reads no family counter and no household-size field, so a
    grown household is checked here for the properties growth is responsible
    for: the size code describes the members present, each family's CPRF/CACF
    and the household's person count agree with its rows, and the role
    structure stayed admissible (at most two partners and one lone parent per
    family, children and relatives only in a family with a parent).
    """
    bad, n = [], len(block)
    if "NPRD" in block.columns:
        nprd = int(pd.to_numeric(block["NPRD"], errors="coerce").iloc[0])
        if nprd != nprd_of(n):
            bad.append(f"nprd:{nprd}!={nprd_of(n)}")
    if "persons_in_household" in block.columns:
        people = int(pd.to_numeric(block["persons_in_household"],
                                   errors="coerce").iloc[0])
        if people != n:
            bad.append(f"persons_in_household:{people}!={n}")
    if key is not None and key in block.columns:
        families = pd.to_numeric(block[key], errors="coerce")
        for value, part in block.groupby(families):
            roles = pd.to_numeric(part["RLHP"], errors="coerce")
            partners = int((roles == 1).sum())
            lone = int((roles == 2).sum())
            if partners > 2:
                bad.append(f"partners:{value}:{partners}")
            if lone > 1:
                bad.append(f"lone_parents:{value}:{lone}")
            if role in (3, 4, 5, 6) and partners + lone == 0:
                bad.append(f"no_parent:{value}")
            if "CPRF" in part.columns:
                cprf = int(pd.to_numeric(part["CPRF"],
                                         errors="coerce").iloc[0])
                if cprf != len(part):
                    bad.append(f"cprf:{value}:{cprf}!={len(part)}")
            if "CACF" in part.columns and "AGEP" in part.columns:
                cacf = int(pd.to_numeric(part["CACF"],
                                         errors="coerce").iloc[0])
                kids = int((pd.to_numeric(part["AGEP"], errors="coerce")
                            < constraint_rules.AGEP_CHILD_THRESHOLD).sum())
                if (cacf != constraint_rules.CACF_RESIDUAL and cacf != kids):
                    bad.append(f"cacf:{value}:{cacf}!={kids}")
    return bad


def transplant_growth(census, shells, donors, reg, global_dom,
                      rlhp_dom, cuts, next_id, required=None, max_donors=4):
    """Insert one donor member into an existing family of a local shell.

    Where transplant_member overwrites an occupied slot, growth appends a
    member: the shell's rows are untouched, the donor profile is solved for the
    *new* row under the role it will hold and the family it joins, and the
    household's size code and family counters are then rewritten from the
    members present.  The whole household is checked before it is emitted.

    The covering row is the new member, so a person-level requirement does not
    have to exist in the shell beforehand; a dwelling-level requirement must,
    because the new member inherits the shell's dwelling terms.  Callers
    therefore filter shells on dwelling requirements only.

    donors: census rows carrying the target cell, best first.  Their person
    fields are both the repair target and the source of the unmodelled
    attributes of the added member.

    Returns (block, deviation, hhid, family) or None.
    """
    req = {str(a).upper(): int(c) for a, c in (required or {}).items()}
    if donors is None or len(donors) == 0 or "RLHP" not in donors.columns:
        return None
    free = [a for a in MODELED_PERSON if a != "RLHP"]
    donor_roles = pd.to_numeric(donors["RLHP"], errors="coerce").dropna()
    donor_roles = donor_roles.astype(int).value_counts()
    tried = 0
    for role, _count in donor_roles.items():
        role = int(role)
        if role not in GROWTH_ROLES:
            continue
        if tried >= int(max_donors):
            break
        tried += 1
        rows_for_role = donors.loc[
            pd.to_numeric(donors["RLHP"], errors="coerce") == role]
        if rows_for_role.empty:
            continue
        donor = rows_for_role.iloc[0]
        lo, hi = GROWTH_ROLE_AGE[role]
        target = {a: int(donor[a]) for a in MODELED_PERSON
                  if a in donor.index and pd.notna(donor[a])}
        for hhid, rows in shells:
            block = census.iloc[rows].reset_index(drop=True).copy()
            n_before = len(block)
            if n_before + 1 > MAX_HH_SIZE:
                continue
            if "RLHP" not in block.columns or "NPRD" not in block.columns:
                continue
            # Only a household that already contains a family can take a
            # member into one: HHCD 3 (lone person), 4 (group), 5 (visitors)
            # and 6 have no family to join, and growing them would change the
            # household composition code rather than its size.
            if "HHCD" in block.columns:
                hhcd = pd.to_numeric(block["HHCD"], errors="coerce").iloc[0]
                if pd.isna(hhcd) or int(hhcd) not in (1, 2):
                    continue
            # A shell whose size code already disagrees with its member count
            # is not grown: rewriting NPRD would turn a pre-existing mismatch
            # into a record that looks audited.
            try:
                nprd_before = int(pd.to_numeric(block["NPRD"],
                                                errors="coerce").iloc[0])
            except (TypeError, ValueError):
                continue
            if nprd_before != nprd_of(n_before):
                continue
            key = family_column(block)
            if key is None:
                continue
            families = growth_families(block, role, key)
            if not families:
                continue
            family, part = families[0]
            fixed = {"RLHP": role}
            for col in [d for d in MODELED_DWELL if d in block.columns]:
                val = block[col].iloc[0]
                fixed[col] = int(val) if pd.notna(val) else -1
            blocked = False
            for attr, code in req.items():
                if attr in fixed:
                    blocked = blocked or int(fixed[attr]) != code
                elif attr not in free:
                    # family or derived field: a grown member cannot set it
                    blocked = True
            if blocked:
                continue
            freedom = prune_domains(free, global_dom, rlhp_dom, role, fixed,
                                    cuts)
            if any(not codes for codes in freedom.values()):
                continue
            for attr, code in req.items():
                if attr in freedom:
                    freedom[attr] = sorted(
                        set(int(c) for c in freedom[attr]) | {code})
            ages = [c for c in freedom.get("AGEP", []) if lo <= int(c) <= hi]
            if not ages:
                continue
            freedom["AGEP"] = ages
            # A required AGEP outside the role's window is left out of the
            # domain above, and solve_nearest_profile then reports the cell
            # infeasible rather than emitting a member in the wrong role.
            profile, dev = solve_nearest_profile(freedom, fixed, target, cuts,
                                                 required=req)
            if profile is None:
                continue
            grown = _grown_row(donor, block, profile, role, part, next_id)
            if len(grown) != n_before + 1:
                continue
            if grown[key].nunique() != block[key].nunique():
                continue                      # growth must not add a family
            if "families_in_household" in block.columns:
                before = pd.to_numeric(block["families_in_household"],
                                       errors="coerce").iloc[0]
                after = pd.to_numeric(grown["families_in_household"],
                                      errors="coerce").iloc[0]
                if int(before) != int(after):
                    continue
            grown["NPRD"] = nprd_of(len(grown))
            grown = recompute_family_counts(grown)
            if validate_household(grown) or growth_violations(grown, role, key):
                continue
            covered = True
            for attr, code in req.items():
                if attr not in grown.columns:
                    covered = False
                    break
                vals = pd.to_numeric(grown[attr], errors="coerce")
                if not bool((vals == code).any()):
                    covered = False
                    break
            if not covered:
                continue
            grown["ABSHID"] = int(next_id)
            if "REGUCP" in grown.columns:
                grown["REGUCP"] = reg
            grown["_source"] = "mip-growth"
            return grown, dev, hhid, family
    return None


# ---------------------------------------------------------------------------
# Opt-in family clone: duplicate a family of the shell under a fresh FNOF
# ---------------------------------------------------------------------------

# Roles that head or belong to a family, so a family holding one can be cloned.
# 7 (non-family member), 9 and 11 are not families: a non-family member is its
# own FNOF group and cloning one would add a person, not a family.
CLONE_ROLES = (1, 2, 3, 4, 5, 6)

# Age window each cloned role admits, in census AGEP coding.  A clone takes the
# template family's structure, so the carrier stays the age its role implies.
CLONE_ROLE_AGE = {1: (15, 37), 2: (15, 37), 3: (0, 14), 4: (15, 24),
                  5: (15, 37), 6: (15, 37)}


def household_composition(families):
    """HHCD a family household should carry, from its family count.

    ABS codes 1 for a one-family household and 2 for two or more; the other
    codes (lone person, group, visitors, other) are not family households at
    all.  Both clone and growth rely on this to tell a consistent household from
    one whose composition code already disagrees with its members.
    """
    return 1 if int(families) <= 1 else 2


def clone_templates(block, role, key="FNOF"):
    """Families of `block` that can be cloned to host `role`.

    A template must already contain the donor's role, so the cloned member can
    take the donor profile in place without changing the family's shape.  A
    lone parent is only cloned from a one-parent family, and every other role
    needs a template with a parent in it.  Smallest first, ties by family code.
    """
    out = []
    if role not in CLONE_ROLES:
        return out
    for value, part in block.groupby(key, sort=True):
        roles = pd.to_numeric(part["RLHP"], errors="coerce")
        partners = int((roles == 1).sum())
        lone = int((roles == 2).sum())
        if not bool((roles == role).any()):
            continue
        if role == 2:
            if lone != 1 or partners:
                continue
        elif partners + lone == 0:
            continue
        out.append((value, part))
    out.sort(key=lambda kv: (len(kv[1]), str(kv[0])))
    return out


def _clone_rows(part, block, fresh_family, next_id, role, profile):
    """The cloned family: a copy of `part` under a fresh family key.

    The copy keeps the template's members, roles and ages, so the clone is a
    real family shape; the member holding the donor's role then takes the solved
    profile.  Identifiers are fresh, and the family id is derived from the
    household id so sibling clones cannot collide.
    """
    clone = part.copy().reset_index(drop=True)
    clone["FNOF"] = int(fresh_family)
    if "ABSFID" in clone.columns:
        clone["ABSFID"] = abs(int(next_id)) * 10 + int(fresh_family)
    if "ABSPID" in clone.columns:
        clone["ABSPID"] = [GROWN_PERSON_ID_BASE + abs(int(next_id)) * 100
                           + len(block) + i for i in range(1, len(clone) + 1)]
    carrier = np.flatnonzero(
        (pd.to_numeric(clone["RLHP"], errors="coerce") == int(role)).to_numpy())
    if carrier.size == 0:
        return None
    for attr, code in profile.items():
        if attr in clone.columns:
            clone.iloc[int(carrier[0]), clone.columns.get_loc(attr)] = code
    return clone


def clone_violations(block, key="FNOF", families_before=None,
                     fmcf_before=None):
    """Checks a cloned household must pass beyond the shared rule set.

    The clone adds a family, so the household's size, composition code, family
    count and per-family counters all change together.  Each is checked here
    against the rows: NPRD describes the member count, HHCD describes the family
    count, every family's CPRF/CACF describes its own members, and the person
    count describes the household.

    ``families_before`` and ``fmcf_before`` are the shell's own values.  FMCF is
    checked for a single increment rather than against the family count: the
    census FMCF does not equal the distinct-FNOF count in most households, so
    the clone adds one to whatever the household counted instead of imposing a
    new definition on it.
    """
    bad, n = [], len(block)
    if "NPRD" in block.columns:
        nprd = int(pd.to_numeric(block["NPRD"], errors="coerce").iloc[0])
        if nprd != nprd_of(n):
            bad.append(f"nprd:{nprd}!={nprd_of(n)}")
    if "persons_in_household" in block.columns:
        people = int(pd.to_numeric(block["persons_in_household"],
                                   errors="coerce").iloc[0])
        if people != n:
            bad.append(f"persons_in_household:{people}!={n}")
    if key is None or key not in block.columns:
        return bad
    families = pd.to_numeric(block[key], errors="coerce")
    n_fam = int(families.nunique())
    if families_before is not None and n_fam != int(families_before) + 1:
        bad.append(f"families:{n_fam}!={int(families_before) + 1}")
    if "HHCD" in block.columns:
        hhcd = int(pd.to_numeric(block["HHCD"], errors="coerce").iloc[0])
        if hhcd != household_composition(n_fam):
            bad.append(f"hhcd:{hhcd}!={household_composition(n_fam)}")
    if fmcf_before is not None and "FMCF" in block.columns:
        fmcf = int(pd.to_numeric(block["FMCF"], errors="coerce").iloc[0])
        if fmcf != int(fmcf_before) + 1:
            bad.append(f"fmcf:{fmcf}!={int(fmcf_before) + 1}")
    for value, part in block.groupby(families):
        roles = pd.to_numeric(part["RLHP"], errors="coerce")
        partners = int((roles == 1).sum())
        lone = int((roles == 2).sum())
        if partners > 2:
            bad.append(f"partners:{value}:{partners}")
        if lone > 1:
            bad.append(f"lone_parents:{value}:{lone}")
        if "CPRF" in part.columns:
            cprf = int(pd.to_numeric(part["CPRF"], errors="coerce").iloc[0])
            if cprf != len(part):
                bad.append(f"cprf:{value}:{cprf}!={len(part)}")
        if "CACF" in part.columns and "AGEP" in part.columns:
            cacf = int(pd.to_numeric(part["CACF"], errors="coerce").iloc[0])
            kids = int((pd.to_numeric(part["AGEP"], errors="coerce")
                        < constraint_rules.AGEP_CHILD_THRESHOLD).sum())
            if cacf != constraint_rules.CACF_RESIDUAL and cacf != kids:
                bad.append(f"cacf:{value}:{cacf}!={kids}")
    return bad


def transplant_family_clone(census, shells, donors, reg, global_dom,
                            rlhp_dom, cuts, next_id, required=None,
                            max_donors=4):
    """Clone one family of a local shell into a second family of that household.

    Growth adds a member; this adds a family.  The template is a family the
    shell already contains, duplicated under a fresh FNOF with fresh family and
    person identifiers, so the added family has a real shape rather than an
    invented one.  The cloned member that holds the donor's role is solved
    against the donor profile, which is what lets the mode cover a cell the
    shell did not contain.  The household's composition code, size code, family
    count and family counters are then rewritten from the rows present and the
    whole household is re-checked.

    A shell is only used when its own composition code already describes its
    families, so cloning never turns a mismatched household into one that looks
    audited.

    Returns (block, deviation, hhid, family) or None.
    """
    req = {str(a).upper(): int(c) for a, c in (required or {}).items()}
    if donors is None or len(donors) == 0 or "RLHP" not in donors.columns:
        return None
    free = [a for a in MODELED_PERSON if a != "RLHP"]
    donor_roles = pd.to_numeric(donors["RLHP"], errors="coerce").dropna()
    donor_roles = donor_roles.astype(int).value_counts()
    tried = 0
    for role, _count in donor_roles.items():
        role = int(role)
        if role not in CLONE_ROLES:
            continue
        if tried >= int(max_donors):
            break
        tried += 1
        rows_for_role = donors.loc[
            pd.to_numeric(donors["RLHP"], errors="coerce") == role]
        if rows_for_role.empty:
            continue
        donor = rows_for_role.iloc[0]
        lo, hi = CLONE_ROLE_AGE[role]
        target = {a: int(donor[a]) for a in MODELED_PERSON
                  if a in donor.index and pd.notna(donor[a])}
        for hhid, rows in shells:
            block = census.iloc[rows].reset_index(drop=True).copy()
            if "RLHP" not in block.columns or "NPRD" not in block.columns:
                continue
            key = family_column(block)
            if key is None:
                continue
            try:
                nprd_before = int(pd.to_numeric(block["NPRD"],
                                                errors="coerce").iloc[0])
                hhcd_before = int(pd.to_numeric(block["HHCD"],
                                                errors="coerce").iloc[0])
            except (TypeError, ValueError):
                continue
            if nprd_before != nprd_of(len(block)):
                continue
            families = pd.to_numeric(block[key], errors="coerce")
            if hhcd_before != household_composition(families.nunique()):
                continue
            templates = clone_templates(block, role, key)
            if not templates:
                continue
            family, part = templates[0]
            if len(block) + len(part) > MAX_HH_SIZE:
                continue
            fixed = {"RLHP": role}
            for col in [d for d in MODELED_DWELL if d in block.columns]:
                val = block[col].iloc[0]
                fixed[col] = int(val) if pd.notna(val) else -1
            blocked = False
            for attr, code in req.items():
                if attr in fixed:
                    blocked = blocked or int(fixed[attr]) != code
                elif attr not in free:
                    blocked = True
            if blocked:
                continue
            freedom = prune_domains(free, global_dom, rlhp_dom, role, fixed,
                                    cuts)
            if any(not codes for codes in freedom.values()):
                continue
            for attr, code in req.items():
                if attr in freedom:
                    freedom[attr] = sorted(
                        set(int(c) for c in freedom[attr]) | {code})
            ages = [c for c in freedom.get("AGEP", [])
                    if lo <= int(c) <= hi]
            if not ages:
                continue
            freedom["AGEP"] = ages
            profile, dev = solve_nearest_profile(freedom, fixed, target, cuts,
                                                 required=req)
            if profile is None:
                continue
            fresh = int(families.max()) + 1
            clone = _clone_rows(part, block, fresh, next_id, role, profile)
            if clone is None:
                continue
            grown = pd.concat([block, clone], ignore_index=True)
            for col in _NUMERIC_AFTER_APPEND + ["FMCF", "HHCD"]:
                if col in grown.columns:
                    grown[col] = pd.to_numeric(grown[col], errors="coerce")
            # One family was added, so the household's own family count grows by
            # one; FMCF is incremented rather than re-derived because the census
            # FMCF does not equal the distinct-FNOF count in most households.
            families_before = int(families.nunique())
            fmcf_before = (int(pd.to_numeric(block["FMCF"],
                                             errors="coerce").iloc[0])
                           if "FMCF" in block.columns else None)
            if fmcf_before is not None:
                grown["FMCF"] = fmcf_before + 1
            n_fam = int(grown[key].nunique())
            if "HHCD" in grown.columns:
                grown["HHCD"] = household_composition(n_fam)
            grown["NPRD"] = nprd_of(len(grown))
            grown = _ensure_counters(grown)
            grown = recompute_family_counts(grown)
            if validate_household(grown) or clone_violations(
                    grown, key, families_before=families_before,
                    fmcf_before=fmcf_before):
                continue
            covered = True
            for attr, code in req.items():
                if attr not in grown.columns:
                    covered = False
                    break
                vals = pd.to_numeric(grown[attr], errors="coerce")
                if not bool((vals == code).any()):
                    covered = False
                    break
            if not covered:
                continue
            grown["ABSHID"] = int(next_id)
            if "REGUCP" in grown.columns:
                grown["REGUCP"] = reg
            grown["_source"] = "mip-clone"
            return grown, dev, hhid, fresh
    return None


# ---------------------------------------------------------------------------
# Verbatim household transplant (for deficits outside the MIP model space)
# ---------------------------------------------------------------------------

def verbatim_transplant(census, donor_rows, reg, next_id, need, rng,
                        size_preference=None):
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

    ``size_preference`` optionally orders the donor households by how well
    their size matches the zone's nprd target mix before sampling.

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
    if size_preference:
        hs = census.loc[census["ABSHID"].isin(hh_ids)].groupby(
            "ABSHID").size()
        hh_ids = np.asarray(sorted(
            hh_ids,
            key=lambda h: (-float(size_preference.get(int(hs.get(h, 0)), 0.0)),
                           str(h))), dtype=hh_ids.dtype)
        picked = pd.Index(hh_ids[:take])
    else:
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
        if cand is None or len(cand) == 0:
            # Empty SA4 slice: structural residual, never a full-pool fallback.
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

# Attributes that the global donor policy sources from the WHOLE census rather
# than region-first.  In strict mode (``--donor-policy strict``) these are
# treated like every other attribute: the zone's own REGUCP slice is
# authoritative, and a band absent there remains a structural residual.
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
                   global_only=False, allow_global=True):
    """Conditional donor ladder for a zero cell (attr, code).

    Global default (``allow_global=True``):
    1. SA4-exact  : donors in the zone's own REGUCP carrying the exact band;
    2. SA4-ordinal: nearest applicable band in that REGUCP, value shifted;
    3. global-exact  : donors anywhere carrying the band;
    4. global-ordinal: nearest applicable band anywhere, value shifted.

    The global steps are the point of expansion: the zone needs a donor for a
    cell the SA4 sample lacks, and the donor profile is imported from the wider
    population while the shell stays local.  The constructed record is then
    tagged to this SA2, so the pool grows without sharing the record with a
    sibling SA2.

    Strict mode (``allow_global=False``) stops after step 2: if the SA4 slice
    has no donor, the cell is a structural residual.

    Steps 2 and 4 are the conditional/ordinal fill: a band absent in context is
    reached by borrowing a real neighbour and shifting the ordinal value, with
    children and not-applicable codes excluded; full individual/household
    feasibility is then enforced by the caller's rule cuts.

    ``global_only`` is a legacy switch for non-geographic attributes; in strict
    mode it is ignored and the SA4 slice is authoritative.  Returns a donor
    frame or None.
    """
    code = int(code)
    if not (allow_global and global_only):
        # 1. SA4-exact
        if attr in creg.columns:
            d = creg[creg[attr] == code]
            if not d.empty:
                return d
        # 2. SA4-ordinal shift
        if ordinal and code not in special and len(creg):
            d = _nearest_ordinal_donors(creg, attr, code, special)
            if d is not None and not d.empty:
                d = d.copy(); d[attr] = code
                report["n_conditional_zero_cells"] = report.get("n_conditional_zero_cells", 0) + 1
                report["n_ordinal_shift_cells"] = report.get("n_ordinal_shift_cells", 0) + 1
                return d
        if not allow_global:
            return None
    # 3. global exact (legacy only)
    d = census[census[attr] == code]
    if not d.empty:
        report["n_global_donor_cells"] = report.get("n_global_donor_cells", 0) + 1
        return d
    # 4. global ordinal shift (legacy only)
    if ordinal and code not in special:
        d = _nearest_ordinal_donors(census, attr, code, special)
        if d is not None and not d.empty:
            d = d.copy(); d[attr] = code
            report["n_ordinal_shift_cells"] = report.get("n_ordinal_shift_cells", 0) + 1
            report["n_global_donor_cells"] = report.get("n_global_donor_cells", 0) + 1
            return d
    return None


def households_to_cover(gap, rep_max):
    """Distinct households needed to close one cell's capacity gap.

    A cell's remaining capacity gap is closed by ``ceil(gap / rep_max)``
    additional households, because each household can be replicated at most
    ``rep_max`` times.  This is the per-cell marginal requirement; it is not a
    tuning parameter and there is no separate per-cell ceiling.
    """
    return int(np.ceil(max(0.0, float(gap)) / max(1, int(rep_max))))


def marginal_zone_budget(cell_targets, rep_max, max_households=None,
                         max_persons=None, hh_size_cap=MAX_HH_SIZE):
    """Per-zone construction budget derived from the marginal gaps.

    ``cell_targets`` are the positive targets of the zone's sample-zero cells.
    Each needs ``ceil(T / rep_max)`` additional households, so the zone's
    household budget is that sum; the person budget multiplies it by the
    structural household-size cap so it cannot bind below coverage.  Explicit
    ``max_households`` / ``max_persons`` restore a legacy fixed ceiling.
    """
    need = int(sum(households_to_cover(t, rep_max) for t in cell_targets))
    households = (int(max_households) if max_households is not None
                  else max(1, need))
    persons = (int(max_persons) if max_persons is not None
               else max(1, need * int(hh_size_cap)))
    return households, persons, need


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(pool_path, census_path, rep_max, out_path,
        max_transplant_cells=None, shells_per_cell=8, zones_limit=None,
        max_per_cell=None, pool_nrows=None, census_nrows=None,
        report_path=None, seed=0, allow_global_donors=True,
        append_only=False, id_offset=0, shell_growth=False,
        family_clone=False):
    if shell_growth and family_clone:
        raise ValueError("shell_growth and family_clone add different things "
                         "to a household; enable one at a time")
    t0 = time.time()
    report = {"rep_max": rep_max, "pool": pool_path, "census": census_path,
              "shell_growth": bool(shell_growth),
              "family_clone": bool(family_clone),
              "budget": {"source": "ceil(gap/rep_max) per cell",
                         "max_transplant_cells": max_transplant_cells,
                         "max_per_cell": max_per_cell},
              "donor_policy": ("global-donor-ladder" if allow_global_donors
                               else "SA4-strict"),
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
                      or 0) + 1) + int(id_offset)
    donor_cache, cells_done = {}, 0
    growth_households = clone_households = slot_households = 0
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
        census_reg_col = census_col.get("regucp")

        padding = _padding_codes()
        is_pad = [(str(r["attr"]).lower(), int(r["code"])) in padding
                  for _, r in deficit_df.iterrows()]
        report["n_deficit_cells_padding"] = int(sum(is_pad))
        report["n_deficit_cells_coverable"] = int(len(is_pad) - sum(is_pad))

        rng = np.random.default_rng(seed)
        modeled_u = set(MODELED_PERSON)
        n_cells_to_process = len(deficit_df)
        if max_transplant_cells is not None:
            n_cells_to_process = min(n_cells_to_process,
                                     int(max_transplant_cells))
        report["n_cells_considered"] = int(n_cells_to_process)
        for _, row in deficit_df.head(n_cells_to_process).iterrows():
            attr, code = row["attr_col"], int(row["code"])
            reg = _norm_scalar(row["reg"])
            zone_id = str(row["zone"])
            gap_hh = households_to_cover(row["gap"], rep_max)
            need = gap_hh if max_per_cell is None else min(gap_hh,
                                                           int(max_per_cell))
            if need <= 0:
                continue
            if not isinstance(attr, str) or attr not in census.columns:
                skipped["attr_not_in_census"] += 1
                continue
            # Source donors local-first, then from the wider census when the
            # global policy is enabled (default).  This is how the constructed
            # pool reflects an SA2 target the SA4 sample cannot supply; each
            # record is tagged to its SA2 below.  Strict mode never borrows.
            global_only = attr in _GLOBAL_DONOR_ATTRS
            reg_key = str(reg)
            if reg_key not in reg_census_cache:
                reg_census_cache[reg_key] = (
                    census[census[census_reg_col].astype(str) == reg_key]
                    if census_reg_col is not None else census.iloc[0:0])
            key = (attr, code, reg_key, bool(allow_global_donors))
            if key not in donor_cache:
                creg = reg_census_cache[reg_key]
                donor_cache[key] = _source_donors(
                    creg, census, attr, int(code),
                    _ORDINAL_SPECIAL.get(attr, set()),
                    attr in _ORDINAL_ATTRS, report,
                    global_only=global_only,
                    allow_global=allow_global_donors)
            donors = donor_cache[key]
            if donors is None or donors.empty:
                if (str(row["attr"]).lower(), code) in padding:
                    skipped["padding_not_applicable"] += 1
                elif allow_global_donors:
                    skipped["no_donor_anywhere"] += 1
                else:
                    skipped["no_sa4_donor"] += 1
                continue
            if attr not in modeled_u:
                # non-modelled table attr: verbatim household transplant
                frames, next_id = verbatim_transplant(
                    census, donors, reg, next_id, need, rng)
                if not frames:
                    skipped["verbatim_no_household"] += 1
                    continue
                blocks.extend(f.assign(_target_sa2=zone_id) for f in frames)
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
            # Neither construction mode needs the donor's role to be present
            # already, so their shells are not filtered on it; the slot path is.
            family_shells = (choose_shells(index, reg, None,
                                           max_shells=shells_per_cell,
                                           any_of=FAMILY_HOST_ROLES,
                                           host_checks=True)
                             if (shell_growth or family_clone) else None)
            shells = choose_shells(index, reg, int(donor_profile["RLHP"]),
                                   max_shells=shells_per_cell)
            if not shells and not family_shells:
                skipped["no_local_shell"] += 1
                continue
            placed = 0
            for _ in range(need):
                if family_shells and shell_growth:
                    grown = transplant_growth(census, family_shells, donors,
                                              reg, global_dom, rlhp_dom,
                                              cuts, next_id)
                    if grown is not None:
                        block, dev, _shell, _family = grown
                        block = block.assign(_target_sa2=zone_id)
                        blocks.append(block)
                        devs.append(int(dev))
                        next_id += 1
                        cells_done += 1
                        growth_households += 1
                        placed += 1
                        continue
                if family_shells and family_clone:
                    cloned = transplant_family_clone(
                        census, family_shells, donors, reg, global_dom,
                        rlhp_dom, cuts, next_id)
                    if cloned is not None:
                        block, dev, _shell, _family = cloned
                        block = block.assign(_target_sa2=zone_id)
                        blocks.append(block)
                        devs.append(int(dev))
                        next_id += 1
                        cells_done += 1
                        clone_households += 1
                        placed += 1
                        continue
                if not shells:
                    break
                res = transplant_member(census, shells, index,
                                        donor_profile, reg, global_dom,
                                        rlhp_dom, cuts, next_id)
                if res is None:
                    break   # same shells will fail again; stop retrying this cell
                block, dev, _ = res
                block = block.drop(columns=["reindexed_slot"])
                block = block.assign(_target_sa2=zone_id)
                blocks.append(block)
                devs.append(int(dev))
                next_id += 1
                cells_done += 1
                slot_households += 1
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
                    blocks.extend(f.assign(_target_sa2=zone_id) for f in frames)
                    devs.extend([0] * len(frames))
                    cells_done += len(frames)
                else:
                    skipped["solve_or_validate_failed"] += 1
    report["n_transplant_households"] = len(blocks)
    report["n_transplant_rows"] = int(sum(len(b) for b in blocks))
    report["n_growth_households"] = int(growth_households)
    report["n_clone_households"] = int(clone_households)
    report["n_slot_households"] = int(slot_households)
    if devs:
        report["deviation_mean"] = float(np.mean(devs))
        report["deviation_max"] = int(max(devs))
    report["skipped"] = dict(skipped)

    # --- merge & write ----------------------------------------------------
    if append_only:
        out = (pd.concat(blocks, ignore_index=True) if blocks
               else pool.iloc[0:0].copy())
    else:
        out = pool.copy()
        if "_source" not in out.columns:
            out["_source"] = "census"
        if blocks:
            out = pd.concat([out] + blocks, ignore_index=True)
    report["append_only"] = bool(append_only)
    report["id_offset"] = int(id_offset)
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


def _optional_limit(value):
    """Parse a limit that may be disabled with 'all', 'none', or 'auto'."""
    text = str(value).strip().lower()
    if text in {"", "all", "none", "auto"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected an integer or 'all', got {value!r}") from exc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=os.path.join(
        HERE, "outputs", "feasible_pool.csv"))
    ap.add_argument("--census", default=config.CENSUS_MICRODATA)
    ap.add_argument("--rep-max", type=int, default=40)
    ap.add_argument("--max-transplant-cells", type=_optional_limit,
                    default=None,
                    help="Legacy ceiling on processed deficit cells; default "
                         "all cells. Use 'all' or omit to disable.")
    ap.add_argument("--shells-per-cell", type=int, default=8)
    ap.add_argument("--max-per-cell", type=_optional_limit, default=None,
                    help="Legacy per-cell ceiling on new households; default "
                         "ceil(gap/rep_max), with no ceiling. Use 'all' or "
                         "omit to keep the marginal-derived requirement.")
    ap.add_argument("--zones-limit", type=int, default=None,
                    help="limit audit to the first N table zones (smoke)")
    ap.add_argument("--zones-file", default=None,
                    help="optional file of zone IDs to audit (one per line)")
    ap.add_argument("--pool-nrows", type=int, default=None)
    ap.add_argument("--census-nrows", type=int, default=None)
    ap.add_argument("--mode", choices=["joint", "per-cell"], default="joint",
                    help="joint (default): one joint-marginal selection per "
                         "zone; per-cell: legacy independent deficit program")
    ap.add_argument("--max-new-households", type=_optional_limit, default=None,
                    help="joint mode: legacy fixed ceiling on selected "
                         "households per zone; default is derived from the "
                         "marginal gap")
    ap.add_argument("--max-new-persons", type=_optional_limit, default=None,
                    help="joint mode: legacy fixed ceiling on selected persons "
                         "per zone; default is derived from the marginal gap")
    ap.add_argument("--donors-per-cell", type=int, default=4,
                    help="joint mode: donor records sampled per zero cell")
    ap.add_argument("--append-menu-candidates", dest="append_menu",
                    action="store_true", default=True,
                    help="joint mode: append the full constructed candidate "
                         "menu per zone, not only the coverage-minimal "
                         "selection (default: on)")
    ap.add_argument("--coverage-only", dest="append_menu",
                    action="store_false",
                    help="joint mode: append only the selected households")
    ap.add_argument("--lambda-composition", type=float, default=1.0,
                    help="joint mode: multiplier on the marginal-deviation "
                         "term; values above 1 make composition matter more "
                         "relative to zero-cell coverage")
    ap.add_argument("--max-size7-share", type=float, default=None,
                    help="joint mode: optional cap on the selected set's "
                         "size-7 household share (use with --coverage-only)")
    ap.add_argument("--max-mtwp17-share", type=float, default=None,
                    help="joint mode: optional cap on the selected set's "
                         "MTWP code-17 person share (use with --coverage-only)")
    ap.add_argument("--min-admitted-households", type=int, default=0,
                    help="joint mode: lower bound on selected households; "
                         "use with --coverage-only to make the MIP select a "
                         "larger composition-controlled set")
    ap.add_argument("--joint-time-limit", type=float, default=30.0,
                    help="joint mode: HiGHS time limit per zone, seconds")
    ap.add_argument("--donor-policy", choices=["global", "strict"],
                    default="global",
                    help="Donor sourcing for expansion. 'global' (default) "
                         "draws donor profiles from the whole census so the "
                         "constructed pool can reflect the SA2 target; 'strict' "
                         "restricts donors to the zone's own REGUCP slice.")
    ap.add_argument("--include-partial-deficits", action="store_true",
                    default=False,
                    help="joint mode: expand every cell below its replication "
                         "capacity, not only zero-capacity cells")
    ap.add_argument("--size-aware-shells", action="store_true",
                    default=False,
                    help="joint mode: prefer shells whose household size "
                         "matches the zone's nprd target mix")
    construction = ap.add_mutually_exclusive_group()
    construction.add_argument(
        "--shell-growth", action="store_true", default=False,
        help="insert the donor member as a new person in an existing family "
             "instead of overwriting the shell's matching slot; the size code "
             "(NPRD) and the family counters are rewritten from the members "
             "and the household is re-validated. Falls back to the slot path "
             "when no family can admit the role.")
    construction.add_argument(
        "--family-clone", dest="family_clone", action="store_true",
        default=False,
        help="clone a family the shell already contains into a second family "
             "under a fresh FNOF, solve the cloned carrier against the donor, "
             "and rewrite the composition code, size code, family count and "
             "family counters from the rows. Falls back to the slot path when "
             "no family can be cloned for the role.")
    ap.add_argument("--append-only", action="store_true",
                    help="Write only newly constructed households, not the "
                         "full input pool (used by zone-sharded runs).")
    ap.add_argument("--id-offset", type=int, default=0,
                    help="Offset added to freshly assigned household IDs, so "
                         "zone-sharded outputs can be merged without collisions.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        HERE, "outputs", "feasible_pool_v5.csv"))
    ap.add_argument("--report", default=os.path.join(
        HERE, "outputs", "mip_pool_report.json"))
    a = ap.parse_args()
    allow_global_donors = a.donor_policy == "global"
    if a.mode == "joint":
        from mip_pool_joint import run_joint
        run_joint(
            pool_path=a.pool, census_path=a.census, rep_max=a.rep_max,
            out_path=a.out, max_transplant_cells=a.max_transplant_cells,
            shells_per_cell=a.shells_per_cell, zones_limit=a.zones_limit,
            max_per_cell=a.max_per_cell, pool_nrows=a.pool_nrows,
            census_nrows=a.census_nrows, report_path=a.report,
            max_new_households=a.max_new_households,
            max_new_persons=a.max_new_persons,
            donors_per_cell=a.donors_per_cell, time_limit=a.joint_time_limit,
            seed=a.seed, zones_file=a.zones_file,
            allow_global_donors=allow_global_donors,
            append_menu_candidates=a.append_menu,
            append_only=a.append_only, id_offset=a.id_offset,
            include_partial_deficits=a.include_partial_deficits,
            size_aware_shells=a.size_aware_shells,
            lambda_composition=a.lambda_composition,
            max_size7_share=a.max_size7_share,
            max_mtwp17_share=a.max_mtwp17_share,
            min_admitted_households=a.min_admitted_households,
            shell_growth=a.shell_growth, family_clone=a.family_clone)
        return
    run(pool_path=a.pool, census_path=a.census, rep_max=a.rep_max,
        out_path=a.out, max_transplant_cells=a.max_transplant_cells,
        shells_per_cell=a.shells_per_cell, zones_limit=a.zones_limit,
        max_per_cell=a.max_per_cell, pool_nrows=a.pool_nrows,
        census_nrows=a.census_nrows, report_path=a.report, seed=a.seed,
        allow_global_donors=allow_global_donors,
        append_only=a.append_only, id_offset=a.id_offset,
        shell_growth=a.shell_growth, family_clone=a.family_clone)


if __name__ == "__main__":
    main()

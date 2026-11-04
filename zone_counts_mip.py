"""zone_counts_mip.py -- one regularised integer program per zone.

Collapses "expand pool" + "fit zone" into a single solve. Instead of a 0/1
selection that a non-replicating fitter then throws away, this chooses an
INTEGER replication count n_j for each candidate household, so a rare carrier
can stand for as many realised copies as the cell needs. The returned counts
ARE the fitted zone population (expand each selected household n_j times).

FORMULATION (per zone z)
------------------------
Variables
    n_j   in {0,...,R_j}   integer replication count of candidate household j
    d_c   >= 0             deviation of controlled control cell c
    miss_c in {0,1}        cell c counted as "not hit"
    g_s   >= 0             deviation of soft (joint / anchor) target s

Objective
    min  L_miss  * sum_c miss_c                 # 1. minimise cells not hit
       + L_dev   * sum_c w_c d_c                 # 2. fit controlled cells
       + L_joint * sum_s v_s g_s                 # 3. joint + out-of-sample anchor
       + L_pars  * sum_j pi_j n_j                # 4. realism / parsimony

Constraints
    d_c  >= | sum_j A[j,c] n_j  - T_c |          controlled 1-way cells
    d_c  <= tol * T_c + M * miss_c               miss indicator
    g_s  >= | sum_j B[j,s] n_j  - U_s |          soft targets (below)
    P(1-e) <= sum_j s_j n_j <= P(1+e)            person-total anchor
    0 <= n_j <= R_j  integer,  d_c,g_s >= 0,  miss_c in {0,1}

The soft block B (J x S), U (S,), v (S,) carries TWO kinds of term, both the
same linear deviation shape, and both are what make the fit generalise to
out-of-sample tables (the thing free replication otherwise overfits away):
  * joint cells B[j,(a=i,b=j)] = members of household j with a=i AND b=j, with
    target U from a census 2-way table where available, else from the pool's own
    joint share x the zone total (entropy-style "stay close to the real joint");
  * anchor cells for attributes NOT in the controls (e.g. bedrd, hhcd, tenlld),
    B[j,(h=v)] = 1 if household j has h=v (household level) or member-count
    (person level), with target U = pool share x zone total.

Capping rep_max and adding these soft terms are the two generalisation levers:
free counts match the controlled margins perfectly but drift on everything else;
the soft block pins the uncontrolled joint to the real data.

Run `python zone_counts_mip.py` for the self-test.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp


def solve_zone_counts(A, T, sizes, *, weights=None, parsimony=None,
                      rep_max=60, tol=0.05, person_tol=0.05, person_target=None,
                      soft=None, L_miss=1e6, L_dev=1.0, L_joint=1.0, L_pars=1e-3,
                      time_limit=60.0, big_m=None):
    """Integer replication counts for one zone.

    Parameters
    ----------
    A : (J, C) array -- candidate j's contribution to controlled cell c.
    T : (C,) array   -- census target per controlled cell.
    sizes : (J,) array -- persons per candidate household.
    weights : (C,) cell weights for the deviation term (default 1/max(T,1)).
    parsimony : (J,) per-candidate penalty (default 0).
    rep_max : int or (J,) -- max copies per candidate.
    soft : optional (B, U, v) -- B is (J, S) incidence for S soft joint/anchor
        cells, U is (S,) their targets, v is (S,) their weights. These are fit
        as pure soft deviations (no miss indicator) and are the generalisation /
        joint term. Pass None to disable.

    Returns
    -------
    dict with n (int counts), miss (0/1 per cell), dev, n_missed, persons,
    soft_dev, objective, solver.
    """
    A = np.asarray(A, dtype=float)
    T = np.asarray(T, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    J, C = A.shape
    if weights is None:
        weights = 1.0 / np.maximum(T, 1.0)
    weights = np.asarray(weights, dtype=float)
    parsimony = (np.zeros(J) if parsimony is None
                 else np.asarray(parsimony, dtype=float))
    R = (np.full(J, int(rep_max), dtype=float) if np.isscalar(rep_max)
         else np.asarray(rep_max, dtype=float))
    P = float(np.dot(sizes, R))  # loose upper scale for big-M / person hi

    if soft is not None:
        B, U, v = soft
        B = np.asarray(B, dtype=float)
        U = np.asarray(U, dtype=float)
        v = np.asarray(v, dtype=float)
        S = B.shape[1] if B.ndim == 2 and B.size else 0
    else:
        B = np.zeros((J, 0)); U = np.zeros(0); v = np.zeros(0); S = 0
    if big_m is None:
        big_m = float(T.sum() + np.abs(A).sum() * R.max() + 1.0)

    # variable layout: [n(J), d(C), miss(C), g(S)]
    nvar = J + C + C + S
    off_d, off_miss, off_g = J, J + C, J + C + C
    c_obj = np.zeros(nvar)
    c_obj[:J] = L_pars * parsimony
    c_obj[off_d:off_d + C] = L_dev * weights
    c_obj[off_miss:off_miss + C] = L_miss
    if S:
        c_obj[off_g:off_g + S] = L_joint * v

    rows, lb, ub = [], [], []

    def add(row, lo, hi):
        rows.append(row); lb.append(lo); ub.append(hi)

    # d_c >= |A n - T|  ->  (A n) - d <= T  and  -(A n) - d <= -T
    for c in range(C):
        r = np.zeros(nvar); r[:J] = A[:, c]; r[off_d + c] = -1.0
        add(r, -np.inf, T[c])
        r = np.zeros(nvar); r[:J] = -A[:, c]; r[off_d + c] = -1.0
        add(r, -np.inf, -T[c])
        # miss_c: d_c - M*miss_c <= tol*T_c
        r = np.zeros(nvar); r[off_d + c] = 1.0; r[off_miss + c] = -big_m
        add(r, -np.inf, tol * T[c])

    # g_s >= |B n - U|  (soft joint / anchor terms, no miss indicator)
    for s in range(S):
        r = np.zeros(nvar); r[:J] = B[:, s]; r[off_g + s] = -1.0
        add(r, -np.inf, U[s])
        r = np.zeros(nvar); r[:J] = -B[:, s]; r[off_g + s] = -1.0
        add(r, -np.inf, -U[s])

    # person total anchor. When a target is given, bind the realised person
    # count to [P(1-person_tol), P(1+person_tol)] so the solve cannot satisfy the
    # control cells at the wrong overall scale; otherwise leave it non-binding.
    r = np.zeros(nvar); r[:J] = sizes
    if person_target and person_target > 0:
        add(r, float(person_target) * (1.0 - person_tol),
            float(person_target) * (1.0 + person_tol))
    else:
        add(r, 0.0, P)

    A_mat = sp.csr_matrix(np.vstack(rows))
    constraints = LinearConstraint(A_mat, np.asarray(lb), np.asarray(ub))
    bounds = Bounds(
        np.concatenate([np.zeros(J), np.zeros(C), np.zeros(C), np.zeros(S)]),
        np.concatenate([R, np.full(C, np.inf), np.ones(C), np.full(S, np.inf)]))
    integrality = np.zeros(nvar, dtype=int)
    integrality[:J] = 1                       # counts are integer
    integrality[off_miss:off_miss + C] = 1    # miss is binary (g stays continuous)
    options = {"time_limit": float(time_limit)} if time_limit else {}
    res = milp(c=c_obj, integrality=integrality, bounds=bounds,
               constraints=constraints, options=options)
    if res is None or res.x is None:
        return {"solver": "failed", "n": np.zeros(J, dtype=int)}
    x = res.x
    n = np.rint(x[:J]).astype(int)
    d = x[off_d:off_d + C]
    miss = np.rint(x[off_miss:off_miss + C]).astype(int)
    g = x[off_g:off_g + S] if S else np.zeros(0)
    return {
        "solver": "optimal" if getattr(res, "success", False) else "incumbent",
        "n": n, "dev": d, "miss": miss,
        "n_missed": int(miss.sum()),
        "persons": int(np.dot(sizes, n)),
        "soft_dev": float(v.dot(g)) if S else 0.0,
        "objective": float(getattr(res, "fun", float("nan"))),
    }


def person_counts_from(n, candidate_members):
    """Expand integer counts into the realised zone population (list of member
    rows), replicating each candidate household n_j times with fresh ids."""
    out, pid = [], 0
    for j, cnt in enumerate(n):
        for _ in range(int(cnt)):
            for member in candidate_members[j]:
                row = dict(member); row["_hh"] = pid; out.append(row)
            pid += 1
    return out


# ---------------------------------------------------------------------------
# Self-test: replication must close a large rare cell that a 0/1 fitter cannot.
# ---------------------------------------------------------------------------
def _self_test():
    # Cells: [rare, common_a, common_b]. Targets: rare=60, common_a=90, cb=120.
    # Clean case: the rare carrier (type 0) is a lone-person household carrying
    # ONLY the rare cell, so replication can close it with no side effects.
    A = np.array([
        [1.0, 0.0, 0.0],   # carrier: 1 rare member, nothing else
        [0.0, 3.0, 0.0],   # common_a household (size 3)
        [0.0, 0.0, 2.0],   # common_b household (size 2)
    ])
    T = np.array([60.0, 90.0, 120.0])
    sizes = np.array([1.0, 3.0, 2.0])

    r1 = solve_zone_counts(A, T, sizes, rep_max=1, time_limit=20)    # no repl.
    r2 = solve_zone_counts(A, T, sizes, rep_max=100, time_limit=20)  # repl.
    print("no-replication (R=1): ", {k: r1[k] for k in ("n", "n_missed", "dev")})
    print("replication   (R=100):", {k: r2[k] for k in ("n", "n_missed", "dev")})
    rare1, rare2 = float(A[:, 0] @ r1["n"]), float(A[:, 0] @ r2["n"])
    print(f"rare-cell realised: R=1 -> {rare1:.0f}/60, R=100 -> {rare2:.0f}/60")
    assert r1["n_missed"] >= 1 and rare1 <= 1, "0/1 selection cannot fill rare"
    assert r2["n_missed"] == 0 and abs(rare2 - 60) <= 1, "replication closes it"
    print("  -> PASS: replication closes a rare cell 0/1 selection cannot.")

    # Coupled case: the only rare carrier ALSO carries common_a, so filling rare
    # overshoots common_a. The one-MIP must trade them off (not flood rare) --
    # the behaviour the composition arm lacked.
    A2 = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
    r3 = solve_zone_counts(A2, T, np.array([2.0, 3.0, 2.0]), rep_max=100,
                           time_limit=20)
    rare3 = float(A2[:, 0] @ r3["n"]); ca3 = float(A2[:, 1] @ r3["n"])
    print(f"coupled case: rare={rare3:.0f}/60, common_a={ca3:.0f}/90, "
          f"missed={r3['n_missed']}  (trades rare-miss to keep common_a exact)")
    assert abs(ca3 - 90) <= 3, "must not overshoot the coupled common cell"
    print("  -> PASS: coupling-aware trade-off, no flooding.")
    print("SELF-TEST PASSED.")


if __name__ == "__main__":
    _self_test()

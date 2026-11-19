"""Donor-count LP: choose how many times each census donor is used.

This is the replication-aware version of the donor-vector balance pass.  For
each zone it solves

    min  sum_c w_c d_c
    s.t. |sum_h a_hc n_h - T_c| <= d_c
         sum_h s_h n_h <= budget
         0 <= n_h <= rep_max

and reports the fractional lower bound and a rounded integer pool.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.optimize import linprog

sys.path.insert(0, r"Z:\Paper_Plan\constraint_first_synth")

import mip_fit as mipf  # noqa: E402
import mip_pool_joint as mj  # noqa: E402

CP = r"Z:\Paper_Plan\runs\2026-11-11_canonical\pools\feasible_pool.csv"
CENSUS = r"Z:\Population_Synth\LINKED_DATA_clean.csv"
ZONES = ["112011236", "112011237", "112011238"]
BUDGET = 0.10
REP_MAX = 3


def tvd_tables(counts_vec, base_vec, cells, T):
    vals = []
    seen = {}
    for i, (attr, _c) in enumerate(cells):
        seen.setdefault(attr, []).append(i)
    for attr, idx in seen.items():
        p = np.array([counts_vec[i] for i in idx], float)
        t = np.array([T[i] for i in idx], float)
        p = p / p.sum() if p.sum() > 0 else p
        t = t / t.sum() if t.sum() > 0 else t
        vals.append(0.5 * np.abs(p - t).sum())
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    ind, hh = mipf.load_control_tables()
    ind_keys = sorted({k for t in ind.values() for k in t.get("keys", [])})
    hh_keys = sorted({k for t in hh.values() for k in t.get("keys", [])})
    sa2_to_reg, _ = mj._load_sa2_to_regucp()

    cp = pd.read_csv(CP, low_memory=False)
    cp.columns = [str(c).lower() for c in cp.columns]
    census = pd.read_csv(CENSUS, low_memory=False)
    census.columns = [str(c).lower() for c in census.columns]

    for zone in ZONES:
        targets = {}
        targets.update(mipf.zone_targets(hh, zone))
        targets.update(mipf.zone_targets(ind, zone))
        if not targets:
            continue

        cp_cand, _ = mipf.zone_candidate_pool(cp, zone, sa2_to_reg)
        Cb, _idb, cells, _sb = mj._candidate_matrix(
            cp_cand, targets, ind_keys, hh_keys)
        base = np.asarray(Cb.sum(axis=1)).ravel()
        T = np.array([targets[c] for c in cells], dtype=float)

        cand, _ = mipf.zone_candidate_pool(census, zone, sa2_to_reg)
        C, ids, _cells2, sizes = mj._candidate_matrix(
            cand, targets, ind_keys, hh_keys)
        A = C.T.tocsr()                     # donors x cells
        n_don = A.shape[0]
        n_cell = A.shape[1]

        counts = {}
        base_total = {}
        for i, (attr, _c) in enumerate(cells):
            counts[attr] = counts.get(attr, 0) + 1
            base_total[attr] = base_total.get(attr, 0.0) + float(base[i])
        w = np.array([1.0 / counts[attr] for attr, _c in cells], float)
        w /= max(w.max(), 1e-12)

        # Work in share units and include the base pool, so the LP minimises
        # the final pool's deviation rather than the added set alone.
        scale = np.array([base_total[attr] if base_total[attr] > 0 else 1.0
                          for attr, _c in cells])
        A_share = A.dot(sp.diags(1.0 / scale)).tocsr()
        rhs_plus = (T - base) / scale
        rhs_minus = (base - T) / scale

        zone_persons = sum(float(v) for (a, _c), v in targets.items()
                           if str(a).lower() == "sexp") or float(sizes.sum())
        budget = BUDGET * zone_persons

        # Constraints:  A n - d <= (T-base)/N ;  -A n - d <= (base-T)/N
        I = sp.eye(n_cell, format="csr")
        A_ub = sp.vstack([
            sp.hstack([A_share.T, -I]),
            sp.hstack([-A_share.T, -I]),
            sp.hstack([sp.csr_matrix(sizes[None, :]),
                       sp.csr_matrix((1, n_cell))]),
        ], format="csr")
        b_ub = np.concatenate([rhs_plus, rhs_minus, [budget]])

        c_obj = np.concatenate([np.zeros(n_don), w])
        bounds = [(0.0, REP_MAX)] * n_don + [(0.0, None)] * n_cell
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method="highs")
        if not res.success:
            print(zone, "LP failed:", res.message, flush=True)
            continue

        n = res.x[:n_don]
        add = np.asarray(A.T.dot(n)).ravel()
        final = base + add

        # Simple integer rounding: floor, then add one replica to the largest
        # fractional remainders until the person budget is exhausted.
        n_int = np.floor(n).astype(int)
        used = float(np.dot(sizes, n_int))
        order = np.argsort(-(n - n_int))
        for j in order:
            if n_int[j] >= REP_MAX:
                continue
            if used + float(sizes[j]) > budget:
                continue
            n_int[j] += 1
            used += float(sizes[j])
            if used >= budget * 0.999:
                break
        add_int = np.asarray(A.T.dot(n_int)).ravel()
        final_int = base + add_int

        print(f"\nzone {zone}: donors={n_don} cells={n_cell} "
              f"budget={budget:.0f} persons", flush=True)
        print(f"  TVD base={tvd_tables(base, base, cells, T):.4f} "
              f"fractional={tvd_tables(final, base, cells, T):.4f} "
              f"rounded={tvd_tables(final_int, base, cells, T):.4f}", flush=True)
        print(f"  donors used frac={int((n > 1e-8).sum())} "
              f"int={int((n_int > 0).sum())}; "
              f"max replica frac={n.max():.2f} int={n_int.max()}", flush=True)


if __name__ == "__main__":
    main()

"""Iterative donor-balance expansion with replication.

At each batch the script recomputes the final-pool gap, scores every donor for
one additional replica, and appends the positive-gain donors until the batch
budget is reached.  A donor may be used up to ``--rep-max`` times.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mip_fit as mipf  # noqa: E402
import mip_pool_joint as mj  # noqa: E402

CP = r"Z:\Paper_Plan\runs\2026-11-11_canonical\pools\feasible_pool.csv"
CENSUS = r"Z:\Population_Synth\LINKED_DATA_clean.csv"
ZONES = r"Z:\Paper_Plan\synth_pop\pbs\zones_367.txt"
OUT = r"Z:\Paper_Plan\runs\2026-11-13_v8\pools\balance_donors_iter.csv"


def select_zone(cp, census, zone, ind, hh, ind_keys, hh_keys, sa2_to_reg,
                budget_fraction=1.0, batch_fraction=0.05, rep_max=3,
                max_donors=5000):
    targets = {}
    targets.update(mipf.zone_targets(hh, zone))
    targets.update(mipf.zone_targets(ind, zone))
    if not targets:
        return None

    cp_cand, _ = mipf.zone_candidate_pool(cp, zone, sa2_to_reg)
    Cb, _idb, cells, _sb = mj._candidate_matrix(
        cp_cand, targets, ind_keys, hh_keys)
    base = np.asarray(Cb.sum(axis=1)).ravel()
    T = np.array([targets[c] for c in cells], dtype=float)

    cand, _ = mipf.zone_candidate_pool(census, zone, sa2_to_reg)
    C, ids, cells2, sizes = mj._candidate_matrix(
        cand, targets, ind_keys, hh_keys)
    if list(cells2) != list(cells):
        raise RuntimeError(f"cell order mismatch in zone {zone}")

    A = C.T.tocsr()
    n_don = A.shape[0]
    counts = {}
    for attr, _c in cells:
        counts[attr] = counts.get(attr, 0) + 1
    w = np.array([1.0 / counts[attr] for attr, _c in cells], float)
    w /= max(w.max(), 1e-12)

    table_idx = {}
    for i, (attr, _c) in enumerate(cells):
        table_idx.setdefault(attr, []).append(i)

    def table_tvds(vec):
        out = {}
        for attr, idx in table_idx.items():
            p = np.array([vec[i] for i in idx], float)
            t = np.array([T[i] for i in idx], float)
            p = p / p.sum() if p.sum() > 0 else p
            t = t / t.sum() if t.sum() > 0 else t
            out[attr] = 0.5 * np.abs(p - t).sum()
        return out

    initial_tvd = table_tvds(base)

    zone_persons = sum(float(v) for (a, _c), v in targets.items()
                       if str(a).lower() == "sexp") or float(sizes.sum())
    total_budget = budget_fraction * zone_persons
    batch_budget = max(1.0, batch_fraction * zone_persons)

    n = np.zeros(n_don, dtype=int)
    cur = base.astype(float).copy()
    used = 0.0
    while used < total_budget - 1e-9:
        # final-pool share gap, recomputed after each batch
        cur_total, target_total = {}, {}
        for i, (attr, _c) in enumerate(cells):
            cur_total[attr] = cur_total.get(attr, 0.0) + cur[i]
            target_total[attr] = target_total.get(attr, 0.0) + T[i]
        gap = np.array([
            (T[i] / target_total[attr] if target_total[attr] > 0 else 0.0)
            - (cur[i] / cur_total[attr] if cur_total[attr] > 0 else 0.0)
            for i, (attr, _c) in enumerate(cells)])
        scale = np.array([cur_total[attr] if cur_total[attr] > 0 else 1.0
                          for attr, _c in cells])

        avail = np.flatnonzero(n < rep_max)
        if not len(avail):
            break
        sub = A[avail].toarray() / scale
        score = np.sum(
            w * (np.abs(gap)[None, :] - np.abs(gap[None, :] - sub)), axis=1)
        order = avail[np.argsort(-score)]
        batch_used = 0.0
        chosen = []
        for j in order:
            if used + batch_used >= total_budget - 1e-9:
                break
            if score[np.searchsorted(avail, j)] <= 0:
                break
            if n[j] >= rep_max:
                continue
            s = float(sizes[j])
            if used + batch_used + s > total_budget:
                continue
            if batch_used + s > batch_budget and chosen:
                continue
            chosen.append(int(j))
            batch_used += s
            if len(chosen) >= max_donors:
                break
        if not chosen:
            break
        for j in chosen:
            n[j] += 1
        add = np.asarray(A[np.array(chosen)].sum(axis=0)).ravel()
        cur = cur + add
        used += batch_used

    sel = np.flatnonzero(n > 0)
    if not len(sel):
        return None
    ids_arr = np.asarray(ids)
    pieces = []
    for j in sel:
        g = cand[cand["abshid"] == ids_arr[j]]
        if not len(g):
            continue
        for _ in range(int(n[j])):
            pieces.append(g)
    if not pieces:
        return None
    block = pd.concat(pieces, ignore_index=True)
    block["_source"] = "balance_donor"
    block["_target_sa2"] = str(zone)
    return block, int(len(sel)), used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones-file", default=ZONES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--budget-fraction", type=float, default=1.0)
    ap.add_argument("--batch-fraction", type=float, default=0.05)
    ap.add_argument("--rep-max", type=int, default=3)
    ap.add_argument("--max-donors", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ind, hh = mipf.load_control_tables()
    ind_keys = sorted({k for t in ind.values() for k in t.get("keys", [])})
    hh_keys = sorted({k for t in hh.values() for k in t.get("keys", [])})
    sa2_to_reg, _ = mj._load_sa2_to_regucp()

    cp = pd.read_csv(CP, low_memory=False)
    cp.columns = [str(c).lower() for c in cp.columns]
    census = pd.read_csv(CENSUS, low_memory=False)
    census.columns = [str(c).lower() for c in census.columns]
    # The census pool uses a shifted QALFP code domain above code 12:
    # 12->13, 13->14, 14->15 in the control-table coding.
    if "qalfp" in census.columns:
        census["qalfp"] = pd.to_numeric(census["qalfp"], errors="coerce").replace(
            {12: 13, 13: 14, 14: 15})

    with open(a.zones_file, encoding="utf-8") as fh:
        zones = [ln.strip() for ln in fh if ln.strip()]
    if a.limit:
        zones = zones[:a.limit]

    blocks, next_id = [], 3_000_000_000
    for i, zone in enumerate(zones, 1):
        got = select_zone(cp, census, zone, ind, hh, ind_keys, hh_keys,
                          sa2_to_reg, a.budget_fraction, a.batch_fraction,
                          a.rep_max, a.max_donors)
        if got is None:
            print(f"  {i}/{len(zones)} zone {zone}: no donor", flush=True)
            continue
        block, n_don, persons = got
        block = block.copy()
        id_map = {old: next_id + k for k, old in enumerate(
            block["abshid"].drop_duplicates().tolist())}
        next_id += len(id_map)
        block["abshid"] = block["abshid"].map(id_map)
        blocks.append(block)
        print(f"  {i}/{len(zones)} zone {zone}: {n_don} donors, "
              f"{persons:.0f} persons", flush=True)

    out = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} rows, {len(blocks)} zones")


if __name__ == "__main__":
    main()

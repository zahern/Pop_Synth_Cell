"""Balance-driven donor expansion.

For each zone, score whole census donor households by how much they reduce the
weighted absolute gap between the current pool and the census marginals, then
append the best positive-gain donors up to a person budget (10% of the zone by
default).  Unlike per-cell carrier construction, a donor is evaluated on its
full attribute vector, so one household can fix several underrepresented cells
without being forced through a single-cell menu.
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
OUT = r"Z:\Paper_Plan\runs\2026-11-13_v8\pools\balance_donors.csv"


def select_zone(cp, census, zone, ind, hh, ind_keys, hh_keys, sa2_to_reg,
                budget_fraction=0.10, max_donors=5000):
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
    n = A.shape[0]
    base_total, target_total, counts = {}, {}, {}
    for i, (attr, _c) in enumerate(cells):
        base_total[attr] = base_total.get(attr, 0.0) + base[i]
        target_total[attr] = target_total.get(attr, 0.0) + T[i]
        counts[attr] = counts.get(attr, 0) + 1
    gap = np.array([
        (T[i] / target_total[attr] if target_total[attr] > 0 else 0.0)
        - (base[i] / base_total[attr] if base_total[attr] > 0 else 0.0)
        for i, (attr, _c) in enumerate(cells)])
    w = np.array([1.0 / (counts[attr] * max(T[i], 1.0))
                  for i, (attr, _c) in enumerate(cells)])
    w /= max(w.max(), 1e-12)
    scale = np.array([base_total[attr] if base_total[attr] > 0 else 1.0
                      for attr, _c in cells])

    score = np.zeros(n, dtype=float)
    for start in range(0, n, 4000):
        block = A[start:start + 4000].toarray() / scale
        score[start:start + 4000] = np.sum(
            w * (np.abs(gap)[None, :] - np.abs(gap[None, :] - block)), axis=1)

    zone_persons = sum(float(v) for (a, _c), v in targets.items()
                       if str(a).lower() == "sexp") or float(np.sum(sizes))
    budget = budget_fraction * zone_persons
    order = np.argsort(-score)
    chosen, used = [], 0.0
    for j in order:
        if len(chosen) >= max_donors or score[j] <= 0:
            break
        s = float(sizes[j])
        if used + s > budget:
            continue
        chosen.append(int(j))
        used += s
    if not chosen:
        return None

    # Build the donor block from the census rows for the chosen household ids.
    ids_arr = np.asarray(ids)
    keep_ids = set(ids_arr[chosen].tolist())
    block = cand[cand["abshid"].isin(keep_ids)].copy()
    block["_source"] = "balance_donor"
    block["_target_sa2"] = str(zone)
    return block, len(chosen), used, float(np.mean(np.abs(gap)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones-file", default=ZONES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--budget-fraction", type=float, default=0.10)
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

    with open(a.zones_file, encoding="utf-8") as fh:
        zones = [ln.strip() for ln in fh if ln.strip()]
    if a.limit:
        zones = zones[:a.limit]

    blocks, next_id = [], 2_000_000_000
    for i, zone in enumerate(zones, 1):
        got = select_zone(cp, census, zone, ind, hh, ind_keys, hh_keys,
                          sa2_to_reg, a.budget_fraction, a.max_donors)
        if got is None:
            print(f"  {i}/{len(zones)} zone {zone}: no donor", flush=True)
            continue
        block, n_don, persons, gap = got
        block = block.copy()
        # fresh household ids, one per census household
        id_map = {old: next_id + k for k, old in enumerate(
            block["abshid"].drop_duplicates().tolist())}
        next_id += len(id_map)
        block["abshid"] = block["abshid"].map(id_map)
        blocks.append(block)
        print(f"  {i}/{len(zones)} zone {zone}: {n_don} donors, "
              f"{persons:.0f} persons, mean|gap| {gap:.4f}", flush=True)

    out = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} rows, {len(blocks)} zones")


if __name__ == "__main__":
    main()

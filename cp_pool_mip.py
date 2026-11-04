"""cp_pool_mip.py -- "CP-pool + MIP-fit" arm (trial).

A drop-in alternative to cp_pool_sa.py / ipu_fit.py that fits each zone with ONE
integer-count program (zone_counts_mip.solve_zone_counts) instead of simulated
annealing or IPU. The optimiser chooses an integer replication count n_j for
every candidate household in the zone's slice so that, over all active control
tables, the number of unmet cells is minimised first and the joint-marginal
deviation second. Because it replicates, a single rare carrier can stand for the
many realised copies a cell needs -- the thing the SA/IPU fitters (selection
without replacement) cannot do.

It reuses mip_fit.build_constraint_matrix for the candidate matrix and the SA
engine's calculate_total_rssz_table for scoring, so the RSSZ it writes is
directly comparable to the base_sa / base_ipu arms and is read by
score_paper_arms.py unchanged (same cp_pool_rssz_summary_chunk<ID>.csv layout).

Usage (one or a few zones; trial on the HPC)
--------------------------------------------
  python cp_pool_mip.py --pool outputs/feasible_pool.csv \
      --zones 311051324 311051325 --rep-max 40 \
      --results-dir outputs/cp_mip_results/chunk_0

Env: SA_CONTROL_DIR (control basis), CP_POOL_CLEAN_CACHE (shared cleaned pool),
     CHUNK_ID (summary suffix). Mirrors cp_pool_sa.py.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config
import mip_fit as mipf
import zone_counts_mip as zc
from cp_pool_sa import assign_infilled_households
from helper_functions import Helper
from sa_only_baseline import (load_control_tables, _load_sa2_to_regucp,
                              _normalize_zone_values, normalize_counts)
from SA_2_level import (build_control_dicts, calculate_total_marginals,
                        calculate_total_rssz_table)


def _build_soft_terms(cand, hh_ids, anchor_attrs, joint_pairs, h_target,
                      min_share=0.01, max_cells=250):
    """Pool-anchored soft joint/anchor incidence for generalisation.

    Builds B (J x S), U (S,), v (S,) where each column is an uncontrolled
    household attribute value (anchor) or an attribute-pair value (joint), with
    target = pool_share x household_target. The shares come from the POOL's own
    records (seed information available for every row), NOT from any held-out
    census control, so this is IPF/entropy-style regularisation, not leakage.
    Household-level attributes only (one value per dwelling); person-level
    anchors are skipped here. Returns None if nothing usable.
    """
    first = cand.groupby("abshid").first().reindex(hh_ids)
    n_hh = max(len(hh_ids), 1)
    cols, U, labels = [], [], []

    def _add_attr_value(mask, share, label):
        if share < min_share or len(cols) >= max_cells:
            return
        cols.append(mask.astype(float).to_numpy())
        U.append(share * float(h_target))
        labels.append(label)

    for h in anchor_attrs:
        if h not in first.columns:
            continue
        vals = first[h]
        for v, cnt in vals.value_counts(dropna=True).items():
            _add_attr_value(vals == v, cnt / n_hh, f"{h}={v}")

    for a, b in joint_pairs:
        if a not in first.columns or b not in first.columns:
            continue
        combo = first[[a, b]].dropna()
        vc = combo.groupby([a, b]).size()
        for (i, j), cnt in vc.items():
            _add_attr_value((first[a] == i) & (first[b] == j),
                            cnt / n_hh, f"{a}={i}&{b}={j}")

    if not cols:
        return None
    B = np.column_stack(cols)
    U = np.asarray(U, dtype=float)
    v = 1.0 / np.maximum(U, 1.0)
    return B, U, v, labels


def run_mip_zone(pop, ind_tables, hh_tables, zone, *, rep_max=40,
                 max_candidates=5000, time_limit=120.0, lambda_dev=1.0,
                 lambda_joint=1.0, anchor_attrs=(), joint_pairs=(), seed=0):
    """Fit one zone with the integer-count MIP. Returns (population_df, row)."""
    available = set(pop.columns)
    usable_ind = {k: v for k, v in ind_tables.items()
                  if set(v.get("keys", [])) <= available}
    usable_hh = {k: v for k, v in hh_tables.items()
                 if set(v.get("keys", [])) <= available}
    # Same per-zone dwelling-universe reconciliation the SA/IPU arms apply, so
    # the control dicts (and therefore the scored RSSZ) match those arms.
    usable_ind, usable_hh = normalize_counts(usable_ind, usable_hh, zones=[zone])
    ind_dicts = build_control_dicts(usable_ind, zone)
    hh_dicts = build_control_dicts(usable_hh, zone)
    if sum(1 for d in ind_dicts.values() if not d) >= 2:
        return None, None
    total_ind, total_hh = calculate_total_marginals(ind_dicts, hh_dicts)
    if total_ind <= 0:
        return None, None

    cand = pop.copy()
    cand.columns = [str(c).lower() for c in cand.columns]
    if "abshid" not in cand.columns:
        return None, None

    # Deduplicate identical households into distinct "types". A zone's candidate
    # universe is its whole SA4 slice (tens of thousands of dwellings), but the
    # CP infill produces many byte-identical households; collapsing them loses
    # nothing, because a type with m real copies -- each usable up to rep_max
    # times -- is exactly ONE integer variable bounded by m*rep_max. So J counts
    # DISTINCT household types, not raw rows, and every candidate the zone has is
    # used rather than truncated at an arbitrary cap. The m*rep_max bound also
    # keeps the generalisation semantics: still <= rep_max copies per real
    # household, just expressed per type.
    id_cols = {"abshid", "abspid", "persid", "person_id", "_source",
               "_target_sa2", "reindexed_slot", "weight", "sa2", "regucp",
               "_hh", "index", "level_0"}
    sig_cols = [c for c in cand.columns if c not in id_cols]
    rowsig = cand[sig_cols].astype(str).agg("|".join, axis=1)
    # Household signature = order-independent join of member row-signatures.
    # Sort rows within each dwelling first, then a C-level join, so this is a
    # single sort + groupby rather than a per-group Python hash (fast on the
    # tens-of-thousands-of-dwelling SA4 slice).
    tmp = (pd.DataFrame({"abshid": cand["abshid"].to_numpy(), "s": rowsig})
           .sort_values(["abshid", "s"]))
    hh_sig = tmp.groupby("abshid")["s"].agg("||".join)
    sig_df = hh_sig.rename("sig").reset_index()
    rep_abshid = sig_df.groupby("sig")["abshid"].first()
    mult = sig_df.groupby("sig")["abshid"].size()
    rep_to_mult = dict(zip(rep_abshid.to_numpy(), mult.to_numpy()))
    reps = set(rep_abshid.to_numpy())
    n_raw = int(cand["abshid"].nunique())

    # Optional post-dedup safety cap (rarely binds). Keep the highest-
    # multiplicity types (most capacity) first, and warn so a bind is never
    # silent -- never a blind truncation below the zone's real diversity.
    capped = False
    if max_candidates and len(reps) > int(max_candidates):
        reps = set(sorted(reps, key=lambda h: -rep_to_mult.get(h, 1)
                          )[:int(max_candidates)])
        capped = True
        print(f"  [warn] zone {zone}: {len(rep_to_mult)} distinct types capped "
              f"to {max_candidates}; raise --max-candidates to use all.")
    cand = cand[cand["abshid"].isin(reps)]

    ind_keys = sorted({k for t in usable_ind.values() for k in t.get("keys", [])})
    hh_keys = sorted({k for t in usable_hh.values() for k in t.get("keys", [])})
    targets = {}
    targets.update(mipf.zone_targets(usable_hh, zone))
    targets.update(mipf.zone_targets(usable_ind, zone))
    if not targets:
        return None, None

    C, hh_ids, cells = mipf.build_constraint_matrix(
        cand, targets, ind_keys, hh_keys)
    if C.shape[0] == 0 or C.shape[1] == 0:
        return None, None
    A = C.T.toarray().astype(float)                      # (J households, C cells)
    T = np.array([targets[c] for c in cells], dtype=float)
    sizes = (cand.groupby("abshid").size().reindex(hh_ids)
             .fillna(1).to_numpy(dtype=float))

    # Per-type replication bound: rep_max copies of each of the type's m real
    # households. Expressed per type, this is R_j = rep_max * multiplicity_j.
    R = np.array([int(rep_max) * int(rep_to_mult.get(h, 1)) for h in hh_ids],
                 dtype=float)

    # Pool-anchored soft joint/anchor terms (generalisation regulariser).
    h_target = total_hh if total_hh and total_hh > 0 else max(1, len(hh_ids))
    soft = None
    n_soft = 0
    if anchor_attrs or joint_pairs:
        built = _build_soft_terms(cand, hh_ids, list(anchor_attrs),
                                  list(joint_pairs), h_target)
        if built is not None:
            B, U, vv, _labels = built
            soft = (B, U, vv)
            n_soft = B.shape[1]

    res = zc.solve_zone_counts(
        A, T, sizes, rep_max=R, person_target=total_ind,
        soft=soft, L_dev=lambda_dev, L_joint=lambda_joint,
        time_limit=time_limit)
    n = np.asarray(res.get("n", np.zeros(len(hh_ids))), dtype=int)

    # Realise the fitted population: replicate each selected household n_j times
    # with a fresh abshid per copy, so households stay whole and distinct.
    groups = {h: g for h, g in cand.groupby("abshid")}
    rows, next_id = [], 1
    for j, cnt in enumerate(n):
        if cnt <= 0:
            continue
        block = groups.get(hh_ids[j])
        if block is None:
            continue
        recs = block.to_dict("records")
        for _ in range(int(cnt)):
            for r in recs:
                rr = dict(r); rr["abshid"] = next_id; rows.append(rr)
            next_id += 1
    if not rows:
        return None, None

    rssz_total, per_table = calculate_total_rssz_table(rows, ind_dicts, hh_dicts)
    dominant = max(per_table, key=per_table.get) if per_table else None
    row = {
        "zone": zone,
        "objective_mode": "mip_counts",
        "solver": res.get("solver"),
        "rep_max": int(rep_max),
        "rssz_total": rssz_total,
        "n_selected_households": int((n > 0).sum()),
        "n_realised_households": int(next_id - 1),
        "n_realised_persons": int(len(rows)),
        "person_target": int(total_ind),
        "n_cells_missed": int(res.get("n_missed", -1)),
        "n_soft_terms": int(n_soft),
        "soft_dev": float(res.get("soft_dev", 0.0)),
        "n_candidates": int(len(hh_ids)),
        "n_raw_households": int(n_raw),
        "n_distinct_types": int(len(hh_ids)),
        "candidates_capped": bool(capped),
        "dominant_table": dominant,
        "dominant_table_rssz": (float(per_table[dominant]) if dominant else None),
        **per_table,
    }
    return pd.DataFrame(rows), row


def _load_clean_pool(pool_path):
    """Cleaned, lower-cased pool -- reuses the CP_POOL_CLEAN_CACHE the SA arm
    writes, so a trial run next to cp_pool_sa pays no extra pool round-trip."""
    cache = os.environ.get("CP_POOL_CLEAN_CACHE", "").strip()
    if cache and os.path.isfile(cache):
        try:
            c = pd.read_pickle(cache)
        except Exception:
            c = None
        if (isinstance(c, dict) and c.get("pool") == os.path.abspath(pool_path)
                and c.get("size") == os.path.getsize(pool_path) and "data" in c):
            print(f"  cleaned-pool cache hit: {cache}")
            return c["data"]
    tmp_dir = os.environ.get("TMPDIR") or os.path.dirname(pool_path) or "."
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f"_mip_clean_{os.getpid()}.csv")
    assign_infilled_households(pd.read_csv(pool_path)).to_csv(tmp, index=False)
    microdata = Helper().clean_my_datagan(tmp)
    try:
        os.remove(tmp)
    except OSError:
        pass
    if cache:
        try:
            pd.to_pickle({"pool": os.path.abspath(pool_path),
                          "size": os.path.getsize(pool_path),
                          "data": microdata}, cache)
        except Exception:
            pass
    return microdata


def main():
    p = argparse.ArgumentParser(description="CP-pool + integer-count MIP fit")
    p.add_argument("--pool", required=True)
    p.add_argument("--zones", nargs="*", type=str, required=True)
    p.add_argument("--rep-max", type=int, default=40)
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Optional safety cap on DISTINCT household types after "
                        "dedup. Default: no cap -- every candidate type the zone "
                        "has is used. Raise rather than lower if a zone warns.")
    p.add_argument("--time-limit", type=float, default=120.0)
    p.add_argument("--lambda-dev", type=float, default=1.0)
    p.add_argument("--lambda-joint", type=float, default=1.0,
                   help="weight on the soft joint/anchor (generalisation) block")
    p.add_argument("--anchor-attrs", nargs="*",
                   default=["bedrd", "hhcd", "tenlld"],
                   help="uncontrolled household attrs to anchor to pool shares "
                        "(generalisation); pass none to disable")
    p.add_argument("--joint-pairs", nargs="*",
                   default=["nprd:hhcd", "nprd:bedrd"],
                   help="attribute pairs 'a:b' to fit jointly against pool "
                        "shares (census targets not required)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", required=True)
    a = p.parse_args()
    pairs = tuple(tuple(s.split(":", 1)) for s in a.joint_pairs
                  if ":" in s)

    results_dir = os.path.abspath(a.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    microdata = _load_clean_pool(a.pool)
    print(f"  Pool records: {len(microdata):,}")
    ind_tables, hh_tables = load_control_tables()
    sa2_to_regucp, _ = _load_sa2_to_regucp()
    has_sa2 = "sa2" in microdata.columns
    has_regucp = "regucp" in microdata.columns

    rows = []
    for zone in [str(z) for z in a.zones]:
        t0 = time.time()
        zp = pd.DataFrame()
        if has_sa2:
            zp = microdata.loc[_normalize_zone_values(microdata["sa2"]) == zone]
        if zp.empty and has_regucp:
            r = sa2_to_regucp.get(zone)
            if r:
                zp = microdata.loc[
                    _normalize_zone_values(microdata["regucp"]) == r]
        if zp.empty:
            print(f"Zone {zone}: no candidates; skipped")
            continue
        try:
            pop_df, row = run_mip_zone(
                zp, ind_tables, hh_tables, zone, rep_max=a.rep_max,
                max_candidates=a.max_candidates, time_limit=a.time_limit,
                lambda_dev=a.lambda_dev, lambda_joint=a.lambda_joint,
                anchor_attrs=tuple(a.anchor_attrs), joint_pairs=pairs,
                seed=a.seed)
        except Exception as exc:
            print(f"Zone {zone}: ERROR {exc}")
            continue
        if row is None:
            print(f"Zone {zone}: no fit")
            continue
        # Name the population the way benchmark_margins.load_cp_mip_zones expects
        # (zone_<zone>_synthetic_population.csv), so the full paper metric suite
        # (RSSZ, TVD, SRMSE, MAE%) computes on it with no extra wiring.
        pop_df.to_csv(
            os.path.join(results_dir, f"zone_{zone}_synthetic_population.csv"),
            index=False)
        rows.append(row)
        print(f"Zone {zone}: rssz_total={row['rssz_total']:.4f} "
              f"missed={row['n_cells_missed']} solver={row['solver']} "
              f"persons={row['n_realised_persons']}/{row['person_target']} "
              f"({time.time()-t0:.0f}s)")

    if not rows:
        print("No zones fitted."); sys.exit(1)
    summary = pd.DataFrame(rows)
    chunk = os.environ.get("CHUNK_ID", "").strip()
    name = (f"cp_pool_rssz_summary_chunk{chunk}.csv" if chunk
            else "cp_pool_rssz_summary.csv")
    out = os.path.join(results_dir, name)
    summary.to_csv(out, index=False)
    v = summary["rssz_total"]
    print(f"\nCP-pool + MIP: {len(v)} zones  mean {v.mean():.4f}  "
          f"median {v.median():.4f}\nSaved: {out}")


if __name__ == "__main__":
    main()

"""
cp_pool_sa.py — "CP-pool + SA" baseline
=========================================
SA applied to the constraint-programming feasible pool (build_feasible_pool_cp)
instead of the GAN-generated pool, so the two *candidate-pool* methods are
directly comparable.  Mirrors sa_only_baseline.py (SA on raw microdata) but
seeds from the CP pool built over the census.

Pipeline position: this is the "CP-pool + SA" row in the benchmark table,
alongside SA-Only, IPU-Only, IPU-GAN, and DATGAN-SA.

Usage
-----
  python cp_pool_sa.py --census LINKED_DATA_national.csv \
         --control-attrs sexp agep mstp heap occp incp lfsp empp hrsp qallp qalfp rlhp mtwp nprd vehrd mrerd rntrd strd hind bedrd hhcd \
         [--zones 301011001 ...] [--max-iter 1000]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config

from sa_only_baseline import (
    load_control_tables, run_sa_zone, _load_sa2_to_regucp, _normalize_zone_values,
)
from helper_functions import Helper
import build_feasible_pool_cp as cp

RESULTS_DIR = config.CP_POOL_RESULTS_DIR
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Household-consistent attributes (shared by all members of a dwelling) used to
# group infilled individuals into new, internally-consistent households.
_HH_SIGNATURE = ["NPRD", "VEHRD", "MRERD", "RNTRD", "STRD", "HIND", "BEDRD",
                 "HHCD", "TENLLD", "DWTD", "FMCF", "CPRF", "CACF", "SSCF"]
_RLHP_PRIORITY = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 7, 11: 8}


def assign_infilled_households(pool, hh_sig_cols=None):
    """Group infilled individuals into new households of their stated size (NPRD).

    Census rows keep their original ABSHID.  Infilled rows (marker `_source ==
    'infilled'`) are grouped by household-attribute signature and chunked into
    households of size NPRD, so each new household is internally consistent.
    """
    m = pool.get("_source") == "infilled"
    if "_source" not in pool.columns or not m.any():
        return pool

    pool = pool.copy()
    base = int(pool.loc[~m, "ABSHID"].max()) + 1 if (~m).any() and "ABSHID" in pool else 1

    sig_cols = [c for c in (hh_sig_cols or _HH_SIGNATURE)
                if c in pool.columns and c != "NPRD"]
    inf = pool[m]
    size = inf["NPRD"].clip(lower=1).astype(int) if "NPRD" in inf else pd.Series(1, index=inf.index)
    sig = inf[sig_cols].astype(str).agg("|".join, axis=1) if sig_cols else pd.Series("all", index=inf.index)

    records = inf.copy()
    records["_sig"] = sig.values
    records["_nprd"] = size.values
    records["_row_order"] = np.arange(len(records), dtype=np.int64)
    if "RLHP" in records.columns:
        records["_rlhp_rank"] = pd.to_numeric(
            records["RLHP"], errors="coerce").map(_RLHP_PRIORITY).fillna(99)
    else:
        records["_rlhp_rank"] = 99

    assigned = {}
    next_id = base
    for signature, group in records.groupby("_sig", sort=True):
        remaining = group.sort_values(["_rlhp_rank", "_row_order"]).copy()
        ordinal = 0
        while not remaining.empty:
            household_size = max(1, int(remaining.iloc[0]["_nprd"]))
            chosen = []
            primary = remaining[remaining["_rlhp_rank"].isin([0, 1])]
            if not primary.empty:
                chosen.append(primary.index[0])
            for idx in remaining.index:
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) >= household_size:
                    break
            household_id = next_id + ordinal
            for idx in chosen:
                assigned[idx] = household_id
            remaining = remaining.drop(index=chosen)
            ordinal += 1
        next_id += max(ordinal, 1)

    pool.loc[list(assigned), "ABSHID"] = pd.Series(assigned, dtype="int64")
    return pool


def build_pool(census, control_attrs, min_count, target_per_cell, seed):
    """Build the CP feasible pool and assign households to infilled rows."""
    pool, report = cp.build_feasible_pool(
        census, control_attrs, min_count=min_count,
        target_per_cell=target_per_cell, infill=True, seed=seed)
    pool = assign_infilled_households(pool)
    return pool, report


def main():
    p = argparse.ArgumentParser(description="CP-pool + SA baseline")
    p.add_argument("--census", type=str, default=config.LINKED_DATA,
                   help="Fused census CSV (national for cell-completeness)")
    p.add_argument("--control-attrs", nargs="+", default=[
        "SEXP", "AGEP", "MSTP", "HEAP", "OCCP", "INCP", "LFSP", "EMPP", "HRSP",
        "QALFP", "QALLP", "RLHP", "MTWP", "NPRD", "VEHRD", "MRERD", "RNTRD",
        "STRD", "HIND", "BEDRD", "HHCD"])
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--target-per-cell", type=int, default=20)
    p.add_argument("--pool", type=str, default=None,
                   help="Prebuilt CP feasible pool (build_feasible_pool_cp.py output). "
                        "When given, that pool is fitted instead of rebuilding one from "
                        "--census, so the refit uses exactly the pool the CP stage "
                        "published -- including its control-table coverage and QALFP "
                        "recoding, which the in-line rebuild does not apply.")
    p.add_argument("--zones", nargs="*", type=str)
    p.add_argument("--csf-map", type=str, default=None)
    p.add_argument("--max-iter", type=int, default=config.SA_MAX_ITER)
    p.add_argument("--initial-temp", type=float, default=config.SA_INITIAL_TEMP)
    p.add_argument("--cooling-rate", type=float, default=config.SA_COOLING_RATE)
    p.add_argument("--objective", type=str, default="rssz_sum",
                   choices=["rssz_sum", "rssz_mean_active"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default=None,
                   help="Isolated output directory for this invocation.")
    a = p.parse_args()

    global RESULTS_DIR, LOG_DIR
    if a.results_dir:
        RESULTS_DIR = os.path.abspath(a.results_dir)
        LOG_DIR = os.path.join(RESULTS_DIR, "logs")
        os.makedirs(RESULTS_DIR, exist_ok=True)

    # Persist the pool (uppercase) and clean it (lowercase + numeric) for SA.
    pool_name = (
        f"cp_feasible_pool_{os.getpid()}.csv" if a.pool
        else "cp_feasible_pool.csv"
    )
    pool_path = os.path.join(RESULTS_DIR, pool_name)
    microdata = None
    # Optional cleaned-pool cache (CP_POOL_CLEAN_CACHE=/path/pool_clean.pkl).
    # A chunked refit otherwise pays the full pool round-trip per chunk: read a
    # multi-hundred-MB CSV, assign infilled households, write it back out, and
    # re-read it to clean. The cache stores the post-clean frame, which is the
    # input every chunk actually needs, so it is keyed on the pool's mtime.
    cache_path = os.environ.get("CP_POOL_CLEAN_CACHE", "").strip()
    if a.pool:
        print(f"Loading prebuilt CP feasible pool ({a.pool})...")
        if not os.path.isfile(a.pool):
            sys.exit(f"ERROR: --pool {a.pool} does not exist; build it with "
                     "build_feasible_pool_cp.py --coverage-tables first.")
        # The cache records which pool it came from. A bare frame keyed only on
        # mtime is not safe to trust: a pool can be rebuilt in place, and a pool
        # with the same name but different contents would silently be fitted.
        if cache_path and os.path.isfile(cache_path):
            try:
                cached = pd.read_pickle(cache_path)
            except Exception:
                cached = None
            if (isinstance(cached, dict)
                    and cached.get("pool") == os.path.abspath(a.pool)
                    and cached.get("size") == os.path.getsize(a.pool)
                    and "data" in cached):
                print(f"  cleaned-pool cache hit: {cache_path}")
                microdata = cached["data"]
            elif cached is not None:
                print(f"  cleaned-pool cache ignored (built from a different pool): "
                      f"{cache_path}")
        # Keep the alias in sync: mip_fit.py reads cp_feasible_pool.csv, so a
        # pool fitted here must be the pool the MIP arm fits. Group infilled rows
        # into whole households first -- build_feasible_pool_cp.py writes one row
        # per infilled person, and fitting singleton infilled rows against an
        # NPRD household-size control inflates the zone population (34.6k fitted
        # against an 18.6k target for 112011236). The in-line build path always
        # applied this; loading a published pool must apply it too, or the refit
        # silently uses a different pool from the one the arm was run on.
        if microdata is None:
            pool_df = assign_infilled_households(pd.read_csv(a.pool))
            pool_df.to_csv(pool_path, index=False)
            sidecar = os.path.splitext(a.pool)[0] + "_build19_report.json"
            if os.path.isfile(sidecar):
                import json
                with open(sidecar, encoding="utf-8") as fh:
                    report = json.load(fh)
                print("CP-pool build report (sidecar):")
                for k, v in report.items():
                    print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    else:
        print("Loading census for the CP feasible pool...")
        census = pd.read_csv(a.census)
        pool, report = build_pool(census, a.control_attrs, a.min_count,
                                  a.target_per_cell, a.seed)
        print("CP-pool build report:")
        for k, v in report.items():
            print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
        pool.drop(columns=["_source"], errors="ignore").to_csv(pool_path, index=False)

    if microdata is None:
        print(f"Cleaning pool via clean_my_datagan ({pool_path})...")
        helper = Helper()
        microdata = helper.clean_my_datagan(pool_path)
        if cache_path:
            try:
                cache_src = os.path.abspath(a.pool) if a.pool else os.path.abspath(pool_path)
                pd.to_pickle({"pool": cache_src,
                              "size": os.path.getsize(cache_src),
                              "data": microdata}, cache_path)
                print(f"  cleaned-pool cache written: {cache_path}")
            except Exception as exc:
                print(f"  WARNING: could not write pool cache ({exc})")
    print(f"  Pool records: {len(microdata):,}")

    ind_tables, hh_tables = load_control_tables()
    print(f"  Individual control tables: {len(ind_tables)}, household: {len(hh_tables)}")

    if a.zones:
        zones = [str(z) for z in a.zones]
    else:
        zones = [str(z) for z in next(iter(ind_tables.values()))["data"]["sa2"].dropna().unique().tolist()]

    has_sa2 = "sa2" in microdata.columns
    has_regucp = "regucp" in microdata.columns
    if not has_sa2 and not has_regucp:
        raise ValueError("CP pool has neither 'sa2' nor 'regucp'; cannot match zones.")

    sa2_to_regucp, _ = _load_sa2_to_regucp(a.csf_map)

    all_rssz, skipped = [], []
    for zone in zones:
        print(f"\nZone {zone}")
        zone_pop = pd.DataFrame()
        if has_sa2:
            zone_pop = microdata.loc[_normalize_zone_values(microdata["sa2"]) == zone]
        if zone_pop.empty and has_regucp:
            zone_pop = microdata.loc[_normalize_zone_values(microdata["regucp"]) == zone]
        if zone_pop.empty and has_regucp and sa2_to_regucp:
            r = sa2_to_regucp.get(zone)
            if r:
                zone_pop = microdata.loc[_normalize_zone_values(microdata["regucp"]) == r]
        if zone_pop.empty:
            print(f"  WARNING: zone {zone} no match — falling back to full CP pool.")
            zone_pop = microdata.copy()

        try:
            result_df, rssz_info = run_sa_zone(
                zone_pop, ind_tables, hh_tables, zone,
                a.max_iter, a.initial_temp, a.cooling_rate, objective_mode=a.objective)
        except Exception as exc:
            print(f"  WARNING: zone {zone} skipped — {exc}")
            skipped.append(zone)
            continue
        if result_df is not None:
            result_df.to_csv(os.path.join(RESULTS_DIR, f"zone_{zone}.csv"), index=False)
            all_rssz.append(rssz_info)
        else:
            skipped.append(zone)

    if not all_rssz:
        print("ERROR: CP-pool baseline produced zero usable zones.")
        sys.exit(1)

    summary = pd.DataFrame(all_rssz)
    _chunk = os.environ.get("CHUNK_ID", "").strip()
    name = f"cp_pool_rssz_summary_chunk{_chunk}.csv" if _chunk else "cp_pool_rssz_summary.csv"
    out = os.path.join(RESULTS_DIR, name)
    summary.to_csv(out, index=False)
    vals = summary["rssz_total"]
    print(f"\n{'='*60}\nCP-pool + SA RSSZ Summary\n{'='*60}")
    print(f"  Zones : {len(vals)}  Mean {vals.mean():.4f}  Median {vals.median():.4f}  "
          f"Min {vals.min():.4f}  Max {vals.max():.4f}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

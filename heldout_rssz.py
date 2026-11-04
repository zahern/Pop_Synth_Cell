"""Held-out RSSZ: fit quality on tables never used by the fitter.

Uses the seven held-out control tables (age5p, assnp, bedrd, hcfmf, hhcd,
lfsp, tenlld) and the same RSSZ normalisation as the fitting objective:
SSZ / (N_k * chi2_critical).  Aggregates per table and per arm over the
367-zone fitted cohort.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from full_cohort_audit import cohort, map_files
from make_fig_heldout import (ALL_TABLES, _census, _counts, _read_zone,
                              load_all_controls)

OUT = os.path.join(HERE, "outputs", "full_cohort_audit")
os.makedirs(OUT, exist_ok=True)

ROOTS = {
    "SA-Only": (os.path.join(HERE, "outputs", "sa_only_results"), "sa_only"),
    "IPU-Only": (os.path.join(HERE, "outputs", "ipu_only_results"), "ipu_only"),
    "CP-only+SA": (os.path.join(HERE, "outputs", "cp_pool_results_base"), "cp_sa"),
    "CP-only+IPU": (os.path.join(HERE, "outputs", "cp_ipu_results_base"), "ipu_only"),
    "MIP+SA": (os.path.join(HERE, "outputs", "cp_pool_results"), "cp_sa"),
    "MIP+IPU": (os.path.join(HERE, "outputs", "cp_ipu_results"), "ipu_only"),
    "Coverage+SA": (os.path.join(HERE, "outputs", "cp_pool_results_coverage"), "cp_sa"),
    "Coverage+IPU": (os.path.join(HERE, "outputs", "cp_ipu_results_coverage"), "ipu_only"),
    "Composition+SA": (os.path.join(HERE, "outputs", "cp_pool_results_composition"), "cp_sa"),
    "Composition+IPU": (os.path.join(HERE, "outputs", "cp_ipu_results_composition"), "ipu_only"),
    "CP-MIP": (os.path.join(HERE, "outputs", "cp_mip_results"), "ipu_only"),
}
PERSON = {"age5p", "assnp", "lfsp"}
HH = {"bedrd", "hcfmf", "hhcd", "tenlld"}


def table_rssz(obs, exp):
    keys = set(obs) | set(exp)
    ssz = 0.0
    n = 0
    df = 0
    for k in keys:
        e = float(exp.get(k, 0.0))
        if e <= 0:
            continue
        o = float(obs.get(k, 0.0))
        ssz += (o - e) ** 2 / e
        n += e
        df += 1
    if n <= 0 or df <= 1:
        return np.nan
    crit = float(chi2.ppf(0.95, df - 1))
    if crit <= 0:
        return np.nan
    return ssz / (n * crit)


def main():
    tables = load_all_controls(os.path.join(HERE, "data_sa2_controls_heldout"))
    for k, ct in tables.items():
        ct["key"] = k.replace("sa2", "", 1)
    tables = {ct["key"]: ct for ct in tables.values()}
    zones = cohort()
    census = {}
    for i, z in enumerate(zones, 1):
        census[z] = {t: _census(tables[t], t, z) for t in ALL_TABLES}
        if i % 100 == 0:
            print("census", i, flush=True)

    rows = []
    for arm, (root, kind) in ROOTS.items():
        files = map_files(root, kind)
        have = [z for z in zones if z in files]
        print(arm, len(have), flush=True)
        # Read each zone once and score all seven held-out tables from it.
        # The previous loop re-read the same CSV once per table (7x I/O) and
        # took ~40 min for ten arms on a network filesystem.
        per_table = {t: [] for t in ALL_TABLES}
        for z in have:
            try:
                df = _read_zone(files[z])
            except Exception:
                continue
            for table in ALL_TABLES:
                obs = _counts(df, table)
                exp = census[z].get(table)
                if not obs or not exp:
                    continue
                v = table_rssz(obs, exp)
                if np.isfinite(v):
                    per_table[table].append(v)
        for table in ALL_TABLES:
            vals = per_table[table]
            if vals:
                rows.append({"arm": arm, "table": table,
                             "zones": len(vals),
                             "mean": float(np.mean(vals)),
                             "median": float(np.median(vals))})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "heldout_rssz.csv"), index=False)
    piv = out.pivot(index="arm", columns="table", values="mean")
    print(piv.round(3).to_string())
    print("wrote", os.path.join(OUT, "heldout_rssz.csv"))


if __name__ == "__main__":
    main()

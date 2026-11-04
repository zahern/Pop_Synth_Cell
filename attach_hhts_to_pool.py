#!/usr/bin/env python3
"""Attach observed HHTS/QTS travel attributes to a synthetic candidate pool.

The census microdata has no bicycle/e-bike fields.  ``step_0b_fuse_national.py
--with-hhts`` attaches the Queensland Travel Survey values to census records by
predictive-mean matching; this script carries those values into a CP/MIP pool
and fills households that were constructed rather than observed.

Household variables: BIKES, EBIKES, ESCOOTER
Person variables:    CARLICENCE, MCLICENCE (where person keys permit)

Observed matches are taken by ABSHID/ABSFID.  Unmatched constructed households
receive a hot-deck donor from the fused census, matched first on
(REGUCP, NPRD, VEHRD, HIND, HHCD, STRD), then on progressively looser keys.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HH_VARS = ["BIKES", "EBIKES", "ESCOOTER"]
PERSON_VARS = ["CARLICENCE", "MCLICENCE"]
HH_KEYS = ["REGUCP", "NPRD", "VEHRD", "HIND", "HHCD", "STRD", "TEND", "BEDRD"]


def _col(df, name):
    for c in df.columns:
        if str(c).lower() == name.lower():
            return c
    return None


def _merge_key(df, cols):
    return df[cols].astype(str).agg("|".join, axis=1)


def _fill_from_hotdeck(pool, donors, keys, variables, rng):
    """Fill missing variables in pool from donor rows sharing `keys`."""
    missing = pool[variables].isna().any(axis=1)
    if not missing.any() or donors.empty:
        return pool, 0
    use_keys = [k for k in keys if k in pool.columns and k in donors.columns]
    if not use_keys:
        if donors.empty:
            return pool, 0
        rows = donors[variables].to_numpy()
        pool_missing = pool.loc[missing]
        fill = rows[rng.integers(0, len(rows), size=len(pool_missing))]
        for j, var in enumerate(variables):
            idx = pool_missing.index[pool_missing[var].isna()]
            pool.loc[idx, var] = fill[:, j][pool_missing[var].isna().to_numpy()]
        return pool, int(pool.loc[missing, variables].notna().any(axis=1).sum())
    donor_groups = {}
    for key, group in donors.groupby(use_keys, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        donor_groups[key] = group[variables].to_numpy()
    pool_missing = pool.loc[missing]
    fill = np.empty((len(pool_missing), len(variables)), dtype=object)
    for i, (_, row) in enumerate(pool_missing.iterrows()):
        key = tuple(str(row[k]) for k in use_keys)
        rows = donor_groups.get(key)
        if rows is None or len(rows) == 0:
            fill[i, :] = np.nan
        else:
            fill[i, :] = rows[rng.integers(0, len(rows))]
    for j, var in enumerate(variables):
        values = fill[:, j]
        ok = pd.notna(values) & pool_missing[var].isna().to_numpy()
        idx = pool_missing.index[ok]
        pool.loc[idx, var] = values[ok]
    return pool, int(pool.loc[missing, variables].notna().any(axis=1).sum())


def _load_location_donors(path, csf_path, variables):
    """HHTS household donors with SA2/REGUCP derived from HOME_SA1_2021."""
    if str(path).lower().endswith(".csv"):
        hh = pd.read_csv(path, low_memory=False)
    else:
        hh = pd.read_excel(path)
    for col in ["BIKES", "EBIKES", "ESCOOTER", "HHSIZE", "HHVEH", "HHWGT_23"]:
        if col in hh.columns:
            hh[col] = pd.to_numeric(hh[col], errors="coerce")
    sa1 = hh["HOME_SA1_2021"].astype(str).str.replace(r"\.0$", "", regex=True)
    hh["sa2"] = sa1.str[:9]
    csf = pd.read_csv(csf_path, dtype={"sa2": str, "regucp": str})
    hh = hh.merge(csf, on="sa2", how="left")
    hh = hh.dropna(subset=["regucp", "HHSIZE", "HHVEH"] + variables)
    hh["NPRD"] = hh["HHSIZE"].astype(int)
    hh["VEHRD"] = hh["HHVEH"].astype(int)
    hh["regucp"] = hh["regucp"].astype(str)
    return hh[["regucp", "NPRD", "VEHRD", "HHWGT_23"] + variables]


def _fill_location_specific(pool, donors, variables, rng):
    """Replace household travel variables with location-conditioned HHTS draws."""
    if donors.empty:
        raise SystemExit("location donor table is empty")
    original_columns = set(pool.columns)
    pool = pool.reset_index(drop=True).copy()
    # The pool uses upper-case census columns; map to temporary canonical names
    # used for grouping.  They are removed before writing so the pool never
    # contains both REGUCP and regucp (which breaks case-insensitive readers).
    for name in ("regucp", "NPRD", "VEHRD"):
        src = _col(pool, name)
        if src is not None:
            pool[name] = (pool[src].astype(str) if name == "regucp"
                          else pd.to_numeric(pool[src], errors="coerce"))
    keys = [k for k in ("regucp", "NPRD", "VEHRD") if k in pool.columns]
    donors = donors.copy()
    donors["regucp"] = donors["regucp"].astype(str)

    levels = [["regucp", "NPRD", "VEHRD"], ["regucp", "NPRD"], ["regucp"]]
    levels = [[k for k in level if k in keys] for level in levels]
    levels = [level for level in levels if level]
    groups = {}
    for level in levels:
        mapping = {}
        for key, sub in donors.groupby(level, dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            mapping[key] = sub
        groups[tuple(level)] = mapping

    values = np.full((len(pool), len(variables)), np.nan, dtype=object)
    assigned = np.zeros(len(pool), dtype=bool)
    used_region = 0
    used_global = 0
    for level in levels:
        key_tuple = tuple(level)
        mapping = groups[key_tuple]
        for key, idx in pool.groupby(level, dropna=False, sort=False).groups.items():
            if not isinstance(key, tuple):
                key = (key,)
            sub = mapping.get(key)
            if sub is None or not len(sub):
                continue
            positions = np.asarray(idx, dtype=int)
            positions = positions[~assigned[positions]]
            if not len(positions):
                continue
            w = sub["HHWGT_23"].fillna(1).to_numpy(dtype=float)
            choices = rng.choice(len(sub), size=len(positions), p=w / w.sum())
            values[positions, :] = sub[variables].to_numpy()[choices]
            assigned[positions] = True
            used_region += len(positions)

    remaining = np.flatnonzero(~assigned)
    if len(remaining):
        global_values = donors[variables].to_numpy()
        w = donors["HHWGT_23"].fillna(1).to_numpy(dtype=float)
        choices = rng.choice(len(donors), size=len(remaining), p=w / w.sum())
        values[remaining, :] = global_values[choices]
        used_global = len(remaining)

    for j, var in enumerate(variables):
        pool[var] = values[:, j]
    for name in ("regucp", "NPRD", "VEHRD"):
        if name not in original_columns and name in pool.columns:
            pool = pool.drop(columns=[name])
    return pool, used_region, used_global


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--hhts-census", required=True,
                    help="Census CSV produced with step_0b --with-hhts")
    ap.add_argument("--hhts-households", default=None,
                    help="Optional 1_QTS_HOUSEHOLDS.xlsx for location-specific "
                         "donors (uses HOME_SA1_2021, HHWGT_23)")
    ap.add_argument("--csf-map", default=None,
                    help="csf_sa.csv mapping SA2 -> REGUCP")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    pool = pd.read_csv(a.pool, low_memory=False)
    hh_vars = [v for v in HH_VARS if v in pd.read_csv(a.hhts_census, nrows=1).columns]
    person_vars = [v for v in PERSON_VARS
                   if v in pd.read_csv(a.hhts_census, nrows=1).columns]
    if not hh_vars:
        raise SystemExit("fused census has no BIKES/EBIKES/ESCOOTER columns")

    usecols = ["ABSHID", "ABSFID"] + hh_vars + person_vars + HH_KEYS
    census = pd.read_csv(a.hhts_census, usecols=lambda c: c in set(usecols),
                         low_memory=False)
    report = {"pool": a.pool, "hhts_census": a.hhts_census,
              "n_pool_rows": int(len(pool))}

    # Household-level values: one row per ABSHID.
    hh_cols = ["ABSHID"] + [v for v in hh_vars if v in census.columns]
    hh_obs = census[hh_cols].drop_duplicates("ABSHID")
    hh_obs = hh_obs.rename(columns={v: f"{v}_hhts" for v in hh_vars})
    pool = pool.merge(hh_obs, on="ABSHID", how="left")
    for v in hh_vars:
        src = f"{v}_hhts"
        if v in pool.columns:
            pool[v] = pool[src].combine_first(pool[v])
            pool = pool.drop(columns=[src])
        else:
            pool[v] = pool[src]
            pool = pool.drop(columns=[src])
    observed_hh = int(pool[hh_vars].notna().any(axis=1).sum())

    # Person-level values: prefer ABSFID, then ABSHID/age/sex hot-deck.
    if "ABSFID" in pool.columns and "ABSFID" in census.columns:
        p_cols = ["ABSFID"] + [v for v in person_vars if v in census.columns]
        p_obs = census[p_cols].drop_duplicates("ABSFID")
        p_obs = p_obs.rename(columns={v: f"{v}_hhts" for v in person_vars})
        pool = pool.merge(p_obs, on="ABSFID", how="left")
        for v in person_vars:
            src = f"{v}_hhts"
            if v in pool.columns:
                pool[v] = pool[src].combine_first(pool[v])
                pool = pool.drop(columns=[src])
            else:
                pool[v] = pool[src]
                pool = pool.drop(columns=[src])
    observed_person = int(pool[person_vars].notna().any(axis=1).sum()) \
        if person_vars else 0

    # Hot-deck the constructed households/persons.
    rng = np.random.default_rng(a.seed)
    donor = census.dropna(subset=hh_vars).copy()
    donor = donor.drop_duplicates("ABSHID")
    filled_total = 0
    for keys in (HH_KEYS, ["REGUCP", "NPRD", "VEHRD"], ["NPRD", "VEHRD"], []):
        if not pool[hh_vars].isna().any().any():
            break
        pool, n = _fill_from_hotdeck(pool, donor, keys, hh_vars, rng)
        filled_total += n
        if not keys:
            break

    location_mode = bool(a.hhts_households)
    region_rows = global_rows = 0
    if location_mode:
        if not a.csf_map:
            raise SystemExit("--csf-map is required with --hhts-households")
        loc = _load_location_donors(a.hhts_households, a.csf_map, hh_vars)
        pool, region_rows, global_rows = _fill_location_specific(
            pool, loc, hh_vars, rng)

    report.update({
        "hh_vars": hh_vars,
        "person_vars": person_vars,
        "location_specific": location_mode,
        "location_rows": int(region_rows),
        "global_fallback_rows": int(global_rows),
        "location_donor_households": int(len(loc)) if location_mode else 0,
        "observed_household_rows": observed_hh,
        "observed_person_rows": observed_person,
        "hotdeck_filled_rows": int(filled_total),
        "households_missing_after": int(pool[hh_vars].isna().any(axis=1).sum()),
        "out": a.out,
    })
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pool.to_csv(a.out, index=False)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

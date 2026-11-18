"""Add the raw HHTS source shares to the HHTS validation matrices.

The published figure compares the fitted population with the attached pool.
This script adds the raw linked-microdata shares so the same cross-tabs can be
read as raw source -> attached pool -> fitted population.
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
TAB = os.path.join(RUN, "tables")
RAW = r"Z:\Population_Synth\Merged_HHTS_CENSUS.csv"

COLS = ["abshid", "agep", "rlhp", "hhcd", "bikes",
        "carlicence", "mclicence"]

# HHTS stores AGEP as 5-year codes 1..18; map each to a representative year so
# the same age-band helpers used for the fitted population apply.
AGE_CODE_TO_YEAR = {
    1: 2, 2: 7, 3: 12, 4: 17, 5: 22, 6: 27, 7: 32, 8: 37, 9: 42,
    10: 47, 11: 52, 12: 57, 13: 62, 14: 67, 15: 72, 16: 77, 17: 82,
    18: 87,
}
AGE_GROUPS = ["0-14", "15-24", "25-44", "45-64", "65+"]
AGE_BANDS = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
BIKE_CATS = ["0", "1", "2", "3+"]
SIZE_CATS = ["1", "2", "3", "4+"]
HHCD_LABELS = {1: "One family", 2: "Two+ families", 3: "Lone person",
               4: "Group household", 5: "Visitors only", 6: "Other",
               7: "Not applicable"}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def age_group(a):
    try:
        a = int(a)
    except (TypeError, ValueError):
        return None
    if a < 15:
        return "0-14"
    if a < 25:
        return "15-24"
    if a <= 28:
        return "25-44"
    if a <= 32:
        return "45-64"
    return "65+"


def age_band(a):
    try:
        a = int(a)
    except (TypeError, ValueError):
        return None
    if 15 <= a <= 24:
        return "15-24"
    if a in (25, 26):
        return "25-34"
    if a in (27, 28):
        return "35-44"
    if a in (29, 30):
        return "45-54"
    if a in (31, 32):
        return "55-64"
    if a >= 33:
        return "65+"
    return None


def bike_cat(v):
    v = float(v) if pd.notna(v) else 0.0
    return "3+" if v >= 2.5 else str(int(max(0, min(2, round(v)))))


def size_cat(n):
    n = int(n)
    return "4+" if n >= 4 else str(max(1, n))


def household_frame(d, key):
    hh = d.drop_duplicates(key).copy()
    size = d.groupby(key).size().rename("hh_size")
    hh = hh.merge(size, left_on=key, right_index=True, how="left")
    ref = d[d["rlhp"] == 1].groupby(key)["agep"].first()
    young = d.groupby(key)["agep"].min()
    hh["ref_age"] = hh[key].map(ref)
    hh["young_age"] = hh[key].map(young)
    hh["bikes_cat"] = hh["bikes"].apply(bike_cat)
    hh["size_cat"] = hh["hh_size"].apply(size_cat)
    hh["ref_band"] = hh["ref_age"].apply(age_group)
    hh["young_band"] = hh["young_age"].apply(age_group)
    return hh


def add_counts(counter, df, cols):
    for row, c in df.groupby(cols, dropna=False).size().items():
        counter[row] += int(c)


def matrix(counter, rows, cols):
    out = pd.DataFrame(0.0, index=rows, columns=cols)
    for (r, c), n in counter.items():
        if r in out.index and c in out.columns:
            out.loc[r, c] += n
    out = out.div(out.sum(axis=1).replace(0, np.nan), axis=0) * 100
    return out


def melt(m, name):
    out = m.reset_index().melt(id_vars="index", var_name="column",
                               value_name="share")
    out.columns = ["row", "col", name]
    return out


def load_raw():
    raw = pd.read_csv(RAW, usecols=[c.upper() for c in COLS],
                      low_memory=False)
    raw.columns = [c.lower() for c in raw.columns]
    raw["agep"] = pd.to_numeric(raw["agep"], errors="coerce").map(
        AGE_CODE_TO_YEAR)
    raw["hh_key"] = raw["abshid"].astype(str)
    return raw


def _clean(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns])


def main() -> None:
    raw = load_raw()
    print("raw rows", len(raw), "households", raw.hh_key.nunique())

    hh = household_frame(raw, "hh_key")
    raw_hhcd, raw_size = Counter(), Counter()
    add_counts(raw_hhcd, hh, ["hhcd", "bikes_cat"])
    add_counts(raw_size, hh, ["size_cat", "bikes_cat"])

    ref = raw[raw["rlhp"] == 1].groupby("hh_key")["agep"].first()
    raw["ref_band"] = raw["hh_key"].map(ref).apply(age_group)
    raw["member_band"] = raw["agep"].apply(age_group)
    raw_refmember = Counter()
    add_counts(raw_refmember, raw, ["ref_band", "member_band"])

    raw["age_band"] = raw["agep"].apply(age_band)
    raw["adult"] = _num(raw["agep"]) >= 15
    raw["car"] = _bool(raw["carlicence"])
    raw["mc"] = _bool(raw["mclicence"])
    raw_age_lic = Counter()
    add_counts(raw_age_lic, raw[raw.adult], ["age_band", "car", "mc"])

    hhcd_rows = [1, 2, 3, 4, 5, 6, 7]
    raw_hhcd_m = matrix(raw_hhcd, hhcd_rows, BIKE_CATS)
    raw_size_m = matrix(raw_size, SIZE_CATS, BIKE_CATS)
    raw_ry_m = matrix(raw_refmember, AGE_GROUPS, AGE_GROUPS)

    hhcd_df = pd.read_csv(os.path.join(TAB, "hhts_hhcd_bikes_matrix.csv"))
    hhcd_df = _clean(hhcd_df, ["raw_share", "pool_raw_pp", "fit_pool_pp",
                               "fit_raw_pp"])
    hhcd_df = hhcd_df.merge(melt(raw_hhcd_m, "raw_share"),
                            on=["row", "col"], how="left")
    hhcd_df["pool_raw_pp"] = hhcd_df.pool_share - hhcd_df.raw_share
    hhcd_df["fit_pool_pp"] = hhcd_df.fitted_share - hhcd_df.pool_share
    hhcd_df["fit_raw_pp"] = hhcd_df.fitted_share - hhcd_df.raw_share
    hhcd_df.to_csv(os.path.join(TAB, "hhts_hhcd_bikes_matrix.csv"), index=False)

    size_df = pd.read_csv(os.path.join(TAB, "hhts_size_bikes_matrix.csv"))
    size_df = _clean(size_df, ["raw_share", "pool_raw_pp", "fit_pool_pp",
                               "fit_raw_pp"])
    size_df = size_df.merge(melt(raw_size_m, "raw_share"),
                            on=["row", "col"], how="left")
    size_df["pool_raw_pp"] = size_df.pool_share - size_df.raw_share
    size_df["fit_pool_pp"] = size_df.fitted_share - size_df.pool_share
    size_df["fit_raw_pp"] = size_df.fitted_share - size_df.raw_share
    size_df.to_csv(os.path.join(TAB, "hhts_size_bikes_matrix.csv"), index=False)

    ry_df = pd.read_csv(os.path.join(TAB, "age_household_matrix.csv"))
    ry_df = _clean(ry_df, ["raw_share", "pool_raw_pp", "fit_pool_pp",
                           "fit_raw_pp"])
    ry_df = ry_df.merge(melt(raw_ry_m, "raw_share"),
                        on=["row", "col"], how="left")
    ry_df["pool_raw_pp"] = ry_df.pool_share - ry_df.raw_share
    ry_df["fit_pool_pp"] = ry_df.fitted_share - ry_df.pool_share
    ry_df["fit_raw_pp"] = ry_df.fitted_share - ry_df.raw_share
    ry_df.to_csv(os.path.join(TAB, "age_household_matrix.csv"), index=False)

    al = pd.read_csv(os.path.join(TAB, "hhts_age_licence_matrix.csv"))
    al = _clean(al, ["raw", "pool_raw_pp", "fit_pool_pp", "fit_raw_pp"])
    raw_rows = []
    for band in AGE_BANDS:
        for label, key in [("Car", "car"), ("Motorcycle", "mc")]:
            num = den = 0
            for (b, car, mc), n in raw_age_lic.items():
                if b != band:
                    continue
                den += n
                if (key == "car" and car) or (key == "mc" and mc):
                    num += n
            raw_rows.append({"age_band": band, "licence": label,
                             "raw": 100 * num / den if den else np.nan})
    al = al.merge(pd.DataFrame(raw_rows), on=["age_band", "licence"], how="left")
    al["pool_raw_pp"] = al.pool - al.raw
    al["fit_pool_pp"] = al.fitted - al.pool
    al["fit_raw_pp"] = al.fitted - al.raw
    al.to_csv(os.path.join(TAB, "hhts_age_licence_matrix.csv"), index=False)

    print("wrote raw shares into the four HHTS matrices")


if __name__ == "__main__":
    main()

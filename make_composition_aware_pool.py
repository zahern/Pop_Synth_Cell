"""Build a composition-aware expansion pool from the fixed-pool shards.

The fixed build appended every menu candidate.  That menu is dominated by
size-7 households and by records carrying MTWP code 17 ("Not stated"), so the
released pool no longer resembles the census composition.

This script keeps the base pool, the infilled records, the MIP-selected coverage
set, and the coverage infill.  From the appended menu it keeps only households
that are neither size 7 nor carry MTWP 17 anywhere.  The result is smaller than
the full-menu pool but remains larger than the base pool, and its household-size
and travel composition stay close to the census.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "outputs", "feasible_pool_expanded_fixed_hhts.csv")
OUT = os.path.join(HERE, "outputs",
                   "feasible_pool_expanded_composition_hhts.csv")


def main():
    df = pd.read_csv(SRC, low_memory=False)
    menu = df["_source"].astype(str) == "mip_joint_menu"
    keep = df[~menu].copy()

    blocked = set()
    for hh, grp in df[menu].groupby("ABSHID", sort=False):
        nprd = pd.to_numeric(grp["NPRD"], errors="coerce").iloc[0]
        mtwp = pd.to_numeric(grp["MTWP"], errors="coerce")
        if int(nprd) == 7 if pd.notna(nprd) else False:
            blocked.add(hh)
        elif (mtwp == 17).any():
            blocked.add(hh)
    extra = df[menu & ~df["ABSHID"].isin(blocked)]
    out = pd.concat([keep, extra], ignore_index=True)
    out.to_csv(OUT, index=False)

    hh = out.drop_duplicates("ABSHID")
    sizes = pd.to_numeric(hh["NPRD"], errors="coerce")
    mtwp = pd.to_numeric(out["MTWP"], errors="coerce")
    print("wrote", OUT)
    print("persons", len(out), "households", len(hh))
    print("size7", round(float((sizes == 7).mean()), 4),
          "mtwp17", round(float((mtwp == 17).mean()), 4))
    print(out["_source"].value_counts().to_dict())


if __name__ == "__main__":
    main()

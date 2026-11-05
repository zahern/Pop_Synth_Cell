"""Fix the nprd code-7 collision in a constructed pool.

The fused microdata use NPRD=7 as a residual/not-applicable code for
visitors-only, other non-classifiable and not-applicable dwellings (HHCD 5-7).
The control tables use code 7 as a seven-person size category.  This script
recomputes NPRD from the actual household membership, keeps the original code
as NPRD_ORIG, and marks non-private/not-applicable dwellings with NPRD_NA so a
fitter can exclude them from the size control.

Usage:
    python fix_nprd_coding.py --pool outputs/feasible_pool.csv \
        --out outputs/feasible_pool_nprdfix.csv
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 200_000


def build_fix(pool: str) -> pd.Series:
    d = pd.read_csv(pool, usecols=["ABSHID", "NPRD", "HHCD"],
                    low_memory=False)
    d.columns = [c.lower() for c in d.columns]
    size = d.groupby("abshid").size()
    hh = d.drop_duplicates("abshid").set_index("abshid")
    orig = hh["nprd"].astype("int64")
    # Minimal targeted fix: code 7 is the residual/not-applicable code, so it
    # becomes NA (0).  Households whose actual membership is 7 or 8+ get the
    # matching size code; smaller households keep their original code.
    fixed = orig.copy()
    fixed[size == 7] = 7
    fixed[size >= 8] = 8
    fixed[orig == 7] = 0
    fixed.name = "nprd_fixed"
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("building household map", flush=True)
    fixed = build_fix(args.pool)
    print(f"households {len(fixed):,}; NA {int((fixed == 0).sum()):,}; "
          f"size7 {int((fixed == 7).sum()):,}; size8+ {int((fixed == 8).sum()):,}",
          flush=True)

    first = True
    written = 0
    for chunk in pd.read_csv(args.pool, chunksize=CHUNK, low_memory=False):
        chunk.columns = [c.lower() for c in chunk.columns]
        orig = chunk["nprd"].astype("int64")
        chunk["NPRD_ORIG"] = orig
        chunk["NPRD_NA"] = (
            (orig == 7) & chunk["hhcd"].isin([5, 6, 7])).astype("int64")
        chunk["nprd"] = chunk["abshid"].map(fixed).fillna(0).astype("int64")
        chunk = chunk.rename(columns={
            "abshid": "ABSHID", "nprd": "NPRD", "hhcd": "HHCD"})
        chunk.to_csv(args.out, mode="a" if not first else "w",
                     header=first, index=False)
        written += len(chunk)
        first = False
        print(f"  {written:,} rows", flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

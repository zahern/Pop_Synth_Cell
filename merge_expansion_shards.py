#!/usr/bin/env python3
"""Merge zone-sharded MIP expansion outputs into one expanded pool.

Each shard is run with ``mip_pool.py --mode joint --append-only``, so it writes
only the households constructed for that shard's zones.  Those records carry a
``_target_sa2`` tag and a shard-specific ``--id-offset``, which keeps household
IDs unique across shards.  The merge restores the base pool once and appends
every shard's constructed rows.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="Base feasible pool")
    ap.add_argument("--shards", nargs="+", required=True,
                    help="Shard append-only CSVs (globs expanded by the shell)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)

    paths = []
    for pattern in a.shards:
        paths.extend(sorted(glob.glob(pattern)))
    paths = [p for p in dict.fromkeys(paths) if os.path.basename(p) !=
             os.path.basename(a.out)]
    if not paths:
        raise SystemExit("no shard files matched")

    base = pd.read_csv(a.base, low_memory=False)
    parts = [base]
    shard_rows = 0
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            continue
        if "_source" in df.columns:
            df = df[df["_source"].astype(str).str.startswith("mip")]
        if df.empty:
            continue
        shard_rows += len(df)
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    # ABSHID is a *household* id and repeats on every person row, so it must
    # never be used to deduplicate the merged pool: doing so collapses a
    # multi-person household to one person.  Shard ids are offset above the
    # base range, so concatenation is already collision-free.
    duplicate_ids = 0
    if "ABSHID" in out.columns and "ABSHID" in base.columns:
        base_ids = set(base["ABSHID"].astype(str))
        shard_ids = set()
        for part in parts[1:]:
            if "ABSHID" in part.columns:
                shard_ids.update(part["ABSHID"].astype(str))
        duplicate_ids = len(base_ids & shard_ids)
    out.to_csv(a.out, index=False)

    report = {
        "base": a.base,
        "shards": paths,
        "n_shards": len(paths),
        "base_rows": int(len(base)),
        "shard_rows": int(shard_rows),
        "merged_rows": int(len(out)),
        "n_unique_households": int(out["ABSHID"].nunique())
        if "ABSHID" in out.columns else None,
        "abshid_collisions_base_shards": int(duplicate_ids),
        "out": a.out,
    }
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

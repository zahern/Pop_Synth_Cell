#!/usr/bin/env python3
"""Split the paper zone-chunk union into disjoint expansion shards."""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", default="pbs/zone_chunks",
                    help="Directory containing chunk_*.txt")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    zones = set()
    for path in glob.glob(os.path.join(a.chunks, "chunk_*.txt")):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                z = line.strip().lstrip("\ufeff")
                if z and not z.startswith("#"):
                    zones.add(z)
    zones = sorted(zones)
    if not zones:
        raise SystemExit("no zones found")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        shard = zones[i::a.n]
        path = out / f"shard_{i}.zones"
        path.write_text("\n".join(shard) + ("\n" if shard else ""),
                        encoding="utf-8")
        print(f"  shard {i:02d}: {len(shard):3d} zones -> {path}")
    (out / "zones.txt").write_text(
        "\n".join(zones) + "\n", encoding="utf-8")
    print(f"  total zones: {len(zones)} across {a.n} disjoint shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

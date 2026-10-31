#!/usr/bin/env python
"""
Score all synthesis arms on the 11 control tables shared by every arm, over a
single common zone cohort.

Why this exists
---------------
The pool x fitter table in the paper compares arms on the 11 shared tables
(heap, mstp, mtwp, occp, rlhp, sexp, hind, mrerd, nprd, rntrd, vehrd) but the
rows were previously taken from different runs with different coverage.  This
script fixes the basis: one table set, one zone cohort, every arm scored the
same way, so a zone count appears once and applies to every row.

The cohort is defined structurally rather than by whichever arm happened to
finish: a zone is in the cohort only if all 11 tables carry a target for it and
every scored arm wrote a population for it.  Arms that do not reach the cohort
are reported with their reach, not silently dropped.

CAUTION before using its output in the paper
-------------------------------------------
The zone populations sitting in the results directories are of mixed vintage:
they were fitted before the October-2026 control-table repairs
(normalize_control_tables.py, the rntrd/strd fix), so a repaired table can no
longer be matched by a stored population.  Re-scoring SA-Only against the
per-zone summaries the run itself wrote shows this: median absolute difference
0.14 per zone but a mean of 2.62 against 0.26, i.e. a handful of zones blow up.
The script is therefore a diagnostic on the stored files, not a way to regenerate
the published RSSZ table, and its numbers must not be placed beside the 353-zone
entries produced at fit time.  Refit before quoting.

Usage
-----
    python score_shared11_rssz.py                 # all arms that have outputs
    python score_shared11_rssz.py --methods sa_only cp_mip
    python score_shared11_rssz.py --limit 20      # quick smoke test

Outputs
-------
    outputs/benchmarking/shared11_rssz_summary.csv   one row per arm
    outputs/benchmarking/shared11_rssz_zones.csv     one row per arm x zone
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import pandas as pd

import config
from SA_2_level import build_control_dicts, calculate_total_rssz_table
from rssz_outlier_analysis import _load_control_table_safe

# The 11 tables carried by every completed arm.  Ordered as in the paper note.
SHARED11 = [
    "heap", "mstp", "mtwp", "occp", "rlhp", "sexp",
    "hind", "mrerd", "nprd", "rntrd", "vehrd",
]
HOUSEHOLD_TABLES = {"hind", "mrerd", "nprd", "rntrd", "vehrd", "bedrd", "hhcd", "strd", "tend"}

OUT_DIR = os.path.join(config.OUTPUTS_DIR, "benchmarking")
SUMMARY_CSV = os.path.join(OUT_DIR, "shared11_rssz_summary.csv")
ZONES_CSV = os.path.join(OUT_DIR, "shared11_rssz_zones.csv")


# --------------------------------------------------------------------------- #
# Control tables
# --------------------------------------------------------------------------- #
def load_shared11():
    """Load exactly the 11 shared tables, split into individual / household."""
    ind, hh = {}, {}
    for attr in SHARED11:
        path = os.path.join(config.SA_CONTROL_DIR, f"sa2{attr}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing shared control table: {path}")
        ct = _load_control_table_safe(path, f"sa2{attr}")
        if ct is None:
            raise RuntimeError(f"Could not load {path}")
        target = hh if attr in HOUSEHOLD_TABLES else ind
        target[attr] = ct
    return ind, hh


def zones_with_full_coverage(ind, hh):
    """Zones that carry a target in every one of the 11 shared tables."""
    per_table = {}
    for attr, ct in {**ind, **hh}.items():
        df = ct["data"]
        per_table[attr] = set(df["sa2"].astype(str).unique())
    zones = set.intersection(*per_table.values())
    return {z for z in zones if z.isdigit()}


# --------------------------------------------------------------------------- #
# Arm outputs
# --------------------------------------------------------------------------- #
# Population files only, in preference order.  The results directories also hold
# per-zone audit/diagnostic CSVs (control_balance_audit, rssz_diagnostics,
# rssz_value) that share the zone_<id>_ prefix but hold no population, so a bare
# zone_*.csv glob would pick them up and score an empty "population".
ZONE_FILE_PATTERNS = (
    (r"^zone_(\d{9})(?:\.0)?_new_population\.csv$", "*_new_population.csv"),
    (r"^zone_(\d{9})(?:\.0)?_synthetic_population\.csv$", "*_synthetic_population.csv"),
    (r"^zone_(\d{9})(?:\.0)?\.csv$", "zone_*.csv"),
)


def arm_dirs():
    """Every arm that has a results directory, in paper order."""
    return [
        ("sa_only", "SA-Only", config.SA_ONLY_RESULTS_DIR),
        ("ipu_only", "IPU-Only", config.IPU_ONLY_RESULTS_DIR),
        ("cp_sa", "CP-pool+SA", config.CP_POOL_RESULTS_DIR),
        ("cp_ipu", "CP-pool+IPU", config.CP_IPU_RESULTS_DIR),
        ("cp_mip", "CP-MIP", config.CP_MIP_RESULTS_DIR),
    ]


def zone_files(directory):
    """Map zone id -> population file, preferring the plain zone_<id>.csv form."""
    found = {}
    if not directory or not os.path.isdir(directory):
        return found
    for pattern, glob_pat in ZONE_FILE_PATTERNS:
        for fpath in sorted(glob.glob(os.path.join(directory, glob_pat))):
            fname = os.path.basename(fpath)
            m = re.match(pattern, fname)
            if not m:
                continue
            zid = m.group(1)
            if zid in found:
                continue
            found[zid] = fpath
    return found


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_zone(fpath, zone, ind, hh):
    try:
        df = pd.read_csv(fpath)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None, None
    if df.empty or "abshid" not in df.columns:
        return None, None
    ind_d = build_control_dicts(ind, zone)
    hh_d = build_control_dicts(hh, zone)
    if not ind_d and not hh_d:
        return None, None
    try:
        total, per_table = calculate_total_rssz_table(df.to_dict("records"), ind_d, hh_d)
    except Exception as exc:                      # noqa: BLE001 - reported, not hidden
        print(f"    WARN {zone}: {exc}")
        return None, None
    # Only the 11 shared tables may contribute; build_control_dicts already
    # restricts to them, but trim defensively in case a table is empty.
    keep = {k: v for k, v in per_table.items() if any(
        t in k.lower() for t in SHARED11)}
    return float(sum(keep.values())), keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="subset of arm keys (default: every arm with outputs)")
    ap.add_argument("--limit", type=int, default=None,
                    help="score at most N zones per arm (smoke test)")
    args = ap.parse_args()

    print("Loading the 11 shared control tables...")
    ind, hh = load_shared11()
    coverage = zones_with_full_coverage(ind, hh)
    print(f"  zones with a target in all 11 tables: {len(coverage)}")

    arms = arm_dirs()
    if args.methods:
        wanted = set(args.methods)
        arms = [a for a in arms if a[0] in wanted]

    files = {key: zone_files(d) for key, _, d in arms}
    for key, label, _ in arms:
        print(f"  {label:14s} {len(files[key]):4d} zone files")

    # Cohort: full 11-table coverage and every scored arm wrote a population.
    cohort = set(coverage)
    for key, _, _ in arms:
        cohort &= set(files[key])
    cohort = sorted(cohort)
    if args.limit:
        cohort = cohort[: args.limit]
    print(f"\nCommon cohort: {len(cohort)} zones (11 tables, every arm present)")
    if not cohort:
        sys.exit("No zone is covered by every requested arm; nothing to score.")

    rows = []
    summary = []
    for key, label, _ in arms:
        totals = []
        for zone in cohort:
            total, per_table = score_zone(files[key][zone], zone, ind, hh)
            if total is None:
                continue
            totals.append(total)
            rec = {"arm": key, "zone": zone, "rssz_total": total}
            for table, value in per_table.items():
                rec[table] = value
            rows.append(rec)
        if not totals:
            summary.append({
                "arm": key, "label": label, "n_zones_scored": 0,
                "rssz_mean": None, "rssz_median": None, "rssz_p90": None,
            })
            print(f"  {label:14s} no zone could be scored")
            continue
        s = pd.Series(totals)
        summary.append({
            "arm": key,
            "label": label,
            "n_zones_scored": len(totals),
            "rssz_mean": round(float(s.mean()), 4),
            "rssz_median": round(float(s.median()), 4),
            "rssz_p90": round(float(s.quantile(0.9)), 4),
        })
        print(f"  {label:14s} n={len(totals):3d}  mean={s.mean():.4f}  "
              f"median={s.median():.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False)
    pd.DataFrame(rows).to_csv(ZONES_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV}\nWrote {ZONES_CSV}")


if __name__ == "__main__":
    main()

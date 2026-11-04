#!/usr/bin/env python
"""Score every paper arm on the common 367-zone / 11-table basis.

Why this exists
---------------
The pool x fitter table compares arms built from different pools, but the
numbers have to be computed the same way: one zone cohort, the same 11 control
tables, per-zone RSSZ summed over those tables, then averaged over zones.  This
script is the single place that does that, including the new fixed size-aware
pool, so a row cannot drift between the text and the table.

Cohort
------
The 371-zone study area minus the four zones excluded from the paper:
304041102, 308051537, 310021279, 311031315.

Usage
-----
    python score_paper_arms.py                 # all arms with outputs
    python score_paper_arms.py --latex         # also print LaTeX rows
    python score_paper_arms.py --arm fixed_sa fixed_ipu

Outputs
-------
    outputs/paper_arms/paper_arms_summary.csv   one row per arm
    outputs/paper_arms/paper_arms_zones.csv     one row per arm x zone
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "outputs")

SHARED11 = ["heap", "mstp", "mtwp", "occp", "rlhp", "sexp",
            "hind", "mrerd", "nprd", "rntrd", "vehrd"]
# All 19 fitted control tables (the basis the fitters actually optimise against,
# per pbs/stage_cp_sa_chunk.pbs). The 11-table subset above silently drops the
# eight tables -- qalfp, qallp, incp, hrsp, strd, indp, typp, stup -- that the
# pool expansion most affects, so rssz11 can look flat or improve while rssz19
# (the real fit) degrades. Both are reported so the gap is explicit.
SHARED19 = ["heap", "mstp", "mtwp", "occp", "rlhp", "sexp",
            "qalfp", "qallp", "incp", "hrsp", "indp", "typp", "stup",
            "hind", "mrerd", "nprd", "rntrd", "vehrd", "strd"]
PERSON_TABLES = {"heap", "mstp", "mtwp", "occp", "rlhp", "sexp",
                 "qalfp", "qallp", "incp", "hrsp", "indp", "typp", "stup"}
# The fitted cohort is the 371-zone scope minus the four zones with no usable
# individual target.  The four are asserted by the fit-time zone lists: the
# expansion shards and the 20 fitting chunks omit 301031014, not 308051537
# (which is not in the 371-zone scope file at all).
EXCLUDED = {"301031014", "304041102", "310021279", "311031315"}


def _cohort() -> set[str]:
    path = os.path.join(HERE, "pbs", "zones_371.txt")
    zones = {line.strip() for line in open(path) if line.strip()}
    return zones - EXCLUDED


def _find_column(df: pd.DataFrame, table: str):
    """Locate a control-table column in either naming convention.

    CP-SA summaries use ``Individual - sa2heap``; CP-IPU summaries use
    ``Individual - heap``.  The relationship table is labelled ``relationship``
    rather than ``rlhp`` in the IPU summaries.  Prefer an exact match, then any
    column whose normalised tail is the table name.
    """
    person = table in PERSON_TABLES
    aliases = {table}
    if table == "rlhp":
        aliases.add("relationship")

    def _match(require_level):
        """Columns carrying this table, optionally constrained to the expected
        Individual/Household level. The level label is not consistent between
        fitters -- e.g. SA writes ``Individual - sa2strd`` while IPU writes
        ``Household - strd`` -- so a strict-level miss falls back to level-free
        matching rather than silently dropping the whole arm."""
        exact, loose = [], []
        for col in df.columns:
            low = str(col).lower()
            if not any(alias in low for alias in aliases):
                continue
            if require_level:
                if person and "individual" not in low:
                    continue
                if not person and "household" not in low:
                    continue
            exact.append(col)
            if any(low.endswith(alias) for alias in aliases):
                loose.append(col)
        return exact, loose

    for require_level in (True, False):
        exact, loose = _match(require_level)
        for group in (loose, exact):
            if group:
                return group[0]
    return None


def load_arm(paths: list[str], name: str) -> pd.DataFrame:
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path, low_memory=False))
        except Exception as exc:
            print(f"  {name}: skipping {path}: {exc}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["zone"] = df["zone"].astype(str).str.strip()
    df = df.drop_duplicates("zone", keep="last")
    return df


def score(df: pd.DataFrame, name: str, cohort: set[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = {}
    missing = []
    for table in SHARED19:
        col = _find_column(df, table)
        if col is None:
            missing.append(table)
        else:
            cols[table] = col
    if missing:
        print(f"  {name}: missing tables {missing}", file=sys.stderr)
        return pd.DataFrame()
    out = pd.DataFrame({"zone": df["zone"]})
    for table, col in cols.items():
        out[table] = pd.to_numeric(df[col], errors="coerce")
    out["rssz11"] = out[SHARED11].sum(axis=1, min_count=len(SHARED11))
    out["rssz19"] = out[SHARED19].sum(axis=1, min_count=len(SHARED19))
    out["arm"] = name
    out = out[out["zone"].isin(cohort) & out["rssz19"].notna()].copy()
    return out


def pool_stats(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, low_memory=False,
                     usecols=lambda c: c.lower() in {"abshid", "nprd"})
    df.columns = [str(c).lower() for c in df.columns]
    hh = df.drop_duplicates("abshid")
    stats = {"persons": int(len(df)), "households": int(len(hh))}
    if "nprd" in hh.columns:
        size = pd.to_numeric(hh["nprd"], errors="coerce")
        stats["size7_share"] = round(float((size == 7).mean()), 4)
        stats["mean_size"] = round(float(size.mean()), 3)
    return stats


def _chunk_paths(directory: str, prefix: str, lo: int = 0,
                 hi: int = 19) -> list[str]:
    """The 20 regular zone-chunk summaries, excluding the 5000+ reruns."""
    paths = []
    for i in range(lo, hi + 1):
        path = os.path.join(directory, f"{prefix}{i}.csv")
        if os.path.exists(path):
            paths.append(path)
    return paths


def _glob(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


ARMS = [
    ("sa_only", _glob(os.path.join(OUT_ROOT, "sa_only_results",
                                   "sa_only_rssz_summary_chunk*.csv"))),
    ("ipu_only", [os.path.join(OUT_ROOT, "ipu_only_rssz_summary.csv")]),
    ("cp_sa", _glob(os.path.join(OUT_ROOT, "cp_pool_results",
                                 "expanded_v3_chunk_*",
                                 "cp_pool_rssz_summary_chunk*.csv"))),
    ("cp_ipu", _chunk_paths(os.path.join(OUT_ROOT, "cp_ipu_results"),
                            "cp_ipu_rssz_summary_chunk")),
    ("cp_mip", _glob(os.path.join(OUT_ROOT, "cp_mip_results",
                                  "cp_pool_rssz_summary_chunk*.csv"))),
    ("large_sa", _glob(os.path.join(OUT_ROOT, "cp_pool_results_large",
                                    "expanded_v3_chunk_*",
                                    "cp_pool_rssz_summary_chunk*.csv"))),
    ("large_ipu", _glob(os.path.join(OUT_ROOT, "cp_ipu_results_large",
                                     "cp_ipu_rssz_summary_chunk*.csv"))),
    ("fixed_sa", _glob(os.path.join(OUT_ROOT, "cp_pool_results_fixed",
                                    "expanded_v3_chunk_*",
                                    "cp_pool_rssz_summary_chunk*.csv"))),
    ("fixed_ipu", _glob(os.path.join(OUT_ROOT, "cp_ipu_results_fixed",
                                     "cp_ipu_rssz_summary_chunk*.csv"))),
    ("coverage_sa", _glob(os.path.join(OUT_ROOT, "cp_pool_results_coverage",
                                       "expanded_v3_chunk_*",
                                       "cp_pool_rssz_summary_chunk*.csv"))),
    ("coverage_ipu", _glob(os.path.join(OUT_ROOT, "cp_ipu_results_coverage",
                                        "cp_ipu_rssz_summary_chunk*.csv"))),
    ("composition_sa", _glob(os.path.join(OUT_ROOT,
                                          "cp_pool_results_composition",
                                          "expanded_v3_chunk_*",
                                          "cp_pool_rssz_summary_chunk*.csv"))),
    ("composition_ipu", _glob(os.path.join(OUT_ROOT,
                                           "cp_ipu_results_composition",
                                           "cp_ipu_rssz_summary_chunk*.csv"))),
    ("base_sa", _glob(os.path.join(OUT_ROOT, "cp_pool_results_base",
                                   "expanded_v3_chunk_*",
                                   "cp_pool_rssz_summary_chunk*.csv"))),
    ("base_ipu", _glob(os.path.join(OUT_ROOT, "cp_ipu_results_base",
                                    "cp_ipu_rssz_summary_chunk*.csv"))),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="*", default=None,
                    help="score only these arm names")
    ap.add_argument("--latex", action="store_true",
                    help="print rows in the paper table's column order")
    args = ap.parse_args()

    cohort = _cohort()
    wanted = set(args.arm) if args.arm else None
    rows = []
    for name, paths in ARMS:
        if wanted and name not in wanted:
            continue
        df = load_arm(paths, name)
        scored = score(df, name, cohort)
        if scored.empty:
            print(f"  {name}: no complete output yet")
            continue
        rows.append(scored)
        print(f"  {name}: {scored.zone.nunique()} zones, "
              f"rssz19 mean {scored.rssz19.mean():.4f} median {scored.rssz19.median():.4f}  |  "
              f"rssz11 mean {scored.rssz11.mean():.4f} median {scored.rssz11.median():.4f}")

    if not rows:
        print("Nothing to score.", file=sys.stderr)
        return 1
    zones = pd.concat(rows, ignore_index=True)
    summary = (zones.groupby("arm")[["rssz19", "rssz11"]]
               .agg(["count", "mean", "median"]))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.rename(columns={"rssz19_count": "count"}).reset_index()
    summary = summary.drop(columns=["rssz11_count"])
    out_dir = os.path.join(OUT_ROOT, "paper_arms")
    os.makedirs(out_dir, exist_ok=True)
    zones.to_csv(os.path.join(out_dir, "paper_arms_zones.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "paper_arms_summary.csv"), index=False)

    pools = {
        "current": os.path.join(OUT_ROOT, "feasible_pool.csv"),
        "large": os.path.join(OUT_ROOT, "feasible_pool_expanded_large_hhts.csv"),
        "fixed": os.path.join(OUT_ROOT, "feasible_pool_expanded_fixed_hhts.csv"),
        "coverage": os.path.join(OUT_ROOT,
                                 "feasible_pool_expanded_fixed_coverage_hhts.csv"),
        "composition": os.path.join(OUT_ROOT,
                                    "feasible_pool_expanded_composition_hhts.csv"),
    }
    print("\nPool composition")
    for label, path in pools.items():
        stats = pool_stats(path)
        if stats:
            print(f"  {label}: {stats}")

    if args.latex:
        label = {"sa_only": "SA-Only", "ipu_only": "IPU-Only",
                 "cp_sa": "CP-pool+SA", "cp_ipu": "CP-pool+IPU",
                 "large_sa": "Large-pool+SA", "large_ipu": "Large-pool+IPU",
                 "fixed_sa": "Large-aware+SA", "fixed_ipu": "Large-aware+IPU"}
        print("\nLaTeX rows (rssz19)")
        for _, row in summary.iterrows():
            print(f"{label.get(row.arm, row.arm):<16} & {int(row['count'])} & "
                  f"{row['rssz19_mean']:.3f} & {row['rssz19_median']:.3f} \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

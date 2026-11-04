"""
Reporting scope
===============
Which zones count toward the statistics reported in the paper, and which are
synthesised only so that the output population stays complete.

Why this exists
---------------
A handful of SA2s in the study area have almost no residents (0-80 people:
airports, industrial estates, parkland).  Their control tables still have
categories, so they still produce a zone file, but every fit statistic computed
on them is noise:

  * RSSZ is a chi-square-type statistic scaled by 1/expected, so a zone with
    3 persons has expected counts of well under 1 per cell.
  * TVD on such a zone is dominated by which single category happens to be
    present -- measured across the current benchmark, zones under 500 persons
    have mean TVD ~0.51-0.60 against ~0.19-0.25 everywhere else, and a median
    of exactly 0.50 (half the mass in the wrong cell).
  * SRMSE divides by max(expected, 1), so it explodes: the same small zones
    carry SRMSE in the hundreds for SA-GAN while the rest of the run sits in
    double digits.

Those zones are not evidence about the method, and averaging them in shifts the
reported means (and distorts SRMSE by more than 20%) for reasons that have
nothing to do with the model.

The two decisions are deliberately separate
-------------------------------------------
  synthesise every zone        -- so the population exported to
                                  Z:/test_runs_tours (new_pop, the tour model's
                                  input) covers the full study area and every
                                  household can be linked downstream;
  report only zones >= floor   -- so the paper's tables, figures and
                                  significance of differences are computed on
                                  zones large enough for the statistic to mean
                                  something.

`reporting_scope()` answers the second question only.  Nothing here removes a
zone from a run, a results directory or an export.

Choosing the floor
------------------
`intended_persons` is the census person count for the zone, taken as the
largest per-zone total across the "full universe" person control tables (see
`_full_universe_tables`).  In the current study area the zone sizes are:

    0, 3, 3, 4, 9, 14, 17, 18, 19, 25, 32, 78 | 391, 602, 724, 851 | 1089, ...

There is a clear gap below ~400 persons, so any floor between roughly 400 and
850 selects the same set.  The default (config.MIN_REPORT_PERSONS = 500) sits
in the middle of that plateau rather than on either edge.

`397979799` is the "no usual address / migratory" pseudo-zone.  It is not a
place, so it is always excluded from reporting regardless of its size.

Usage
-----
    from reporting_scope import zone_person_totals, reporting_scope, filter_reported

    totals = zone_person_totals(control_tables)          # from load_all_controls
    scope  = reporting_scope(control_tables, min_persons=500)
    reported_df, exclusions = filter_reported(metrics_df, scope)

CLI (writes the scope table so a paper table can cite it):

    python reporting_scope.py [--min-persons 500] [--control-dir ...] [--out ...]

To filter any existing per-zone result table (an RSSZ summary, rssz_by_zone.csv,
a metrics file) without touching the script that produced it:

    python reporting_scope.py --filter-csv outputs/cp_pool_results/cp_pool_rssz_summary.csv

which writes `<name>_reported.csv` next to it and prints the statistic before
and after the floor.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config

# Zone codes that are not places.  397979799 is the ABS "migratory / no usual
# address" pseudo-SA2; it has residents in the marginals but no geography, so it
# can never be a reported study area.
NON_GEOGRAPHIC_ZONES = {"397979799"}

# A table counts the full person universe when its national total is within this
# fraction of the largest person total seen.  The full-universe tables
# (indp/hrsp/qalfp/typp/occp/mtwp) agree to ~0.02%; tables built on a restricted
# universe (age5p/sexp/heap cover persons 15+ only, ~84% of the total; strd
# covers a quarter of it) fall well outside.
_FULL_UNIVERSE_TOL = 0.01


def _normalise_zone(series):
    return (series.astype(str).str.strip()
            .str.replace(r"\.0$", "", regex=True))


def _table_zone_totals(ct, zone_level="sa2"):
    """Per-zone sum of `count` for one loaded control table (or None)."""
    df = ct.get("data")
    if df is None or "count" not in df.columns:
        return None
    zone_col = ct.get("zone_col") or zone_level
    if zone_col not in df.columns:
        zone_col = zone_level
    if zone_col not in df.columns:
        return None
    counts = pd.to_numeric(df["count"], errors="coerce")
    out = counts.groupby(_normalise_zone(df[zone_col])).sum()
    # Keep genuine zeros: a zone whose every control category is empty is the
    # strongest possible case for exclusion, so it must not vanish here.
    return out if len(out) else None


def _full_universe_tables(control_tables, zone_level="sa2"):
    """Names of the control tables that count the whole person universe.

    Calibrated from the data rather than hardcoded: the biggest national person
    total defines the universe, and every table within `_FULL_UNIVERSE_TOL` of
    it counts the same people.  Household tables sit at roughly 40% of the
    person total and are rejected by the same test.
    """
    totals = {}
    for name, ct in control_tables.items():
        g = _table_zone_totals(ct, zone_level=zone_level)
        if g is not None:
            totals[name] = float(g.sum())
    if not totals:
        return [], {}
    top = max(totals.values())
    if top < 1e5:
        print(f"  [WARN] largest control-table total is only {top:,.0f}; "
              "person/household universes may be indistinguishable")
    keep = sorted(n for n, v in totals.items() if v >= top * (1.0 - _FULL_UNIVERSE_TOL))
    return keep, totals


def zone_person_totals(control_tables, zone_level="sa2"):
    """Intended census population per zone.

    Returns a Series indexed by zone code.  For each zone the value is the
    largest count across the full-universe person tables, so a zone is never
    under-counted because one table happens to omit a category.
    """
    names, _ = _full_universe_tables(control_tables, zone_level=zone_level)
    if not names:
        raise SystemExit(
            "reporting_scope: no full-universe person control table found; "
            "cannot size the zones.  Check SA_CONTROL_DIR."
        )
    frames = []
    for name in names:
        g = _table_zone_totals(control_tables[name], zone_level=zone_level)
        if g is not None:
            frames.append(g.rename(name))
    totals = pd.concat(frames, axis=1).max(axis=1)
    totals.name = "intended_persons"
    return totals.sort_values(ascending=False)


def reporting_scope(control_tables, min_persons=None, zone_level="sa2"):
    """Per-zone reporting decision.

    Columns: zone, intended_persons, reported (bool), reason.
    `reason` is empty for reported zones and names the exclusion rule otherwise.
    """
    if min_persons is None:
        min_persons = getattr(config, "MIN_REPORT_PERSONS", 500)
    totals = zone_person_totals(control_tables, zone_level=zone_level)
    out = pd.DataFrame({"zone": totals.index.astype(str),
                        "intended_persons": totals.values})
    out["reported"] = True
    out["reason"] = ""
    non_geo = out["zone"].isin(NON_GEOGRAPHIC_ZONES)
    too_small = out["intended_persons"] < int(min_persons)
    out.loc[too_small, "reason"] = f"below_{int(min_persons)}_persons"
    out.loc[non_geo, "reason"] = "non_geographic_zone"
    out.loc[too_small | non_geo, "reported"] = False
    out = out.sort_values(["reported", "intended_persons"],
                          ascending=[True, True]).reset_index(drop=True)
    out.attrs["min_persons"] = int(min_persons)
    return out


def filter_reported(metrics_df, scope):
    """Split a benchmark metrics table into the reported subset and exclusions.

    `metrics_df` needs a `zone` column (any dtype; it is normalised).  Returns
    (reported_df, exclusions_df) where exclusions_df lists every zone dropped,
    with its intended size and the rule that dropped it.  The reported frame
    also gains an `intended_persons` column so tables can report the floor.
    """
    if metrics_df is None or metrics_df.empty:
        return metrics_df, pd.DataFrame()
    df = metrics_df.copy()
    df["zone"] = _normalise_zone(df["zone"])
    sc = scope.copy()
    sc["zone"] = _normalise_zone(sc["zone"])
    lookup = sc.set_index("zone")["intended_persons"]
    df["intended_persons"] = df["zone"].map(lookup)
    keep = set(sc.loc[sc["reported"], "zone"])
    reported = df[df["zone"].isin(keep)].copy()
    exclusions = sc.loc[~sc["reported"],
                        ["zone", "intended_persons", "reason"]].copy()
    missing = sorted(set(df["zone"]) - set(sc["zone"]))
    if missing:
        # A zone file the control tables cannot size cannot be certified as
        # large enough to report on, so it is excluded -- loudly, because a
        # missing size usually means a zone/control-table mismatch rather than
        # a genuinely unsized area.
        print(f"  [WARN] {len(missing)} zone(s) present in the metrics have no "
              f"census size and were excluded from the reported set: {missing[:5]}"
              + (" ..." if len(missing) > 5 else ""))
        exclusions = pd.concat([
            exclusions,
            pd.DataFrame({"zone": missing,
                          "intended_persons": np.nan,
                          "reason": "no_census_size"}),
        ], ignore_index=True)
    return reported, exclusions


def summarise(scope, metrics_df=None):
    """Human-readable summary of the scope (and of its effect, if metrics given)."""
    lines = []
    n_all = len(scope)
    n_rep = int(scope["reported"].sum())
    floor = scope.attrs.get("min_persons", "?")
    lines.append(f"Reporting floor: {floor} persons")
    lines.append(f"Zones total: {n_all}   reported: {n_rep}   "
                 f"excluded: {n_all - n_rep}")
    excluded = scope.loc[~scope["reported"]]
    if len(excluded):
        sizes = ["?" if pd.isna(v) else str(int(v))
                 for v in excluded["intended_persons"]]
        lines.append(f"Excluded sizes (persons): {sizes}")
    if metrics_df is not None and not metrics_df.empty:
        rep, _ = filter_reported(metrics_df, scope)
        if not rep.empty:
            lines.append(f"Metrics rows kept: {len(rep):,}/{len(metrics_df):,} "
                         f"({100*len(rep)/len(metrics_df):.1f}%)")
    return "\n".join(lines)


def filter_csv(path, scope, stat_cols=None):
    """Apply the reporting scope to an existing per-zone CSV.

    Writes `<stem>_reported.csv` beside the input and returns (all, reported)
    frames.  A zone column named `zone` or `Zone` is required.
    """
    df = pd.read_csv(path)
    zone_col = next((c for c in df.columns if c.lower() == "zone"), None)
    if zone_col is None:
        raise SystemExit(f"{path}: no 'zone' column, cannot apply the reporting scope")
    df = df.rename(columns={zone_col: "zone"})
    reported, exclusions = filter_reported(df, scope)
    stem, ext = os.path.splitext(path)
    out = f"{stem}_reported{ext or '.csv'}"
    reported.to_csv(out, index=False)
    exl_path = f"{stem}_reporting_exclusions.csv"
    exclusions.to_csv(exl_path, index=False)

    if stat_cols is None:
        stat_cols = ["rssz_total", "rssz"]
    cols = [c for c in stat_cols
            if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
    print(f"\n{os.path.basename(path)}: {len(df)} zones -> {len(reported)} reported "
          f"({len(df) - len(reported)} excluded)")
    for c in cols:
        v_all = pd.to_numeric(df[c], errors="coerce").dropna()
        v_rep = pd.to_numeric(reported[c], errors="coerce").dropna()
        if not len(v_rep):
            continue
        print(f"  {c:12s} all: mean={v_all.mean():.4f} median={v_all.median():.4f} "
              f"(n={len(v_all)})   reported: mean={v_rep.mean():.4f} "
              f"median={v_rep.median():.4f} (n={len(v_rep)})")
    print(f"  wrote {out}")
    print(f"  wrote {exl_path}")
    return df, reported


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-persons", type=int, default=None,
                    help="Reporting floor in persons (default: config.MIN_REPORT_PERSONS)")
    ap.add_argument("--control-dir", default=None,
                    help="Control-table directory (default: config.SA_CONTROL_DIR)")
    ap.add_argument("--zone-level", choices=["sa2", "sa1"], default="sa2")
    ap.add_argument("--out", default=None,
                    help="Output CSV (default: outputs/reporting_scope/zone_reporting_scope.csv)")
    ap.add_argument("--filter-csv", action="append", default=None, metavar="CSV",
                    help="Apply the scope to an existing per-zone CSV (repeatable); "
                         "writes <name>_reported.csv alongside it")
    a = ap.parse_args()

    import benchmark_margins as bm
    control_dir = a.control_dir or (config.SA_CONTROL_DIR if a.zone_level == "sa2"
                                    else config.SA1_CONTROL_DIR)
    print(f"Loading control tables from: {control_dir}")
    control_tables = bm.load_all_controls(control_dir, zone_level=a.zone_level)
    if not control_tables:
        raise SystemExit("No control tables loaded.")

    names, totals = _full_universe_tables(control_tables, zone_level=a.zone_level)
    print(f"Full-universe person tables ({len(names)}): {', '.join(names)}")

    scope = reporting_scope(control_tables, min_persons=a.min_persons,
                            zone_level=a.zone_level)

    out = a.out or os.path.join(config.OUTPUTS_DIR, "reporting_scope",
                                "zone_reporting_scope.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    scope.to_csv(out, index=False)

    print()
    print(summarise(scope))
    print(f"\nWrote {len(scope)} zones -> {out}")
    print("\nExcluded zones:")
    ex = scope.loc[~scope["reported"]]
    if ex.empty:
        print("  (none)")
    else:
        for _, r in ex.iterrows():
            pid = r["intended_persons"]
            pid_s = "    ?" if pd.isna(pid) else f"{int(pid):>5}"
            print(f"  {r['zone']}  persons={pid_s}  {r['reason']}")
    if a.filter_csv:
        for path in a.filter_csv:
            if not os.path.exists(path):
                print(f"\n[SKIP] {path} not found")
                continue
            filter_csv(path, scope)

    print("\nNote: excluded zones are still synthesised and still exported to the "
          "travel model;\n      this table only governs which zones enter reported "
          "statistics.")


if __name__ == "__main__":
    main()

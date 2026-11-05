"""Add CP-MIP to the held-out figure suite.

The held-out pass in make_fig_heldout.py predates the one-MIP arm, so the TVD
figure and the category panels carry four arms.  This script scores CP-MIP on
the same 367-zone cohort and the same seven held-out tables, appends it to the
released per-zone metrics, and redraws the summary figure and category panels
with five arms.

Outputs (overwrites the four-arm versions, keeping a _4arm backup):
  outputs/paper_figures/heldout_metrics_all_arms.csv
  outputs/paper_figures/heldout_tvd_summary.csv
  outputs/paper_figures/heldout_category_shares.csv
  outputs/paper_figures/fig_heldout_tvd.png
  outputs/paper_figures/fig_heldout_categories_*.png
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from full_cohort_audit import cohort, map_files
from make_fig_heldout import (ALL_TABLES, ARM_LABELS, FIG_DIR, _census,
                              _counts, _read_zone, _tvd, load_all_controls,
                              panels_from_shares, write_panels)
from make_fig_all_variables import ARM_ORDER, plot_grouped

ARM = "cp_mip"
ARM_LABEL = ARM_LABELS[ARM]
COLOURS = {k: c for k, _, c in ARM_ORDER}
LABELS = {k: l for k, l, _ in ARM_ORDER}
KEYS = ["sa_only", "ipu_only", "cp_sa", "cp_ipu", ARM]


def main():
    ctrl_dir = os.path.join(HERE, "data_sa2_controls_heldout")
    tables = load_all_controls(ctrl_dir)
    for k, ct in tables.items():
        ct["key"] = k.replace("sa2", "", 1)
    tables = {ct["key"]: ct for ct in tables.values()}

    zones = cohort()
    print(f"cohort: {len(zones)} zones", flush=True)
    files = map_files(os.path.join(HERE, "outputs", "cp_mip_results"), ARM)
    print(f"cp_mip files: {len(files)}", flush=True)

    metrics_path = os.path.join(FIG_DIR, "heldout_metrics_all_arms.csv")
    old = pd.read_csv(metrics_path)
    old["zone"] = old["zone"].astype(str)
    old = old[old.arm != ARM]
    allowed = old.dropna(subset=["tvd"]).groupby("table")["zone"].apply(set)

    census = {}
    rows = []
    totals = {t: {} for t in ALL_TABLES}
    zone_hits = {t: set() for t in ALL_TABLES}

    for i, z in enumerate(zones, 1):
        census[z] = {t: _census(tables[t], t, z) for t in ALL_TABLES}
        path = files.get(z)
        if path is None:
            continue
        try:
            df = _read_zone(path)
        except Exception as exc:
            print(f"  [read fail] {z}: {exc}", flush=True)
            continue
        for table in ALL_TABLES:
            obs = _counts(df, table)
            exp = census[z].get(table)
            if not obs or not exp:
                continue
            rows.append({"zone": z, "table": table, "arm": ARM,
                         "tvd": _tvd(obs, exp),
                         "n_census": int(sum(exp.values())),
                         "n_synth": int(sum(obs.values()))})
            if z in allowed.get(table, set()):
                zone_hits[table].add(z)
                for c, n in obs.items():
                    # Category keys are integers for some held-out tables and
                    # strings for others; normalise so the share join below
                    # matches the label written to the CSV.
                    key = str(c).strip()
                    totals[table][key] = totals[table].get(key, 0) + n
        if i % 50 == 0:
            print(f"  {i}/{len(zones)} zones", flush=True)

    new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    # Keep the released four-arm file for reference.
    for name in ("heldout_metrics_all_arms.csv",
                 "heldout_tvd_summary.csv",
                 "heldout_category_shares.csv"):
        src = os.path.join(FIG_DIR, name)
        bak = src.replace(".csv", "_4arm.csv")
        if os.path.exists(src) and not os.path.exists(bak):
            shutil.copy2(src, bak)
    new.to_csv(metrics_path, index=False)

    summary = (new.groupby(["table", "arm"])["tvd"]
                  .agg(["mean", "median", "count"]))
    summary.to_csv(os.path.join(FIG_DIR, "heldout_tvd_summary.csv"))
    mean = summary["mean"].unstack("arm").reindex(
        index=ALL_TABLES, columns=KEYS)
    print("\nmean held-out TVD (367-zone cohort):")
    print(mean.round(4).to_string())

    # Summary figure: the six tables every arm supports.
    arm_tables = [t for t in ALL_TABLES if t != "hcfmf"]
    sub = mean.loc[arm_tables]
    series = [(LABELS[k], sub[k].fillna(0.0).values, COLOURS[k])
              for k in KEYS]
    out = os.path.join(FIG_DIR, "fig_heldout_tvd.png")
    plot_grouped(arm_tables, series, out,
                 "Held-out control tables: mean TVD (367-zone cohort)",
                 xlabels=arm_tables, ylabel="Mean TVD", height=4.6,
                 dpi=300, pct_axis=False)
    print("wrote", out)

    # Category shares: add the CP-MIP column on the same zone set the other
    # arms use for that table, then rebuild the stacked panels.
    shares_path = os.path.join(FIG_DIR, "heldout_category_shares.csv")
    shares = pd.read_csv(shares_path)
    added = []
    for table in ALL_TABLES:
        ok = zone_hits[table]
        tot = totals[table]
        denom = sum(tot.values()) or 1.0
        sub_sh = shares[shares.table == table].copy()
        sub_sh[f"{ARM}_share"] = sub_sh["category"].map(
            lambda c: tot.get(str(c).strip(), 0.0) / denom)
        added.append(sub_sh)
        print(f"  {table}: cp_mip shares over {len(ok)} zones")
    shares = pd.concat(added, ignore_index=True)
    shares.to_csv(shares_path, index=False)

    write_panels(panels_from_shares(shares_path), 300)


if __name__ == "__main__":
    main()

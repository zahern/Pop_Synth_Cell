"""Held-out control tables: census versus every arm on a matched cohort.

All four non-generative arms are scored on the 368 SA2 zones they share.  Two
category-key defects in the earlier held-out pass are repaired here:

  age5p   the control table's labels were flattened to band lower bounds
          (0, 5, ... 37) by the generic loader; they are mapped back to the
          ABS 5-year labels before comparison, and the synthetic side is
          derived from the agep band codes.  The previous 1.000 TVD was a
          text-vs-integer artefact, not a synthesis result.
  tenlld  the control table stores "Not applicable" as text while the
          population uses code 9; the two are aligned.

hcfmf cannot be shown for the two raw arms: neither population carries
families_in_household on the matched cohort.  The three constructed-pool arms
do carry it, so that table is plotted for the census and those three arms only.

Writes:
  outputs/paper_figures/fig_heldout_<table>.png          one panel per table
  outputs/paper_figures/fig_heldout_categories_person.png    age5p, assnp, lfsp
  outputs/paper_figures/fig_heldout_categories_dwelling.png  bedrd, hcfmf
  outputs/paper_figures/fig_heldout_categories_household.png hhcd, tenlld
  outputs/paper_figures/fig_heldout_tvd.png
  outputs/paper_figures/heldout_category_shares.csv
  outputs/paper_figures/heldout_metrics_all_arms.csv
  outputs/paper_figures/heldout_tvd_summary.csv
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from benchmark_margins import load_all_controls, _derived_obs
from make_marginal_histograms import (ARMS as ARM_DIRS, find_zone_files,
                                      census_zone_counts, _norm_cat)
from make_fig_all_variables import ARM_ORDER, plot_grouped, draw_grouped, FIG_DIR
from matplotlib.patches import Patch

CENSUS_COLOUR = "#9E9E9E"
ARM_LABELS = {k: label for k, label, _ in ARM_ORDER}
ARM_COLOURS = {k: colour for k, _, colour in ARM_ORDER}
ARM_KEYS = [k for k, _, _ in ARM_ORDER]
CP_KEYS = ["cp_sa", "cp_ipu"]

ALL_TABLES = ["age5p", "assnp", "bedrd", "hhcd", "lfsp", "tenlld", "hcfmf"]
ALL_ARM_TABLES = [t for t in ALL_TABLES if t != "hcfmf"]

AGE5P_ORDER = [
    "0-4 years", "5-9 years", "10-14 years", "15-19 years", "20-24 years",
    "25-29 years", "30-34 years", "35-39 years", "40-44 years", "45-49 years",
    "50-54 years", "55-59 years", "60-64 years", "65-69 years", "70-74 years",
    "75-79 years", "80-84 years", "85 years and over",
]
AGE5P_CODE = {
    0: "0-4 years", 5: "5-9 years", 10: "10-14 years", 15: "15-19 years",
    20: "20-24 years", 25: "25-29 years", 26: "30-34 years", 27: "35-39 years",
    28: "40-44 years", 29: "45-49 years", 30: "50-54 years", 31: "55-59 years",
    32: "60-64 years", 33: "65-69 years", 34: "70-74 years", 35: "75-79 years",
    36: "80-84 years", 37: "85 years and over",
}

ORDER = {
    "assnp": [1, 2, 3, 4, 7],
    "bedrd": [0, 1, 2, 3, 4, 5, 6, 7],
    "hhcd": [1, 2, 3, 4, 5, 6, 7],
    "lfsp": [1, 2, 3, 4, 5, 6],
    "tenlld": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}
LABELS = {
    "assnp": {1: "1", 2: "2", 3: "3", 4: "4", 7: "Not applicable"},
    "bedrd": {0: "None", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5+",
              6: "Not stated", 7: "Not applicable"},
    "hhcd": {1: "One family", 2: "Two+ families", 3: "Lone person",
             4: "Group", 5: "Visitors only", 6: "Other",
             7: "Not applicable"},
    "lfsp": {1: "Employed", 2: "Unemployed", 3: "Not in labour force",
             4: "Not stated", 5: "Not applicable", 6: "Overseas visitor"},
    "tenlld": {**{i: str(i) for i in range(1, 12)}, 9: "Not applicable"},
}
HCFMF_ORDER = ["One family household", "Two family household",
               "Three or more family household", "Not applicable"]
HCFMF_LABELS = {"One family household": "One family",
                "Two family household": "Two families",
                "Three or more family household": "Three+ families",
                "Not applicable": "Not applicable"}
ATTR_LABELS = {
    "age5p": "Age in five-year groups",
    "assnp": "Need for assistance",
    "bedrd": "Bedrooms",
    "hhcd": "Household composition",
    "lfsp": "Labour force status",
    "tenlld": "Landlord type",
    "hcfmf": "Family household composition",
}


COLS = {"abshid", "agep", "assnp", "bedrd", "hhcd", "lfsp", "tenlld",
        "families_in_household"}

# Tables whose unit of observation is the DWELLING, not the person.  Synthetic
# populations repeat dwelling attributes on every member row, so counting rows
# overstates them by the mean household size (here ~2.3x) and inflates their
# held-out RSSZ.  Deduplicate to one row per abshid before counting.
HH_TABLES = {"bedrd", "hcfmf", "hhcd", "tenlld"}


def _read_zone(path):
    # Only the held-out source columns are needed.  Restricting the parser here
    # is ~15x faster than reading the full 100+ column population file.
    df = pd.read_csv(path, usecols=lambda c: str(c).lower() in COLS,
                     low_memory=False)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _counts(df, table):
    # Dwelling attributes are repeated on every member row; count one row per
    # household for household-level tables (bedrd, hcfmf, hhcd, tenlld).
    if table in HH_TABLES and "abshid" in df.columns:
        df = df.drop_duplicates("abshid")
    if table in ("age5p", "hcfmf"):
        got = _derived_obs(df, table)
        return got if got else None
    if table not in df.columns:
        return None
    s = df[table].dropna()
    if s.empty:
        return None
    return {_norm_cat(c): int(n) for c, n in s.value_counts().items()}


def _census(ct, table, zone):
    got = census_zone_counts(ct, zone)
    out = {}
    for (_, cat), n in got.items():
        if table == "age5p":
            cat = AGE5P_CODE.get(cat, cat)
        elif table == "tenlld" and cat == "Not applicable":
            cat = 9
        elif table == "assnp" and cat == "Not applicable":
            cat = 7
        out[cat] = out.get(cat, 0) + n
    return out


def _tvd(obs, exp):
    keys = set(obs) | set(exp)
    so = sum(obs.values()) or 1.0
    se = sum(exp.values()) or 1.0
    return 0.5 * sum(abs(obs.get(k, 0) / so - exp.get(k, 0) / se) for k in keys)


def _ordered(table, present):
    if table == "age5p":
        base = [c for c in AGE5P_ORDER if c in present]
        return base + sorted(c for c in present if c not in base)
    if table == "hcfmf":
        base = [c for c in HCFMF_ORDER if c in present]
        return base + sorted(c for c in present if c not in base)
    order = [c for c in ORDER[table] if c in present]
    rest = sorted((c for c in present if c not in order),
                  key=lambda c: (0, float(c)) if isinstance(c, (int, float)) else (1, 0))
    return order + rest


def _short_age(label):
    s = str(label)
    if s.endswith(" years and over"):
        return s.replace(" years and over", "+")
    return s.replace(" years", "")


def _xlabels(table, cats):
    if table == "age5p":
        return [_short_age(c) for c in cats]
    if table == "hcfmf":
        return [HCFMF_LABELS.get(c, str(c)) for c in cats]
    labels = LABELS.get(table, {})
    out = []
    for c in cats:
        if c in labels:
            out.append(labels[c])
            continue
        try:
            out.append(labels.get(int(c), str(c)))
        except (TypeError, ValueError):
            out.append(str(c))
    return out


def _share(counts, total):
    return {c: n / total for c, n in counts.items()} if total else {}


def _tick_space(labels, fontsize, rot):
    """Vertical room a rotated tick label needs, in inches.

    The labels are drawn under the axis, so the space reserved for them must be
    at least their rendered extent; otherwise the longest label runs into the
    title of the panel below (which is how ``Two+ families`` came to be printed
    through ``(b) Landlord type``).  Character width is approximated at 0.58 em
    for the DejaVu Sans used by matplotlib, with a little padding.
    """
    width = 0.58 * max(len(str(label)) for label in labels) * fontsize / 72.0
    if rot == 90:
        return width + 0.10
    # Right-aligned labels at rot degrees hang down-left from the tick; the
    # vertical extent is the text width projected onto the y axis plus one
    # line of glyph height.
    return (width * np.sin(np.deg2rad(rot)) + fontsize / 72.0 + 0.10)


def compose(panels, order, out_path, dpi, height_ratios, hspace=None,
            font_scale=1.35):
    """Stack the per-table category panels into one appendix figure.

    Every panel gets its own block of vertical space: title, axes, then the
    rotated tick labels.  The layout is computed in inches because the labels
    are the tallest element and their length differs by panel; a fixed
    ``hspace`` plus ``tight_layout`` cannot reserve that (and warns that the
    axes are incompatible with it).
    """
    rows = [t for t in order if t in panels]
    if not rows:
        return None

    tick_fs = 7 * font_scale
    axes_h = [1.5 * r for r in height_ratios]
    rots, label_h = [], []
    for table in rows:
        n_cats = len(panels[table]["cats"])
        rot = 90 if n_cats > 10 else 30
        rots.append(rot)
        label_h.append(_tick_space(_xlabels(table, panels[table]["cats"]),
                                   tick_fs, rot))

    title_h = 0.34
    top_margin, bottom_margin = 0.42, 0.10
    blocks = [title_h + a + l for a, l in zip(axes_h, label_h)]
    fig_h = sum(blocks) + top_margin + bottom_margin
    fig = plt.figure(figsize=(9.6, fig_h))

    left, right = 0.115, 0.985
    width = right - left
    y_top = 1.0 - top_margin / fig_h
    for i, table in enumerate(rows):
        y_top -= title_h / fig_h
        h = axes_h[i] / fig_h
        ax = fig.add_axes([left, y_top - h, width, h])
        y_top -= h + label_h[i] / fig_h
        d = panels[table]
        draw_grouped(ax, d["cats"], d["series"],
                     title=(f"({chr(97 + i)}) {ATTR_LABELS[table]} "
                            f"({d['zones']} zones)"),
                     xlabels=_xlabels(table, d["cats"]),
                     font_scale=font_scale, legend=False, title_pad=6,
                     rot=rots[i])

    handles = [Patch(facecolor=colour, edgecolor="black", linewidth=0.35,
                     label=label)
               for label, _, colour in panels[rows[0]]["series"]]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0 - 0.06 / fig_h), ncol=len(handles),
               frameon=False, fontsize=9 * font_scale,
               handlelength=1.2, columnspacing=1.0, handletextpad=0.45)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_path)}")
    return out_path


def panels_from_shares(path):
    """Rebuild the panel data from heldout_category_shares.csv (fast path)."""
    df = pd.read_csv(path)
    panels = {}
    for table, sub in df.groupby("table", sort=False):
        cats = list(dict.fromkeys(sub["category"].astype(str)))
        one = sub.drop_duplicates("category").set_index("category")
        cens = one["census_share"].reindex(cats).fillna(0.0)
        series = [("Census", cens.values, CENSUS_COLOUR)]
        for key in ARM_KEYS:
            col = f"{key}_share"
            if col not in sub.columns or sub[col].isna().all():
                continue
            s = one[col].reindex(cats).fillna(0.0)
            series.append((ARM_LABELS[key], s.values, ARM_COLOURS[key]))
        panels[table] = {"cats": cats, "series": series,
                         "zones": int(sub["n_zones"].iloc[0])}
    return panels


def write_panels(panels, dpi):
    """Write the stacked appendix figures from the per-table panel data."""
    compose(panels, ["age5p", "assnp", "lfsp"],
            os.path.join(FIG_DIR, "fig_heldout_categories_person.png"),
            dpi, height_ratios=[2.3, 1.0, 1.15])
    compose(panels, ["bedrd", "hcfmf"],
            os.path.join(FIG_DIR, "fig_heldout_categories_dwelling.png"),
            dpi, height_ratios=[1.45, 1.0])
    compose(panels, ["hhcd", "tenlld"],
            os.path.join(FIG_DIR, "fig_heldout_categories_household.png"),
            dpi, height_ratios=[1.25, 1.55])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--from-shares", action="store_true",
                    help="rebuild only the stacked panels from "
                         "heldout_category_shares.csv (no zone rescan)")
    args = ap.parse_args()

    if args.from_shares:
        shares = os.path.join(FIG_DIR, "heldout_category_shares.csv")
        if not os.path.exists(shares):
            raise SystemExit(f"missing {shares}")
        write_panels(panels_from_shares(shares), args.dpi)
        return

    tables = load_all_controls(os.path.join(HERE, "data_sa2_controls_heldout"))
    for k, ct in tables.items():
        ct["key"] = k.replace("sa2", "", 1)
    tables = {ct["key"]: ct for ct in tables.values()}
    print(f"control tables: {sorted(tables)}")

    files = {arm: find_zone_files(ARM_DIRS[arm]) for arm in ARM_KEYS}
    shared = sorted(set.intersection(*[set(f) for f in files.values()]))
    # Restrict to the paper's 367-zone fitted cohort (371-zone scope minus the
    # four zones with no usable individual target), so the table and figures
    # match the cohort used everywhere else.
    cohort_path = os.path.join(HERE, "pbs", "zones_371.txt")
    if os.path.exists(cohort_path):
        scope = {ln.strip() for ln in open(cohort_path) if ln.strip()}
        excluded = {"301031014", "304041102", "310021279", "311031315"}
        shared = sorted(set(shared) & (scope - excluded))
    print(f"matched cohort: {len(shared)} zones")

    census = {}
    for i, zone in enumerate(shared, 1):
        census[zone] = {t: _census(tables[t], t, zone) for t in ALL_TABLES}
        if i % 100 == 0:
            print(f"  census: {i}/{len(shared)} zones", flush=True)

    agg = {}
    for arm in ARM_KEYS:
        agg[arm] = {}
        for i, zone in enumerate(shared, 1):
            df = _read_zone(files[arm][zone])
            agg[arm][zone] = {t: _counts(df, t) for t in ALL_TABLES}
            if i % 100 == 0:
                print(f"  {arm}: {i}/{len(shared)} zones", flush=True)

    metric_rows = []
    share_rows = []
    panels = {}

    def analyse(table, table_arms, title):
        zones = [z for z in shared
                 if census[z].get(table)
                 and all(agg[a][z].get(table) for a in table_arms)]
        if not zones:
            print(f"  [skip] {table}: no common zones with the attribute")
            return None
        for arm in table_arms:
            for z in zones:
                exp = census[z][table]
                obs = agg[arm][z][table]
                metric_rows.append({
                    "zone": z, "table": table, "arm": arm,
                    "tvd": _tvd(obs, exp),
                    "n_census": int(sum(exp.values())),
                    "n_synth": int(sum(obs.values())),
                })

        totals = {a: defaultdict(float) for a in ["census"] + table_arms}
        for z in zones:
            for c, n in census[z][table].items():
                totals["census"][c] += n
            for arm in table_arms:
                for c, n in agg[arm][z][table].items():
                    totals[arm][c] += n

        present = set().union(*(set(totals[k]) for k in totals))
        cats = [c for c in _ordered(table, present)
                if sum(totals[k].get(c, 0) for k in totals) > 0]
        # Drop categories that are invisible at this scale (below 0.05% in
        # every series); they remain in the per-zone metrics CSV.
        cats = [c for c in cats
                if max(totals[k].get(c, 0) / max(sum(totals[k].values()), 1)
                       for k in totals) >= 0.0005]
        cens = _share(totals["census"], sum(totals["census"].values()))
        series = [("Census", np.array([cens.get(c, 0.0) for c in cats]),
                   CENSUS_COLOUR)]
        for arm in table_arms:
            s = _share(totals[arm], sum(totals[arm].values()))
            series.append((ARM_LABELS[arm],
                           np.array([s.get(c, 0.0) for c in cats]),
                           ARM_COLOURS[arm]))

        out = os.path.join(FIG_DIR, f"fig_heldout_{table}.png")
        plot_grouped(cats, series, out,
                     f"{table}: {ATTR_LABELS[table]} (held-out, {len(zones)} zones)",
                     xlabels=_xlabels(table, cats), height=4.6,
                     dpi=args.dpi)
        print(f"  wrote {os.path.basename(out)}  ({len(zones)} zones, "
              f"{len(table_arms)} arms)")
        panels[table] = {"cats": cats, "series": series, "zones": len(zones)}
        for c in cats:
            row = {"table": table, "category": c, "census_share": cens.get(c, 0.0),
                   "n_zones": len(zones)}
            for arm in table_arms:
                row[f"{arm}_share"] = _share(
                    totals[arm], sum(totals[arm].values())).get(c, 0.0)
            share_rows.append(row)
        return zones

    for table in ALL_ARM_TABLES:
        analyse(table, ARM_KEYS, table)

    # hcfmf: only the constructed-pool arms carry families_in_household.
    analyse("hcfmf", CP_KEYS, "hcfmf")

    # Stacked appendix panels: person-carried tables, then the dwelling and
    # household tables split over two figures so the category labels do not
    # collide.  Each panel carries the census target and every arm the table
    # supports, in the same order and colours as Figure 4.
    write_panels(panels, args.dpi)

    pd.DataFrame(share_rows).to_csv(
        os.path.join(FIG_DIR, "heldout_category_shares.csv"), index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(FIG_DIR, "heldout_metrics_all_arms.csv"),
                   index=False)
    summary = (metrics.groupby(["table", "arm"])["tvd"]
                      .agg(["mean", "median", "count"]))
    summary.to_csv(os.path.join(FIG_DIR, "heldout_tvd_summary.csv"))
    mean = summary["mean"].unstack("arm").reindex(index=ALL_TABLES,
                                                  columns=ARM_KEYS)
    print("\nmean held-out TVD (matched cohort):")
    print(mean.round(3).to_string())

    # Summary figure: the six tables every arm supports.
    sub = mean.loc[ALL_ARM_TABLES]
    series = [(ARM_LABELS[k], sub[k].fillna(0.0).values, ARM_COLOURS[k])
              for k in ARM_KEYS]
    out = os.path.join(FIG_DIR, "fig_heldout_tvd.png")
    plot_grouped(ALL_ARM_TABLES, series, out,
                 "Held-out control tables: mean TVD (matched 368-zone cohort)",
                 xlabels=ALL_ARM_TABLES, ylabel="Mean TVD", height=4.6,
                 dpi=args.dpi, pct_axis=False)
    print(f"  wrote {os.path.basename(out)}")


if __name__ == "__main__":
    main()

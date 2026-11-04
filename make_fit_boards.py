#!/usr/bin/env python3
"""make_fit_boards.py -- "cell board" figures for the fitted populations.

The pool boards in make_paper_boards.py draw what the pipeline *supplies*
(which control cells the pool can reach).  These two draw what the fitters
*achieve* with it, in the same board idiom, for the four completed arms of
main_edit3.tex (SA-Only, IPU-Only, CP-pool+SA, CP-pool+IPU):

  fig_board_fit.pdf
      One board per arm, one row per shared control table, one column per
      zone in the 367-zone fitted cohort.  A square's shade is that zone's
      RSSZ for that table, so a dark band is a table an arm fails on across
      many zones rather than in one.  Rows are split into person tables and
      household tables, and each arm carries a bar of its per-table mean, so
      the household-vs-individual question is readable off one page.

  fig_board_fit_cell.pdf
      One zone in detail.  Rows are the same eleven tables, a square is a
      category, and each category carries four squares, one per arm.  A
      square states whether that arm hit the census count for the cell, fell
      short, overshot, left a positive target empty, or invented a category
      the census does not record there.  This is the cell-level statement the
      RSSZ numbers aggregate.

Sources
  outputs/paper_arms/paper_arms_zones.csv     per arm x zone x table RSSZ
                                             (the basis of tab:pool_fitter_matrix)
  outputs/{sa_only_results,ipu_only_results,cp_pool_results/expanded_v3_chunk_*,
           cp_ipu_results}/zone_<sa2>...(csv) fitted populations, for the cell board
  data_sa2_controls/sa2<attr>.csv             census control tables

Usage
  python make_fit_boards.py                       # both boards, copy into the paper
  python make_fit_boards.py --zone 301011004      # different worked example
  python make_fit_boards.py --no-cells            # zone board only (fast)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config
from benchmark_margins import load_all_controls
from make_marginal_histograms import aggregate_zone, find_zone_files

PAPER_DIR = os.path.normpath(os.path.join(HERE, "..", "Population_Synthesiser___GAN_SA"))
OUTDIR = os.path.join(config.OUTPUTS_DIR, "paper_figures")
ZONES_CSV = os.path.join(config.OUTPUTS_DIR, "paper_arms", "paper_arms_zones.csv")

SHARED11 = ["heap", "mstp", "mtwp", "occp", "rlhp", "sexp",
            "hind", "mrerd", "nprd", "rntrd", "vehrd"]
PERSON_TABLES = ["heap", "mstp", "mtwp", "occp", "rlhp", "sexp"]
HH_TABLES = ["hind", "mrerd", "nprd", "rntrd", "vehrd"]

TABLE_LABEL = {
    "heap": "heap  educational attainment",
    "mstp": "mstp  marital status",
    "mtwp": "mtwp  method of travel to work",
    "occp": "occp  occupation",
    "rlhp": "rlhp  relationship in household",
    "sexp": "sexp  sex",
    "hind": "hind  household income",
    "mrerd": "mrerd  mortgage repayment",
    "nprd": "nprd  persons per dwelling",
    "rntrd": "rntrd  rent",
    "vehrd": "vehrd  vehicles per dwelling",
}

# Arm order is the paper's table order, and it is used for the four sub-squares
# in the cell board so the columns of the two figures agree.
ARM_ORDER = ["sa_only", "ipu_only", "cp_sa", "cp_ipu"]
ARM_LABEL = {"sa_only": "SA-Only", "ipu_only": "IPU-Only",
             "cp_sa": "CP-pool+SA", "cp_ipu": "CP-pool+IPU"}
ARM_ROOTS = {
    "sa_only": [config.SA_ONLY_RESULTS_DIR],
    "ipu_only": [config.IPU_ONLY_RESULTS_DIR],
    "cp_sa": sorted(__import__("glob").glob(
        os.path.join(config.CP_POOL_RESULTS_DIR, "expanded_v3_chunk_*"))),
    "cp_ipu": [config.CP_IPU_RESULTS_DIR],
}

# RSSZ bands.  The bulk of the per-zone values sit below 0.05, so the bands are
# geometric: a linear ramp would put four fifths of the board in one shade.
BANDS = [0.0, 0.01, 0.05, 0.15, 0.50, 1.50, np.inf]
BAND_LABELS = ["$<0.01$", "$0.01$--$0.05$", "$0.05$--$0.15$",
               "$0.15$--$0.50$", "$0.50$--$1.5$", "$>1.5$"]
BAND_COLOURS = ["#f2f5f4", "#c7dcd2", "#8fbfa6", "#e0b062", "#c1732c", "#8c2f27"]

# Cell states for the detail board.
STATE_COLOURS = {
    "hit": "#3f8f5f",       # within 10% of the census count
    "close": "#a9cfa5",     # within 30%
    "short": "#e0b062",     # under by more than 30%
    "over": "#c1732c",      # over by more than 30%
    "empty": "#8c2f27",     # positive target, nothing in the population
    "invented": "#7d3a8c",  # population carries a category the census does not
    "none": "#f4f4f4",      # no target and nothing emitted
}
STATE_LABELS = {
    "hit": "cell reproduced ($\\leq$10% of the census count)",
    "close": "within 30%",
    "short": "short by more than 30%",
    "over": "over by more than 30%",
    "empty": "positive target, no record emitted",
    "invented": "category emitted where the census has none",
    "none": "no target, nothing emitted",
}

INK = "#1a1a1a"
EDGE = "#b9b9b9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 7.5,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
    "text.color": INK,
})


# --------------------------------------------------------------------------- #
# The zone x table board
# --------------------------------------------------------------------------- #
def load_zone_board(verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Per arm x zone x table RSSZ, in one frame ordered by mean RSSZ."""
    if not os.path.exists(ZONES_CSV):
        raise SystemExit(f"missing {ZONES_CSV}: run score_paper_arms.py first")
    df = pd.read_csv(ZONES_CSV)
    df["zone"] = df["zone"].astype(str).str.strip()
    df = df[df["arm"].isin(ARM_ORDER)]
    missing = [t for t in SHARED11 if t not in df.columns]
    if missing:
        raise SystemExit(f"{ZONES_CSV} has no columns for {missing}")

    # Zones are ordered by the four-arm mean, so the panels share one column
    # order and a dark patch can be read across arms.
    per_zone = (df.groupby("zone")["rssz11"].mean().sort_values())
    order = per_zone.index.tolist()
    if verbose:
        counts = df.groupby("arm")["zone"].nunique().to_dict()
        print(f"  [zones] {len(order)} zones; " +
              ", ".join(f"{ARM_LABEL[a]} {counts.get(a, 0)}" for a in ARM_ORDER))
    return df, order


def figure_zone_board(df: pd.DataFrame, order: list[str],
                      out_pdf: str, out_png: str) -> dict:
    nz = len(order)
    rows = PERSON_TABLES + HH_TABLES
    nrow = len(rows)

    arr = {}
    for arm in ARM_ORDER:
        sub = (df[df["arm"] == arm].set_index("zone")
               .reindex(order)[SHARED11])
        arr[arm] = sub.to_numpy(dtype=float).T          # (nrow, nz)

    cmap = ListedColormap(BAND_COLOURS)
    cmap.set_bad(NOT_SCORED)
    norm = BoundaryNorm(BANDS, cmap.N)

    # A table whose per-zone values are all exactly zero across the cohort is
    # not a perfect fit: it is a table the run did not score and defaulted to
    # zero.  Such rows are masked and drawn grey instead of in the best band.
    unscored = {}
    for arm in ARM_ORDER:
        a = arr[arm]
        rows_off = [rows[i] for i in range(nrow) if np.all(a[i] == 0)]
        if rows_off:
            unscored[arm] = rows_off

    row_h = 0.132
    panel_h = nrow * row_h
    head_h = 0.34
    left, heat_w, bar_w, gap = 0.95, 4.30, 0.46, 0.09
    fig_w = left + heat_w + gap + bar_w + 0.10
    fig_h = 0.10 + len(ARM_ORDER) * (head_h + panel_h + 0.20) + 0.72

    fig = plt.figure(figsize=(fig_w, fig_h))

    # One shared bar scale, so a table's bar is comparable across arms.
    all_means = np.concatenate([np.nanmean(arr[a], axis=1) for a in ARM_ORDER])
    bar_max = float(np.nanmax(all_means)) * 1.08

    y = fig_h - 0.10
    for panel, arm in enumerate(ARM_ORDER):
        y -= head_h
        zone_tot = np.nansum(arr[arm], axis=0)
        fig.text(left / fig_w, (y + 0.06) / fig_h, ARM_LABEL[arm],
                 fontsize=7.6, fontweight="bold", ha="left", va="bottom")
        note = ""
        if arm in unscored:
            note = (f"   ({len(rows) - len(unscored[arm])} of {len(rows)} tables "
                    f"scored)")
        fig.text((left + 1.30) / fig_w, (y + 0.06) / fig_h,
                 f"mean zone total {np.nanmean(zone_tot):.3f}"
                 f"   median {np.nanmedian(zone_tot):.3f}" + note,
                 fontsize=6.5, ha="left", va="bottom", color="#444444")

        y_bot = y - panel_h
        ax = fig.add_axes([left / fig_w, y_bot / fig_h,
                           heat_w / fig_w, panel_h / fig_h])
        block = np.ma.masked_array(arr[arm], mask=~np.isfinite(arr[arm]))
        for i, table in enumerate(rows):
            if table in unscored.get(arm, []):
                block[i, :] = np.ma.masked
        ax.imshow(block, aspect="auto", cmap=cmap, norm=norm,
                  interpolation="nearest", extent=(0, nz, nrow, 0))
        ax.set_xticks([])
        ax.set_yticks(np.arange(nrow) + 0.5)
        ax.set_yticklabels([TABLE_LABEL[t] for t in rows] if panel == 0 else [],
                           fontsize=5.9)
        ax.tick_params(axis="y", length=0, pad=2)
        # One rule between the person block and the household block.
        ax.axhline(len(PERSON_TABLES), color="#555555", linewidth=0.8)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_linewidth(0.5)


        # Per-table mean as a bar beside the board.
        axb = fig.add_axes([(left + heat_w + gap) / fig_w, y_bot / fig_h,
                            bar_w / fig_w, panel_h / fig_h])
        means = np.nanmean(arr[arm], axis=1)
        axb.barh(np.arange(nrow) + 0.5, means, height=0.66,
                 color=ARM_COLOURS[arm], linewidth=0)
        axb.set_ylim(nrow, 0)
        axb.set_yticks([])
        axb.set_xticks([0, bar_max])
        axb.set_xticklabels(["0", f"{bar_max:.2f}"], fontsize=5.2)
        axb.tick_params(axis="x", length=1.5, pad=1)
        axb.set_xlim(0, bar_max)
        for side in ("top", "right", "left"):
            axb.spines[side].set_visible(False)
        axb.axhline(len(PERSON_TABLES), color="#555555", linewidth=0.8)
        y = y_bot - 0.20

    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none")
               for c in BAND_COLOURS]
    labels = list(BAND_LABELS)
    if unscored:
        handles.append(Rectangle((0, 0), 1, 1, facecolor=NOT_SCORED,
                                 edgecolor="none"))
        labels.append("not scored in this run")
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=6.6, handlelength=1.0,
               handleheight=0.85, columnspacing=1.0, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.30 / fig_h),
               title="per-zone RSSZ for one control table", title_fontsize=6.6)
    fig.text(0.5, 0.14 / fig_h,
             f"{nz} zones, ordered by mean RSSZ across the four arms: "
             f"best-fitting zone at the left, worst at the right",
             ha="center", va="bottom", fontsize=6.0, color="#555555")
    fig.text(0.5, 0.02 / fig_h,
             "rows 1--6 person tables, 7--11 household tables; "
             "the bars give each table's mean on one scale shared by all four boards",
             ha="center", va="bottom", fontsize=6.0, color="#555555")

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=230, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    summary = []
    for panel, arm in enumerate(ARM_ORDER):
        a = arr[arm]
        for i, table in enumerate(rows):
            summary.append({
                "arm": arm, "arm_label": ARM_LABEL[arm], "table": table,
                "group": "person" if table in PERSON_TABLES else "household",
                "mean_rssz": float(np.nanmean(a[i])),
                "median_rssz": float(np.nanmedian(a[i])),
                "p90_rssz": float(np.nanpercentile(a[i], 90)),
                "zones_above_0.5": int(np.nansum(a[i] > 0.5)),
                "zones": int(np.sum(~np.isnan(a[i]))),
            })
    print(f"  wrote {os.path.basename(out_pdf)}  ({len(ARM_ORDER)} boards, "
          f"{nrow} tables x {nz} zones)")
    return {"rows": summary}


ARM_COLOURS = {"sa_only": "#1b6ca8", "ipu_only": "#c0392b",
               "cp_sa": "#d68910", "cp_ipu": "#0e7c7b"}
NOT_SCORED = "#cfcfcf"


# --------------------------------------------------------------------------- #
# The cell board for one zone
# --------------------------------------------------------------------------- #
def _shared_tables():
    """The eleven shared control tables, keyed as the marginals code expects."""
    tables = load_all_controls(config.SA_CONTROL_DIR)
    single = {}
    for key, ct in tables.items():
        ct = dict(ct)
        ct["key"] = re.sub(r"^sa[12]", "", str(key).lower())
        if len(ct.get("attr_cols", [])) == 1:
            single[ct["key"]] = ct
    return single


def build_cells(zone: str, tables: dict, verbose: bool = True):
    """(table, category) -> census count and each arm's count, for one zone."""
    from make_marginal_histograms import census_zone_counts, _norm_cat

    census = {}
    for key in SHARED11:
        ct = tables.get(key)
        if ct is None:
            raise SystemExit(f"control table for {key} not found")
        for (_, cat), n in census_zone_counts(ct, zone).items():
            census[(key, cat)] = census.get((key, cat), 0) + int(n)

    arm_counts, arm_files = {}, {}
    for arm in ARM_ORDER:
        found = find_zone_files(ARM_ROOTS[arm])
        path = found.get(str(zone))
        if not path:
            print(f"  [skip] {ARM_LABEL[arm]}: no population for {zone}",
                  file=sys.stderr)
            continue
        got = aggregate_zone(path, {k: tables[k] for k in SHARED11})
        if not got:
            print(f"  [skip] {ARM_LABEL[arm]}: could not read {path}",
                  file=sys.stderr)
            continue
        arm_counts[arm] = {k: v for k, v in got.items() if k[0] in SHARED11}
        arm_files[arm] = path
        if verbose:
            print(f"  [{zone}] {ARM_LABEL[arm]:<12} {os.path.basename(path)}")

    cats = {}
    for (key, cat) in census:
        cats.setdefault(key, set()).add(cat)
    for got in arm_counts.values():
        for (key, cat) in got:
            cats.setdefault(key, set()).add(cat)
    for key in cats:
        cats[key] = sorted(cats[key],
                           key=lambda c: (0, c, "") if isinstance(c, (int, float))
                           else (1, 0, str(c)))
    return census, arm_counts, cats, arm_files


def _state(exp: int, got: int) -> str:
    if exp == 0 and got == 0:
        return "none"
    if exp == 0:
        return "invented"
    if got == 0:
        return "empty"
    err = abs(got - exp) / exp
    if err <= 0.10:
        return "hit"
    if err <= 0.30:
        return "close"
    return "short" if got < exp else "over"


def figure_cell_board(zone: str, census, arm_counts, cats, arm_files,
                      out_pdf: str, out_png: str) -> dict:
    """One zone, one square per (arm, category).

    The four arms are stacked inside each category rather than placed side by
    side: abreast, the widest table (occupation) makes the board forty inches
    wide, and a square that small cannot be read.  Stacked, the board is a
    couple of inches across and the four arms read down a column.
    """
    rows = [r for r in (PERSON_TABLES + HH_TABLES) if r in cats]
    arms = [a for a in ARM_ORDER if a in arm_counts]

    # Categories that neither the census nor any arm populates carry no
    # information and would only widen the board.
    kept, dropped = {}, 0
    for table in rows:
        keep = []
        for cat in cats[table]:
            exp = census.get((table, cat), 0)
            if exp or any(arm_counts[a].get((table, cat), 0) for a in arms):
                keep.append(cat)
            else:
                dropped += 1
        kept[table] = keep
    rows = [t for t in rows if kept[t]]

    sq, cat_gap, row_gap = 0.072, 0.022, 0.105
    label_w = 1.05
    row_h = len(arms) * sq
    board_w = max(len(kept[t]) for t in rows) * (sq + cat_gap)
    board_h = len(rows) * (row_h + row_gap)
    legend_h = 1.05
    foot_h = 0.46
    fig_w = label_w + board_w + 0.08
    fig_h = foot_h + board_h + legend_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([label_w / fig_w, (foot_h + legend_h) / fig_h,
                       board_w / fig_w, board_h / fig_h])
    ax.set_xlim(0, board_w)
    ax.set_ylim(board_h, 0)
    ax.axis("off")

    counts = {a: {s: 0 for s in STATE_COLOURS} for a in arms}
    y = 0.0
    for group, block in (("person", [r for r in rows if r in PERSON_TABLES]),
                         ("household", [r for r in rows if r in HH_TABLES])):
        if not block:
            continue
        for table in block:
            x = 0.0
            for cat in kept[table]:
                exp = census.get((table, cat), 0)
                for j, arm in enumerate(arms):
                    got = arm_counts[arm].get((table, cat), 0)
                    st = _state(exp, got)
                    counts[arm][st] += 1
                    # Fill gives the fit state, the border the arm, so a
                    # square is identifiable without counting rows.
                    ax.add_patch(Rectangle(
                        (x, y + j * sq), sq * 0.88, sq * 0.88,
                        facecolor=STATE_COLOURS[st],
                        edgecolor=ARM_COLOURS[arm], linewidth=0.45))
                x += sq + cat_gap
            ax.text(-0.07, y + row_h / 2, TABLE_LABEL.get(table, table),
                    ha="right", va="center", fontsize=5.8)
            y += row_h + row_gap
        y += 0.03


    handles, labels = [], []
    for st in ("hit", "close", "short", "over", "empty", "invented", "none"):
        handles.append(Rectangle((0, 0), 1, 1, facecolor=STATE_COLOURS[st],
                                 edgecolor=STATE_COLOURS[st]
                                 if st not in ("none",) else EDGE,
                                 linewidth=0.3))
        labels.append(STATE_LABELS[st])
    for arm in arms:
        handles.append(Rectangle((0, 0), 1, 1, facecolor="white",
                                 edgecolor=ARM_COLOURS[arm], linewidth=1.2))
        labels.append(ARM_LABEL[arm])
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=5.9, handlelength=0.9, handleheight=0.8,
               columnspacing=0.9, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.205),
               title="fill: how the cell fits      outline: which arm",
               title_fontsize=6.0)

    tally = "   ".join(
        f"{ARM_LABEL[a]}  {counts[a]['hit']}/{sum(len(kept[t]) for t in rows)} hit"
        for a in arms)
    fig.text(0.5, 0.072, tally, ha="center", va="bottom", fontsize=6.0,
             color="#444444")
    fig.text(0.5, 0.046,
             f"zone {zone}: {len(rows)} tables, {sum(len(kept[t]) for t in rows)} "
             f"categories carrying a census count or an emitted record"
             + (f" ({dropped} padding cells omitted)" if dropped else ""),
             ha="center", va="bottom", fontsize=5.8, color="#555555")
    fig.text(0.5, 0.024,
             "States are the per-cell relative error; RSSZ squares the deviation "
             "over target cells only and divides by the zone total,",
             ha="center", va="bottom", fontsize=5.6, color="#777777")
    fig.text(0.5, 0.006,
             "so a table can carry many amber cells at a small RSSZ, "
             "and a violet cell lies outside the target set altogether.",
             ha="center", va="bottom", fontsize=5.6, color="#777777")

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_pdf)}  ({zone}, {len(rows)} tables, "
          f"{sum(len(cats[t]) for t in rows)} cells)")
    return counts


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="112011236",
                    help="zone for the cell board (default: the paper's worked example)")
    ap.add_argument("--no-zones", action="store_true", help="skip the zone x table board")
    ap.add_argument("--no-cells", action="store_true", help="skip the cell board")
    ap.add_argument("--no-copy", action="store_true",
                    help="leave the PDFs in outputs/paper_figures only")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    made, summary_rows = [], []

    if not a.no_zones:
        df, order = load_zone_board()
        pdf = os.path.join(OUTDIR, "fig_board_fit.pdf")
        info = figure_zone_board(df, order, pdf,
                                 os.path.join(OUTDIR, "fig_board_fit.png"))
        summary_rows.extend(info["rows"])
        made.append(pdf)

    if not a.no_cells:
        tables = _shared_tables()
        census, arm_counts, cats, arm_files = build_cells(str(a.zone), tables)
        if not arm_counts:
            print("  no populations for the cell board", file=sys.stderr)
        else:
            pdf = os.path.join(OUTDIR, "fig_board_fit_cell.pdf")
            figure_cell_board(str(a.zone), census, arm_counts, cats, arm_files,
                              pdf,
                              os.path.join(OUTDIR, "fig_board_fit_cell.png"))
            made.append(pdf)

    if summary_rows:
        path = os.path.join(OUTDIR, "fit_board_summary.csv")
        pd.DataFrame(summary_rows).to_csv(path, index=False)
        print(f"  wrote {path}")

    if made and not a.no_copy:
        dest = os.path.join(PAPER_DIR, "files")
        if os.path.isdir(dest):
            for path in made:
                shutil.copy2(path, dest)
                print(f"  copied {os.path.basename(path)} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

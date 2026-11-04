#!/usr/bin/env python3
"""make_paper_boards.py -- exportable "cell board" figures for the paper.

Two vector figures, rendered straight from whatever job output is on disk, so
they can be regenerated as new shards land instead of being redrawn by hand:

  fig_board_pool.pdf
      One board, one square per (control table, category) cell, one row per
      table.  Squares are coloured by provenance: kept from an observed census
      record, written in by CP construction, or written in by a zone's MIP.
      The point of the figure is the keep-and-add reading: the pool is larger
      than the extract because squares are written in, not because records are
      removed.

  fig_pool_matching.pdf
      Companion panel: sample households as columns, control cells as rows,
      the fitter's binary selection x_h, and the SA2 target bars E_{z,k,m}.
      It shows how the board's cell supply is used to match the marginals.

  fig_board_expansion.pdf
      One square per control cell with a positive census target that the zone's
      pool slice cannot supply at all (a zero-capacity cell).
      Top   : one zone in detail; green = the joint selection covered the cell,
              amber = left uncovered, labelled with the cell.
      Bottom: every audited zone so far, one row each.

  files/expansion_totals.tex
      The same full-run totals as LaTeX macros, so the paragraph that quotes
      them and the board that draws them cannot drift apart.

Sources
  outputs/feasible_pool.csv                     census + CP rows, `_source`
  outputs/feasible_pool_expanded_v3.csv         rows tagged `mip_joint`
  outputs/cp_expansion_shards/report_*.json     per-zone expansion reports
  outputs/cp_expansion_zero_priority/report_*.json
  outputs/full10_joint_report.json              released ten-zone audit

Usage
  python make_paper_boards.py                 # render both, copy into paper/files
  python make_paper_boards.py --no-pool       # expansion board only (fast)
  python make_paper_boards.py --watch 300     # regenerate whenever a source changes
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import statistics
import sys
import time

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.normpath(os.path.join(HERE, "..", "Population_Synthesiser___GAN_SA"))
OUTDIR = os.path.join(HERE, "outputs", "paper_figures")
CACHE = os.path.join(OUTDIR, "board_cells_cache.json")

POOL = os.path.join(HERE, "outputs", "feasible_pool.csv")
EXPANDED = os.path.join(HERE, "outputs", "feasible_pool_expanded_v3.csv")
FULL10 = os.path.join(HERE, "outputs", "full10_joint_report.json")
# The current cohort is the 20-shard run. Older report directories describe
# earlier pools and introduce zones outside the 367-zone cohort (for example
# 301031014), so they are only a fallback when no shard report exists.
REPORT_GLOBS = [
    os.path.join(HERE, "outputs", "expansion_shards", "shard_*.report.json"),
]
LEGACY_REPORT_GLOBS = [
    os.path.join(HERE, "outputs", "cp_expansion_shards", "report_*.json"),
    os.path.join(HERE, "outputs", "cp_expansion_zero_priority", "report_*.json"),
]

# Person-level and household-level control tables carried by the construction.
PERSON_ATTRS = ["sexp", "agep", "mstp", "heap", "occp", "incp", "lfsp", "empp",
                "hrsp", "qallp", "qalfp", "rlhp", "mtwp"]
HH_ATTRS = ["nprd", "vehrd", "mrerd", "rntrd", "strd", "hind", "bedrd", "hhcd", "tend"]
ATTRS = PERSON_ATTRS + HH_ATTRS

BLUE = "#2f5d86"      # supplied by an observed census record
GREEN = "#3f8f5f"     # written in by CP construction
AMBER = "#d08512"     # written in by the zone-level MIP
AMBER_L = "#f0c882"   # zero-capacity cell left uncovered
EDGE = "#a8a8a8"
INK = "#1a1a1a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 7.5,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
    "text.color": INK,
})


# --------------------------------------------------------------------------- #
# Cell boards for the pool comparison
# --------------------------------------------------------------------------- #
def _categories(series: pd.Series) -> set:
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() >= 0.9:
        return set(num.dropna().astype(int).tolist())
    return set(series.dropna().astype(str).str.strip().tolist())


def _attr_table(path: str, attrs: list[str], sources: set[str] | None = None):
    """Return {attr: set(categories)} restricted to rows whose _source is in
    `sources`, plus the number of rows that survived."""
    wanted = {a.upper() for a in attrs} | {"_SOURCE"}
    df = pd.read_csv(path, usecols=lambda c: c.upper() in wanted, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    if sources is not None and "_source" in df.columns:
        df = df[df["_source"].astype(str).str.lower().isin(sources)]
    cells = {a: _categories(df[a]) for a in attrs if a in df.columns}
    return cells, len(df)


def load_cell_stages(verbose: bool = True) -> dict:
    """Census / CP / MIP cell sets, cached against source file mtimes."""
    key = {}
    for path in (POOL, EXPANDED):
        if os.path.exists(path):
            st = os.stat(path)
            key[os.path.basename(path)] = [int(st.st_mtime), st.st_size]
    if os.path.exists(CACHE):
        try:
            cached = json.load(open(CACHE))
            if cached.get("key") == key and "stages" in cached:
                if verbose:
                    print("  [cells] cached board cells reused")
                return {s: {a: set(v) for a, v in d.items()}
                        for s, d in cached["stages"].items()}, cached["rows"]
        except Exception:
            pass

    if not os.path.exists(POOL):
        raise SystemExit(f"pool not found: {POOL}")
    t0 = time.time()
    census, n_pool = _attr_table(POOL, ATTRS, sources={"census"})
    pool, _ = _attr_table(POOL, ATTRS)
    rows = {"pool_rows": n_pool, "census_rows": n_pool - 0}
    stages = {"census": census, "cp": pool}

    if os.path.exists(EXPANDED):
        mip, n_exp = _attr_table(EXPANDED, ATTRS, sources={"mip_joint"})
        stages["mip"] = mip
        rows["expanded_rows"] = n_exp
    else:
        stages["mip"] = {a: set() for a in ATTRS}
        rows["expanded_rows"] = 0

    # recount the census subset for the caption
    rows["census_rows"] = len(pd.read_csv(
        POOL, usecols=lambda c: c.lower() == "_source", low_memory=False
    ).query("_source == 'census'"))

    os.makedirs(OUTDIR, exist_ok=True)
    json.dump({"key": key,
               "rows": rows,
               "stages": {s: {a: sorted(v) for a, v in d.items()} for s, d in stages.items()}},
              open(CACHE, "w"))
    if verbose:
        print(f"  [cells] read pool in {time.time() - t0:.1f}s")
    return stages, rows


def _row_layout(all_cells: set) -> list[tuple[str, list]]:
    """Staircase layout: person tables, then household tables, longest first."""
    rows = []
    for group in (PERSON_ATTRS, HH_ATTRS):
        cats_by_attr = {}
        for attr in group:
            cats_by_attr[attr] = sorted({c for a, c in all_cells if a == attr})
        for attr in sorted(group, key=lambda a: (-len(cats_by_attr[a]), a)):
            rows.append((attr, cats_by_attr[attr]))
    return rows


def _draw_rows(ax, rows, state, label_fs=6.2):
    """Unit-chart board: one row per control table, one square per category,
    stacked left to right.  state(attr, cat) -> (face, edge, width)."""
    blocks = [("person tables", [r for r in rows if r[0] in PERSON_ATTRS]),
              ("household tables", [r for r in rows if r[0] not in PERSON_ATTRS])]
    y_top = 0.0
    for title, block in blocks:
        if not block:
            continue
        ax.text(-0.45, y_top - 0.5, title, ha="right", va="center", fontsize=6.2,
                style="italic", color="#555555")
        y_top -= 1.0
        for attr, cats in block:
            for j, cat in enumerate(cats):
                fc, ec, lw = state(attr, cat)
                # A small inset makes the unit cells legible as cells rather
                # than as one continuous bar when adjacent cells share a state.
                ax.add_patch(Rectangle((j + 0.06, y_top - 0.94), 0.88, 0.88,
                                       facecolor=fc, edgecolor=ec, linewidth=lw,
                                       joinstyle="miter"))
            ax.text(-0.45, y_top - 0.5, attr, ha="right", va="center",
                    fontsize=label_fs, color=INK)
            y_top -= 1.0
        y_top -= 0.45
    ax.set_xlim(-0.06, max(len(c) for _, c in rows) + 0.06)
    ax.set_ylim(y_top + 0.45, 0.05)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_sample_matching(ax) -> None:
    """Schematic: sample households -> selected/replicated set -> SA2 target.

    The board above says which control cells the pool can supply.  This panel
    says how the fitter uses that supply: each column is a sample household,
    a filled square is a cell it carries, and the row sums after selection and
    replication are compared with E_{z,k,m}.
    """
    cells = ["mstp:1", "mstp:3", "tend:6", "vehrd:6"]
    households = ["h_1", "h_2", "h_3", "h_4", "h_5"]
    # rows x columns: 1 means the household carries the control cell.
    marks = [
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
    ]
    provenance = [BLUE, BLUE, GREEN, BLUE, AMBER]
    selected = [1, 1, 1, 0, 1]
    target = [12, 8, 4, 6]

    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 5.7)
    ax.axis("off")
    ax.set_aspect("auto")

    ax.text(0.0, 5.35, "(b)  from sample households to the SA2 marginal",
            fontsize=7.0, fontweight="bold", ha="left", va="center")
    ax.text(0.0, 4.98, "sample households", fontsize=6.2, ha="left",
            va="center", color="#444444", style="italic")

    x0, dx = 2.15, 0.72
    y0, dy = 4.18, 0.78
    for j, household in enumerate(households):
        ax.text(x0 + j * dx, 4.56, f"${household}$", fontsize=6.0,
                ha="center", va="center")
        for i, row in enumerate(marks):
            ax.add_patch(Rectangle(
                (x0 + j * dx - 0.22, y0 - i * dy - 0.22), 0.44, 0.44,
                facecolor=provenance[j] if row[j] else "white",
                edgecolor=EDGE, linewidth=0.45))
    for i, cell in enumerate(cells):
        ax.text(x0 - 0.30, y0 - i * dy, cell, fontsize=6.0, ha="right",
                va="center")

    # Selection row: one binary per sample household.
    ax.text(x0 - 0.30, 0.75, r"$x_h$", fontsize=6.2, ha="right", va="center")
    for j, value in enumerate(selected):
        ax.add_patch(Rectangle((x0 + j * dx - 0.22, 0.53), 0.44, 0.44,
                               facecolor="white", edgecolor=EDGE, linewidth=0.45))
        ax.text(x0 + j * dx, 0.75, str(value), fontsize=5.8,
                ha="center", va="center")

    ax.annotate("", xy=(8.15, 2.55), xytext=(5.95, 2.55),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    ax.text(6.05, 2.72, "select, then\nreplicate $\\leq r_{\\max}$",
            fontsize=5.8, ha="left", va="bottom", color="#444444")

    # Target marginal for the same cells.
    ax.text(8.35, 4.98, "zone target", fontsize=6.2, ha="left",
            va="center", color="#444444", style="italic")
    ax.text(10.25, 4.56, r"$E_{z,k,m}$", fontsize=6.0, ha="center",
            va="center")
    bar_x, bar_max = 8.45, 2.45
    for i, value in enumerate(target):
        y = y0 - i * dy
        width = bar_max * value / max(target)
        ax.add_patch(Rectangle((bar_x, y - 0.16), width, 0.32,
                               facecolor="#dce6ef", edgecolor="#8aa2b8",
                               linewidth=0.45))
        ax.text(bar_x + width + 0.08, y, str(value), fontsize=5.9,
                ha="left", va="center")
    ax.text(0.0, 0.02,
            "Each filled square is one household's contribution $c_{h,k,m}$; "
            "the fitter chooses $x_h$ and the number of replicas so the row "
            "sums approach the target bars.",
            fontsize=5.6, ha="left", va="bottom", color="#555555")


def figure_pool(stages: dict, rows_meta: dict, out_pdf: str, out_png: str) -> None:
    all_cells = set().union(*(set((a, c) for a, v in st.items() for c in v)
                              for st in stages.values()))
    rows = _row_layout(all_cells)

    sq = 0.128
    label_w, legend_h, pad = 0.95, 0.34, 0.06
    n_units = len(rows) + 2 + 0.9               # tables + two headings + gaps
    board_w = max(len(c) for _, c in rows) * sq
    board_h = (n_units - 0.45) * sq
    fig_w = label_w + board_w + 0.14
    fig_h = pad + board_h + legend_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([label_w / fig_w, (pad + legend_h) / fig_h,
                       board_w / fig_w, board_h / fig_h])
    ax.set_title("(a)  pool supply by control cell", loc="left",
                 fontsize=7.0, fontweight="bold", pad=3)

    census, cp, mip = stages["census"], stages["cp"], stages["mip"]

    def state(attr, cat):
        if cat in census.get(attr, ()):
            return BLUE, BLUE, 0.4
        if cat in cp.get(attr, ()):
            return GREEN, GREEN, 0.4
        if cat in mip.get(attr, ()):
            return AMBER, AMBER, 0.4
        return "white", EDGE, 0.5

    _draw_rows(ax, rows, state)

    n_all = sum(len(c) for _, c in rows)
    n_census = sum(1 for (a, c) in all_cells if c in census.get(a, ()))
    n_cp = sum(1 for (a, c) in all_cells if c in cp.get(a, ()) and c not in census.get(a, ()))
    n_mip = sum(1 for (a, c) in all_cells if c in mip.get(a, ()) and c not in cp.get(a, ()))

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor=BLUE, linewidth=0.4),
        Rectangle((0, 0), 1, 1, facecolor=GREEN, edgecolor=GREEN, linewidth=0.4),
        Rectangle((0, 0), 1, 1, facecolor=AMBER, edgecolor=AMBER, linewidth=0.4),
    ]
    labels = [f"kept from an observed census record ({n_census} of {n_all} cells)",
              f"written in by CP construction (+{n_cp})",
              f"written in by the zone MIP (+{n_mip})"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=6.9, handlelength=1.0, handleheight=1.0,
               columnspacing=1.2, handletextpad=0.5,
               bbox_to_anchor=(0.5, (pad * 0.3) / fig_h))

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_pdf)}  "
          f"(census {n_census}/{n_all} cells, CP +{n_cp}, MIP +{n_mip})")


def figure_pool_matching(out_pdf: str, out_png: str) -> None:
    """Standalone panel (b): sample households, cells and the SA2 target."""
    fig = plt.figure(figsize=(6.6, 1.55))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    _draw_sample_matching(ax)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_pdf)}")


def load_zone_reports(verbose: bool = True):
    """Newest per-zone expansion record across every report on disk."""
    files = []
    for pattern in REPORT_GLOBS:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        for pattern in LEGACY_REPORT_GLOBS:
            files.extend(sorted(glob.glob(pattern)))
        if os.path.exists(FULL10):
            files.append(FULL10)

    zones: dict[str, dict] = {}
    meta = {}
    for path in files:
        try:
            st = os.stat(path)
            doc = json.load(open(path))
        except Exception:
            continue
        meta[path] = st.st_mtime
        for rec in doc.get("zone_reports", []) or []:
            if str(rec.get("status", "ok")).lower() != "ok":
                continue
            zone = str(rec.get("zone", "")).strip()
            if not zone:
                continue
            rec = dict(rec)
            rec["_mtime"] = st.st_mtime
            prev = zones.get(zone)
            if prev is None or rec["_mtime"] > prev["_mtime"]:
                zones[zone] = rec
    if verbose:
        print(f"  [zones] {len(zones)} audited zones from {len(files)} report(s)")
    return zones


def aggregate_zones(zones: dict) -> dict:
    """Full-run totals over every audited zone (and the uncovered-cell census)."""

    def n(rec, key):
        try:
            return int(rec.get(key, 0) or 0)
        except Exception:
            return 0

    zeros = [n(r, "n_zero_cells") for r in zones.values()]
    covered = [n(r, "n_zero_cells_covered") for r in zones.values()]
    # The audit's "added" set is the whole appended menu, not only the
    # coverage-minimal selection; prefer the transplant totals when present.
    hhs = [n(r, "n_transplant_households") or n(r, "n_selected_households")
           for r in zones.values()]
    persons = [n(r, "n_transplant_rows") or n(r, "n_selected_persons")
               for r in zones.values()]
    dev_sum = [float(r.get("marginal_deviation_sum") or 0.0) for r in zones.values()
               if r.get("marginal_deviation_sum") is not None]
    dev_mean = [float(r.get("marginal_deviation_mean") or 0.0) for r in zones.values()
                if r.get("marginal_deviation_mean") is not None]
    uncovered = {}
    for rec in zones.values():
        for cell in (rec.get("uncovered_cells") or []):
            uncovered[str(cell)] = uncovered.get(str(cell), 0) + 1
    tot_zero = sum(zeros)
    tot_cov = sum(covered)

    def stat(values, fn):
        return fn(values) if values else 0

    return {
        "zones": len(zones),
        "optimal": sum(1 for r in zones.values()
                       if str(r.get("solver", "")).lower() == "optimal"),
        "zero_cells": tot_zero,
        "covered": tot_cov,
        "uncovered": tot_zero - tot_cov,
        "cover_pct": round(100.0 * tot_cov / tot_zero) if tot_zero else 0,
        "households": sum(hhs),
        "persons": sum(persons),
        "zero_min": stat(zeros, min), "zero_med": stat(zeros, statistics.median),
        "zero_max": stat(zeros, max),
        "hh_med": stat(hhs, statistics.median), "hh_max": stat(hhs, max),
        "persons_max": stat(persons, max),
        "dev_sum_min": min(dev_sum) if dev_sum else 0.0,
        "dev_sum_med": statistics.median(dev_sum) if dev_sum else 0.0,
        "dev_sum_max": max(dev_sum) if dev_sum else 0.0,
        "dev_mean_min": min(dev_mean) if dev_mean else 0.0,
        "dev_mean_med": statistics.median(dev_mean) if dev_mean else 0.0,
        "dev_mean_max": max(dev_mean) if dev_mean else 0.0,
        "uncovered_by_cell": dict(sorted(uncovered.items(),
                                         key=lambda kv: (-kv[1], kv[0]))),
    }


def _texnum(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return f"{int(value):,}".replace(",", "{,}")


def _texstat(value) -> str:
    """A median: whole numbers plain, halves kept to one decimal."""
    value = float(value)
    if value.is_integer():
        return _texnum(int(value))
    return f"{value:.1f}"


def write_totals_tex(totals: dict, path: str) -> None:
    """Write the numbers the prose quotes, so text and board cannot disagree."""
    cells = list(totals["uncovered_by_cell"].items())
    labels = [f"\\texttt{{{c}}} ({_texnum(k)})" for c, k in cells]
    if len(labels) > 1:
        breakdown = ", ".join(labels[:-1]) + " and " + labels[-1]
    else:
        breakdown = labels[0] if labels else "none"

    lines = [
        "% Generated by make_paper_boards.py -- do not edit by hand.",
        "% Full-run totals for the pool-expansion audit. Regenerate after new",
        "% shards land:  cd synth_pop && python make_paper_boards.py --no-pool",
    ]
    for name, value in (
        ("ExpZones", totals["zones"]),
        ("ExpOptimalSolves", totals["optimal"]),
        ("ExpZeroCells", totals["zero_cells"]),
        ("ExpCoveredCells", totals["covered"]),
        ("ExpUncoveredCells", totals["uncovered"]),
        ("ExpCoverPct", totals["cover_pct"]),
        ("ExpAddedHouseholds", totals["households"]),
        ("ExpAddedPersons", totals["persons"]),
        ("ExpZeroMin", totals["zero_min"]),
        ("ExpZeroMedian", int(totals["zero_med"])),
        ("ExpZeroMax", totals["zero_max"]),
        ("ExpMedianHouseholds", int(totals["hh_med"])),
        ("ExpMaxHouseholds", totals["hh_max"]),
        ("ExpMaxPersons", totals["persons_max"]),
        ("ExpDevSumMin", float(totals["dev_sum_min"])),
        ("ExpDevSumMedian", float(totals["dev_sum_med"])),
        ("ExpDevSumMax", float(totals["dev_sum_max"])),
        ("ExpDevMeanMin", float(totals["dev_mean_min"])),
        ("ExpDevMeanMedian", float(totals["dev_mean_med"])),
        ("ExpDevMeanMax", float(totals["dev_mean_max"])),
    ):
        lines.append(f"\\newcommand{{\\{name}}}{{{_texnum(value)}}}")
    lines.append(f"\\newcommand{{\\ExpUncoveredCodeCount}}{{{_texnum(len(cells))}}}")
    lines.append(f"\\newcommand{{\\ExpUncoveredBreakdown}}{{{breakdown}}}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {os.path.basename(path)}  "
          f"({totals['zones']} zones, {totals['covered']:,}/{totals['zero_cells']:,} covered)")


def write_expansion_table(zones: dict, totals: dict, path: str) -> None:
    """The per-zone audit as a longtable, two zones per row."""

    def n(rec, key):
        try:
            return int(rec.get(key, 0) or 0)
        except Exception:
            return 0

    def block(zone, rec):
        return [str(zone), str(n(rec, "n_zero_cells")),
                str(n(rec, "n_zero_cells_covered")),
                str(n(rec, "n_selected_households")),
                str(n(rec, "n_selected_persons")),
                f"{float(rec.get('marginal_deviation_sum') or 0.0):.3f}"]

    recs = sorted(zones.items(), key=lambda kv: str(kv[0]))
    header = ("Zone & Zero & Cov. & HH & Pers. & Dev. & "
              "Zone & Zero & Cov. & HH & Pers. & Dev. \\\\")

    body = []
    for i in range(0, len(recs), 2):
        left = block(*recs[i])
        right = block(*recs[i + 1]) if i + 1 < len(recs) else [""] * 6
        body.append(" & ".join(left + right) + " \\\\")

    covered = [n(r, "n_zero_cells_covered") for r in zones.values()]
    hhs = [n(r, "n_selected_households") for r in zones.values()]
    devs = [float(r.get("marginal_deviation_sum") or 0.0) for r in zones.values()]
    total = ["Total", _texnum(totals["zero_cells"]), _texnum(totals["covered"]),
             _texnum(totals["households"]), _texnum(totals["persons"]),
             _texnum(sum(devs))]
    median = ["Median", _texstat(totals["zero_med"]), _texstat(statistics.median(covered)),
              _texstat(statistics.median(hhs)), _texstat(statistics.median(
                  [n(r, "n_selected_persons") for r in zones.values()])),
              f"{statistics.median(devs):.3f}"]
    body.append("\\midrule")
    body.append(" & ".join(total + [""] * 6) + " \\\\")
    body.append(" & ".join(median + [""] * 6) + " \\\\")

    caption = (
        f"Per-zone candidate-pool expansion over the {totals['zones']} zones processed so "
        "far, two zones per row. \\emph{Zero} is the number of positive-target control "
        "cells with zero attainable capacity in that zone's pool slice; \\emph{Cov.} the "
        "number the support block covered; \\emph{HH} and \\emph{Pers.} the "
        "coverage-selected households and person rows; \\emph{Dev.} the weighted joint-marginal deviation of "
        "the added set (lower is better). Every support solve returned an optimal "
        "solution. The last rows give the run totals and per-zone medians; blank entries "
        "after the final zone are padding."
    )

    lines = [
        "% Generated by make_paper_boards.py -- do not edit by hand.",
        "% Per-zone expansion audit; regenerate after new shards land:",
        "%   cd synth_pop && python make_paper_boards.py --no-pool",
        "{\\footnotesize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{longtable}{@{}lrrrrrrrrrrrr@{}}",
        f"\\caption{{{caption}}}",
        "\\label{tab:pool_expansion_audit}\\\\",
        "\\toprule",
        header,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "\\multicolumn{12}{@{}l}{\\emph{Table \\thetable\\ (continued)}}\\\\",
        header,
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{12}{r@{}}{\\emph{continued on the next page}}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ] + body + [
        "\\end{longtable}",
        "}",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {os.path.basename(path)}  ({len(recs)} zones, "
          f"{len(body) - 2} rows)")


def figure_expansion(zones: dict, out_pdf: str, out_png: str, focus: str | None = None,
                     max_rows: int | None = None) -> None:
    """The expansion board.

    (a) one zone's zero-capacity cells, green where the selected set covered
        them and amber where nothing in the slice carries them;
    (b) every cell the whole run left open, counted by control code --- the
        residual is a property of ten codes, not of ten zones.

    `max_rows` is accepted for CLI compatibility and is no longer used: the
    per-zone table in the appendix carries the zone-by-zone detail.
    """
    if not zones:
        raise SystemExit("no expansion zone reports found; nothing to draw")

    def n(rec, key):
        try:
            return int(rec.get(key, 0) or 0)
        except Exception:
            return 0

    ranked = sorted(zones.items(), key=lambda kv: (-n(kv[1], "n_zero_cells"), kv[0]))
    focus_zone, focus_rec = ranked[0]
    if focus and focus in zones:
        focus_zone, focus_rec = focus, zones[focus]

    tot_zero = sum(n(r, "n_zero_cells") for r in zones.values())
    tot_cov = sum(n(r, "n_zero_cells_covered") for r in zones.values())
    tot_hh = sum(n(r, "n_selected_households") for r in zones.values())
    tot_ps = sum(n(r, "n_selected_persons") for r in zones.values())

    cells = list(aggregate_zones(zones)["uncovered_by_cell"].items())
    codes = [c for c, _ in cells]
    counts = [k for _, k in cells]
    n_codes = max(len(codes), 1)

    nz = max(n(focus_rec, "n_zero_cells"), 1)
    ncov = min(n(focus_rec, "n_zero_cells_covered"), nz)
    uncovered = [str(c) for c in (focus_rec.get("uncovered_cells") or [])]

    fig_w = 6.2
    label_w, stat_w = 0.66, 2.15
    sq = min(0.26, (fig_w - label_w - stat_w - 0.15) / nz)
    bar_h = 0.185
    b_h = bar_h * n_codes + 0.08

    h_a_title, h_a_labels, h_b_title, h_foot = 0.17, 0.58, 0.21, 0.50
    fig_h = (h_a_title + sq + h_a_labels + 0.16 + h_b_title + b_h + h_foot)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # ---- (a) one zone in detail -------------------------------------------- #
    y_b_bottom = h_foot
    y_b_top = y_b_bottom + b_h
    y_a_labels = y_b_top + h_b_title + 0.02
    y_a_strip = y_a_labels + h_a_labels

    ax_a = fig.add_axes([label_w / fig_w, (y_a_strip - 3.2 * sq) / fig_h,
                         (nz * sq) / fig_w, (4.2 * sq) / fig_h])
    for j in range(nz):
        covered = j < ncov
        ax_a.add_patch(Rectangle((j + 0.07, 0), 0.86, 0.86,
                                 facecolor=GREEN if covered else AMBER_L,
                                 edgecolor=GREEN if covered else AMBER,
                                 linewidth=0.6))
    for k, cell in enumerate(uncovered[:max(nz - ncov, 0)]):
        ax_a.text(ncov + k + 0.5, -0.10, cell, rotation=90, ha="center", va="top",
                  fontsize=5.6, color=INK)
    ax_a.set_xlim(0, nz)
    ax_a.set_ylim(-3.2, 1.0)
    ax_a.set_aspect("equal")
    ax_a.axis("off")

    fig.text(label_w / fig_w - 0.06, (y_a_strip + 0.43 * sq) / fig_h, f"SA2 {focus_zone}",
             ha="right", va="center", fontsize=7.0)
    fig.text((label_w + nz * sq + 0.12) / fig_w, (y_a_strip + 0.43 * sq) / fig_h,
             f"{ncov} of {nz} covered by\n{n(focus_rec, 'n_selected_households')} households"
             f" ({n(focus_rec, 'n_selected_persons')} persons)",
             ha="left", va="center", fontsize=6.6, color="#444444", linespacing=1.3)
    fig.text(0.5, (y_a_strip + sq + 0.05) / fig_h,
             "(a)  the zero-capacity cells of one zone's pool slice",
             ha="center", va="bottom", fontsize=8.0)

    # ---- (b) what stayed open across the whole run -------------------------- #
    ax_b = fig.add_axes([(label_w + 0.10) / fig_w, y_b_bottom / fig_h,
                         (fig_w - label_w - 0.95) / fig_w, b_h / fig_h])
    ypos = list(range(len(codes)))
    ax_b.barh(ypos, counts, height=0.64, color=AMBER_L, edgecolor=AMBER, linewidth=0.5)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels(codes, fontsize=6.2, fontfamily="monospace")
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, max(counts + [1]) * 1.14 if counts else 1)
    ax_b.set_ylim(len(codes) - 0.4, -0.6)
    for s in ("top", "right", "bottom", "left"):
        ax_b.spines[s].set_visible(False)
    ax_b.set_xticks([])
    ax_b.tick_params(left=False, pad=2)
    for y, c in zip(ypos, counts):
        ax_b.text(c + max(counts) * 0.02, y, f"{c:,}", va="center", ha="left",
                  fontsize=6.2, color="#333333")

    fig.text(0.5, (y_b_top + 0.05) / fig_h,
             f"(b)  every cell the run left open, counted by control code",
             ha="center", va="bottom", fontsize=8.0)

    # ---- legend and run totals ---------------------------------------------- #
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=GREEN, edgecolor=GREEN, linewidth=0.5),
        Rectangle((0, 0), 1, 1, facecolor=AMBER_L, edgecolor=AMBER, linewidth=0.5),
    ]
    fig.legend(handles,
               ["zero-capacity cell covered by the selected set",
                "left open: no observed record in the pool carries it"],
               loc="lower center", ncol=2, frameon=False, fontsize=6.6,
               handlelength=1.0, handleheight=1.0, columnspacing=1.4,
               handletextpad=0.5, bbox_to_anchor=(0.5, 0.20 / fig_h))

    fig.text(0.5, 0.02 / fig_h,
             f"all {len(zones)} audited zones: {tot_cov:,} of {tot_zero:,} zero-capacity "
             f"cells covered with {tot_hh:,} households ({tot_ps:,} persons); "
             f"the {sum(counts):,} left open are all in the codes above",
             ha="center", va="bottom", fontsize=6.4, color=INK)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"  wrote {os.path.basename(out_pdf)}  "
          f"({len(zones)} zones, {tot_cov:,}/{tot_zero:,} zero cells covered, "
          f"{len(codes)} unbalanced codes)")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _sources_stamp() -> tuple:
    stamp = []
    for path in [POOL, EXPANDED, FULL10] + [p for g in REPORT_GLOBS for p in glob.glob(g)]:
        try:
            st = os.stat(path)
            stamp.append((path, int(st.st_mtime), st.st_size))
        except OSError:
            pass
    return tuple(stamp)


def render(args) -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    targets = []
    if not args.no_pool:
        stages, rows_meta = load_cell_stages()
        pdf = os.path.join(OUTDIR, "fig_board_pool.pdf")
        figure_pool(stages, rows_meta, pdf, os.path.join(OUTDIR, "fig_board_pool.png"))
        targets.append(pdf)
        match_pdf = os.path.join(OUTDIR, "fig_pool_matching.pdf")
        figure_pool_matching(match_pdf,
                             os.path.join(OUTDIR, "fig_pool_matching.png"))
        targets.append(match_pdf)
    if not args.no_expansion:
        zones = load_zone_reports()
        pdf = os.path.join(OUTDIR, "fig_board_expansion.pdf")
        figure_expansion(zones, pdf, os.path.join(OUTDIR, "fig_board_expansion.png"),
                         focus=args.zone, max_rows=args.max_rows)
        targets.append(pdf)
        if args.paper_dir:
            totals = aggregate_zones(zones)
            write_totals_tex(totals,
                             os.path.join(args.paper_dir, "files", "expansion_totals.tex"))
            write_expansion_table(zones, totals,
                                  os.path.join(args.paper_dir, "files",
                                               "pool_expansion_table.tex"))

    if args.paper_dir:
        dest = os.path.join(args.paper_dir, "files")
        os.makedirs(dest, exist_ok=True)
        for pdf in targets:
            shutil.copy2(pdf, os.path.join(dest, os.path.basename(pdf)))
            print(f"  copied -> {os.path.join(dest, os.path.basename(pdf))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-dir", default=PAPER_DIR,
                    help="paper folder to copy the PDFs into (default: %(default)s)")
    ap.add_argument("--no-paper-dir", action="store_true", help="write only to outputs/")
    ap.add_argument("--no-pool", action="store_true", help="skip the pool board")
    ap.add_argument("--no-expansion", action="store_true", help="skip the expansion board")
    ap.add_argument("--zone", default=None, help="SA2 to feature in panel (a)")
    ap.add_argument("--max-rows", type=int, default=12, help="zones drawn in panel (b)")
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                    help="re-render whenever a source file changes")
    args = ap.parse_args()
    if args.no_paper_dir:
        args.paper_dir = None

    print(f"make_paper_boards  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    render(args)
    if args.watch <= 0:
        return 0

    stamp = _sources_stamp()
    print(f"  watching every {args.watch}s ... (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(args.watch)
            current = _sources_stamp()
            if current != stamp:
                stamp = current
                print(f"\n  change detected {time.strftime('%H:%M:%S')} - re-rendering")
                try:
                    render(args)
                except SystemExit as exc:
                    print(f"  skipped: {exc}")
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

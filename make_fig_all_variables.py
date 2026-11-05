"""Per-table marginal comparison: census plus every arm, one grouped plot each.

Six series per category: the census target (grey) and the five non-generative
arms (SA-Only, IPU-Only, CP-pool+SA, CP-pool+IPU, CP-MIP).  Grouped bars rather
than overlaid curves, so over- and under-production read directly as bar height.

Reads outputs/paper_figures/marginal_category_shares.csv (written by
make_marginal_histograms.py) and writes:

  outputs/paper_figures/fig_marginals_<table>.png   one figure per control table
  outputs/paper_figures/fig_marginals_all_tables.png  mean TVD by table and arm
  outputs/paper_figures/marginal_tvd_arms.csv        the values behind that figure
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FIG_DIR = os.path.join(HERE, "outputs", "paper_figures")

# (key, label, colour).  The generative arms are deliberately absent: the
# manuscript treats generative augmentation only in the literature review, so
# no result figure carries it.
ARM_ORDER = [
    ("sa_only", "SA-Only", "#0072B2"),
    ("ipu_only", "IPU-Only", "#D55E00"),
    ("cp_sa", "CP-pool + SA", "#E69F00"),
    ("cp_ipu", "CP-pool + IPU", "#009E73"),
    ("cp_mip", "CP-MIP", "#34495E"),
]
ARM_KEYS = [a[0] for a in ARM_ORDER]
CENSUS_COLOUR = "#9E9E9E"

TABLE_ORDER = [
    "heap", "hind", "hrsp", "incp", "indp", "mrerd", "mstp", "mtwp",
    "nprd", "occp", "qalfp", "qallp", "rlhp", "rntrd", "sexp", "strd",
    "stup", "typp", "vehrd",
]

ATTR_LABELS = {
    "heap":  "Highest educational attainment",
    "hind":  "Household income (weekly)",
    "hrsp":  "Hours worked",
    "incp":  "Personal income (weekly)",
    "indp":  "Industry of employment",
    "mrerd": "Mortgage repayment (monthly)",
    "mstp":  "Marital status",
    "mtwp":  "Method of travel to work",
    "nprd":  "Dwellings by persons resident",
    "occp":  "Occupation",
    "qalfp": "Field of study",
    "qallp": "Non-school qualification",
    "rlhp":  "Relationship in household",
    "rntrd": "Rent (weekly)",
    "sexp":  "Sex",
    "strd":  "Dwelling structure",
    "stup":  "Educational institution attending",
    "typp":  "Non-private dwelling type",
    "vehrd": "Vehicles per dwelling",
}


def _sort_categories(values):
    """Numeric codes in numeric order, then any text categories."""
    def key(c):
        s = str(c)
        try:
            return (0, float(s), "")
        except ValueError:
            return (1, 0.0, s)
    return sorted(values, key=key)


def draw_grouped(ax, categories, series, title=None, xlabels=None,
                 ylabel="Share of population", pct_axis=True,
                 font_scale=1.0, legend=True, title_pad=36, rot=None):
    """Draw the grouped bars for one panel on an existing axis."""
    n_cats = len(categories)
    n_series = len(series)
    if n_cats == 0 or n_series == 0:
        return ax

    x = np.arange(n_cats)
    group_width = 0.80
    bar_width = group_width / n_series

    for i, (label, values, colour) in enumerate(series):
        offset = (i - (n_series - 1) / 2.0) * bar_width
        ax.bar(x + offset, np.asarray(values, dtype=float), width=bar_width * 0.94,
               color=colour, edgecolor="black", linewidth=0.35, label=label,
               zorder=3)

    ax.set_xlim(-0.60, n_cats - 0.40)
    ax.set_xticks(x)
    if xlabels is None:
        xlabels = [str(c) for c in categories]
    if rot is None:
        rot = 90 if n_cats > 10 else 30
    ha = "center" if rot == 90 else "right"
    ax.set_xticklabels(xlabels, rotation=rot, ha=ha, fontsize=7 * font_scale)
    ax.set_ylabel(ylabel, fontsize=9 * font_scale)
    if pct_axis:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="y", labelsize=8 * font_scale)
    ax.grid(axis="y", color="0.85", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ymax = max((np.max(v) if len(v) else 0.0) for _, v, _ in series)
    ax.set_ylim(0, max(ymax * 1.18, 1e-9))

    if legend:
        handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.35, label=l)
                   for l, _, c in series]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                  ncol=min(n_series, 6), frameon=False, fontsize=8.5 * font_scale,
                  handlelength=1.2, columnspacing=1.1, handletextpad=0.5)
    if title:
        ax.set_title(title, fontsize=10 * font_scale, pad=title_pad, loc="left")
    return ax


def plot_grouped(categories, series, out_path, title, xlabels=None,
                 ylabel="Share of population", height=4.8, dpi=300,
                 pct_axis=True, width=None, font_scale=1.0):
    """Draw one grouped-bar panel and save it.

    categories : ordered category values (only used for the x positions)
    series     : list of (label, values, colour), in plotting order
    """
    n_cats = len(categories)
    n_series = len(series)
    if n_cats == 0 or n_series == 0:
        return None

    width = width or max(8.0, 0.80 * n_cats + 3.0)
    fig, ax = plt.subplots(figsize=(width, height))
    draw_grouped(ax, categories, series, title=title, xlabels=xlabels,
                 ylabel=ylabel, pct_axis=pct_axis, font_scale=font_scale)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def table_series(table, df, code_order=None, labels=None):
    """Build (categories, xlabels, series) for one control table."""
    sub = df[(df["table"] == table) & (df["arm"].isin(ARM_KEYS))].copy()
    if sub.empty:
        return None

    # The shares CSV stores category keys as strings (some tables carry text
    # categories), so normalise the requested code order to the same dtype
    # before reindexing.  Otherwise an integer code_order silently selects
    # nothing and every bar plots as zero.
    sub["category"] = sub["category"].astype(str)
    if code_order is not None:
        code_order = [str(c) for c in code_order]

    present = [a for a in ARM_ORDER if a[0] in set(sub["arm"])]
    if code_order is None:
        code_order = _sort_categories(sub["category"].unique())

    census = (sub.drop_duplicates("category")
                 .set_index("category")["census_share"]
                 .reindex(code_order).fillna(0.0))

    series = [("Census", census.values, CENSUS_COLOUR)]
    for key, label, colour in present:
        s = (sub[sub["arm"] == key].set_index("category")["synth_share"]
             .reindex(code_order).fillna(0.0))
        series.append((label, s.values, colour))

    if labels is None:
        labels = [str(c) for c in code_order]
    return code_order, labels, series


def plot_table(table, df, out_path, code_order=None, labels=None,
               title=None, height=4.8, dpi=300, width=None, font_scale=1.0):
    built = table_series(table, df, code_order=code_order, labels=labels)
    if built is None:
        print(f"  [skip] {table}: no rows for the requested arms")
        return None
    cats, xlabels, series = built
    title = title or f"{table}: {ATTR_LABELS.get(table, table)}"
    path = plot_grouped(cats, series, out_path, title, xlabels=xlabels,
                        height=height, dpi=dpi, width=width,
                        font_scale=font_scale)
    if path:
        tvds = {label: 0.5 * float(np.sum(np.abs(np.asarray(vals) - np.asarray(series[0][1]))))
                for label, vals, _ in series[1:]}
        tv = ", ".join(f"{k} {v:.3f}" for k, v in tvds.items())
        print(f"  wrote {os.path.basename(path)}  [{tv}]")
    return path


def plot_overview(df, out_path, dpi=300):
    """TVD per control table, one horizontal bar per arm.

    Horizontal bars keep all nineteen table names legible when the figure is
    included at text width; a 19-group vertical chart would be either very wide
    or very cramped.
    """
    d = df[df["arm"].isin(ARM_KEYS)].copy()
    if d.empty:
        return None
    piv = d.pivot_table(index="table", columns="arm", values="tvd", aggfunc="mean")
    tables = [t for t in TABLE_ORDER if t in piv.index]
    tables += [t for t in piv.index if t not in tables]
    piv = piv.reindex(tables)

    series = []
    for key, label, colour in ARM_ORDER:
        if key in piv.columns:
            series.append((label, piv[key].fillna(0.0).values, colour))

    n = len(tables)
    fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.34 * n + 1.9)))
    y = np.arange(n)
    group_height = 0.80
    bar_h = group_height / max(len(series), 1)
    for i, (label, vals, colour) in enumerate(series):
        offset = (i - (len(series) - 1) / 2.0) * bar_h
        ax.barh(y + offset, np.asarray(vals, dtype=float), height=bar_h * 0.94,
                color=colour, edgecolor="black", linewidth=0.35, label=label,
                zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(tables, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("TVD (pooled shares)", fontsize=9)
    ax.set_xlim(0, max(0.55, float(np.nanmax(piv.values)) * 1.15))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="0.85", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.35, label=l)
               for l, _, c in series]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
              ncol=3, frameon=False, fontsize=8.5,
              handlelength=1.2, columnspacing=1.1, handletextpad=0.5)
    ax.set_title("Total-variation distance by control table (pooled shares)",
                 fontsize=10, pad=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    piv.round(6).to_csv(os.path.join(FIG_DIR, "marginal_tvd_arms.csv"))
    print(f"  wrote {os.path.basename(out_path)}")
    return out_path


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", nargs="+", default=None,
                    help="subset of control tables (default: all in the shares CSV)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    shares = os.path.join(FIG_DIR, "marginal_category_shares.csv")
    if not os.path.exists(shares):
        raise SystemExit(f"missing {shares}; run make_marginal_histograms.py first")
    df = pd.read_csv(shares)

    tables = args.tables or [t for t in TABLE_ORDER if t in set(df["table"])]
    tables += [t for t in sorted(set(df["table"])) if t not in tables]
    for table in tables:
        out = os.path.join(FIG_DIR, f"fig_marginals_{table}.png")
        plot_table(table, df, out, dpi=args.dpi)

    tvd_path = os.path.join(FIG_DIR, "marginal_tvd_by_table.csv")
    if os.path.exists(tvd_path):
        tvd = pd.read_csv(tvd_path)
    else:
        rows = []
        for table in tables:
            built = table_series(table, df)
            if built is None:
                continue
            _, _, series = built
            cens = np.asarray(series[0][1])
            for label, vals, _ in series[1:]:
                rows.append({"table": table, "arm_label": label,
                             "tvd": 0.5 * float(np.sum(np.abs(np.asarray(vals) - cens)))})
        tvd = pd.DataFrame(rows)
        key_by_label = {label: key for key, label, _ in ARM_ORDER}
        tvd["arm"] = tvd["arm_label"].map(key_by_label)

    plot_overview(tvd, os.path.join(FIG_DIR, "fig_marginals_all_tables.png"),
                  dpi=args.dpi)


if __name__ == "__main__":
    main()

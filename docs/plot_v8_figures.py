"""Result figures for the v8 corrected-control run.

Reads only the collected summary tables (``tables/arm_zones.csv`` and
``tables/rssz_by_table.csv``) so it is cheap to rerun while arms finish.

    python plot_v8_figures.py [--paper-dir ...]

Writes ``fig_rssz_by_arm.png``, ``fig_zone_size.png`` and
``fig_fit_example.png`` into the paper folder.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PAPER = os.path.abspath(os.path.join(
    RUN, "..", "..", "Population_Synthesiser___GAN_SA"))

# bottom-to-top order in the boxplot; (key, label, colour)
ARMS = [
    ("base_sa", "SA-Only", "#4C82A8"),
    ("base_ipu", "IPU-Only", "#E08A4C"),
    ("cp_sa", "CP-only + SA", "#E8B84B"),
    ("cp_ipu", "CP-only + IPU", "#4FB3A0"),
    ("cp_mip", "CP-MIP (replication)", "#8FC7E6"),
    ("cp_mip_unique", "CP-MIP (unique)", "#9AA6B2"),
    ("cp_mip_fallback", "CP-MIP + fallback", "#5B9BD5"),
    ("mip_sa", "MIP + SA", "#A98BC4"),
    ("mip_ipu", "MIP + IPU", "#E3A0BC"),
    ("mip_mip_unique", "MIP (direct)", "#7B5EA7"),
    ("mip_mip_fallback_unique", "MIP (direct) + fallback", "#C0A6D8"),
]
SEPARATORS_AFTER = {"base_ipu", "cp_mip_fallback"}

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 180,
})


def _zones() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RUN, "tables", "arm_zones.csv"))
    df["zone"] = df["zone"].astype(str).str.zfill(9)
    return df


GROUPS = {
    "raw": ["sa_only", "ipu_only"],
    "cp": ["cp_sa", "cp_ipu", "cp_mip", "cp_mip_unique",
           "cp_mip_fallback"],
    "expanded": ["mip_sa", "mip_ipu", "mip_mip_unique",
                 "mip_mip_fallback_unique"],
}
GROUP_LABELS = {"raw": "Raw microdata",
                "cp": "CP feasible pool",
                "expanded": "MIP-expanded pool"}
METHOD = {
    "sa_only": "SA", "cp_sa": "SA", "mip_sa": "SA",
    "ipu_only": "IPU", "cp_ipu": "IPU", "mip_ipu": "IPU",
    "cp_mip": "MIP (rep)", "cp_mip_fallback": "MIP (rep) + fallback",
    "cp_mip_unique": "MIP (unique)", "mip_mip_unique": "MIP (unique)",
    "mip_mip_fallback_unique": "MIP (unique) + fallback",
}
METHOD_COLOUR = {
    "SA": "#E8A33D",
    "IPU": "#3FA796",
    "MIP (rep)": "#4C72B0",
    "MIP (unique)": "#9DB4D6",
}


def _method_colour(key: str) -> str:
    method = METHOD[key]
    if method.startswith("MIP (rep)"):
        return METHOD_COLOUR["MIP (rep)"]
    if method.startswith("MIP (unique)"):
        return METHOD_COLOUR["MIP (unique)"]
    return METHOD_COLOUR[method]


def fig_rssz_by_arm(df: pd.DataFrame, out: str) -> None:
    present = set(df.arm)
    panels = []
    for group, keys in GROUPS.items():
        rows = [k for k in keys if k in present]
        if rows:
            panels.append((GROUP_LABELS[group], rows))
    if not panels:
        return

    widths = [max(len(rows), 2) for _title, rows in panels]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(10.8, 4.6), sharex=True,
        gridspec_kw={"width_ratios": widths, "wspace": 0.45})
    if len(panels) == 1:
        axes = [axes]

    reference = float(df.loc[df.arm == "cp_sa", "rssz19"].median()) \
        if "cp_sa" in present else None

    for ax, (title, rows) in zip(axes, panels):
        labels = []
        for y, key in enumerate(rows):
            v = pd.to_numeric(df.loc[df.arm == key, "rssz19"],
                              errors="coerce").dropna()
            if v.empty:
                continue
            ax.boxplot(
                [v.to_numpy()], positions=[y], widths=0.58,
                vert=False, patch_artist=True, showfliers=True,
                flierprops=dict(marker=".", markersize=1.3, alpha=0.32,
                                markerfacecolor="#555555",
                                markeredgecolor="none"),
                medianprops=dict(color="#111111", linewidth=1.2),
                boxprops=dict(facecolor=_method_colour(key),
                              edgecolor="#333333", linewidth=0.55),
                whiskerprops=dict(color="#333333", linewidth=0.65),
                capprops=dict(color="#333333", linewidth=0.65),
            )
            ax.scatter([v.mean()], [y], marker="D", s=10,
                       facecolor="white", edgecolor="#111111",
                       linewidth=0.6, zorder=4)
            labels.append((y, f"{METHOD[key]}\n{v.median():.3f} / "
                              f"{v.mean():.3f}"))
        ax.set_yticks([y for y, _ in labels])
        ax.set_yticklabels([lab for _y, lab in labels], fontsize=7.2)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(title, fontsize=8.8, fontweight="bold",
                     color="#333333", pad=8)
        ax.set_xscale("log")
        ax.set_xlim(0.008, 1.25)
        ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0])
        ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1"],
                           fontsize=7.5)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if reference is not None:
            ax.axvline(reference, color="#999999", linewidth=0.7,
                       linestyle=(0, (2, 2)), zorder=1)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c,
                             edgecolor="#333333", linewidth=0.5)
               for c in METHOD_COLOUR.values()]
    labels = list(METHOD_COLOUR)
    if reference is not None:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="#999999", linewidth=0.9,
                              linestyle=(0, (2, 2))))
        labels.append(f"CP-only+SA median ({reference:.3f})")
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.965), ncol=len(handles), frameon=False,
               fontsize=7.8, handlelength=1.1, columnspacing=1.4)
    fig.supxlabel("19-table RSSZ per zone (log scale; lower is better)",
                  fontsize=8.5, y=0.045)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.80, bottom=0.16)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_zone_size(df: pd.DataFrame, out: str) -> None:
    """Share of zones with RSSZ > 1 by census-population quintile."""
    size = (df[df.arm == "cp_mip"][["zone", "person_target"]]
            .dropna().drop_duplicates("zone"))
    size = size[size.person_target > 0]
    if size.empty:
        return
    qs = np.quantile(size.person_target, [0.2, 0.4, 0.6, 0.8])
    size["quintile"] = np.digitize(size.person_target, qs) + 1
    d = df.merge(size[["zone", "quintile"]], on="zone", how="inner")

    have = [(k, lab, c) for k, lab, c in ARMS if k in set(d.arm)]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    width = 0.8 / max(len(have), 1)
    for i, (key, label, colour) in enumerate(have):
        sub = d[d.arm == key]
        share = (sub.assign(bad=sub.rssz19 > 1)
                 .groupby("quintile").bad.mean().reindex([1, 2, 3, 4, 5]))
        ax.bar(np.arange(5) + i * width - 0.4 + width / 2,
               share.to_numpy() * 100, width=width, label=label,
               color=colour, edgecolor="#333333", linewidth=0.3)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["Q1\nsmallest", "Q2", "Q3", "Q4", "Q5\nlargest"])
    ax.set_ylabel("Zones with RSSZ $>1$ (\\%)")
    ax.set_ylim(0, max(12.0, ax.get_ylim()[1] * 1.25))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.2, ncol=5, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.0,
              handlelength=1.2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


TABLE_ORDER = ["heap", "hrsp", "incp", "indp", "mstp", "mtwp", "occp",
               "qalfp", "qallp", "rlhp", "sexp", "strd", "stup", "typp",
               "hind", "mrerd", "nprd", "rntrd", "vehrd"]


def _zone_table_row(patterns: dict[str, str], zone: str) -> dict:
    import glob
    for _key, pat in patterns.items():
        for path in glob.glob(pat):
            try:
                d = pd.read_csv(path)
            except Exception:
                continue
            d["zone"] = d["zone"].astype(str).str.zfill(9)
            row = d[d.zone == zone]
            if not row.empty:
                return row.iloc[0].to_dict()
    return {}


def fig_fit_example(df: pd.DataFrame, out: str) -> None:
    """Zone-level table residuals for CP-only+SA against CP-MIP."""
    import glob
    zone = "305031124"          # AT the SA median and near the MIP median in v8
    patterns = {
        "cp_sa": os.path.join(RUN, "fits", "cp_sa",
                              "cp_pool_rssz_summary_chunk*.csv"),
        "cp_mip": os.path.join(RUN, "fits", "cp_mip",
                               "cp_pool_rssz_summary_chunk*.csv"),
    }
    rows = {k: _zone_table_row({k: p}, zone)
            for k, p in patterns.items()}
    if not all(rows.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4),
                             gridspec_kw={"width_ratios": [1.05, 1.6]})

    # (a) 19-table RSSZ by strategy for this zone
    have = [(k, lab, c) for k, lab, c in ARMS if k in set(df.arm)]
    vals = []
    for key, _lab, _c in have:
        r = df[(df.arm == key) & (df.zone == zone)]
        vals.append(float(r.rssz19.iloc[0]) if not r.empty else np.nan)
    ax = axes[0]
    ax.barh(np.arange(len(have)), vals,
            color=[c for _k, _l, c in have], edgecolor="#333333",
            linewidth=0.4)
    ax.set_yticks(np.arange(len(have)))
    ax.set_yticklabels([lab for _k, lab, _c in have])
    ax.set_xlabel("19-table RSSZ")
    ax.set_title(f"(a) zone {zone}", fontsize=8)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)

    # (b) per-table residual
    ax = axes[1]
    x = np.arange(len(TABLE_ORDER))
    w = 0.4
    for i, (key, colour, lab) in enumerate([("cp_sa", "#E8B84B", "CP-only + SA"),
                                           ("cp_mip", "#8FC7E6",
                                            "CP-MIP (replication)")]):
        vals = []
        for t in TABLE_ORDER:
            for col in rows[key]:
                if str(col).lower().endswith("sa2" + t):
                    vals.append(float(rows[key][col]))
                    break
            else:
                vals.append(0.0)
        ax.bar(x + i * w - w / 2, np.maximum(vals, 1e-4), width=w,
               color=colour, edgecolor="#333333", linewidth=0.3,
               label=lab)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_xticks(x)
    ax.set_xticklabels(TABLE_ORDER, rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("Table RSSZ (log scale)")
    ax.set_title(f"(b) zone {zone}", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=DEFAULT_PAPER)
    args = ap.parse_args()
    os.makedirs(args.paper_dir, exist_ok=True)
    df = _zones()
    fig_rssz_by_arm(df, os.path.join(args.paper_dir, "fig_rssz_by_arm.png"))
    fig_zone_size(df, os.path.join(args.paper_dir, "fig_zone_size.png"))
    fig_fit_example(df, os.path.join(args.paper_dir, "fig_fit_example.png"))
    print("wrote figures to", args.paper_dir)


if __name__ == "__main__":
    main()

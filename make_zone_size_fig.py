"""Tail share by zone-size quintile.

The median is not the point of this figure: it is the share of zones whose
11-table RSSZ exceeds 1.0.  That is where the MIP+SA blow-ups live, and it rises
with zone size even though the median is flat.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "outputs", "paper_arms", "zone_size_analysis.csv")
OUT = os.path.join(HERE, "outputs", "paper_figures", "fig_zone_size.png")

ARMS = [
    ("sa_only", "SA-Only", "#0072B2", "o", "-"),
    ("ipu_only", "IPU-Only", "#D55E00", "s", "--"),
    ("base_sa", "CP-only + SA", "#E69F00", "^", "-."),
    ("composition_sa", "Comp. pool + SA", "#5E3C99", "D", ":"),
    ("cp_mip", "CP-MIP", "#34495E", "*", "-"),
    ("cp_sa", "MIP + SA", "#8E44AD", "v", "-"),
    ("cp_ipu", "MIP + IPU", "#CC79A7", "P", "--"),
]
BINS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
TAIL = 1.0


def main():
    d = pd.read_csv(CSV)
    d["size_bin"] = pd.Categorical(d["size_bin"], categories=BINS, ordered=True)
    x = np.arange(len(BINS))

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    handles = []
    for key, label, colour, marker, linestyle in ARMS:
        share = (d[key] > TAIL).groupby(
            d["size_bin"], observed=True).mean().reindex(BINS) * 100
        line, = ax.plot(x, share.to_numpy(), marker=marker, markersize=5.5,
                        linewidth=1.8, linestyle=linestyle, color=colour,
                        label=label)
        handles.append(line)
    ax.set_xticks(x)
    ax.set_xticklabels(BINS)
    ax.set_xlabel("census-population quintile (smallest to largest)")
    ax.set_ylabel(f"zones with 11-table RSSZ > {TAIL:g} (%)")
    ax.set_ylim(bottom=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print("wrote", OUT)
    for key, label, _, _, _ in ARMS:
        share = (d[key] > TAIL).groupby(
            d["size_bin"], observed=True).mean().reindex(BINS) * 100
        print(f"{label:18s}", share.round(1).to_dict())


if __name__ == "__main__":
    main()

"""Zone-size scaling: median RSSZ and win rate by census-population quintile."""
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
    ("sa_only", "SA-Only", "#0072B2"),
    ("ipu_only", "IPU-Only", "#D55E00"),
    ("base_sa", "CP-only + SA", "#E69F00"),
    ("composition_sa", "Composition + SA", "#8E44AD"),
    ("cp_mip", "CP-MIP", "#34495E"),
]
BINS = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def main():
    d = pd.read_csv(CSV)
    d["size_bin"] = pd.Categorical(d["size_bin"], categories=BINS, ordered=True)
    med = d.groupby("size_bin", observed=True)[[a for a, _, _ in ARMS]].median()
    wins = {}
    for a, _, _ in ARMS:
        if a == "sa_only":
            continue
        wins[a] = (d[a] < d["sa_only"]).groupby(
            d["size_bin"], observed=True).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.7),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(BINS))
    for a, label, colour in ARMS:
        ax1.plot(x, med[a].reindex(BINS).to_numpy(), marker="o", linewidth=1.8,
                 color=colour, label=label)
    ax1.set_xticks(x)
    ax1.set_xticklabels(BINS)
    ax1.set_xlabel("census-population quintile (smallest to largest)")
    ax1.set_ylabel("median 11-table RSSZ")
    ax1.legend(frameon=False, fontsize=7.5, ncol=2)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    for a, label, colour in ARMS:
        if a == "sa_only":
            continue
        ax2.plot(x, wins[a].reindex(BINS).to_numpy() * 100, marker="o",
                 linewidth=1.8, color=colour, label=label)
    ax2.axhline(50, linestyle=":", color="#7f8c8d", linewidth=1)
    ax2.set_ylim(0, 105)
    ax2.set_xticks(x)
    ax2.set_xticklabels(BINS)
    ax2.set_xlabel("census-population quintile")
    ax2.set_ylabel("zones beating SA-Only (%)")
    ax2.legend(frameon=False, fontsize=7.5, ncol=2)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print("wrote", OUT)
    print(med.round(3).to_string())
    print({k: v.round(3).to_dict() for k, v in wins.items()})


if __name__ == "__main__":
    main()

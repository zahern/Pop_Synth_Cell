"""Per-zone RSSZ distribution by strategy, with mean and SD in the labels.

All nine paper arms, 367-zone fitted cohort, 11-table reporting basis.  The
box panel is deliberately the whole figure: the mean/SD is printed beside each
arm, but the distribution shows the tail zones that the mean hides.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "outputs", "paper_arms", "paper_arms_zones.csv")
OUT = os.path.join(HERE, "outputs", "paper_figures", "fig_rssz_by_arm.png")

ARMS = [
    ("sa_only", "SA-Only", "#0072B2"),
    ("composition_sa", "Comp. pool + SA", "#5E3C99"),
    ("cp_mip", "CP-MIP", "#34495E"),
    ("base_sa", "CP-only + SA", "#E69F00"),
    ("composition_ipu", "Comp. pool + IPU", "#B2ABD2"),
    ("cp_ipu", "MIP + IPU", "#CC79A7"),
    ("base_ipu", "CP-only + IPU", "#009E73"),
    ("ipu_only", "IPU-Only", "#D55E00"),
    ("cp_sa", "MIP + SA", "#8E44AD"),
]


def main():
    d = pd.read_csv(CSV)
    stats = {}
    for key, label, colour in ARMS:
        s = d[d.arm == key].rssz11.dropna().to_numpy()
        stats[key] = {
            "label": label,
            "colour": colour,
            "values": s,
            "mean": float(np.mean(s)),
            "sd": float(np.std(s, ddof=1)),
            "median": float(np.median(s)),
        }

    order = sorted(stats, key=lambda k: stats[k]["mean"])
    labels = [
        f"{stats[k]['label']}\n{stats[k]['mean']:.3f} $\\pm$ {stats[k]['sd']:.3f}"
        for k in order
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    data = [stats[k]["values"] for k in order]
    colours = [stats[k]["colour"] for k in order]
    bp = ax.boxplot(
        data, vert=False, widths=0.58, patch_artist=True, whis=1.5,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2.5, alpha=0.35,
                        markerfacecolor="0.35", markeredgecolor="none"),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="0.35", linewidth=0.9),
        capprops=dict(color="0.35", linewidth=0.9))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.55)
        patch.set_edgecolor("0.25")
    for i, key in enumerate(order):
        ax.plot(stats[key]["mean"], i + 1, marker="D", markersize=4.5,
                color="black", markerfacecolor="white", markeredgewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 40)
    ax.set_yticks(np.arange(1, len(order) + 1))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("11-table RSSZ (log scale)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print("wrote", OUT)
    for key in order:
        st = stats[key]
        print(f"{st['label']:18s} mean={st['mean']:.4f} sd={st['sd']:.4f} "
              f"median={st['median']:.4f}")


if __name__ == "__main__":
    main()

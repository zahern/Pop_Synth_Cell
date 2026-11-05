"""Cohort RSSZ by strategy: mean +/- SD and per-zone distribution.

All nine paper arms, 367-zone fitted cohort, 11-table reporting basis.  The
mean/SD panel shows why the mean alone is misleading: CP-MIP and the IPU arms
have large SDs driven by the same tail zones that the box panel exposes.
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

    # Sort by mean so the left panel reads monotonically.
    order = sorted(stats, key=lambda k: stats[k]["mean"])
    y = np.arange(1, len(order) + 1)
    labels = [
        f"{stats[k]['label']}\n{stats[k]['mean']:.3f} $\\pm$ {stats[k]['sd']:.3f}"
        for k in order
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1]})

    floor = 0.02
    for i, key in enumerate(order):
        st = stats[key]
        lo = max(floor, st["mean"] - st["sd"])
        hi = st["mean"] + st["sd"]
        ax1.errorbar(
            st["mean"], i + 1, xerr=[[st["mean"] - lo], [hi - st["mean"]]],
            fmt="o", markersize=5.5, color=st["colour"], ecolor=st["colour"],
            elinewidth=1.4, capsize=3, capthick=1.0)
        ax1.plot(st["median"], i + 1, marker="|", markersize=9, color="black",
                 markeredgewidth=1.2)
    ax1.set_xscale("log")
    ax1.set_xlim(floor, 40)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("11-table RSSZ (log scale)")
    ax1.set_title("mean $\\pm$ 1 SD", fontsize=9)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    data = [stats[k]["values"] for k in order]
    colours = [stats[k]["colour"] for k in order]
    bp = ax2.boxplot(
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
        ax2.plot(stats[key]["mean"], i + 1, marker="D", markersize=4.5,
                 color="black", markerfacecolor="white", markeredgewidth=0.9)
    # boxplot resets the shared y-axis ticks; restore the arm labels.
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlim(floor, 40)
    ax2.set_xlabel("11-table RSSZ (log scale)")
    ax2.set_title("per-zone distribution", fontsize=9)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print("wrote", OUT)
    for key in order:
        st = stats[key]
        print(f"{st['label']:18s} mean={st['mean']:.4f} sd={st['sd']:.4f} "
              f"median={st['median']:.4f}")


if __name__ == "__main__":
    main()

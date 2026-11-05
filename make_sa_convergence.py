"""Convergence of SA on raw microdata vs the constructed/extended pool.

The x-axis is the SA annealer's iteration budget in every panel: each pool is
built once (MIP-expanded, coverage-only, or composition-aware) and then fitted
by the same SA implementation, so the curves differ only in the pool they
anneal. The MIP itself is a single solve and has no iterations.

Panel A: one zone (301011004), best RSSZ at the 200-iteration checkpoints,
linear scale.
Panel B: all zones common to the runs, each zone's best RSSZ divided by its
own final best, summarised by the median and interquartile band.
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "outputs", "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

ZONE = "301011004"
PAT = re.compile(r"Zone (\d+): iter (\d+)/1000\s+RSSZ=([0-9.]+)\s+best=([0-9.]+)")

RUNS = {
    "SA-Only": sorted(glob.glob(os.path.join(HERE, "s3b_c*.o27308*"))),
    "CP-pool + SA": sorted(glob.glob(os.path.join(
        HERE, "pbs_logs", "non_gan_fitters_v3", "cp_sa_*.log"))),
    "Coverage pool + SA": sorted(glob.glob(os.path.join(
        HERE, "pbs_logs", "coverage_fitters", "cp_sa_*.log"))),
    "Composition pool + SA": sorted(glob.glob(os.path.join(
        HERE, "pbs_logs", "composition_fitters", "cp_sa_*.log"))),
}


def read_run(paths):
    rows = []
    for path in paths:
        text = open(path, errors="ignore").read()
        for m in PAT.finditer(text):
            rows.append((m.group(1), int(m.group(2)), float(m.group(3)),
                         float(m.group(4))))
    return pd.DataFrame(rows, columns=["zone", "iteration", "current", "best"])


def main():
    data = {arm: read_run(paths) for arm, paths in RUNS.items()}
    for arm, df in data.items():
        print(arm, "zones", df.zone.nunique(), "rows", len(df))

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    colors = {"SA-Only": "#1b6ca8", "CP-pool + SA": "#d68910",
              "Coverage pool + SA": "#8e44ad",
              "Composition pool + SA": "#2e8b57"}

    # Panel A: one zone, linear scale. The four traces sit at clearly
    # different levels, so no log axis is needed.
    for arm, df in data.items():
        tr = df[df.zone.astype(str) == ZONE].sort_values("iteration")
        axA.plot(tr.iteration, tr.best, marker="o", markersize=5,
                 linewidth=1.8, color=colors[arm], label=arm)
    axA.set_xlabel("SA iteration checkpoint")
    axA.set_ylabel("best RSSZ (sum over 19 tables)")
    axA.set_xticks([200, 400, 600, 800, 1000])
    axA.set_ylim(0, None)
    axA.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # Panel B: common zones, normalised to each zone's own final best.
    common = set.intersection(*[
        set(df.loc[df.iteration == 1000, "zone"]) for df in data.values()])
    print("common zones", len(common))
    for arm, df in data.items():
        sub = df[df.zone.isin(common)]
        piv = sub.pivot_table(index="zone", columns="iteration",
                              values="best", aggfunc="last")
        piv = piv.dropna()
        final = piv[1000]
        norm = piv.div(final, axis=0)
        med = norm.median(axis=0)
        q25 = norm.quantile(0.25, axis=0)
        q75 = norm.quantile(0.75, axis=0)
        med_index = med.index.to_numpy()
        print(arm, "median ratios", med.round(3).to_dict())
        axB.plot(med_index, med.values, marker="o",
                 markersize=5, linewidth=1.8, color=colors[arm], label=arm)
        axB.fill_between(med_index, q25.values, q75.values,
                         color=colors[arm], alpha=0.16, linewidth=0)
    axB.axhline(1.0, color="#555555", linewidth=0.8, linestyle=":")
    axB.set_xlabel("SA iteration checkpoint")
    axB.set_ylabel("best RSSZ / zone's final best")
    axB.set_xticks([200, 400, 600, 800, 1000])
    axB.set_ylim(0.99, None)
    axB.set_xlim(150, 1050)
    axB.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_sa_convergence.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())

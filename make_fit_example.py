"""Worked example of the zone-fitting process, from the current CP-pool+SA run.

Panel A: the SA convergence trace for zone 301011004, from the 200-iteration
checkpoints in the chunk-1 log. The solid line is the current solution and the
dashed line is the best seen so far. At this sampling resolution the two
coincide; the per-iteration schedule diagnostic in fig_sa_schedule shows the
uphill moves that the coarse sampling hides.

Panel B: the per-table RSSZ decomposition of the same zone's final population.

No titles: the figure is described in the caption.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "outputs", "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

ZONE = "301011004"
LOG = os.path.join(HERE, "pbs_logs", "non_gan_fitters_v3", "cp_sa_1.log")
SUMMARY = os.path.join(HERE, "outputs", "cp_pool_results",
                       "expanded_v3_chunk_1", "cp_pool_rssz_summary_chunk1.csv")


def load_trace():
    text = open(LOG, errors="ignore").read()
    block = text.split(f"Zone {ZONE}", 1)[1]
    block = block.split("\nZone ", 1)[0]
    rows = []
    for m in re.finditer(
        r"iter (\d+)/\d+\s+RSSZ=([0-9.]+)\s+best=([0-9.]+)", block
    ):
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    if not rows:
        raise SystemExit(f"no checkpoints found for zone {ZONE} in {LOG}")
    return rows


def main():
    trace = load_trace()
    df = pd.read_csv(SUMMARY)
    row = df[df["zone"].astype(str) == ZONE].iloc[0]
    table_cols = [c for c in df.columns
                  if c.startswith(("Individual - ", "Household - "))]
    vals = {c.split(" - ")[-1].replace("sa2", "", 1): float(row[c])
            for c in table_cols}
    hh = {c.split(" - ")[-1].replace("sa2", "", 1)
          for c in table_cols if c.startswith("Household")}
    order = sorted(vals, key=vals.get, reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    its = [t[0] for t in trace]
    cur = [t[1] for t in trace]
    best = [t[2] for t in trace]
    ax1.plot(its, cur, marker="o", color="#d68910", linewidth=1.8,
             label="current solution")
    ax1.plot(its, best, linestyle="--", color="#34495e", linewidth=1.4,
             label="best solution")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("RSSZ (sum over 19 tables)")
    ax1.legend(frameon=False, fontsize=8.5)
    # Lower left is empty: the trace starts at the top-left and descends.  The
    # earlier 0.72 placement put the note straight through the line and the
    # 200-iteration marker.
    ax1.text(0.03, 0.10,
             "200-iteration checkpoints\n(coarse sampling)",
             transform=ax1.transAxes, fontsize=8.5, va="bottom")
    ax1.set_ylim(min(cur) - 0.02, max(cur) + 0.02)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    labels = order[::-1]
    values = [vals[k] for k in labels]
    colors = ["#1b6ca8" if k in hh else "#7fb3d5" for k in labels]
    ax2.barh(range(len(labels)), values, color=colors)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("per-table RSSZ")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.axvline(0.1, linestyle=":", color="#c0392b", linewidth=1)
    # The old floating "dwelling-level" label sat in empty space in the middle
    # of the panel and named only one of the two colours.  A key under the axis
    # gives both meanings without covering a bar or the threshold line.
    ax2.legend(handles=[Patch(facecolor="#1b6ca8", label="Dwelling-level"),
                        Patch(facecolor="#7fb3d5", label="Person-level")],
               loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
               frameon=False, fontsize=8.5)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_fit_example.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print("trace", trace)
    print(f"zone {ZONE}: rssz_total={float(row['rssz_total']):.4f} "
          f"dominant={row['dominant_table']}")


if __name__ == "__main__":
    raise SystemExit(main())

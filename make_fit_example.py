"""Worked example of what the pipeline stages add, for one SA2 zone.

Zone 301011004. Panel A: 11-table RSSZ by arm for the same zone, showing what
each stage contributes. Panel B: the same zone's per-table residuals under
CP-pool + SA and CP-MIP, showing where the one-MIP fit removes the residual.

No titles: the figure is described in the caption.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "outputs", "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

ZONE = "301011004"
ARMS = pd.read_csv(os.path.join(HERE, "outputs", "paper_arms",
                                "paper_arms_zones.csv"))
ARMS["zone"] = ARMS["zone"].astype(str)
LADDER = [  # (arm key, label, colour) in pipeline order
    ("ipu_only", "IPU-Only", "#D55E00"),
    ("sa_only", "SA-Only", "#0072B2"),
    ("cp_ipu", "CP-pool + IPU", "#009E73"),
    ("cp_sa", "CP-pool + SA", "#E69F00"),
    ("cp_mip", "CP-MIP", "#34495E"),
]


def load_tables(results_glob):
    rows = []
    for path in sorted(glob.glob(results_glob)):
        df = pd.read_csv(path)
        df["zone"] = df["zone"].astype(str)
        hit = df[df.zone == ZONE]
        if len(hit):
            rows.append(hit.iloc[0])
    if not rows:
        raise SystemExit(f"zone {ZONE} not found under {results_glob}")
    row = rows[0]
    cols = [c for c in row.index
            if c.startswith(("Individual - ", "Household - "))]
    return {c.split(" - ")[-1].replace("sa2", "", 1): float(row[c])
            for c in cols}, float(row["rssz_total"])


def main():
    # Panel A: per-arm RSSZ for this zone.
    zone_scores = ARMS[ARMS.zone == ZONE].set_index("arm")["rssz11"]

    # Panel B: per-table residuals, CP-pool + SA and CP-MIP.
    sa_tables, sa_total = load_tables(os.path.join(
        HERE, "outputs", "cp_pool_results", "expanded_v3_chunk_*",
        "cp_pool_rssz_summary_chunk*.csv"))
    mip_tables, mip_total = load_tables(os.path.join(
        HERE, "outputs", "cp_mip_results", "cp_pool_rssz_summary_chunk*.csv"))
    print(f"cp_sa 19-table {sa_total:.4f} / cp_mip 19-table {mip_total:.4f}")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.6]})

    labels = [lab for _, lab, _ in LADDER][::-1]
    colors = [c for _, _, c in LADDER][::-1]
    values = [zone_scores[a] for a, _, _ in LADDER][::-1]
    axA.barh(range(len(labels)), values, color=colors, edgecolor="black",
             linewidth=0.4)
    axA.set_yticks(range(len(labels)))
    axA.set_yticklabels(labels, fontsize=9)
    axA.set_xlabel("11-table RSSZ")
    axA.grid(axis="x", color="0.85", linewidth=0.6)
    axA.set_axisbelow(True)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    order = sorted(sa_tables, key=sa_tables.get, reverse=True)
    x = np.arange(len(order))
    sa_vals = [sa_tables[t] for t in order]
    mip_vals = [mip_tables.get(t, 0.0) for t in order]
    axB.bar(x - 0.2, sa_vals, width=0.4, color="#E69F00", edgecolor="black",
            linewidth=0.3, label="CP-pool + SA")
    axB.bar(x + 0.2, mip_vals, width=0.4, color="#34495E", edgecolor="black",
            linewidth=0.3, label="CP-MIP")
    axB.set_xticks(x)
    axB.set_xticklabels(order, rotation=90, fontsize=7.5)
    axB.set_ylabel("per-table RSSZ")
    axB.legend(frameon=False, fontsize=8.5)
    axB.grid(axis="y", color="0.85", linewidth=0.6)
    axB.set_axisbelow(True)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_fit_example.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print("ladder", {a: round(float(zone_scores[a]), 4) for a, _, _ in LADDER})


if __name__ == "__main__":
    raise SystemExit(main())

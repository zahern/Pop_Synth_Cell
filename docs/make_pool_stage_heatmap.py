"""High-level pool-stage marginal distance by control table."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA\fig_pool_stage_heatmap.png"

LABELS = {
    "heap": "Highest education",
    "hind": "Household income",
    "hrsp": "Hours worked",
    "incp": "Personal income",
    "indp": "Industry",
    "mrerd": "Mortgage repayment",
    "mstp": "Marital status",
    "mtwp": "Travel to work",
    "nprd": "Persons per dwelling",
    "occp": "Occupation",
    "qalfp": "Field of study",
    "qallp": "Qualification level",
    "rlhp": "Relationship in household",
    "rntrd": "Rent",
    "sexp": "Sex",
    "strd": "Dwelling structure",
    "stup": "Student status",
    "typp": "Education institution",
    "vehrd": "Vehicles",
}
STAGES = ["Raw", "CP", "Expanded", "CP + fallback", "Expanded + fallback"]
SHORT = ["Raw", "CP", "Expanded", "CP +\ncarriers", "Expanded +\ncarriers"]

plt.rcParams.update({"font.size": 8.5})
d = pd.read_csv(os.path.join(RUN, "tables", "pool_score_by_zone_table.csv"))
REL = [f"{s} relative" for s in STAGES]
den = d["max_tvd"].to_numpy(dtype=float)
for s, rel in zip(STAGES, REL):
    num = d[s].to_numpy(dtype=float)
    d[rel] = np.where(den > 1e-9, 100.0 * num / den, np.nan)
piv = d.groupby("table")[REL].mean()
keys = [t for t in LABELS if t in piv.index]
piv = piv.reindex(keys)
piv.index = [LABELS[t] for t in keys]
zc = pd.read_csv(os.path.join(RUN, "tables", "zero_cell_target_share.csv"))
zc = zc.set_index("table").reindex(keys).fillna(0.0)

fig, ax = plt.subplots(figsize=(8.6, 6.6))
vals = piv[REL].to_numpy()
vmax = max(1.0, float(np.nanmax(vals)))
im = ax.imshow(vals, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
ax.set_xticks(np.arange(len(STAGES)))
ax.set_xticklabels(SHORT, fontsize=8)
ax.set_yticks(np.arange(len(piv)))
ax.set_yticklabels(piv.index, fontsize=8)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        ax.text(j, i, f"{vals[i, j]:.0f}%", ha="center", va="center",
                fontsize=7,
                color="white" if vals[i, j] > 0.62 * vmax else "#222222")
ax.set_title("TVD as % of maximum marginal TVD", fontsize=10, pad=8)
ax.tick_params(length=0)
ax.set_yticks(np.arange(-.5, len(piv), 1), minor=True)
ax.grid(which="minor", axis="y", color="white", linewidth=0.8)
ax.tick_params(which="minor", length=0)
ax.set_xlabel("Pool stage", fontsize=8.5)

# Side table: how much of each table's census target sits in zero-capacity
# cells, i.e. the part the carrier stage has to add rather than reweight.
zvals = zc["zero_share"].to_numpy(dtype=float)
zcells = zc["zero_cells"].to_numpy(dtype=float)
ax.set_xlim(-0.5, 5.95)
ax.axvline(4.58, color="#CCCCCC", linewidth=0.7)
ax.text(4.70, -0.95, "Zero-cell addition", fontsize=8.2,
        fontweight="bold", clip_on=False)
ax.text(4.70, -0.55, "share", fontsize=6.8, color="#555555",
        clip_on=False)
ax.text(5.38, -0.55, "cells", fontsize=6.8, color="#555555",
        clip_on=False)
for i, (v, n) in enumerate(zip(zvals, zcells)):
    ax.text(4.70, i, f"{v:.1f}%" if v >= 0.05 else "0",
            va="center", fontsize=7.0, color="#333333", clip_on=False)
    ax.text(5.38, i, f"{int(n)}", va="center", fontsize=7.0,
            color="#333333", clip_on=False)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12,
                    fraction=0.04)
cbar.set_label("TVD as % of maximum TVD from the marginal (lower is better)",
               fontsize=7.5)
cbar.ax.tick_params(labelsize=7)

fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT)

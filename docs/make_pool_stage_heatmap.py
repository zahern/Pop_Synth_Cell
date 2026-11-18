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
piv = d.groupby("table")[STAGES].mean()
piv = piv.reindex([t for t in LABELS if t in piv.index])
piv.index = [LABELS[t] for t in piv.index]

fig, ax = plt.subplots(figsize=(7.6, 6.4))
vals = (piv[STAGES].div(piv["Raw"], axis=0)).to_numpy()
im = ax.imshow(vals, aspect="auto", cmap="RdBu_r",
               norm=TwoSlopeNorm(vcenter=1.0, vmin=0.8, vmax=1.2))
ax.set_xticks(np.arange(len(STAGES)))
ax.set_xticklabels(SHORT, fontsize=8)
ax.set_yticks(np.arange(len(piv)))
ax.set_yticklabels(piv.index, fontsize=8)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        ax.text(j, i, f"{vals[i, j]:.2f}×", ha="center", va="center",
                fontsize=7,
                color="white" if abs(vals[i, j] - 1.0) > 0.55 else "#222222")
ax.set_title("Mean TVD relative to the raw stage for the same control table",
             fontsize=10, pad=8)
ax.tick_params(length=0)
ax.set_xlabel("Pool stage", fontsize=8.5)

cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
cbar.set_label("TVD / raw TVD (below 1 is better)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT)

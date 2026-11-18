"""Change in pool-stage TVD relative to the raw microdata, by control table."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA\fig_pool_stage_raw_delta.png"

LABELS = {
    "heap": "Highest education", "hind": "Household income",
    "hrsp": "Hours worked", "incp": "Personal income",
    "indp": "Industry", "mrerd": "Mortgage repayment",
    "mstp": "Marital status", "mtwp": "Travel to work",
    "nprd": "Persons per dwelling",
    "occp": "Occupation", "qalfp": "Field of study",
    "qallp": "Qualification level", "rlhp": "Relationship in household",
    "rntrd": "Rent", "sexp": "Sex", "strd": "Dwelling structure",
    "stup": "Student status",
    "typp": "Education institution", "vehrd": "Vehicles",
}
STAGES = ["Raw", "CP", "Expanded", "CP + fallback", "Expanded + fallback"]
COLS = ["CP", "Expanded", "CP + fallback", "Expanded + fallback"]

d = pd.read_csv(os.path.join(RUN, "tables", "pool_score_by_zone_table.csv"))
piv = d.groupby("table")[STAGES].mean()
piv = piv.reindex([t for t in LABELS if t in piv.index])
piv.index = [LABELS[t] for t in piv.index]
vals = piv[COLS].to_numpy() - piv["Raw"].to_numpy()[:, None]

lim = max(0.002, float(np.nanmax(np.abs(vals))))
fig, ax = plt.subplots(figsize=(7.4, 6.4))
im = ax.imshow(vals, aspect="auto", cmap="RdBu_r",
               norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
ax.set_xticks(np.arange(len(COLS)))
ax.set_xticklabels(["CP", "Expanded", "CP +\ncarriers", "Expanded +\ncarriers"],
                   fontsize=8)
ax.set_yticks(np.arange(len(piv)))
ax.set_yticklabels(piv.index, fontsize=8)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.55 * lim else "#222222")
ax.set_title("Change in mean TVD relative to the raw microdata", fontsize=10, pad=8)
ax.tick_params(length=0)
ax.set_xlabel("Pool stage", fontsize=8.5)

cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
cbar.set_label("TVD difference (blue = better than raw)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT)

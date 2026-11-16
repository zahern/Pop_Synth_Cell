"""Heatmap of mean per-zone RSSZ by method and control table (v8 cohort)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA"

ARMS = [
    ("cp_sa", "CP pool, SA"),
    ("cp_ipu", "CP pool, IPU"),
    ("mip_sa", "Expanded, SA"),
    ("mip_ipu", "Expanded, IPU"),
    ("cp_mip_fallback", "Expanded, MIP"),
]
TABLES = ["heap", "mstp", "mtwp", "occp", "rlhp", "sexp", "qalfp", "qallp",
          "incp", "hrsp", "indp", "typp", "stup", "hind", "mrerd", "nprd",
          "rntrd", "vehrd", "strd"]

d = pd.read_csv(os.path.join(RUN, "tables", "rssz_by_table.csv"))
pv = d.pivot_table(index="arm", columns="table", values="mean")
pv = pv.reindex(index=[a for a, _l in ARMS], columns=TABLES)
vals = pv.to_numpy(dtype=float)
masked = np.ma.masked_invalid(vals)

fig, ax = plt.subplots(figsize=(11.4, 4.9))
cmap = plt.get_cmap("YlGnBu").copy()
cmap.set_bad("#EDEDED")
norm = LogNorm(vmin=1e-4, vmax=0.12)
im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)

ax.set_xticks(np.arange(len(TABLES)))
ax.set_xticklabels(TABLES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(np.arange(len(ARMS)))
ax.set_yticklabels([lab for _a, lab in ARMS], fontsize=8)

for i in range(len(ARMS)):
    for j in range(len(TABLES)):
        v = vals[i, j]
        if not np.isfinite(v):
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.6,
                    color="#666666")
            continue
        # white text on dark cells, dark text on light cells
        frac = (np.log10(max(v, 1e-4)) - np.log10(1e-4)) / (
            np.log10(0.12) - np.log10(1e-4))
        colour = "white" if frac > 0.55 else "#222222"
        txt = f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                color=colour)

# best arm per table, marked in the cell corner
for j in range(len(TABLES)):
    col = vals[:, j]
    if np.all(~np.isfinite(col)):
        continue
    i = int(np.nanargmin(col))
    ax.scatter(j + 0.33, i - 0.33, marker="*", s=24, color="#D62728",
               zorder=5, edgecolor="white", linewidth=0.3)

ax.axhline(1.5, color="#333333", linewidth=0.8)
ax.set_title("Mean per-zone RSSZ by pool and fitting method "
             "(362-zone cohort; lower is better; red star = best per table)",
             fontsize=10, pad=10)
ax.set_xlabel("Control table", fontsize=8)
ax.set_ylabel("Configuration", fontsize=8)
ax.tick_params(length=0)

cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
cbar.set_label("Mean RSSZ (log scale)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.tight_layout()
path = os.path.join(OUT, "fig_method_table_heatmap.png")
fig.savefig(path, dpi=190, bbox_inches="tight")
plt.close(fig)
print("wrote", path)

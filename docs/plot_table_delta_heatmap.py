"""Per-table change in mean RSSZ relative to CP-only (the block ablations)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA"

ROWS = [
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
pv = pv.reindex(columns=TABLES)
base = pv.loc["cp_sa"]
vals = np.vstack([(pv.loc[k] - base).to_numpy() for k, _l in ROWS])

lim = max(0.01, float(np.nanmax(np.abs(vals))))
norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
fig, ax = plt.subplots(figsize=(11.4, 3.2))
im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", norm=norm)

ax.set_xticks(np.arange(len(TABLES)))
ax.set_xticklabels(TABLES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(np.arange(len(ROWS)))
ax.set_yticklabels([lab for _k, lab in ROWS], fontsize=8)

for i in range(len(ROWS)):
    for j in range(len(TABLES)):
        v = vals[i, j]
        if not np.isfinite(v):
            continue
        txt = f"{v:+.4f}" if abs(v) >= 1e-4 else "0"
        ax.text(j, i, txt, ha="center", va="center", fontsize=5.4,
                color="white" if abs(v) > 0.55 * lim else "#222222")

ax.set_title("Change in mean per-zone RSSZ relative to the CP pool with SA "
             "(blue = better, red = worse)", fontsize=10, pad=10)
ax.set_xlabel("Control table", fontsize=8)
ax.set_ylabel("Configuration", fontsize=8)
ax.tick_params(length=0)

cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
cbar.set_label("RSSZ difference", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.tight_layout()
path = os.path.join(OUT, "fig_table_delta_heatmap.png")
fig.savefig(path, dpi=190, bbox_inches="tight")
plt.close(fig)
print("wrote", path)

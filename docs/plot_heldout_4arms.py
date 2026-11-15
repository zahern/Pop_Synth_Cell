"""Held-out TVD for the unified method and its three block ablations."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA"
ARMS = [
    ("base_sa", "CP-only", "#E8A33D"),
    ("cp_mip", "CP-MIP, no fallback", "#4C72B0"),
    ("cp_mip_fallback", "CP-MIP, fallback", "#1F6FB2"),
    ("mip_fallback", "CP-MIP, unique fallback", "#9DB4D6"),
]
TABLES = ["age5p", "assnp", "bedrd", "hhcd", "lfsp", "tenlld", "hcfmf"]

d = pd.read_csv(os.path.join(RUN, "paper_figures", "heldout_tvd_summary.csv"))
pv = d.pivot_table(index="table", columns="arm", values="mean")
pv = pv.reindex(index=TABLES, columns=[a for a, _l, _c in ARMS])

fig, ax = plt.subplots(figsize=(8.6, 4.0))
x = np.arange(len(TABLES))
w = 0.8 / len(ARMS)
for i, (arm, label, colour) in enumerate(ARMS):
    ax.bar(x + i * w - 0.4 + w / 2, pv[arm].to_numpy(), width=w,
           label=label, color=colour, edgecolor="#333333", linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels(TABLES, fontsize=8)
ax.set_ylabel("Mean held-out TVD")
ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper left")
ax.set_title("Held-out tables: mean TVD for the unified method and ablations",
             fontsize=9.5)
fig.tight_layout()
path = os.path.join(OUT, "fig_heldout_tvd.png")
fig.savefig(path, dpi=190, bbox_inches="tight")
plt.close(fig)
print("wrote", path)

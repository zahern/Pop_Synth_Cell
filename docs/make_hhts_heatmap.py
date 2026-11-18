"""Heatmap version of the HHTS validation figure."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
TAB = os.path.join(RUN, "tables")
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA"

hhcd = pd.read_csv(os.path.join(TAB, "hhts_hhcd_bikes_matrix.csv"))
size = pd.read_csv(os.path.join(TAB, "hhts_size_bikes_matrix.csv"))
alic = pd.read_csv(os.path.join(TAB, "hhts_age_licence_matrix.csv"))
ry = pd.read_csv(os.path.join(TAB, "age_household_matrix.csv"))

HHCD_ORDER = [1, 2, 3, 4, 5, 6]
HHCD_LABELS = {1: "One family", 2: "Two+ families", 3: "Lone person",
               4: "Group household", 5: "Visitors only", 6: "Other",
               7: "N/A"}
BIKES = ["0", "1", "2", "3+"]
SIZES = ["1", "2", "3", "4+"]
AGES = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_GROUPS = ["0-14", "15-24", "25-44", "45-64", "65+"]


def _mat(df, row_col, col_col, val_col, rows, cols):
    m = df.pivot(index=row_col, columns=col_col, values=val_col)
    return m.reindex(index=rows, columns=cols)


def _heat(ax, mat, row_labels, col_labels, title, xlabel, ylabel,
          fmt="+0.1f", vmax=None):
    vals = mat.to_numpy(dtype=float)
    if vmax is None:
        vmax = max(0.3, float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any()
                   else 1.0)
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7.5)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_title(title, fontsize=9.2, pad=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(length=0)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = vals[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6.4)
                continue
            ax.text(j, i, format(v, fmt), ha="center", va="center",
                    fontsize=6.4,
                    color="white" if abs(v) > 0.55 * vmax else "#222222")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Fitted − attached pool (pp)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)


fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4))

m = _mat(hhcd, "row", "col", "diff_pp", HHCD_ORDER, BIKES)
_heat(axes[0, 0], m, [HHCD_LABELS[i] for i in HHCD_ORDER], BIKES,
      "(a) Household composition × bicycles", "Bicycles in household",
      "Household composition", vmax=5.0)

m = _mat(size, "row", "col", "diff_pp", SIZES, BIKES)
_heat(axes[0, 1], m, SIZES, BIKES,
      "(b) Household size × bicycles", "Bicycles in household",
      "Household size", vmax=1.5)

m = alic.pivot(index="age_band", columns="licence", values="diff_pp")
m = m.reindex(index=AGES, columns=["Car", "Motorcycle"])
_heat(axes[1, 0], m, AGES, ["Car", "Motorcycle"],
      "(c) Age band × licence", "Licence", "Age band", vmax=2.0)

# reference age x youngest member age; drop the impossible 0-14 reference row
m = _mat(ry, "row", "col", "diff_pp", AGE_GROUPS, AGE_GROUPS)
keep = [g for g in AGE_GROUPS if np.isfinite(m.loc[g]).any()
        and abs(np.nansum(m.loc[g].to_numpy())) > 0]
m = m.loc[keep]
_heat(axes[1, 1], m, keep, AGE_GROUPS,
      "(d) Reference-person age × member age",
      "Household member", "Reference person", vmax=4.0)

fig.suptitle("Household travel survey attributes: pre-fit attached pool → "
             "expanded fitted population", fontsize=11, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.955))
out = os.path.join(OUT, "fig_hhts_validation.png")
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote", out)

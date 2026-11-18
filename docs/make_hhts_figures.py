"""HHTS validation figures from the corrected household-key tables."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
TAB = os.path.join(RUN, "tables")
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA"

overall = pd.read_csv(os.path.join(TAB, "hhts_overall_validation.csv"))
size = pd.read_csv(os.path.join(TAB, "hhts_size_validation.csv"))
age = pd.read_csv(os.path.join(TAB, "hhts_age_validation.csv"))
hhcd = pd.read_csv(os.path.join(TAB, "hhts_hhcd_validation.csv"))
quint = pd.read_csv(os.path.join(TAB, "hhts_quintile_validation.csv"))
sa4 = pd.read_csv(os.path.join(TAB, "hhts_sa4_validation.csv"))
cov = pd.read_csv(os.path.join(TAB, "hhts_donor_coverage.csv"))

FIT = "#1F6FB2"
POOL = "#9DB4D6"
INK = "#222222"
GRID = "#DDDDDD"


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0, labelsize=7.5)
    ax.yaxis.label.set_size(8)
    ax.xaxis.label.set_size(8)


def _bars(ax, labels, fitted, pool, title, ylabel, rotation=0):
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, np.asarray(pool) * 100, w, label="Attached pool",
           color=POOL, edgecolor="#333333", linewidth=0.3)
    ax.bar(x + w / 2, np.asarray(fitted) * 100, w,
           label="Fitted population", color=FIT, edgecolor="#333333",
           linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation,
                       ha="right" if rotation else "center")
    ax.set_title(title, fontsize=9.2, pad=7)
    ax.set_ylabel(ylabel)
    _style(ax)


# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
ax = axes[0, 0]
labels = ["Bikes 0", "Bikes 1", "Bikes 2", "Bikes 3+", "E-bike 1+",
          "E-scooter 1+", "Car licence", "Moto licence"]
_bars(ax, labels, overall["fitted"], overall["pool"],
      "(a) Overall HHTS attribute shares", "Share (%)", rotation=30)
ax.legend(fontsize=7.2, frameon=False, loc="upper left")

ax = axes[0, 1]
_bars(ax, size["size"].astype(str), size["any_bike_fitted"],
      size["any_bike_pool"], "(b) Bicycle ownership by household size",
      "Households with a bicycle (%)")
ax.set_xlabel("Household size")

ax = axes[1, 0]
_bars(ax, age["age_band"], age["fitted"], age["pool"],
      "(c) Car licence by age group (AGEP band)", "Car licence (%)")

ax = axes[1, 1]
h = hhcd[hhcd["hhcd"] != 7].copy()
_bars(ax, h["label"], h["any_bike_fitted"], h["any_bike_pool"],
      "(d) Bicycle ownership by household composition",
      "Households with a bicycle (%)", rotation=25)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_hhts_validation.png"), dpi=220,
            bbox_inches="tight")
plt.close(fig)
print("wrote fig_hhts_validation.png")

# ---------------------------------------------------------------- figure 2
metrics = [("bikes3p", "Bikes 3+"), ("ebike1p", "E-bike 1+"),
           ("escooter1p", "E-scooter 1+"), ("car_licence", "Car licence")]
sa4 = sa4.sort_values("sa4_name")
diff = np.array([[100 * (sa4[f"{k}_fitted"].iloc[i] -
                         sa4[f"{k}_pool"].iloc[i])
                  for k, _ in metrics] for i in range(len(sa4))])

fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2),
                         gridspec_kw={"width_ratios": [1.15, 1.0]})
ax = axes[0]
norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
im = ax.imshow(diff, cmap="RdBu_r", norm=norm, aspect="auto")
ax.set_xticks(np.arange(len(metrics)))
ax.set_xticklabels([m[1] for m in metrics], rotation=25, ha="right",
                   fontsize=7.5)
ax.set_yticks(np.arange(len(sa4)))
ax.set_yticklabels(sa4["sa4_name"], fontsize=7.0)
for i in range(diff.shape[0]):
    for j in range(diff.shape[1]):
        ax.text(j, i, f"{diff[i, j]:+.1f}", ha="center", va="center",
                fontsize=6.4, color=INK)
ax.set_title("(a) Fitted minus attached pool by SA4 (pp)", fontsize=9.2,
             pad=7)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.ax.tick_params(labelsize=7)
cb.set_label("Difference (pp)", fontsize=7.5)

ax = axes[1]
x = np.arange(len(quint))
w = 0.2
colors = ["#1F6FB2", "#E08A1E", "#3C9A5F", "#B23A48"]
for k, ((key, label), col) in enumerate(zip(metrics, colors)):
    vals = 100 * (quint[f"{key}_fitted"] - quint[f"{key}_pool"])
    ax.bar(x + (k - 1.5) * w, vals, w, label=label, color=col,
           edgecolor="#333333", linewidth=0.25)
ax.axhline(0, color="#444444", linewidth=0.7)
ax.set_xticks(x)
ax.set_xticklabels(quint["quintile"], rotation=20, ha="right", fontsize=7.5)
ax.set_ylabel("Fitted minus pool (pp)")
ax.set_title("(b) Difference by zone-size quintile", fontsize=9.2, pad=7)
_style(ax)
ax.legend(fontsize=7.0, frameon=False, ncol=2, loc="upper right")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_hhts_spatial.png"), dpi=220,
            bbox_inches="tight")
plt.close(fig)
print("wrote fig_hhts_spatial.png")

# ---------------------------------------------------------------- figure 3
fig, ax = plt.subplots(figsize=(9.2, 4.2))
labels = cov["sa4_name"].tolist()
vals = cov["donor_per_100_target"].to_numpy()
bars = ax.bar(np.arange(len(labels)), vals, color=FIT, edgecolor="#333333",
              linewidth=0.3, width=0.65)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}",
            ha="center", va="bottom", fontsize=7.2)
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5)
ax.set_ylabel("Observed HHTS donor households\nper 100 target households")
ax.set_title("HHTS donor coverage by donor region", fontsize=9.5, pad=8)
_style(ax)
ax.set_ylim(0, max(vals) * 1.18)
ax.text(0.01, 0.95,
        "All 362 zones draw on observed HHTS donor households",
        transform=ax.transAxes, fontsize=7.4, va="top", color="#333333")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_hhts_coverage.png"), dpi=220,
            bbox_inches="tight")
plt.close(fig)
print("wrote fig_hhts_coverage.png")

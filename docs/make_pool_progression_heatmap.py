"""Heatmap version of the five-zone pool progression figure.

Five representative zones (one per population-size quintile) are shown for
field of study and highest education.  Each panel is a category x zone heatmap
of the stage share minus the census share in percentage points; the three
columns are the raw SA4 slice, the CP pool and the expanded pool.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA"
SYNTH = r"Z:\Paper_Plan\synth_pop"

sys.path.insert(0, RUN)
sys.path.insert(0, SYNTH)

from zone_size_progression_v8 import (  # noqa: E402
    CSF, ZONES, read_pool, sa4_shares, zone_targets, RAW, CP, MIP,
)

TABLES = [
    ("qalfp", "Field of study", {
        1: "Nat. sciences", 2: "IT", 3: "Engineering", 4: "Architecture",
        5: "Agriculture", 6: "Health", 7: "Education", 8: "Management",
        9: "Society & culture", 10: "Creative arts", 11: "Food & hosp.",
        12: "Mixed field", 13: "Inadequately desc.", 14: "Not stated",
        15: "Not applicable", 16: "Overseas visitor",
    }),
    ("heap", "Highest education", {
        1: "Postgraduate", 2: "Grad dip / cert", 3: "Bachelor",
        4: "Adv dip / diploma", 5: "Cert I & II", 6: "Year 10+",
        7: "Cert III & IV", 8: "Year 9 or below", 9: "Supplementary",
        10: "Not stated", 11: "Not applicable",
    }),
]
STAGES = [("Raw", "Raw SA4 slice"), ("CP", "CP pool"),
          ("MIP", "Expanded pool")]


def main() -> None:
    raw, cp, mip = read_pool(RAW), read_pool(CP), read_pool(MIP)
    raw_s, cp_s, mip_s = sa4_shares(raw), sa4_shares(cp), sa4_shares(mip)
    tgt = zone_targets()

    geo = pd.read_csv(os.path.join(SYNTH, "sa2_zones_reference_357.csv"),
                      dtype={"sa2_code": str, "sa4_code": str})
    geo["sa2_code"] = geo.sa2_code.str.zfill(9)
    geo["sa4_code"] = geo.sa4_code.str.zfill(3)
    csf = pd.read_csv(CSF, dtype=str)
    csf["sa2"] = csf.sa2.str.zfill(9)
    csf["regucp"] = csf.regucp.str.zfill(2)
    both = geo.merge(csf, left_on="sa2_code", right_on="sa2", how="inner")
    sa4_to_reg = both.drop_duplicates("sa4_code").set_index("sa4_code").regucp
    geo["regucp"] = geo.sa4_code.map(sa4_to_reg)
    zone_sa4 = dict(zip(geo.sa2_code, geo.regucp))

    z = pd.read_csv(ZONES)
    z = z[z.arm == "cp_mip"][["zone", "person_target"]].dropna()
    z["zone"] = z.zone.astype(str).str.zfill(9)
    z = z.drop_duplicates("zone")
    z = z[z.zone.isin(tgt)]
    z["sa4"] = z.zone.map(zone_sa4)
    z = z.dropna(subset=["sa4"])
    z["sa4"] = z.sa4.str.zfill(2)
    z = z[z.sa4.isin(cp_s)]
    qs = np.quantile(z.person_target, [0.2, 0.4, 0.6, 0.8])
    z["quintile"] = np.digitize(z.person_target, qs) + 1

    reps = []
    for q in sorted(z.quintile.unique()):
        sub = z[z.quintile == q]
        med = sub.person_target.median()
        reps.append(sub.iloc[(sub.person_target - med).abs().argsort()[:1]])
    reps = pd.concat(reps)
    print(reps[["zone", "person_target", "quintile"]].to_string(index=False))

    fig, axes = plt.subplots(2, 3, figsize=(7.8, 7.4))
    all_vals = []
    panels = []
    for row, (table, title, labels) in enumerate(TABLES):
        codes0 = sorted(labels)
        mean_target = {c: 0.0 for c in codes0}
        for _, r in reps.iterrows():
            tt = tgt[r.zone][table]
            for c in codes0:
                mean_target[c] += tt.get(c, 0.0) / len(reps)
        codes = sorted(codes0, key=lambda c: -mean_target[c])
        for col, (stage, stage_label) in enumerate(STAGES):
            shares = {"Raw": raw_s, "CP": cp_s, "MIP": mip_s}[stage]
            mat = np.array([
                [100.0 * (shares[r.sa4][table].get(c, 0.0)
                          - tgt[r.zone][table].get(c, 0.0))
                 for _, r in reps.iterrows()]
                for c in codes
            ])
            all_vals.append(mat)
            panels.append((row, col, title, stage_label, labels, codes, mat))

    lim = max(1.0, float(np.ceil(np.max(np.abs(np.concatenate(
        [d.ravel() for d in all_vals]))))))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)

    for row, col, title, stage_label, labels, codes, mat in panels:
        ax = axes[row, col]
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(np.arange(len(reps)))
        ax.set_xticklabels([f"Q{int(q)}" for q in reps.quintile],
                           fontsize=7.0)
        ax.set_yticks(np.arange(len(codes)))
        if col == 0:
            ax.set_yticklabels([labels[c] for c in codes], fontsize=6.8)
        else:
            ax.set_yticklabels([])
        if row == 0:
            ax.set_title(stage_label, fontsize=9.0, pad=5)
        ax.tick_params(length=0)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center",
                        fontsize=5.6,
                        color="white" if abs(mat[i, j]) > 0.55 * lim
                        else "#222222")
        ax.set_xticks(np.arange(-.5, len(reps), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(codes), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)

    cax = fig.add_axes([0.24, 0.025, 0.52, 0.013])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Stage share minus census share (percentage points)",
                   fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    fig.suptitle("Pool progression for five representative zones across the "
                 "population-size quintiles", fontsize=9.2, y=0.985)
    fig.text(0.012, 0.72, "Field of study", rotation=90, va="center",
             fontsize=9.0, fontweight="bold")
    fig.text(0.012, 0.31, "Highest education", rotation=90, va="center",
             fontsize=9.0, fontweight="bold")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.09, top=0.89,
                        wspace=0.18, hspace=0.36)
    path = os.path.join(OUT, "fig_pool_progression_by_size.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()

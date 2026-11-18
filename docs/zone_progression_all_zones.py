"""All-zone category progression heatmap for four control tables.

Accumulates the Raw SA4 slice, the CP pool and the expanded pool over every
fitted zone, weighted by the zone's person target, and plots each category's
pooled share minus the pooled census share in percentage points.

The single-zone version (fig_zone_progression_all.png) was hard to read because
one zone's sampling noise dominated the pattern.  Pooling all zones removes
that noise and keeps the category detail.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import glob

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA"
SYNTH = r"Z:\Paper_Plan\synth_pop"

sys.path.insert(0, RUN)
sys.path.insert(0, SYNTH)

from zone_size_progression_v8 import (  # noqa: E402
    ZONES, read_pool, sa4_shares, zone_targets, RAW, CP, MIP,
)
import make_marginal_histograms as mmh  # noqa: E402

TABLES = [
    ("qalfp", "Field of study", {
        1: "Natural sciences", 2: "IT", 3: "Engineering", 4: "Architecture",
        5: "Agriculture", 6: "Health", 7: "Education", 8: "Management",
        9: "Society & culture", 10: "Creative arts", 11: "Food & hospitality",
        12: "Mixed field", 13: "Inadequately described", 14: "Not stated",
        15: "Not applicable", 16: "Overseas visitor",
    }),
    ("rlhp", "Relationship in household", {
        1: "Partner", 2: "Lone parent", 3: "Child under 15",
        4: "Dependent student", 5: "Non-dependent child",
        6: "Other relative", 7: "Lone / group / unrelated",
        9: "Visitor / other", 10: "Not applicable", 11: "Overseas visitor",
    }),
    ("stup", "Student status", {
        1: "Not attending", 2: "Full-time", 3: "Part-time",
        4: "Status not stated",
    }),
    ("heap", "Highest education", {
        1: "Postgraduate", 2: "Grad dip / cert", 3: "Bachelor",
        4: "Adv dip / diploma", 5: "Cert I & II", 6: "Year 10+",
        7: "Cert III & IV", 8: "Year 9 or below", 9: "Supplementary",
        10: "Not stated", 11: "Not applicable",
    }),
]
STAGES = [("Raw", "Raw"), ("CP", "CP"), ("MIP", "Expanded"),
          ("FIT", "Fitted")]


def main() -> None:
    raw, cp, mip = read_pool(RAW), read_pool(CP), read_pool(MIP)
    raw_s, cp_s, mip_s = sa4_shares(raw), sa4_shares(cp), sa4_shares(mip)

    tgt = zone_targets()
    zpt = pd.read_csv(os.path.join(RUN, "tables", "zone_pool_tvd.csv"),
                      usecols=["zone", "sa4"], dtype=str)
    zpt["zone"] = zpt.zone.str.zfill(9)
    zpt["sa4"] = zpt.sa4.str.zfill(2)
    zone_sa4 = dict(zip(zpt.zone, zpt.sa4))

    z = pd.read_csv(ZONES)
    z = z[z.arm == "cp_mip"][["zone", "person_target"]].dropna()
    z["zone"] = z.zone.astype(str).str.zfill(9)
    z = z.drop_duplicates("zone")
    z = z[z.zone.isin(tgt)]
    z["sa4"] = z.zone.map(zone_sa4)
    z = z.dropna(subset=["sa4"])
    z["sa4"] = z.sa4.str.zfill(2)
    z = z[z.sa4.isin(cp_s)]
    print(f"zones {len(z)}")

    controls = mmh.load_all_controls(os.path.join(SYNTH,
                                                  "data_sa2_controls_v8"))
    single = {}
    for k, ct in controls.items():
        ct["key"] = __import__("re").sub(r"^sa[12]", "", str(k).lower())
        if len(ct["attr_cols"]) == 1:
            single[ct["key"]] = ct

    def to_shares(counts):
        tot = sum(counts.values())
        return {k: v / tot for k, v in counts.items()} if tot else {}

    fit_share = {}
    table_keys = [t for t, _title, _labels in TABLES]
    for i, zone in enumerate(z.zone, 1):
        path = glob.glob(os.path.join(
            RUN, "fits", "mip_mip_fallback_unique", "**",
            f"zone_{zone}_synthetic_population.csv"), recursive=True)[0]
        got = mmh.aggregate_zone(path, single) or {}
        for table in table_keys:
            counts = {c: n for (k, c), n in got.items() if k == table}
            fit_share[(zone, table)] = to_shares(counts)
        if i % 50 == 0:
            print(f"  fitted shares {i}/{len(z)}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.4))
    axes = axes.ravel()
    all_vals = []
    panels = []
    for table, title, labels in TABLES:
        codes0 = sorted(labels)
        target_mean = {c: 0.0 for c in codes0}
        dev_sum = {name: {c: 0.0 for c in codes0} for name, _ in STAGES}
        n_zones = 0
        for _, r in z.iterrows():
            zone, sa4 = r.zone, r.sa4
            tt = tgt.get(zone, {}).get(table)
            if not tt:
                continue
            n_zones += 1
            for c in codes0:
                target_mean[c] += tt.get(c, 0.0)
            for name, shares in (("Raw", raw_s), ("CP", cp_s),
                                 ("MIP", mip_s), ("FIT", None)):
                sh = (fit_share[(zone, table)] if name == "FIT"
                      else shares[sa4][table])
                for c in codes0:
                    dev_sum[name][c] += 100.0 * (sh.get(c, 0.0)
                                                 - tt.get(c, 0.0))

        target_mean = {c: v / n_zones for c, v in target_mean.items()}
        codes = sorted(codes0, key=lambda c: -target_mean[c])
        dev = np.zeros((len(codes), len(STAGES)))
        for j, (name, _label) in enumerate(STAGES):
            for i, c in enumerate(codes):
                dev[i, j] = dev_sum[name][c] / n_zones
        all_vals.append(dev)
        panels.append((table, title, labels, codes, target_mean, dev))

    stacked = np.concatenate([d.ravel() for d in all_vals])
    lim = max(1.0, float(np.ceil(np.max(np.abs(stacked)))))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    cmap = LinearSegmentedColormap.from_list(
        "zero_blue_tails_red",
        [(0.0, "#B2182B"), (0.25, "#F7F7F7"), (0.5, "#2166AC"),
         (0.75, "#F7F7F7"), (1.0, "#B2182B")])

    for ax, (table, title, labels, codes, target, dev) in zip(axes, panels):
        im = ax.imshow(dev, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(len(STAGES)))
        ax.set_xticklabels([s for _n, s in STAGES], rotation=30, ha="right",
                           fontsize=7.0)
        ax.set_yticks(np.arange(len(codes)))
        ax.set_yticklabels([labels[c] for c in codes], fontsize=6.8)
        ax.set_title(title, fontsize=8.6, fontweight="bold", pad=5)
        ax.tick_params(length=0)
        for i in range(len(codes)):
            for j in range(len(STAGES)):
                v = dev[i, j]
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=5.7,
                        color="white" if (abs(v) > 0.55 * lim
                                          or abs(v) < 0.12 * lim)
                        else "#222222")
        ax.set_xticks(np.arange(-.5, len(STAGES), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(codes), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)

    cax = fig.add_axes([0.24, 0.028, 0.52, 0.012])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Gap magnitude (blue = near zero, red = large gap)",
                   fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    fig.suptitle("Mean per-zone progression from the raw slice to the fitted "
                 "population", fontsize=8.6, y=0.98)
    fig.subplots_adjust(bottom=0.10, hspace=0.42, wspace=0.62)
    path = os.path.join(OUT, "fig_zone_progression_all.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()

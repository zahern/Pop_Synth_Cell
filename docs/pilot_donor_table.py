"""Per-table, per-zone donor-balance check for the three pilot zones."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, r"Z:\Paper_Plan\runs\2026-11-13_v8")
sys.path.insert(0, r"Z:\Paper_Plan\synth_pop")

import pool_score_summary as pss  # noqa: E402

RUN = r"Z:\Paper_Plan\runs\2026-11-13_v8"
CP = r"Z:\Paper_Plan\runs\2026-11-11_canonical\pools\feasible_pool.csv"
DON = r"C:\Users\ahernz\AppData\Local\Temp\opencode\balance_donors_pilot.csv"
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA\fig_pilot_donor_check.png"
TABLE = os.path.join(RUN, "tables", "pilot_donor_check.csv")

ZONES = ["112011236", "112011237", "112011238"]
LABELS = {
    "heap": "Highest education", "hind": "Household income",
    "hrsp": "Hours worked", "incp": "Personal income", "indp": "Industry",
    "mrerd": "Mortgage repayment", "mstp": "Marital status",
    "mtwp": "Travel to work", "nprd": "Persons per dwelling",
    "occp": "Occupation", "qalfp": "Field of study",
    "qallp": "Qualification level", "rlhp": "Relationship in household",
    "rntrd": "Rent", "sexp": "Sex", "strd": "Dwelling structure",
    "stup": "Student status", "typp": "Education institution",
    "vehrd": "Vehicles",
}


def main() -> None:
    cp = pd.read_csv(CP, low_memory=False)
    cp.columns = [str(c).lower() for c in cp.columns]
    cp["regucp"] = cp.regucp.astype(str).str.zfill(2)
    don = pd.read_csv(DON, low_memory=False)
    don.columns = [str(c).lower() for c in don.columns]
    don["_target_sa2"] = don._target_sa2.astype(str).str.zfill(9)

    geo = pd.read_csv(r"Z:\Paper_Plan\synth_pop\sa2_zones_reference_357.csv",
                      dtype={"sa2_code": str, "sa4_code": str})
    geo["sa2_code"] = geo.sa2_code.str.zfill(9)
    geo["sa4_code"] = geo.sa4_code.str.zfill(3)
    csf = pd.read_csv(pss.CSF, dtype=str)
    csf["sa2"] = csf.sa2.str.zfill(9)
    csf["regucp"] = csf.regucp.str.zfill(2)
    both = geo.merge(csf, left_on="sa2_code", right_on="sa2", how="inner")
    s2r = both.drop_duplicates("sa4_code").set_index("sa4_code").regucp
    geo["regucp"] = geo.sa4_code.map(s2r)
    zone_sa4 = dict(zip(geo.sa2_code, geo.regucp))

    tgt = pss.targets()
    rows = []
    for zone in ZONES:
        sa4 = zone_sa4[zone]
        base_df = cp[cp.regucp == sa4]
        don_df = don[don._target_sa2 == zone]
        for table, hh in pss.TABLES:
            tt = tgt[zone][table]
            base = pss.counts(base_df, table, bool(hh))
            add = pss.counts(don_df, table, bool(hh)) if len(don_df) else {}
            both = {k: base.get(k, 0) + add.get(k, 0)
                    for k in set(base) | set(add)}
            max_tvd = 1.0 - min(v for v in tt.values() if v == v)
            b = pss.tvd(pss.to_shares(base), tt)
            f = pss.tvd(pss.to_shares(both), tt)
            rows.append({
                "zone": zone, "table": table,
                "base_pct": 100 * b / max_tvd,
                "donor_pct": 100 * f / max_tvd,
                "delta_pp": 100 * (f - b) / max_tvd,
            })

    d = pd.DataFrame(rows)
    d.to_csv(TABLE, index=False)
    print(d.groupby("table")[["base_pct", "donor_pct", "delta_pp"]]
          .mean().round(2).to_string())

    pv = d.pivot(index="table", columns="zone", values="delta_pp")
    pv = pv.reindex([t for t in LABELS if t in pv.index])
    fig, ax = plt.subplots(figsize=(5.8, 6.8))
    vals = pv.to_numpy(dtype=float)
    lim = max(0.02, float(np.nanmax(np.abs(vals))))
    im = ax.imshow(vals, cmap="RdBu_r", aspect="auto",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
    ax.set_xticks(np.arange(pv.shape[1]))
    ax.set_xticklabels(pv.columns, fontsize=8)
    ax.set_yticks(np.arange(pv.shape[0]))
    ax.set_yticklabels([LABELS[t] for t in pv.index], fontsize=8)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:+.1f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(vals[i, j]) > 0.55 * lim else "#222222")
    ax.set_title("Donor-balanced minus CP relative TVD (pp)\n"
                 "blue = improvement, red = worse", fontsize=9)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Change in relative TVD (pp)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT, TABLE)


if __name__ == "__main__":
    main()

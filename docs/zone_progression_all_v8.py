"""Marginal progression for SA2 112011236 across four control tables.

Each panel shows the raw SA4 slice, the CP pool and the expanded pool as a
difference from the census marginal (percentage points).  The symmetric-log
axis keeps both sub-point and double-digit gaps visible.  Tables with a known
code-domain mismatch (nprd, strd) are deliberately excluded here; the v9 run
fixes strd and the nprd leak is reported separately.

Run locally over the Z: share.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ZONE = "112011236"
REGUCP = "4"
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA"

RAW = r"Z:\Population_Synth\LINKED_DATA_clean_hhts_qalfp.csv"
CP = r"Z:\Paper_Plan\runs\2026-11-11_canonical\pools\feasible_pool.csv"
MIP = r"Z:\Paper_Plan\runs\2026-11-13_v8\pools\feasible_pool_expanded_released_hhts.csv"
CTRL = r"Z:\Paper_Plan\synth_pop\data_sa2_controls_v8"
FIT = (r"Z:\Paper_Plan\runs\2026-11-13_v8\fits\cp_mip_fallback\chunk_0"
       rf"\zone_{ZONE}_synthetic_population.csv")

TABLES = [
    ("qalfp", "Field of study", False),
    ("rlhp", "Relationship in household", False),
    ("stup", "Student status", False),
    ("heap", "Highest education", False),
]

LABELS = {
    "qalfp": {1: "Nat. sci.", 2: "IT", 3: "Eng.", 4: "Arch.", 5: "Agri.",
              6: "Health", 7: "Educ.", 8: "Mgmt", 9: "Society",
              10: "Creative", 11: "Food/hosp.", 12: "Mixed",
              13: "Inad. desc.", 14: "Not stated", 15: "N/A",
              16: "Overseas"},
    "rlhp": {1: "Partner", 2: "Lone parent", 3: "Child <15",
             4: "Dep. student", 5: "Non-dep. child", 6: "Other related",
             7: "Non-family", 9: "Visitor/other", 10: "N/A",
             11: "Overseas"},
    "stup": {1: "Not attending", 2: "Full-time", 3: "Part-time",
             4: "Not stated", 5: "Overseas"},
    "heap": {1: "Postgrad", 2: "Grad dip", 3: "Bachelor", 4: "Adv dip",
             5: "Cert I/II", 6: "Yr 10+", 7: "Cert III/IV", 8: "Yr 9-",
             9: "Suppl.", 10: "Not stated", 11: "N/A", 12: "Overseas"},
}

sys.path.insert(0, r"Z:\Paper_Plan\synth_pop")
from CensusRecordImporter import CensusControlTable  # noqa: E402

_ct = CensusControlTable.__new__(CensusControlTable)
_ct.define_mapping()
MAPS = _ct.mappings


def shares_from_codes(codes: pd.Series) -> dict[int, float]:
    vc = codes.dropna().astype(int).value_counts(normalize=True)
    return {int(k): float(v) for k, v in vc.items()}


def control_target(table: str) -> dict[int, float]:
    d = pd.read_csv(os.path.join(CTRL, f"sa2{table}.csv"), skiprows=9,
                    usecols=[0, 1, 2, 3], dtype=str)
    d.columns = ["counting", "sa2", "label", "count"]
    d = d[d.sa2.str.zfill(9) == ZONE]
    d["code"] = d.label.map(MAPS[table])
    d["count"] = pd.to_numeric(d["count"], errors="coerce")
    d = d.dropna(subset=["code", "count"])
    tot = d.groupby(d.code.astype(int))["count"].sum()
    return {int(k): float(v / tot.sum()) for k, v in tot.items()}


def pool_shares(df: pd.DataFrame, table: str, household: bool) -> dict[int, float]:
    if household:
        s = df.groupby("abshid")[table].first()
    else:
        s = df[table]
    return shares_from_codes(s)


def read_subset(path: str, wanted: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c.lower() in wanted,
                     low_memory=False)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def main() -> None:
    wanted = {"regucp", "abshid"} | {t for t, _l, _h in TABLES}
    raw = read_subset(RAW, wanted)
    raw = raw[raw.regucp.astype(str).str.zfill(2) == REGUCP.zfill(2)]
    cp = read_subset(CP, wanted)
    cp = cp[cp.regucp.astype(str).str.zfill(2) == REGUCP.zfill(2)]
    mip = read_subset(MIP, wanted)
    mip = mip[mip.regucp.astype(str).str.zfill(2) == REGUCP.zfill(2)]

    fit = pd.read_csv(FIT, low_memory=False)
    fit.columns = [str(c).lower() for c in fit.columns]

    rows = []
    panels = []
    for table, title, household in TABLES:
        tgt = control_target(table)
        series = {
            "Raw": pool_shares(raw, table, household),
            "CP": pool_shares(cp, table, household),
            "MIP": pool_shares(mip, table, household),
        }
        fitted = pool_shares(fit, table, household)
        codes = sorted(set(tgt) | {c for s in series.values() for c in s})
        codes = [c for c in codes if c in LABELS[table]]
        panels.append((table, title, codes, tgt, series))
        for c in codes:
            rows.append({
                "table": table, "code": c, "label": LABELS[table][c],
                "target": 100 * tgt.get(c, 0.0),
                **{f"{k.lower()}": 100 * s.get(c, 0.0)
                   for k, s in series.items()},
                "fitted": 100 * fitted.get(c, 0.0),
            })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "zone_progression_all_v8.csv"), index=False)

    colours = {"Raw": "#4D4D4D", "CP": "#00796B", "MIP": "#5E35B1"}
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4))
    linthresh = 0.1
    for ax, (table, title, codes, tgt, series) in zip(axes.ravel(), panels):
        y = np.arange(len(codes))
        ax.axvspan(-0.2, 0.2, color="#EEEEEE", zorder=0, linewidth=0)
        ax.axvline(0, color="#333333", linewidth=0.9, zorder=2)
        for name, s in series.items():
            diff = np.array([100 * (s.get(c, 0.0) - tgt.get(c, 0.0))
                             for c in codes])
            ax.scatter(diff, y, s=34, color=colours[name], label=name,
                       edgecolor="white", linewidth=0.5, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[table][c] for c in codes], fontsize=7)
        ax.set_title(f"{table}: {title}", fontsize=9, fontweight="bold")
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_xlim(-60, 60)
        ax.set_xticks([-50, -10, -1, -0.1, 0, 0.1, 1, 10, 50])
        ax.set_xticklabels(["-50", "-10", "-1", "-0.1", "0", "0.1",
                            "1", "10", "50"], fontsize=7)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-0.6, len(codes) - 0.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"Marginal progression, SA2 {ZONE}: each pool stage minus the "
                 f"census marginal", fontsize=11, y=0.945)
    fig.supxlabel("Share minus census marginal (percentage points; symmetric "
                  "log scale; grey band is $\\pm$0.2 pp)", fontsize=8.5,
                  y=0.035)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    path = os.path.join(OUT, "fig_zone_progression_all.png")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

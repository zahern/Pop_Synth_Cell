"""Pool-target marginal distance by zone size, and representative-zone progressions.

For every fitted zone, compare the census marginal of each of the 19 control
tables with the raw SA4 slice, the CP pool and the expanded pool.  The pools are
constant within an SA4, so this is cheap once the SA4 shares are computed.

Writes:
  zone_pool_tvd.csv                 per-zone TVD for Raw/CP/MIP
  fig_pool_tvd_by_size.png          mean TVD by population quintile
  fig_pool_progression_by_size.png  qalfp and nprd progression for 5 zones
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = r"Z:\Paper_Plan\runs\2026-11-13_v8"
OUT = r"Z:\Paper_Plan\Population_Synthesiser___GAN_SA"
SYNTH = r"Z:\Paper_Plan\synth_pop"

RAW = r"Z:\Population_Synth\LINKED_DATA_clean_hhts_qalfp.csv"
CP = r"Z:\Paper_Plan\runs\2026-11-11_canonical\pools\feasible_pool.csv"
MIP = r"Z:\Paper_Plan\runs\2026-11-13_v8\pools\feasible_pool_expanded_released_hhts.csv"
CTRL = os.path.join(SYNTH, "data_sa2_controls_v8")
CSF = os.path.join(SYNTH, "csf_sa.csv")
ZONES = os.path.join(HERE, "tables", "arm_zones.csv")

sys.path.insert(0, SYNTH)
from CensusRecordImporter import CensusControlTable  # noqa: E402

TABLES = [("heap", 0), ("mstp", 0), ("mtwp", 0), ("occp", 0), ("rlhp", 0),
          ("sexp", 0), ("qalfp", 0), ("qallp", 0), ("incp", 0), ("hrsp", 0),
          ("indp", 0), ("typp", 0), ("stup", 0), ("hind", 1), ("mrerd", 1),
          ("nprd", 1), ("rntrd", 1), ("vehrd", 1), ("strd", 1)]
HH = {t for t, h in TABLES if h}

_ct = CensusControlTable.__new__(CensusControlTable)
_ct.define_mapping()
MAPS = _ct.mappings


def read_pool(path: str) -> pd.DataFrame:
    wanted = {"regucp", "abshid"} | {t for t, _h in TABLES}
    df = pd.read_csv(path, usecols=lambda c: c.lower() in wanted,
                     low_memory=False)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def shares(codes: pd.Series) -> dict[int, float]:
    vc = codes.dropna().astype(int).value_counts(normalize=True)
    return {int(k): float(v) for k, v in vc.items()}


def sa4_shares(df: pd.DataFrame) -> dict[str, dict[str, dict[int, float]]]:
    out: dict[str, dict[str, dict[int, float]]] = {}
    df = df.copy()
    df["regucp"] = df["regucp"].astype(str).str.zfill(2)
    for reg, sub in df.groupby("regucp"):
        per_table = {}
        for table, _h in TABLES:
            if table in HH:
                s = sub.groupby("abshid")[table].first()
            else:
                s = sub[table]
            per_table[table] = shares(s)
        out[reg] = per_table
    return out


def zone_targets() -> dict[str, dict[str, dict[int, float]]]:
    out: dict[str, dict[str, dict[int, float]]] = {}
    for table, _h in TABLES:
        d = pd.read_csv(os.path.join(CTRL, f"sa2{table}.csv"), skiprows=9,
                        usecols=[0, 1, 2, 3], dtype=str)
        d.columns = ["c", "sa2", "label", "n"]
        d["n"] = pd.to_numeric(d["n"], errors="coerce")
        d["code"] = d["label"].map(MAPS[table])
        d = d.dropna(subset=["code", "n"])
        d["sa2"] = d["sa2"].astype(str).str.zfill(9)
        d["code"] = d.code.astype(int)
        for z, sub in d.groupby("sa2"):
            g = sub.groupby("code")["n"].sum()
            tot = g.sum()
            if tot <= 0:
                continue
            out.setdefault(z, {})[table] = {int(k): float(v / tot)
                                            for k, v in g.items()}
    return out


def tvd(a: dict[int, float], b: dict[int, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def main() -> None:
    raw = read_pool(RAW)
    cp = read_pool(CP)
    mip = read_pool(MIP)
    raw_s = sa4_shares(raw)
    cp_s = sa4_shares(cp)
    mip_s = sa4_shares(mip)

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
    print("zones", len(z))

    rows = []
    for _, r in z.iterrows():
        zone, sa4 = r.zone, r.sa4
        if sa4 not in cp_s:
            continue
        for table, _h in TABLES:
            t = tgt[zone].get(table)
            if not t:
                continue
            rows.append({
                "zone": zone, "sa4": sa4, "table": table,
                "person_target": r.person_target,
                "raw": tvd(raw_s[sa4][table], t),
                "cp": tvd(cp_s[sa4][table], t),
                "mip": tvd(mip_s[sa4][table], t),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "tables", "zone_pool_tvd.csv"), index=False)
    print(out.groupby("table")[["raw", "cp", "mip"]].mean().round(4).to_string())

    # size quintiles over zones
    zq = z.copy()
    qs = np.quantile(zq.person_target, [0.2, 0.4, 0.6, 0.8])
    zq["quintile"] = np.digitize(zq.person_target, qs) + 1
    d = out.merge(zq[["zone", "quintile"]], on="zone", how="inner")

    g = (d.groupby(["quintile", "table"])[["raw", "cp", "mip"]].mean()
         .reset_index())
    # mean over all 19 control tables
    tab_mean = d.groupby("quintile")[["raw", "cp", "mip"]].mean()
    print("\nmean TVD over 19 tables by quintile")
    print(tab_mean.round(4).to_string())

    colours = {"raw": "#9E9E9E", "cp": "#4FB3A0", "mip": "#7B5EA7"}
    labels = {"raw": "Raw SA4 slice", "cp": "CP pool", "mip": "Expanded pool"}
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for key in ("raw", "cp", "mip"):
        ax.plot(tab_mean.index, tab_mean[key], marker="o", color=colours[key],
                label=labels[key], linewidth=1.6, markersize=5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["Q1\nsmallest", "Q2", "Q3", "Q4", "Q5\nlargest"])
    ax.set_ylabel("Mean TVD to the census marginal (19 tables)")
    ax.set_xlabel("Zone population quintile")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Pool-target marginal distance by zone size", fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_pool_tvd_by_size.png")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)

    # representative zone per quintile: closest to the quintile median size
    reps = []
    for q in sorted(zq.quintile.unique()):
        sub = zq[zq.quintile == q]
        med = sub.person_target.median()
        reps.append(sub.iloc[(sub.person_target - med).abs().argsort()[:1]])
    reps = pd.concat(reps)
    print("\nrepresentative zones")
    print(reps[["zone", "person_target", "quintile"]].to_string(index=False))

    # qalfp and heap progression for the representatives
    panel_tables = [("qalfp", "Field of study"), ("heap", "Highest education")]
    short = {
        "qalfp": {1: "Nat. sci.", 2: "IT", 3: "Eng.", 4: "Arch.",
                  5: "Agri.", 6: "Health", 7: "Educ.", 8: "Mgmt",
                  9: "Society", 10: "Creative", 11: "Food/hosp.",
                  12: "Mixed", 13: "Inad. desc.", 14: "Not stated",
                  15: "N/A", 16: "Overseas"},
        "heap": {1: "Postgrad", 2: "Grad dip", 3: "Bachelor", 4: "Adv dip",
                 5: "Cert I/II", 6: "Yr 10+", 7: "Cert III/IV", 8: "Yr 9-",
                 9: "Suppl.", 10: "Not stated", 11: "N/A", 12: "Overseas"},
    }
    colours = {"Raw": "#4D4D4D", "CP": "#00796B", "MIP": "#5E35B1"}
    fig, axes = plt.subplots(len(reps), 2, figsize=(10.6, 2.25 * len(reps)),
                             squeeze=False)
    for i, (_, r) in enumerate(reps.iterrows()):
        zone, sa4 = r.zone, zone_sa4.get(r.zone)
        for j, (table, title) in enumerate(panel_tables):
            ax = axes[i, j]
            t = tgt.get(zone, {}).get(table, {})
            series = {
                "Raw": raw_s[sa4][table],
                "CP": cp_s[sa4][table],
                "MIP": mip_s[sa4][table],
            }
            codes = sorted(t)
            y = np.arange(len(codes))
            ax.axvspan(-0.2, 0.2, color="#EEEEEE", zorder=0, linewidth=0)
            ax.axvline(0, color="#333333", linewidth=0.8, zorder=2)
            for name, s in series.items():
                diff = np.array([100 * (s.get(c, 0.0) - t.get(c, 0.0))
                                 for c in codes])
                ax.scatter(diff, y, s=22, color=colours[name], label=name,
                           edgecolor="white", linewidth=0.4, zorder=4)
            ax.set_yticks(y)
            ax.set_yticklabels([short[table].get(c, str(c)) for c in codes],
                               fontsize=5.5)
            ax.set_xscale("symlog", linthresh=0.1)
            ax.set_xlim(-60, 60)
            ax.set_xticks([-50, -10, -1, -0.1, 0, 0.1, 1, 10, 50])
            ax.set_xticklabels(["-50", "-10", "-1", "-0.1", "0", "0.1",
                                "1", "10", "50"], fontsize=6)
            ax.grid(axis="x", color="#EEEEEE", linewidth=0.4)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylim(-0.6, len(codes) - 0.4)
            if i == 0:
                ax.set_title(title, fontsize=8.5, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"Q{int(r.quintile)} · {int(r.person_target):,}\n"
                              f"{zone}", fontsize=6.5)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Pool progression for representative zones across the size "
                 "distribution: each stage minus the census marginal",
                 fontsize=10, y=0.965)
    fig.supxlabel("Share minus census marginal (percentage points; symmetric "
                  "log scale; grey band is $\\pm$0.2 pp)", fontsize=8, y=0.012)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    path = os.path.join(OUT, "fig_pool_progression_by_size.png")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()

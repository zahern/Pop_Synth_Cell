"""Generalised pool score: mean marginal TVD over the 19 control tables.

Stages compared before fitting:
  raw SA4 slice, CP pool, expanded pool, CP + fallback, expanded + fallback.

The fallback carriers are zone-tagged, so the last two stages are built per zone
(SA4 slice + that zone's carriers).  Writes a CSV and a compact bar chart.
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
FB = r"Z:\Paper_Plan\runs\2026-11-13_v8\pools\fallback_carriers.csv"
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


def read_subset(path: str, extra=()) -> pd.DataFrame:
    wanted = {"regucp", "abshid", "_target_sa2"} | {t for t, _h in TABLES} | set(extra)
    df = pd.read_csv(path, usecols=lambda c: c.lower() in wanted,
                     low_memory=False)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def counts(df: pd.DataFrame, table: str, household: bool) -> dict[int, int]:
    if household:
        s = df.groupby("abshid")[table].first()
    else:
        s = df[table]
    vc = s.dropna().astype(int).value_counts()
    return {int(k): int(v) for k, v in vc.items()}


def to_shares(c: dict[int, int]) -> dict[int, float]:
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()} if tot else {}


def tvd(a: dict[int, float], b: dict[int, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def targets() -> dict[str, dict[str, dict[int, float]]]:
    out = {}
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
            if g.sum() > 0:
                out.setdefault(z, {})[table] = to_shares(
                    {int(k): int(v) for k, v in g.items()})
    return out


def main() -> None:
    raw = read_subset(RAW)
    cp = read_subset(CP)
    mip = read_subset(MIP)
    fb = read_subset(FB)

    raw["regucp"] = raw.regucp.astype(str).str.zfill(2)
    cp["regucp"] = cp.regucp.astype(str).str.zfill(2)
    mip["regucp"] = mip.regucp.astype(str).str.zfill(2)
    fb["_target_sa2"] = fb._target_sa2.astype(str).str.zfill(9)

    raw_by = {r: g for r, g in raw.groupby("regucp")}
    cp_by = {r: g for r, g in cp.groupby("regucp")}
    mip_by = {r: g for r, g in mip.groupby("regucp")}
    fb_by = {z: g for z, g in fb.groupby("_target_sa2")}

    # per-SA4 counts
    raw_c = {r: {t: counts(g, t, t in HH) for t, _h in TABLES}
             for r, g in raw_by.items()}
    cp_c = {r: {t: counts(g, t, t in HH) for t, _h in TABLES}
            for r, g in cp_by.items()}
    mip_c = {r: {t: counts(g, t, t in HH) for t, _h in TABLES}
             for r, g in mip_by.items()}
    fb_c = {z: {t: counts(g, t, t in HH) for t, _h in TABLES}
            for z, g in fb_by.items()}

    tgt = targets()
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

    stages = ["Raw", "CP", "Expanded", "CP + fallback", "Expanded + fallback"]
    rows = []
    for _, r in z.iterrows():
        zone, sa4 = r.zone, r.sa4
        if sa4 not in cp_c:
            continue
        for table, _h in TABLES:
            t = tgt[zone].get(table)
            if not t:
                continue
            f = fb_c.get(zone, {}).get(table, {})
            # combine counts, then shares
            cp_plus = {k: cp_c[sa4][table].get(k, 0) + f.get(k, 0)
                       for k in set(cp_c[sa4][table]) | set(f)}
            mip_plus = {k: mip_c[sa4][table].get(k, 0) + f.get(k, 0)
                        for k in set(mip_c[sa4][table]) | set(f)}
            vals = {
                "Raw": tvd(to_shares(raw_c[sa4][table]), t),
                "CP": tvd(to_shares(cp_c[sa4][table]), t),
                "Expanded": tvd(to_shares(mip_c[sa4][table]), t),
                "CP + fallback": tvd(to_shares(cp_plus), t),
                "Expanded + fallback": tvd(to_shares(mip_plus), t),
            }
            max_tvd = 1.0 - min(v for v in t.values() if v == v)
            rows.append({"zone": zone, "table": table, "max_tvd": max_tvd,
                         **vals})

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "tables", "pool_score_by_zone_table.csv"),
             index=False)

    summary = d[stages].mean().to_frame("mean_tvd")
    summary["pct_vs_raw"] = 100 * (summary.mean_tvd / summary.loc["Raw", "mean_tvd"] - 1)
    summary.to_csv(os.path.join(HERE, "tables", "pool_score_summary.csv"))
    print(summary.round(4).to_string())

    colours = ["#4D4D4D", "#00796B", "#5E35B1", "#2E8B57", "#8E44AD"]
    coverage = {"Raw": 0.0, "CP": 0.0, "Expanded": 13.3,
                "CP + fallback": 100.0, "Expanded + fallback": 100.0}
    cov = np.array([coverage[s] for s in stages])
    tvd_score = 1 - summary.mean_tvd.to_numpy() / summary.mean_tvd.max()
    composite = 0.5 * tvd_score + 0.5 * cov / 100.0

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    x = np.arange(len(stages))
    ax = axes[0]
    bars = ax.bar(x, summary.mean_tvd, color=colours, edgecolor="#333333",
                  linewidth=0.5, width=0.62)
    for i, (b, v) in enumerate(zip(bars, summary.mean_tvd)):
        pct = summary.pct_vs_raw.iloc[i]
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0005,
                f"{v:.4f}\n({pct:+.1f}%)", ha="center", va="bottom",
                fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=6.8, rotation=22, ha="right")
    ax.set_ylabel("Mean TVD to the census marginal (19 tables)")
    ax.set_ylim(0, summary.mean_tvd.max() * 1.24)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(a) Marginal distance (lower is better)", fontsize=9)

    ax = axes[1]
    bars = ax.bar(x, cov, color=colours, edgecolor="#333333",
                  linewidth=0.5, width=0.62)
    for b, v in zip(bars, cov):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=6.8, rotation=22, ha="right")
    ax.set_ylabel("Zero-capacity cells with a carrier (%)")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(b) Support coverage (higher is better)", fontsize=9)

    ax = axes[2]
    bars = ax.bar(x, composite, color=colours, edgecolor="#333333",
                  linewidth=0.5, width=0.62)
    for b, v in zip(bars, composite):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=6.8, rotation=22, ha="right")
    ax.set_ylabel("Composite pool score (0-1)")
    ax.set_ylim(0, 0.62)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(c) Equal-weight composite", fontsize=9)

    fig.suptitle("Pool score before fitting: marginal distance, support "
                 "coverage and their equal-weight composite", fontsize=10.5,
                 y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_pool_score_summary.png")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    print("composite:", dict(zip(stages, composite.round(3))))


if __name__ == "__main__":
    main()

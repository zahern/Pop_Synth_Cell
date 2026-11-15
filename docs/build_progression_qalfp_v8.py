"""Build per-zone qalfp marginal progression for the v8 dashboard.

Same payload shape as ``build_progression_qalfp.py`` but sourced from the v8
run: the corrected control tables, the canonical CP pool, the v8 expanded pool,
and the CP-MIP+fallback fitted populations (the arm used in the paper's worked
example).

Writes dashboard/data/progression_qalfp.js.
"""
from __future__ import annotations

import glob
import json
import os
import re

import pandas as pd

PAPER = r"Z:/Paper_Plan/Population_Synthesiser___GAN_SA"
SYNTH = r"Z:/Paper_Plan/synth_pop"
RUN = r"Z:/Paper_Plan/runs/2026-11-13_v8"
OUT = os.path.join(PAPER, "dashboard", "data")
os.makedirs(OUT, exist_ok=True)

MICRO = r"Z:/Population_Synth/LINKED_DATA_clean_hhts_qalfp.csv"
CP = r"Z:/Paper_Plan/runs/2026-11-11_canonical/pools/feasible_pool.csv"
EXP = os.path.join(RUN, "pools", "feasible_pool_expanded_released_hhts.csv")
CTRL = os.path.join(SYNTH, "data_sa2_controls_v8", "sa2qalfp.csv")
FIT_ROOT = os.path.join(RUN, "fits", "cp_mip_fallback")

QALFP = {
    "Natural and Physical Sciences": 1,
    "Information Technology": 2,
    "Engineering and Related Technologies": 3,
    "Architecture and Building": 4,
    "Agriculture, Environmental and Related Studies": 5,
    "Health": 6,
    "Education": 7,
    "Management and Commerce": 8,
    "Society and Culture": 9,
    "Creative Arts": 10,
    "Food, Hospitality and Personal Services": 11,
    "Mixed Field Programmes": 12,
    "Field of study inadequately described": 13,
    "Field of study not stated": 14,
    "Not applicable": 15,
    "Overseas visitor": 16,
}
INV = {v: k for k, v in QALFP.items()}

zones = [ln.strip() for ln in
         open(os.path.join(SYNTH, "pbs", "zones_367.txt")) if ln.strip()]
csf = pd.read_csv(os.path.join(SYNTH, "csf_sa.csv"), dtype=str)
zone_sa4 = dict(zip(csf["sa2"].str.zfill(9), csf["regucp"].astype(str)))

ct = pd.read_csv(CTRL, skiprows=9, usecols=[0, 1, 2, 3])
ct.columns = ["counting", "sa2", "category", "count"]
ct["sa2"] = ct["sa2"].astype(str).str.zfill(9)
ct["code"] = ct["category"].map(QALFP)
ct = ct.dropna(subset=["code"])
ct["code"] = ct["code"].astype(int)
target = {z: {int(c): float(v)
              for c, v in sub.groupby("code")["count"].sum().items()}
          for z, sub in ct.groupby("sa2")}

# The HHTS extract is already recoded to the control codebook, so no shift.
raw = pd.read_csv(MICRO, usecols=["REGUCP", "QALFP"], low_memory=False)
raw["REGUCP"] = raw["REGUCP"].astype(str)
raw["QALFP"] = pd.to_numeric(raw["QALFP"], errors="coerce")
raw = raw.dropna(subset=["QALFP"])
raw_c = {reg: sub["QALFP"].astype(int).value_counts().to_dict()
         for reg, sub in raw.groupby("REGUCP")}

cp = pd.read_csv(CP, usecols=["REGUCP", "QALFP"], low_memory=False)
cp["REGUCP"] = cp["REGUCP"].astype(str)
cp["QALFP"] = pd.to_numeric(cp["QALFP"], errors="coerce")
cp = cp.dropna(subset=["QALFP"])
cp_c = {reg: sub["QALFP"].astype(int).value_counts().to_dict()
        for reg, sub in cp.groupby("REGUCP")}

exp = pd.read_csv(EXP, usecols=["REGUCP", "QALFP", "_target_sa2"],
                  low_memory=False)
exp["QALFP"] = pd.to_numeric(exp["QALFP"], errors="coerce")
exp = exp.dropna(subset=["QALFP"])
exp["QALFP"] = exp["QALFP"].astype(int)
exp["_target_sa2"] = pd.to_numeric(exp["_target_sa2"], errors="coerce")
base = exp[exp["_target_sa2"].isna()].copy()
base["REGUCP"] = base["REGUCP"].astype(str)
base_c = {reg: sub["QALFP"].value_counts().to_dict()
          for reg, sub in base.groupby("REGUCP")}
added = exp[exp["_target_sa2"].notna()].copy()
added["_target_sa2"] = (added["_target_sa2"].astype(int).astype(str)
                        .str.zfill(9))
added_c = {z: sub["QALFP"].value_counts().to_dict()
           for z, sub in added.groupby("_target_sa2")}

fit_files = {}
for p in glob.glob(os.path.join(FIT_ROOT, "**", "zone_*.csv"), recursive=True):
    m = re.search(r"zone_(\d{9})(?:_(?:synthetic|new)_population)?\.csv$",
                  os.path.basename(p))
    if not m:
        continue
    fit_files[m.group(1)] = p


def pct(counts):
    total = sum(counts.values())
    return {int(k): (100.0 * v / total if total else 0.0)
            for k, v in counts.items()}


def vec(counts):
    return [round(pct(counts).get(c, 0.0), 3) for c in range(1, 17)]


payload = {"categories": [str(c) for c in range(1, 17)],
           "labels": [INV[c] for c in range(1, 17)],
           "zones": {}}
for z in zones:
    fit_counts = {}
    fit_reg = None
    if z in fit_files:
        d = pd.read_csv(fit_files[z], low_memory=False)
        col = "QALFP" if "QALFP" in d.columns else "qalfp"
        vals = pd.to_numeric(d[col], errors="coerce").dropna().astype(int)
        fit_counts = vals.value_counts().to_dict()
        regcol = next((c for c in ("regucp", "REGUCP") if c in d.columns), None)
        if regcol:
            rv = pd.to_numeric(d[regcol], errors="coerce").dropna()
            if not rv.empty:
                fit_reg = str(int(rv.iloc[0]))
    sa4 = zone_sa4.get(z) or fit_reg
    if sa4 is None:
        continue
    cp_counts = cp_c.get(sa4, {})
    mip_counts = dict(cp_counts)
    for c, n in added_c.get(z, {}).items():
        mip_counts[c] = mip_counts.get(c, 0) + n
    payload["zones"][z] = {
        "target": vec(target.get(z, {})),
        "raw": vec(raw_c.get(sa4, {})),
        "cp": vec(cp_counts),
        "mip": vec(mip_counts),
        "fitted": vec(fit_counts),
    }

with open(os.path.join(OUT, "progression_qalfp.js"), "w",
          encoding="utf-8") as fh:
    fh.write("window.DASH_PROGRESSION=")
    json.dump(payload, fh, separators=(",", ":"))
    fh.write(";\n")
print("wrote progression_qalfp.js", len(payload["zones"]), "zones")

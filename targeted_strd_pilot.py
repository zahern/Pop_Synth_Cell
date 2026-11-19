"""Targeted dwelling-structure top-up pilot.

Observed donors are only ~12% semi-detached against a 30% target in the test
zones.  This script takes renter households from the global census, recodes the
dwelling structure on every member row, and appends them until the target share
is approached.  It is a pilot for the construction pass, not yet rule-checked.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"Z:\Paper_Plan\runs\2026-11-13_v8")
sys.path.insert(0, r"Z:\Paper_Plan\synth_pop")

import pool_score_summary as pss  # noqa: E402

CP = r"Z:\Paper_Plan\runs\2026-11-11_canonical\pools\feasible_pool.csv"
DON = r"C:\Users\ahernz\AppData\Local\Temp\opencode\balance_donors_iter_qalfp.csv"
CENSUS = r"Z:\Population_Synth\LINKED_DATA_clean.csv"
OUT = r"C:\Users\ahernz\AppData\Local\Temp\opencode\targeted_strd_pilot.csv"
ZONES = ["112011236", "112011237", "112011238"]


def main() -> None:
    cp = pd.read_csv(CP, low_memory=False)
    cp.columns = [str(c).lower() for c in cp.columns]
    cp["regucp"] = cp.regucp.astype(str).str.zfill(2)
    don = pd.read_csv(DON, low_memory=False)
    don.columns = [str(c).lower() for c in don.columns]
    don["_target_sa2"] = don._target_sa2.astype(str).str.zfill(9)
    cen = pd.read_csv(CENSUS, low_memory=False)
    cen.columns = [str(c).lower() for c in cen.columns]
    if "qalfp" in cen.columns:
        cen["qalfp"] = pd.to_numeric(cen["qalfp"], errors="coerce").replace(
            {12: 13, 13: 14, 14: 15})

    # zone -> SA4
    geo = pd.read_csv(r"Z:\Paper_Plan\synth_pop\sa2_zones_reference_357.csv",
                      dtype={"sa2_code": str, "sa4_code": str})
    geo["sa2_code"] = geo.sa2_code.str.zfill(9)
    geo["sa4_code"] = geo.sa4_code.str.zfill(3)
    csf = pd.read_csv(pss.CSF, dtype=str)
    csf["sa2"] = csf.sa2.str.zfill(9)
    csf["regucp"] = csf.regucp.str.zfill(9).str.zfill(2)
    both = geo.merge(csf, left_on="sa2_code", right_on="sa2", how="inner")
    s2r = both.drop_duplicates("sa4_code").set_index("sa4_code").regucp
    geo["regucp"] = geo.sa4_code.map(s2r)
    zmap = dict(zip(geo.sa2_code, geo.regucp))

    # household-level census frame
    hh = cen.groupby("abshid", sort=False).first()
    hh = hh.reset_index()

    tgt = pss.targets()
    blocks, next_id = [], 4_000_000_000
    for z in ZONES:
        sa4 = zmap[z]
        base = pss.counts(cp[cp.regucp == sa4], "strd", True)
        ddf = don[don._target_sa2 == z]
        add = pss.counts(ddf, "strd", True) if len(ddf) else {}
        cur = {k: base.get(k, 0) + add.get(k, 0) for k in set(base) | set(add)}
        tt = tgt[z]["strd"]
        N = sum(cur.values())
        for code in (2, 3):
            share = tt.get(code, 0.0)
            have = cur.get(code, 0)
            if share <= 0:
                continue
            need = int(np.ceil((share * N - have) / max(1e-9, 1 - share)))
            need = max(0, min(need, 6000))
            if need == 0:
                continue
            # Prefer renters for semi/flats; fall back to any household.
            pool = hh[(hh["strd"] != code) & (hh["rntrd"].fillna(0) > 0)]
            if len(pool) < need:
                pool = hh[hh["strd"] != code]
            sel = pool.sample(n=need, random_state=7, replace=False)
            ids = set(sel["abshid"].tolist())
            block = cen[cen["abshid"].isin(ids)].copy()
            block["strd"] = code
            block["_source"] = "targeted_strd"
            block["_target_sa2"] = z
            id_map = {old: next_id + k for k, old in enumerate(
                block["abshid"].drop_duplicates().tolist())}
            next_id += len(id_map)
            block["abshid"] = block["abshid"].map(id_map)
            blocks.append(block)
            print(f"zone {z} strd {code}: need {need}, added {len(block)} rows",
                  flush=True)

    out = pd.concat(blocks, ignore_index=True)
    out.to_csv(OUT, index=False)
    print("wrote", OUT, len(out), "rows")


if __name__ == "__main__":
    main()

"""Full-cohort joint and diversity audit for the completed arms.

Reads each arm's released zone populations, computes on the 367-zone fitted
cohort:
  * mean joint TVD over eight designed attribute pairs against the microdata;
  * mean absolute Cramer's V difference on the same pairs;
  * within-zone identical-person share (duplicate 20-attribute tuples);
  * cross-zone cloning share (distinct tuples appearing in more than one zone).

Outputs outputs/full_cohort_audit/full_cohort_audit.csv.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config
from benchmark_margins import _cramers_v

OUT = os.path.join(config.OUTPUTS_DIR, "full_cohort_audit")
os.makedirs(OUT, exist_ok=True)

PAIRS = [("sexp", "mstp"), ("agep", "mstp"), ("heap", "occp"),
         ("mstp", "rlhp"), ("nprd", "vehrd"), ("hind", "rntrd"),
         ("rlhp", "vehrd"), ("heap", "hind")]
TUPLE_COLS = ["agep", "mstp", "sexp", "heap", "hrsp", "incp", "mtwp",
              "mrerd", "nprd", "occp", "qalfp", "qallp", "rlhp", "hind",
              "rntrd", "strd", "tenlld", "vehrd", "empp", "gngp"]
MICRO = os.path.join(config.POP_SYNTH_DIR, "LINKED_DATA_clean.csv")
CSF = os.path.join(HERE, "csf_sa.csv")

ARMS = {
    "sa_only": os.path.join(config.OUTPUTS_DIR, "sa_only_results"),
    "ipu_only": os.path.join(config.OUTPUTS_DIR, "ipu_only_results"),
    "cp_sa": os.path.join(config.OUTPUTS_DIR, "cp_pool_results"),
    "cp_ipu": os.path.join(config.OUTPUTS_DIR, "cp_ipu_results"),
    "coverage_sa": os.path.join(config.OUTPUTS_DIR,
                                "cp_pool_results_coverage"),
    "coverage_ipu": os.path.join(config.OUTPUTS_DIR,
                                 "cp_ipu_results_coverage"),
    "composition_sa": os.path.join(config.OUTPUTS_DIR,
                                   "cp_pool_results_composition"),
    "composition_ipu": os.path.join(config.OUTPUTS_DIR,
                                    "cp_ipu_results_composition"),
    "cp_mip": os.path.join(config.OUTPUTS_DIR, "cp_mip_results"),
}

ZONE_RE = re.compile(r"zone_(\d{9})(?:_(?:new|synthetic)_population)?\.csv$")


def cohort():
    z = {line.strip() for line in open(os.path.join(HERE, "pbs",
                                                    "zones_371.txt"))
         if line.strip()}
    return z - {"301031014", "304041102", "310021279", "311031315"}


def map_files(root, arm):
    best = {}
    for path in glob.glob(os.path.join(root, "**", "zone_*.csv"),
                          recursive=True):
        base = os.path.basename(path)
        if any(tag in base for tag in ("rssz", "audit", "diagnostic")):
            continue
        m = ZONE_RE.search(base)
        if not m:
            continue
        zone = m.group(1)
        score = 0
        if arm in ("cp_sa", "coverage_sa", "composition_sa"):
            score = 2 if "expanded_v3_chunk_" in path else 1
        elif arm == "cp_mip":
            score = 3 if base.endswith("_synthetic_population.csv") else 1
        elif arm.endswith("ipu"):
            score = 3 if base.endswith("_synthetic_population.csv") else 2
        if zone not in best or score > best[zone][0]:
            best[zone] = (score, path)
    return {z: p for z, (_s, p) in best.items()}


def pair_dist(df, a, b):
    if a not in df.columns or b not in df.columns:
        return None
    s = df[[a, b]].dropna()
    if s.empty:
        return None
    vc = s.groupby([a, b]).size()
    return vc / vc.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None,
                    help="run only these arm keys (default: all)")
    ap.add_argument("--out", default=os.path.join(OUT, "full_cohort_audit.csv"))
    args = ap.parse_args()
    zones = cohort()
    csf = pd.read_csv(CSF)
    csf["sa2"] = csf["sa2"].astype(str)
    csf["regucp"] = csf["regucp"].astype(str)
    zone_reg = dict(zip(csf["sa2"], csf["regucp"]))
    regs = {zone_reg[z] for z in zones if z in zone_reg}
    micro = pd.read_csv(MICRO, low_memory=False)
    micro.columns = [str(c).lower() for c in micro.columns]
    micro["regucp"] = pd.to_numeric(micro["regucp"], errors="coerce")
    micro = micro.dropna(subset=["regucp"])
    micro["regucp"] = micro["regucp"].astype(int).astype(str)
    # Use every reference region; some cohort zones are missing from csf_sa and
    # inherit their REGUCP from the fitted population, so restricting the
    # reference to the csf keys would silently drop 32 zones.
    regs = set(micro["regucp"].unique())
    ref = micro.copy()
    print("cohort", len(zones), "regions", len(regs), "ref rows", len(ref))

    # Reference pair distributions and Cramer's V by REGUCP.
    ref_dist = {}
    ref_v = {}
    for reg, sub in ref.groupby("regucp", sort=False):
        for pair in PAIRS:
            d = pair_dist(sub, *pair)
            if d is not None:
                ref_dist[(reg, pair)] = d
                try:
                    ref_v[(reg, pair)] = _cramers_v(sub[pair[0]],
                                                   sub[pair[1]])
                except Exception:
                    pass

    rows = []
    for arm, root in ARMS.items():
        if args.arms and arm not in args.arms:
            continue
        files = map_files(root, arm)
        have = [z for z in zones if z in files]
        print(f"{arm}: {len(have)}/{len(zones)} populations", flush=True)
        tuple_zones = {}
        dup = 0
        persons = 0
        used = 0
        tvd_vals = []
        dv_vals = []
        for i, zone in enumerate(have, 1):
            try:
                df = pd.read_csv(files[zone], low_memory=False)
            except Exception as exc:
                print("  skip", zone, exc, file=sys.stderr)
                continue
            df.columns = [str(c).lower() for c in df.columns]
            reg = zone_reg.get(zone)
            if reg is None and "regucp" in df.columns:
                vals = pd.to_numeric(df["regucp"], errors="coerce").dropna()
                if not vals.empty:
                    reg = str(int(vals.iloc[0]))
            if reg is None:
                continue
            for p in PAIRS:
                if not all(c in df.columns for c in p):
                    continue
                pair_df = df[list(p)].dropna()
                if pair_df.empty:
                    continue
                s_counts = pair_df.groupby(list(p)).size()
                r = ref_dist.get((reg, p))
                if r is None:
                    continue
                s = s_counts / s_counts.sum()
                cats = set(s.index) | set(r.index)
                tvd_vals.append(0.5 * sum(
                    abs(float(s.get(c, 0.0)) - float(r.get(c, 0.0)))
                    for c in cats))
                try:
                    from scipy.stats import chi2_contingency
                    xs = np.array([k[0] for k in s_counts.index])
                    ys = np.array([k[1] for k in s_counts.index])
                    ws = np.array(list(s_counts.values), dtype=float)
                    tab = pd.crosstab(pd.Series(xs), pd.Series(ys),
                                      values=pd.Series(ws),
                                      aggfunc="sum").fillna(0.0)
                    n = tab.values.sum()
                    if n > 0 and min(tab.shape) > 1:
                        chi2 = chi2_contingency(tab.values,
                                                correction=False)[0]
                        v_syn = float(np.sqrt(
                            chi2 / (n * (min(tab.shape) - 1))))
                        v_ref = ref_v.get((reg, p))
                        if v_ref is not None:
                            dv_vals.append(abs(v_syn - v_ref))
                except Exception:
                    pass
            if all(c in df.columns for c in TUPLE_COLS):
                sub = df[TUPLE_COLS].fillna(-999)
                h = pd.util.hash_pandas_object(sub, index=False).to_numpy()
                uniq = np.unique(h)
                dup += int(len(h) - len(uniq))
                persons += int(len(h))
                for hv in uniq:
                    tuple_zones[int(hv)] = tuple_zones.get(int(hv), 0) + 1
                used += 1
            if i % 50 == 0:
                print(f"  {arm}: {i}/{len(have)}", flush=True)

        distinct = len(tuple_zones)
        clones = sum(1 for v in tuple_zones.values() if v > 1)
        rows.append({
            "arm": arm,
            "zones": len(have),
            "tuples_zones": used,
            "joint_tvd": float(np.mean(tvd_vals)) if tvd_vals else np.nan,
            "abs_dv": float(np.mean(dv_vals)) if dv_vals else np.nan,
            "identical_persons": dup,
            "persons": persons,
            "identical_share": dup / persons if persons else np.nan,
            "distinct_tuples": distinct,
            "cloned_tuples": clones,
            "cross_zone_clone_share": clones / distinct if distinct else np.nan,
        })
        print(rows[-1], flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string(index=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

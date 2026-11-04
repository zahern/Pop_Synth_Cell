"""Joint-structure metrics (joint TVD, mean |dCramersV|) on a fixed zone cohort.

Fills the two open columns of the matched-protocol table without disturbing the
shared benchmarking outputs: results go to outputs/joint60/ only.

For each of eight designed attribute pairs, compares the arm's pooled joint
distribution over the cohort against the census microdata restricted to the
cohort's REGUCP codes, and compares Cramer's V the same way.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, r"Z:\Paper_Plan\synth_pop")
import config
from benchmark_margins import _cramers_v
from make_marginal_histograms import ARMS, find_zone_files

COHORT_N = 60
OUT = os.path.join(config.OUTPUTS_DIR, "joint60")
os.makedirs(OUT, exist_ok=True)
MICRO = r"z:\Population_Synth\LINKED_DATA_clean.csv"

PAIRS = [("sexp", "mstp"), ("agep", "mstp"), ("heap", "occp"),
         ("mstp", "rlhp"), ("nprd", "vehrd"), ("hind", "rntrd"),
         ("rlhp", "vehrd"), ("heap", "hind")]

micro = pd.read_csv(MICRO, low_memory=False)
micro.columns = [str(c).lower() for c in micro.columns]

# cohort = zones shared by every arm that has data
files = {a: f for a, f in ((a, find_zone_files(ARMS[a])) for a in ARMS) if len(f) >= 40}
shared = sorted(set.intersection(*[set(f) for f in files.values()]))[:COHORT_N]
print(f"arms: {list(files)}  cohort: {len(shared)} zones")

# restrict microdata to the cohort's REGUCPs
regs = set()
for z in shared:
    try:
        d = pd.read_csv(files[next(iter(files))][z], usecols=lambda c: c.lower() == "regucp",
                        low_memory=False)
        regs |= set(pd.to_numeric(d["regucp"], errors="coerce").dropna().unique())
    except Exception:
        pass
mr = pd.to_numeric(micro["regucp"], errors="coerce")
ref = micro[mr.isin(regs)] if regs else micro
print(f"microdata reference rows: {len(ref):,}")


def pair_dist(df, a, b):
    if a not in df.columns or b not in df.columns:
        return None
    s = df[[a, b]].dropna()
    if s.empty:
        return None
    vc = s.groupby([a, b]).size()
    return vc / vc.sum()


rows = []
for arm, fmap in files.items():
    acc = {p: {} for p in PAIRS}
    for i, z in enumerate(shared, 1):
        try:
            df = pd.read_csv(fmap[z], low_memory=False)
        except Exception:
            continue
        df.columns = [str(c).lower() for c in df.columns]
        for p in PAIRS:
            d = pair_dist(df, *p)
            if d is None:
                continue
            for k, v in d.items():
                acc[p][k] = acc[p].get(k, 0.0) + v
        if i % 20 == 0:
            print(f"  {arm}: {i}/{len(shared)}", flush=True)
    tvds, vs = [], []
    for p in PAIRS:
        r = pair_dist(ref, *p)
        s = acc[p]
        if not s or r is None:
            continue
        tot = sum(s.values()) or 1.0
        ps = {k: v / tot for k, v in s.items()}
        cats = set(ps) | set(r.index)
        tvds.append(0.5 * sum(abs(ps.get(c, 0.0) - float(r.get(c, 0.0))) for c in cats))
        # Cramer's V on the same pair
        a, b = p
        sub_r = ref[[a, b]].dropna()
        try:
            from scipy.stats import chi2_contingency
            xs = np.array([k[0] for k in s.keys()])
            ys = np.array([k[1] for k in s.keys()])
            ws = np.array([s[k] for k in s.keys()], dtype=float)
            tab = pd.crosstab(pd.Series(xs), pd.Series(ys),
                              values=pd.Series(ws), aggfunc="sum").fillna(0.0)
            n = tab.values.sum()
            if n > 0 and min(tab.shape) > 1:
                chi2 = chi2_contingency(tab.values, correction=False)[0]
                v_syn = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))
                v_ref = _cramers_v(sub_r[a], sub_r[b])
                vs.append(abs(v_syn - v_ref))
        except Exception:
            pass
    rows.append({"arm": arm, "zones": len(shared),
                 "joint_tvd": float(np.mean(tvds)) if tvds else np.nan,
                 "abs_dv": float(np.mean(vs)) if vs else np.nan})

out = pd.DataFrame(rows).sort_values("joint_tvd")
print()
print(out.round(4).to_string(index=False))
out.to_csv(os.path.join(OUT, "joint_metrics_60.csv"), index=False)
print("\nwrote", os.path.join(OUT, "joint_metrics_60.csv"))

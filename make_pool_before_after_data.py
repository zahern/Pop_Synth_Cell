"""Write outputs/pool_before_after_zone.json for the paper's before/after figure.

For one SA2, compare each positive-target control cell's attainable capacity in
the base pool and in the expanded pool. Run from synth_pop with the paper control
directory in SA_CONTROL_DIR.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import mip_fit as mipf
from sa_only_baseline import load_control_tables, _load_sa2_to_regucp

ZONE = "112011236"
REP = 40
ind, hh = load_control_tables()
ind_keys = sorted({k for t in ind.values() for k in t.get("keys", [])})
hh_keys = sorted({k for t in hh.values() for k in t.get("keys", [])})
mapping, _ = _load_sa2_to_regucp()
tgt = {}
tgt.update(mipf.zone_targets(ind, ZONE))
tgt.update(mipf.zone_targets(hh, ZONE))
states_by_stage = {}
pool_stats = {}
for label, filename in [("before", "feasible_pool.csv"),
                        ("after", "feasible_pool_expanded.csv")]:
    pool = pd.read_csv(os.path.join(HERE, "outputs", filename), low_memory=False)
    pool.columns = [str(c).lower() for c in pool.columns]
    cand, _ = mipf.zone_candidate_pool(pool, ZONE, mapping)
    C, hhids, cells = mipf.build_constraint_matrix(cand, tgt, ind_keys, hh_keys)
    cap = REP * np.asarray(C.sum(axis=1)).ravel()
    states = []
    for i, cell in enumerate(cells):
        e = float(tgt[cell])
        if e <= 0:
            continue
        c = float(cap[i])
        states.append({"attr": str(cell[0]), "category": str(cell[1]),
                       "target": e, "capacity": c,
                       "state": "sufficient" if c >= e else ("short" if c > 0 else "zero")})
    states_by_stage[label] = states
    pool_stats[label] = {
        "persons": int(len(cand)),
        "households": int(cand["abshid"].nunique()) if "abshid" in cand.columns else 0,
    }
out_path = os.path.join(HERE, "outputs", "pool_before_after_zone.json")
with open(out_path, "w") as fh:
    json.dump({"zone": ZONE, "states": states_by_stage,
               "pool_stats": pool_stats}, fh)
print(json.dumps({k: len(v) for k, v in states_by_stage.items()}))

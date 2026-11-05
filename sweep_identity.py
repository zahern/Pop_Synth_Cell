"""Identical-person share across the CP-MIP sweep settings, same 19 zones."""
import glob
import os

import numpy as np
import pandas as pd

HERE = r"Z:\Paper_Plan\synth_pop\outputs\cp_mip_sweep"
TUPLE_COLS = ["agep", "mstp", "sexp", "heap", "hrsp", "incp", "mtwp",
              "mrerd", "nprd", "occp", "qalfp", "qallp", "rlhp", "hind",
              "rntrd", "strd", "tenlld", "vehrd", "empp", "gngp"]

rows = []
for setting in sorted(glob.glob(os.path.join(HERE, "rep*_lam*"))):
    files = sorted(glob.glob(os.path.join(setting, "zone_*.csv")))
    dup = persons = 0
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [str(c).lower() for c in df.columns]
        cols = [c for c in TUPLE_COLS if c in df.columns]
        if len(cols) < 5:
            continue
        sub = df[cols].fillna(-999)
        h = pd.util.hash_pandas_object(sub, index=False).to_numpy()
        dup += int(len(h) - len(np.unique(h)))
        persons += int(len(h))
    name = os.path.basename(setting)
    rep = int(name.split("_")[0].replace("rep", ""))
    lam = float(name.split("lam")[1])
    rows.append({"setting": name, "rep_max": rep, "lambda_joint": lam,
                 "zones": len(files), "identical_share": dup / persons,
                 "persons": persons})
    print(rows[-1], flush=True)

out = pd.DataFrame(rows).sort_values(["rep_max", "lambda_joint"])
out.to_csv(os.path.join(HERE, "sweep_identity.csv"), index=False)
print()
print(out[["setting", "rep_max", "lambda_joint", "zones",
           "identical_share"]].round(3).to_string(index=False))

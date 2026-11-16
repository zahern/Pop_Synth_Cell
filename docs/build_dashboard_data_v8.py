"""Build the paper's population-fit dashboard data from the v8 run.

Same payload schema as ``build_dashboard_data.py``, sourced from
``runs/2026-11-13_v8``.  Run locally; it only reads the Z: share and writes the
paper folder.
"""
from __future__ import annotations

import json
import os

import geopandas as gpd
import pandas as pd

PAPER = r"Z:/Paper_Plan/Population_Synthesiser___GAN_SA"
SYNTH = r"Z:/Paper_Plan/synth_pop"
RUN = r"Z:/Paper_Plan/runs/2026-11-13_v8"
OUT = os.path.join(PAPER, "dashboard", "data")
os.makedirs(OUT, exist_ok=True)

ARMS = ["base_sa", "base_ipu", "mip_sa", "mip_ipu", "cp_mip_fallback",
        "mip_fallback"]

META = {
    "base_sa": dict(label="CP pool, SA", pool="CP feasible pool", fitting="SA",
                    settings="1,000 iterations per zone; Metropolis; best-so-far",
                    desc="CP construction with the conventional SA fit."),
    "base_ipu": dict(label="CP pool, IPU", pool="CP feasible pool", fitting="IPU",
                     settings="IPU weights on the CP feasible pool",
                     desc="CP construction with the conventional IPU fit."),
    "mip_sa": dict(label="Expanded, SA", pool="CP pool + expansion",
                   fitting="SA",
                   settings="1,000 iterations per zone; Metropolis; best-so-far",
                   desc="Expanded pool fitted by SA."),
    "mip_ipu": dict(label="Expanded, IPU", pool="CP pool + expansion",
                    fitting="IPU",
                    settings="IPU weights on the expanded pool",
                    desc="Expanded pool fitted by IPU."),
    "cp_mip_fallback": dict(label="Expanded, MIP",
                            pool="CP pool + expansion",
                            fitting="Integer MIP",
                            settings="REP_MAX 4, lambda_joint 20; replication allowed",
                            desc="Reported method: CP construction, MIP expansion and the fitting MIP."),
    "mip_fallback": dict(label="Expanded, MIP unique",
                         pool="CP pool + expansion",
                         fitting="Integer MIP (unique)",
                         settings="REP_MAX 1 + adult uniqueness",
                         desc="Uniqueness-constrained MIP; the population propagated to the ABM."),
}

TABLES = [
    ("heap", "Highest education"), ("mstp", "Marital status"),
    ("mtwp", "Travel to work"), ("occp", "Occupation"),
    ("rlhp", "Relationship in household"), ("sexp", "Sex"),
    ("qalfp", "Field of study"), ("qallp", "Qualification level"),
    ("incp", "Personal income"), ("hrsp", "Hours worked"),
    ("indp", "Industry"), ("typp", "Education institution"),
    ("stup", "Student status"), ("hind", "Household income"),
    ("mrerd", "Mortgage repayment"), ("nprd", "Household size"),
    ("rntrd", "Rent"), ("vehrd", "Vehicles"), ("strd", "Dwelling structure"),
]
TABLE_IDS = [t[0] for t in TABLES]
SHARED11 = ["heap", "mstp", "mtwp", "occp", "rlhp", "sexp",
            "hind", "mrerd", "nprd", "rntrd", "vehrd"]

scope = [ln.strip() for ln in
         open(os.path.join(SYNTH, "pbs", "zones_371.txt")) if ln.strip()]
arm_zones = pd.read_csv(os.path.join(RUN, "tables", "arm_zones.csv"))
fitted = set(arm_zones.loc[arm_zones["arm"] == "cp_mip_fallback", "zone"]
             .astype(str).str.zfill(9))
special = sorted(set(scope) - fitted)
print(f"zones: {len(scope)} shown, {len(fitted)} fitted, {len(special)} special")

# ---------------------------------------------------------------------------
# zones
# ---------------------------------------------------------------------------
shp = os.path.join(SYNTH, "boundaries", "SA2_2021_AUST_GDA2020.shp")
gdf = gpd.read_file(shp)
gdf["SA2_CODE21"] = gdf["SA2_CODE21"].astype(str).str.zfill(9)
gdf = gdf[gdf["SA2_CODE21"].isin(set(scope))].to_crs(4326)
gdf["geometry"] = gdf.geometry.simplify(0.001, preserve_topology=True)

zone_payload = []
minx = miny = 1e9
maxx = maxy = -1e9
for _, row in gdf.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    rings = []
    for poly in polys:
        try:
            coords = [[round(float(x), 4), round(float(y), 4)]
                      for x, y in poly.exterior.coords]
        except Exception:
            continue
        if len(coords) < 4:
            continue
        rings.append(coords)
        for x, y in coords:
            minx = min(minx, x); maxx = max(maxx, x)
            miny = min(miny, y); maxy = max(maxy, y)
    if rings:
        zone_payload.append({
            "code": row["SA2_CODE21"],
            "name": row.get("SA2_NAME21", ""),
            "special": row["SA2_CODE21"] in set(special),
            "rings": rings,
        })
zone_payload.sort(key=lambda z: z["code"])
with open(os.path.join(OUT, "zones.js"), "w", encoding="utf-8") as fh:
    fh.write("window.DASH_ZONES=")
    json.dump({"bounds": [minx, miny, maxx, maxy], "zones": zone_payload},
              fh, separators=(",", ":"))
    fh.write(";\n")
print(f"zones.js: {len(zone_payload)} polygons")

# ---------------------------------------------------------------------------
# per-zone / per-table RSSZ
# ---------------------------------------------------------------------------
zt = pd.read_csv(os.path.join(RUN, "tables", "zone_table_rssz.csv"))
zt["zone"] = zt["zone"].astype(str).str.zfill(9)
rssz = {}
for arm in ARMS:
    sub = zt[zt["arm"] == arm]
    rssz[arm] = {}
    for _, row in sub.iterrows():
        tables = {}
        for t in TABLE_IDS:
            tables[t] = round(float(row[t]), 4) if t in row and pd.notna(row[t]) else None
        rssz[arm][row["zone"]] = {
            "total": round(float(row["rssz_total"]), 4),
            "tables": tables,
        }

# ---------------------------------------------------------------------------
# method summaries
# ---------------------------------------------------------------------------
summary = pd.read_csv(os.path.join(RUN, "tables", "arm_summary.csv"))
summary = summary.set_index("arm")
held_path = os.path.join(RUN, "audit", "heldout_rssz_all.csv")
held_mean = {}
if os.path.exists(held_path):
    held = pd.read_csv(held_path)
    held_mean = held.groupby("arm")["mean"].mean().to_dict()
audit_path = os.path.join(RUN, "tables", "full_cohort_audit_v8.csv")
if not os.path.exists(audit_path):
    audit_path = os.path.join(RUN, "audit", "full_cohort_audit_v8.csv")
audit = {}
if os.path.exists(audit_path):
    au = pd.read_csv(audit_path)
    for _, r in au.iterrows():
        audit[r["arm"]] = dict(joint=r["joint_tvd"], dV=r["abs_dv"],
                               identical=100 * r["identical_share"],
                               median_zones=r["tuple_zone_median"],
                               single_zone=100 * r["persons_single_zone_share"])

HELD_KEY = {"base_sa": "CP-only+SA", "base_ipu": "CP-only+IPU",
            "mip_sa": "MIP+SA", "mip_ipu": "MIP+IPU",
            "cp_mip_fallback": "CP-MIP+fallback",
            "mip_fallback": "MIP (direct)+fallback"}
AUDIT_KEY = {"base_sa": "base_sa", "base_ipu": "base_ipu",
             "mip_sa": "cp_mip_expansion_sa",
             "mip_ipu": "cp_mip_expansion_ipu",
             "cp_mip_fallback": "cp_mip_fallback",
             "mip_fallback": "mip_mip_fallback_unique"}

SUMMARY_KEY = {"base_sa": "cp_sa", "base_ipu": "cp_ipu",
               "mip_sa": "mip_sa", "mip_ipu": "mip_ipu",
               "cp_mip_fallback": "cp_mip_fallback",
               "mip_fallback": "mip_mip_fallback_unique"}

methods = []
for arm in ARMS:
    skey = SUMMARY_KEY.get(arm, arm)
    if skey not in summary.index:
        continue
    row = summary.loc[skey]
    sub = zt[zt["arm"] == arm]
    r11 = sub[[t for t in SHARED11 if t in sub.columns]].sum(axis=1)
    a = audit.get(AUDIT_KEY.get(arm, arm), {})
    methods.append({
        "id": arm, **META[arm],
        "median19": round(float(row["median_362"]), 4),
        "mean19": round(float(row["mean_362"]), 4),
        "median11": round(float(r11.median()), 4) if len(r11) else None,
        "mean11": round(float(r11.mean()), 4) if len(r11) else None,
        "heldout": round(float(held_mean[HELD_KEY[arm]]), 3)
        if HELD_KEY.get(arm) in held_mean else None,
        "joint": round(float(a["joint"]), 3) if a else None,
        "dV": round(float(a["dV"]), 3) if a else None,
        "identical": round(float(a["identical"]), 1) if a else None,
        "median_zones": float(a["median_zones"]) if a else None,
        "single_zone": round(float(a["single_zone"]), 1) if a else None,
        "half_zones": None,
    })

# ---------------------------------------------------------------------------
# cohort-level observed vs synthesized marginal shares
# ---------------------------------------------------------------------------
marginals = {}
shares_path = os.path.join(RUN, "paper_figures", "marginal_category_shares.csv")
if os.path.exists(shares_path):
    shares = pd.read_csv(shares_path)
    shares = shares[shares["arm"].isin(ARMS)]
    for table in TABLE_IDS:
        sub = shares[shares["table"] == table]
        if sub.empty:
            continue
        cats = sorted(sub["category"].astype(str).unique(),
                      key=lambda x: (len(x), x))
        census = None
        per_arm = {}
        for arm in ARMS:
            a = sub[sub["arm"] == arm]
            if a.empty:
                continue
            a = a.set_index(a["category"].astype(str))
            vals = [float(a.loc[c, "synth_share"]) if c in a.index else 0.0
                    for c in cats]
            per_arm[arm] = [round(v, 5) for v in vals]
            if census is None:
                census = [round(float(a.loc[c, "census_share"])
                                if c in a.index else 0.0, 5) for c in cats]
        marginals[table] = {"categories": cats, "census": census or [],
                            "methods": per_arm}

payload = {
    "tables": [{"id": t, "label": label} for t, label in TABLES],
    "methods": methods,
    "rssz": rssz,
    "marginals": marginals,
    "meta": {
        "cohort": 362,
        "shown": len(zone_payload),
        "special": special,
        "fitting_tables": 19,
        "heldout_tables": 7,
        "note": ("v8 corrected-control run. Five configurations: the CP pool "
                 "with SA and IPU, and the expanded pool with SA, IPU and MIP. "
                 "RSSZ is per-zone on the 19-table fitting basis. Marginal "
                 "shares are cohort-level across the 362 fitted zones, not per "
                 "zone."),
    },
}
with open(os.path.join(OUT, "results.js"), "w", encoding="utf-8") as fh:
    fh.write("window.DASH_RESULTS=")
    json.dump(payload, fh, separators=(",", ":"))
    fh.write(";\n")
print("results.js written:", len(methods), "methods,", len(marginals), "tables")

"""
built_form_conditioning.py — condition zone seeding on built form (A2)
======================================================================

Purpose
-------
`overture_built_form.py` produces per-SA2 built-form features.  This script turns
those features into two artefacts that make the *inner* zone sampling realistic:

  1. `zone_built_form_archetype.csv` — every SA2 assigned to a built-form
     archetype (inner-urban / suburban / peri-urban / rural) via clustering.

  2. `zone_regucp_weights.csv` — a weighted distribution over REGUCP (SA4) regions
     for each zone, learned from the built-form archetype.  This replaces the
     current single-REGUCP concordance (csf_sa.csv) whose fallback is "use the
     full seed pool", which ignores built form entirely.

  3. `zone_dwelling_targets.csv` — an independent dwelling-count estimate per
     zone (from building footprints) that can anchor the household-level control
     total, addressing the household-count mismatch seen in diff_latest.png.

Integration into SA_2_level.py
------------------------------
The seeding code currently does (pseudo):

    regucp = concordance.get(zone)           # from csf_sa.csv
    if regucp is None or regucp not in pool:
        seed = full_pool                    # <-- coarse fallback

Replace the fallback (and optionally the whole concordance) with:

    weights = load_zone_regucp_weights(zone)   # {regucp: weight}
    seed = pool_records.sample(weights)        # built-form-aware draw

so that an inner-urban zone seeds from REGUCPs whose built form is similar to it,
rather than from the entire state.

Usage
-----
  python built_form_conditioning.py --built-form outputs/sa2_built_form.csv \
         --concordance csf_sa.csv --n-clusters 5 --out-dir outputs/built_form/

"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------
# Public helper used by SA_2_level.py
# ---------------------------------------------------------------------------

def load_zone_regucp_weights(weights_csv):
    """
    Load the zone -> weighted REGUCP distribution produced by this script.

    Returns a dict {zone_code: {regucp: weight}} suitable for built-form-aware
    pool seeding.  `regucp` values are strings; weights sum to 1.
    """
    df = pd.read_csv(weights_csv)
    out = {}
    for zone, grp in df.groupby("sa2_code"):
        w = dict(zip(grp["regucp"].astype(str), grp["weight"].astype(float)))
        total = sum(w.values()) or 1.0
        out[str(zone)] = {k: v / total for k, v in w.items()}
    return out


def seed_pool_by_built_form(pool_df, zone, weights, regucp_col="REGUCP", rng=None):
    """
    Draw pool records for `zone` weighted by the built-form-derived REGUCP
    distribution, falling back to the full pool when no weight is available.

    This is a drop-in replacement for the "fallback to full seed pool" branch in
    SA_2_level.py.  It does NOT change the marginal-fitting step; it only biases
    which candidate records are *available* to the zone, making inner sampling
    follow the zone's built form.
    """
    rng = rng or np.random.default_rng(0)
    w = weights.get(str(zone))
    if not w:
        return pool_df
    pool = pool_df.copy()
    mask = pool[regucp_col].astype(str).isin(w)
    if mask.sum() == 0:
        return pool_df
    subset = pool[mask]
    p = np.array([w[r] for r in subset[regucp_col].astype(str)])
    p = p / p.sum()
    # Sample with replacement to the size of the zone's candidate pool.
    idx = rng.choice(len(subset), size=len(pool), p=p, replace=True)
    return subset.iloc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature -> artefact construction
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
    "area_km2", "n_buildings", "area_mean", "height_mean",
    "multi_unit_share", "dwelling_density", "urban_class",
]


def assign_archetypes(feats, n_clusters=5, seed=0):
    """Cluster SA2s into built-form archetypes on the standardised feature vector."""
    cols = [c for c in _FEATURE_COLS if c in feats.columns]
    X = feats[cols].copy()
    # Land-use / POI columns are optional; include them if present.
    extra = [c for c in feats.columns if c.startswith("land_") or c == "n_places"]
    if extra:
        X = pd.concat([X, feats[extra].copy()], axis=1)
    X = X.fillna(0.0)
    Xn = (X - X.mean()) / (X.std() + 1e-9)

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        feats["archetype"] = km.fit_predict(Xn)
    except Exception as exc:
        # Fallback: quantile binning on dwelling density.
        print(f"[warn] sklearn unavailable ({exc}); falling back to density bins.")
        feats["archetype"] = pd.qcut(
            feats["dwelling_density"].rank(method="first"),
            n_clusters, labels=False, duplicates="drop").fillna(0).astype(int)
    return feats


def build_regucp_weights(feats, concordance_csv, sa2_col="sa2_code"):
    """
    For each zone, learn P(REGUCP | archetype) from the zones that DO have a
    concordance entry, and export a weighted REGUCP distribution per zone.
    """
    concord = pd.read_csv(concordance_csv)
    # Detect column names heuristically.
    zcol = next((c for c in concord.columns if "sa2" in c.lower() or "zone" in c.lower()),
                concord.columns[0])
    rcol = next((c for c in concord.columns if "regucp" in c.lower() or "region" in c.lower()
                 or "sa4" in c.lower()), concord.columns[1])
    concord = concord[[zcol, rcol]].dropna()
    concord[zcol] = concord[zcol].astype(str).str.zfill(9)
    concord[rcol] = concord[rcol].astype(str)

    merged = feats.merge(concord, left_on="sa2_code", right_on=zcol, how="left")

    # P(REGUCP | archetype)
    archetype_regucp = (
        merged.dropna(subset=[rcol])
              .groupby(["archetype", rcol]).size().reset_index(name="count")
    )
    archetype_regucp["weight"] = archetype_regucp.groupby("archetype")["count"] \
        .transform(lambda s: s / s.sum())

    rows = []
    for _, zrow in merged.iterrows():
        arch = zrow["archetype"]
        sub = archetype_regucp[archetype_regucp["archetype"] == arch]
        for _, r in sub.iterrows():
            rows.append({"sa2_code": zrow["sa2_code"], "regucp": r[rcol],
                         "weight": r["weight"]})
    return pd.DataFrame(rows)


def build_dwelling_targets(feats, occupancy=2.5):
    """
    Estimate an independent dwelling/household count per zone from building
    footprints.  Multi-unit buildings house more dwellings than detached homes, so
    the raw building count is scaled by a multi-unit factor before applying the
    average persons-per-household occupancy.
    """
    feats = feats.copy()
    # dwellings ~= n_buildings * (1 + multi_unit_share * 8)  (heuristic unit factor)
    multi_factor = 1.0 + feats.get("multi_unit_share", 0.0).fillna(0.0) * 8.0
    feats["dwelling_estimate"] = (feats["n_buildings"].fillna(0) * multi_factor).round()
    feats["hh_persons_estimate"] = (feats["dwelling_estimate"] * occupancy).round()
    return feats[["sa2_code", "n_buildings", "dwelling_estimate", "hh_persons_estimate"]]


def main():
    ap = argparse.ArgumentParser(description="Condition zone seeding on built form (A2)")
    ap.add_argument("--built-form", required=True, help="CSV from overture_built_form.py")
    ap.add_argument("--concordance", required=True, help="SA2->REGUCP concordance CSV (csf_sa.csv)")
    ap.add_argument("--n-clusters", type=int, default=5)
    ap.add_argument("--occupancy", type=float, default=2.5)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    out_dir = a.out_dir or os.path.join(HERE, "outputs", "built_form")
    os.makedirs(out_dir, exist_ok=True)

    feats = pd.read_csv(a.built_form)
    feats["sa2_code"] = feats["sa2_code"].astype(str).str.zfill(9)

    feats = assign_archetypes(feats, n_clusters=a.n_clusters)
    feats.to_csv(os.path.join(out_dir, "zone_built_form_archetype.csv"), index=False)

    weights = build_regucp_weights(feats, a.concordance)
    weights.to_csv(os.path.join(out_dir, "zone_regucp_weights.csv"), index=False)

    dw = build_dwelling_targets(feats, occupancy=a.occupancy)
    dw.to_csv(os.path.join(out_dir, "zone_dwelling_targets.csv"), index=False)

    # Summary of archetypes (for the paper / sanity check)
    summary = feats.groupby("archetype").agg(
        n_zones=("sa2_code", "count"),
        mean_density=("dwelling_density", "mean"),
        mean_multi_unit=("multi_unit_share", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(out_dir, "archetype_summary.csv"), index=False)

    print(f"Archetype summary:\n{summary.to_string(index=False)}")
    print(f"Wrote {len(weights)} zone->REGUCP weight rows, "
          f"{len(dw)} dwelling-target rows to {out_dir}")


if __name__ == "__main__":
    main()

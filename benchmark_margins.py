"""
Benchmark Margins Comparison
============================
Compares each strategy's synthetic population against census SA2 marginals
across all control tables.  Produces (under outputs/benchmarking/):

  metrics_sa2.csv           — per-zone × per-table: RSSZ, TVD, SRMSE, MAE%
  metrics_sa4.csv           — same metrics aggregated to SA4 geography
  strategy_summary.csv      — mean/median/max for each strategy × metric,
                              plus repair rate and pMSE ratio
  repair_rate.csv           — per strategy: records violating ≥1 hard constraint
  repair_rate_by_constraint.csv — per strategy × constraint breakdown
  pmse.csv                  — per strategy: pMSE, null pMSE, normalised ratio
  joint_representation.csv      — pairwise Cramér's V: synthetic vs census microdata
  joint_representation_sa4.csv  — same, pooled by SA4 region (geographic breakdown)
  benchmark_metrics_boxplot.png
  strategy_utility_comparison.png — repair rate and pMSE ratio by strategy
  sa4_heatmap_{strategy}_{metric}.png
  joint_cramer_scatter.png
  joint_cramer_sa4_heatmap.png  — SA4 × strategy: mean |Δ Cramér's V| vs microdata

Metrics
-------
  TVD   (Total Variation Distance): 0.5 * Σ|p_i − q_i|.  Range [0, 1].
        Measures distributional divergence; independent of population size.
  SRMSE (Standardised RMSE): √mean[(obs−exp)² / max(exp,1)].
        Penalises large absolute errors relative to expected counts.
  MAE%  (Mean Absolute Error %): mean(|obs−exp| / max(exp,1)) × 100.
        Interpretable as an average relative error across categories.
  Repair rate: percentage of records violating at least one hard logical
        constraint (child with an occupation, non-worker with hours worked,
        household both renting and mortgaged, …).  The GAN's raw output is
        repaired before use, so this quantifies how much repair each method
        needs; a method that enforces constraints at construction (CP pool)
        should be near 0%.
  pMSE  (propensity score MSE): a logistic regression separates synthetic from
        real census records; pMSE = mean((p̂ − 0.5)²) on held-out data,
        normalised by the null pMSE c/(1+c)².  Ratio ≈ 1 means the synthetic
        records are indistinguishable from the census; higher is worse.

SA4 geography
-------------
  SA2 codes are 9-digit integers (e.g. 301011001).
  SA4 codes are the first 3 digits (e.g. 301).
  All zone-level metrics are averaged across SA2 zones within each SA4.

Joint representation
--------------------
  For every pair of attributes present in both the synthetic population and
  census microdata, Cramér's V is computed.  Points lying close to the
  diagonal in joint_cramer_scatter.png indicate the GAN/SA preserved the
  pairwise associations.  Large deviations flag attributes whose joint
  structure was distorted.

Addresses
  Reviewer 1: "Additional validation … comparing distributions with the
               original microdata, evaluating joint attribute relationships."
  Reviewer 2: P31 SA-Only vs SA-GAN comparison.

Usage
-----
    python benchmark_margins.py [--strategies sa_gan sa_only ipu_gan ipu_only]
                                [--microdata path/to/microdata_for_exp.csv]
                                [--ipu-gan-dir path/to/zones_ipu_gan_main]
                                [--ipu-only-dir path/to/zones_ipu_only_main]
"""

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config
import reporting_scope

from CensusRecordImporter import CensusControlTable

OUTPUT_DIR = os.path.join(config.OUTPUTS_DIR, "benchmarking")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Control-table loader
# ---------------------------------------------------------------------------

def _sa4(sa2_code):
    """Derive 3-digit SA4 code from a 9-digit SA2 code."""
    return str(sa2_code).strip()[:3]


def _load_control_table(path, key, zone_level="sa2"):
    """
    Load one census CSV control table.
    The generic 'N-digit' column from ABS files is renamed to the actual
    attribute name (derived from the filename) BEFORE validate_and_clean()
    so that the DataMappings text→integer conversions are applied correctly.
    """
    try:
        ct = CensusControlTable(path, zone_level=zone_level)
        attr = re.sub(r"^sa[12]", "", key)          # "sa2heap" → "heap"
        for generic in ("1-digit", "2-digit"):
            if generic in ct.control_table.columns:
                ct.control_table = ct.control_table.rename(columns={generic: attr})
        ct.validate_and_clean()
        df = ct.control_table.copy()
        # Belt-and-braces: rename any remaining generic column after validation
        for generic in ("1-digit", "2-digit"):
            if generic in df.columns:
                df = df.rename(columns={generic: attr})
        zone_col = getattr(ct, "sa_num", zone_level)
        if zone_col not in df.columns and zone_level in df.columns:
            zone_col = zone_level
        if zone_col not in df.columns:
            raise KeyError(f"Zone column '{zone_col}' not found in control table")

        df[zone_col] = df[zone_col].astype(str)
        attr_cols = [c for c in df.columns if c not in (zone_col, "count")]
        return {"data": df, "attr_cols": attr_cols, "attr": attr, "zone_col": zone_col}
    except Exception as exc:
        warnings.warn(f"Skipping {os.path.basename(path)}: {exc}")
        return None


def load_all_controls(control_dir, zone_level="sa2"):
    """Load all control tables from a directory for a given zone level."""
    tables = {}
    root = control_dir
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Control directory not found: {root}")
    for fname in sorted(os.listdir(root)):
        if not fname.endswith(".csv"):
            continue
        if " copy" in fname.lower() or fname.startswith("~"):
            continue
        key = fname.replace(".csv", "").lower()
        ct = _load_control_table(os.path.join(root, fname), key, zone_level=zone_level)
        if ct is not None:
            tables[key] = ct
    print(f"  Loaded {len(tables)} control tables")
    return tables


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalise(count_dict):
    total = max(sum(count_dict.values()), 1)
    return {k: v / total for k, v in count_dict.items()}


# ---------------------------------------------------------------------------
# Control-table category decoding
# ---------------------------------------------------------------------------
# A handful of benchmark control tables (the held-out household/person tables
# bedrd, tend, lfsp, hhcd) store category *labels* as text, whereas every
# synthetic population and every fitted control table uses the ABS integer
# codes. Comparing integer obs keys against text exp keys makes the key sets
# fully disjoint -> TVD = 1.0, spuriously implying the GAN-augmented methods
# "can't do" these attributes when in fact they generate them (100% populated,
# all categories present). These label->code maps decode the control labels
# back to codes so obs and exp align and TVD reflects the true fit.
#
# Codes come from the ABS "2021 Census 5% basic microdata" data item list
# (the coding the GAN was trained on); the label STRINGS are the exact ones the
# control tables carry (some are ABS short forms, e.g. "Owned with a mortgage").
_LABEL_TO_CODE = {
    "bedrd": {
        "None (includes studio apartments or bedsitters)": 0, "One bedroom": 1,
        "Two bedrooms": 2, "Three bedrooms": 3, "Four bedrooms": 4,
        "Five bedrooms or more": 5, "Not stated": 6, "Not applicable": 7,
    },
    "hhcd": {
        "One family household": 1, "Two or more family household": 2,
        "Lone person household": 3, "Group household": 4, "Visitors only": 5,
        "Other non-classifiable": 6, "Not applicable": 7,
    },
    "tend": {
        "Owned outright": 1, "Owned with a mortgage": 2, "Rented": 3,
        "Other": 4, "Not stated": 5, "Not applicable": 6,
    },
    "lfsp": {
        "Employed": 1, "Unemployed": 2, "Not in the labour force": 3,
        "Not stated": 4, "Not applicable": 5, "Overseas visitor": 6,
    },
}


def tvd(obs_counts, exp_counts):
    """Total Variation Distance.  Range [0, 1]."""
    keys = set(obs_counts) | set(exp_counts)
    obs_p = _normalise(obs_counts)
    exp_p = _normalise(exp_counts)
    return 0.5 * sum(abs(obs_p.get(k, 0.0) - exp_p.get(k, 0.0)) for k in keys)


def srmse(obs_counts, exp_counts):
    """Standardised RMSE: sqrt(mean((obs−exp)² / max(exp,1)))."""
    keys = set(obs_counts) | set(exp_counts)
    if not keys:
        return np.nan
    sq = [
        (obs_counts.get(k, 0) - exp_counts.get(k, 0)) ** 2
        / max(exp_counts.get(k, 0), 1)
        for k in keys
    ]
    return float(np.sqrt(np.mean(sq)))


def mae_pct(obs_counts, exp_counts):
    """Mean Absolute Error as a percentage of expected counts."""
    keys = set(obs_counts) | set(exp_counts)
    if not keys:
        return np.nan
    vals = [
        abs(obs_counts.get(k, 0) - exp_counts.get(k, 0))
        / max(exp_counts.get(k, 0), 1) * 100.0
        for k in keys
    ]
    return float(np.mean(vals))


def _obs_counts(pop_df, attr_cols):
    """Extract marginal counts for attr_cols from a synthetic-population DataFrame."""
    missing = [c for c in attr_cols if c not in pop_df.columns]
    if missing:
        return {}
    grp = pop_df.groupby(attr_cols).size()
    return {k: int(v) for k, v in grp.items()}


# ---------------------------------------------------------------------------
# Derived control-table observations
# ---------------------------------------------------------------------------
# A few census control tables have no verbatim column in the synthetic
# populations but are directly derivable from one that does, and store TEXT
# category labels (so they are compared against text exp keys, not codes):
#   age5p (5-year age groups)  <- agep (ABS codes 0-24 = single years 0-24,
#                                  25-37 = 5-year bands 25-29 ... 85+)
#   hcfmf (family household composition) <- families_in_household (1/2/3+)
# Without this the population lacks an 'age5p'/'hcfmf' column, _obs_counts
# returns {} and TVD defaults to 0.5 for every method -- a measurement
# artifact, not a synthesis failure. Methods lacking the source column
# (e.g. the 13-attribute SA-Only/IPU-Only pools have no families_in_household)
# still return {} and are correctly reported as unable to reproduce the table.
def _age5p_label(a):
    a = int(a)
    if a <= 24:
        lo = (a // 5) * 5
        return f"{lo}-{lo + 4} years"
    if a >= 37:
        return "85 years and over"
    lo = 25 + (a - 25) * 5
    return f"{lo}-{lo + 4} years"


def _hcfmf_label(f):
    return {1: "One family household", 2: "Two family household"}.get(
        int(f), "Three or more family household")


def _derived_obs(pop_df, attr):
    """Return derived text-label counts for age5p/hcfmf, or None if not derivable."""
    if attr == "age5p" and "agep" in pop_df.columns:
        s = pd.to_numeric(pop_df["agep"], errors="coerce").dropna()
        return s.map(_age5p_label).value_counts().to_dict()
    if attr == "hcfmf" and "families_in_household" in pop_df.columns:
        s = pd.to_numeric(pop_df["families_in_household"], errors="coerce").dropna()
        return s.map(_hcfmf_label).value_counts().to_dict()
    return None


# ---------------------------------------------------------------------------
# Per-zone metrics
# ---------------------------------------------------------------------------

def zone_metrics(pop_df, control_tables, zone_id):
    """
    For each control table, compute TVD, SRMSE, MAE% between the synthetic
    population (pop_df) and the census marginals for zone_id.
    """
    records = []
    for tbl_key, ct in control_tables.items():
        attr_cols = ct["attr_cols"]
        if not attr_cols:
            continue
        zone_rows = ct["data"][ct["data"]["sa2"].astype(str) == str(zone_id)]
        if zone_rows.empty:
            continue
        # If this table stores text category labels (bedrd/tend/lfsp/hhcd), decode
        # them to the ABS integer codes so exp keys align with the population's
        # integer obs keys. Single-attribute tables only.
        decode = _LABEL_TO_CODE.get(str(ct.get("attr", "")).lower()) if len(attr_cols) == 1 else None
        exp = {}
        for _, row in zone_rows.iterrows():
            if len(attr_cols) > 1:
                key = tuple(row[c] for c in attr_cols)
            else:
                key = row[attr_cols[0]]
                if decode is not None:
                    key = decode.get(str(key).strip(), key)
            exp[key] = exp.get(key, 0) + int(row["count"])
        if not exp:
            continue
        # age5p/hcfmf have no verbatim population column but are derivable from
        # agep / families_in_household (text labels align with the control table).
        derived = _derived_obs(pop_df, str(ct.get("attr", "")).lower())
        obs = derived if derived is not None else _obs_counts(pop_df, attr_cols)
        records.append({
            "zone":      zone_id,
            "table":     tbl_key,
            "tvd":       tvd(obs, exp),
            "srmse":     srmse(obs, exp),
            "mae_pct":   mae_pct(obs, exp),
            "n_census":  sum(exp.values()),
            "n_synth":   sum(obs.values()),
        })
    return records


# ---------------------------------------------------------------------------
# Joint-representation: Cramér's V
# ---------------------------------------------------------------------------

def _cramers_v(x, y):
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    denom = n * (min(ct.shape) - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(chi2 / denom))


def joint_representation(synth_df, micro_df, attr_cols):
    """
    For every pair of shared attributes, compute Cramér's V in both the
    synthetic population and the census microdata.

    A value near the y=x diagonal indicates the synthetic population
    preserved the pairwise association; a point above the line means the
    synthetic population *over*-represents the association; below = under.
    """
    # Build case-insensitive lookup: lowercase key -> actual column name in each df
    # This handles zone CSVs (lowercase) vs microdata loaded with .upper().
    synth_cols = {c.lower(): c for c in synth_df.columns}
    micro_cols  = {c.lower(): c for c in micro_df.columns}
    attr_lower  = sorted({c.lower() for c in attr_cols})
    shared = [c for c in attr_lower if c in synth_cols and c in micro_cols]
    results = []
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            a, b = shared[i], shared[j]
            v_s = _cramers_v(synth_df[synth_cols[a]], synth_df[synth_cols[b]])
            v_m = _cramers_v(micro_df[micro_cols[a]],  micro_df[micro_cols[b]])
            results.append({
                "attr_a":           a,
                "attr_b":           b,
                "cramers_v_synth":  v_s,
                "cramers_v_micro":  v_m,
                "abs_diff": abs(v_s - v_m)
                    if (not np.isnan(v_s) and not np.isnan(v_m)) else np.nan,
            })
    return results


def joint_representation_sa4(zones_dict, micro_df, attr_cols):
    """
    SA4-level breakdown of joint_representation(): zones sharing an SA4 prefix
    (first 3 digits of the SA2 code) are pooled before computing pairwise
    Cramér's V, so each region gets its own synth-vs-microdata comparison.

    This complements the zone-level TVD/SRMSE/MAE% SA4 aggregation
    (sa4_aggregate) with a joint-distribution view — addressing Reviewer 1's
    request to see whether the GAN improves joint, not just marginal, fit,
    and whether that effect varies geographically.
    """
    by_sa4 = {}
    for zone_id, df in zones_dict.items():
        by_sa4.setdefault(_sa4(zone_id), []).append(df)

    results = []
    for sa4, dfs in sorted(by_sa4.items()):
        pooled = pd.concat(dfs, ignore_index=True)
        for rec in joint_representation(pooled, micro_df, attr_cols):
            rec["sa4"] = sa4
            rec["n_zones"] = len(dfs)
            results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Zone-file loaders
# ---------------------------------------------------------------------------

def _read_zone_csv(fpath):
    try:
        return pd.read_csv(fpath)
    except Exception as exc:
        warnings.warn(f"Could not read {fpath}: {exc}")
        return None


def load_sa_only_zones():
    out = {}
    for fpath in sorted(glob.glob(os.path.join(config.SA_ONLY_RESULTS_DIR, "zone_*.csv"))):
        zone_id = os.path.basename(fpath).replace("zone_", "").replace(".csv", "")
        df = _read_zone_csv(fpath)
        if df is not None:
            out[zone_id] = df
    return out


def load_sa_gan_zones():
    out = {}
    patterns = (
        os.path.join(config.SA_GAN_RESULTS_DIR, "zone_*.csv"),
        os.path.join(config.SA_GAN_RESULTS_DIR, "zone_*_synthetic_population.csv"),
    )

    for pattern in patterns:
        for fpath in sorted(glob.glob(pattern)):
            fname = os.path.basename(fpath)

            # Extract SA2 zone ID from known filename forms.
            # Examples:
            #   zone_301011001.csv
            #   301011001_synthetic_population.csv
            match = re.match(r"^(?:zone_)?(\d{9})(?:\.0)?(?:_|\.csv)", fname)
            if not match:
                continue
            zone_id = match.group(1)

            if zone_id in out:
                continue

            df = _read_zone_csv(fpath)
            if df is None:
                continue

            # Guard against non-population CSVs slipping in.
            if "abshid" not in df.columns:
                continue

            out[zone_id] = df
    return out


def load_blended_zones():
    """Blended strategy (SA-GAN + SA-Only hybrid) is not yet implemented.
    Returns an empty dict so the strategy is silently skipped."""
    return {}


def _load_ipu_zones(result_dir):
    """Generic loader for IPU output directories (same naming convention as SA-GAN)."""
    if not result_dir or not os.path.isdir(result_dir):
        return {}
    out = {}
    for fpath in sorted(glob.glob(os.path.join(result_dir, "zone_*_synthetic_population.csv"))):
        fname = os.path.basename(fpath)
        match = re.match(r"^(?:zone_)?(\d{9})(?:\.0)?(?:_|\.csv)", fname)
        if not match:
            continue
        zone_id = match.group(1)
        if zone_id in out:
            continue
        df = _read_zone_csv(fpath)
        if df is None or "abshid" not in df.columns:
            continue
        out[zone_id] = df
    return out


def load_ipu_gan_zones(ipu_gan_dir=None):
    """Load IPU-GAN outputs (SA fitted using GAN-augmented microdata as seed)."""
    d = ipu_gan_dir or getattr(config, "IPU_GAN_RESULTS_DIR", None)
    return _load_ipu_zones(d)


def load_ipu_only_zones(ipu_only_dir=None):
    """Load IPU-Only outputs (SA fitted using raw census microdata as seed)."""
    d = ipu_only_dir or getattr(config, "IPU_ONLY_RESULTS_DIR", None)
    return _load_ipu_zones(d)


def load_cp_sa_zones(cp_sa_dir=None):
    """CP-pool + SA outputs (constraint-programming pool, SA fitting).

    Uses the same zone_<zone>.csv convention as SA-Only.
    """
    d = cp_sa_dir or getattr(config, "CP_POOL_RESULTS_DIR", None)
    if not d or not os.path.isdir(d):
        return {}
    out = {}
    for fpath in sorted(glob.glob(os.path.join(d, "zone_*.csv"))):
        zone_id = os.path.basename(fpath).replace("zone_", "").replace(".csv", "")
        df = _read_zone_csv(fpath)
        if df is None or "abshid" not in df.columns:
            continue
        out[zone_id] = df
    return out


def load_cp_mip_zones(cp_mip_dir=None):
    """CP-pool + MIP outputs (constraint-programming pool, integer-program fitting).

    Uses the zone_<zone>_synthetic_population.csv convention.
    """
    d = cp_mip_dir or getattr(config, "CP_MIP_RESULTS_DIR", None)
    return _load_ipu_zones(d)


def load_cp_ipu_zones(cp_ipu_dir=None):
    """CP-feasible pool + IPU outputs.

    The CP-IPU method uses the same IPU fitter as IPU-GAN/IPU-Only, making
    this a candidate-pool ablation rather than a new fitting algorithm.
    """
    d = cp_ipu_dir or getattr(config, "CP_IPU_RESULTS_DIR", None)
    return _load_ipu_zones(d)


def _load_microdata(path):
    """Load census microdata for pMSE and joint-representation analysis.

    Prefers the full training dataset (cleaned_dataset_hhid.csv) because it
    covers all GAN-output columns; the fallback `microdata_for_exp.csv` only
    carries the SA control attributes and gives a degenerate comparison.
    Identifier/geography columns are dropped and everything is coerced to
    numeric so the propensity model and Cramér's V see the same coding as the
    zone files.
    """
    full_micro = os.path.join(HERE, "cleaned_dataset_hhid.csv")
    micro_df = None
    for cand in [full_micro, path]:
        if not cand or not os.path.exists(cand):
            continue
        try:
            raw = pd.read_csv(cand)
            raw.columns = [c.upper() for c in raw.columns]
            id_cols = {"AREAENUM", "ABSHID", "ABSFID", "ABSPID", "FAMILY_ID",
                       "HOUSEHOLD_ID", "PERSON_ID", "REGUCP", "STATE_X", "STATE_Y",
                       "PERSONS_IN_HOUSEHOLD", "FAMILIES_IN_HOUSEHOLD",
                       "FINASF", "FMCF", "FNOF", "FRLF", "SPLF", "SSCF",
                       "CACF", "CPAF", "CPRF", "CPAD", "CPLTHRD",
                       "BEDRD", "HHCD", "HIED", "INGDWTD", "LLDD", "MV1D", "MV5D",
                       "STATE", "TEND", "DWTD", "BIKES", "EBIKES", "ESCOOTER"}
            raw = raw[[c for c in raw.columns
                       if c not in id_cols
                       and raw[c].notna().any()
                       and 2 <= raw[c].nunique() <= 60]]
            raw = raw.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            micro_df = raw
            print(f"\nMicrodata loaded for pMSE / joint analysis: {len(micro_df):,} "
                  f"records × {micro_df.shape[1]} columns  "
                  f"(source: {os.path.basename(cand)})")
            break
        except Exception as exc:
            warnings.warn(f"Could not load microdata ({cand}): {exc}")
    if micro_df is None:
        warnings.warn("No microdata available — pMSE and joint analysis will be skipped.")
    return micro_df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def run_strategy(name, zones_dict, control_tables):
    print(f"\n{'-'*60}\nStrategy: {name}  ({len(zones_dict)} zones)")
    all_records = []
    for zone_id, pop_df in sorted(zones_dict.items()):
        recs = zone_metrics(pop_df, control_tables, zone_id)
        all_records.extend(recs)
    if not all_records:
        print(f"  [WARN] No metrics computed for {name}.")
        return pd.DataFrame()
    df = pd.DataFrame(all_records)
    df["strategy"] = name
    return df


def sa4_aggregate(metrics_df):
    """Average TVD/SRMSE/MAE% across SA2 zones within each SA4 region."""
    if metrics_df.empty:
        return pd.DataFrame()
    df = metrics_df.copy()
    df["sa4"] = df["zone"].apply(_sa4)
    return (
        df.groupby(["strategy", "sa4", "table"], as_index=False)
        .agg(
            tvd_mean=("tvd",     "mean"),
            srmse_mean=("srmse", "mean"),
            mae_pct_mean=("mae_pct", "mean"),
            tvd_max=("tvd",      "max"),
            srmse_max=("srmse",  "max"),
            n_census=("n_census", "sum"),
            n_synth=("n_synth",   "sum"),
            n_zones=("zone",      "count"),
        )
    )


def strategy_summary(metrics_df):
    if metrics_df.empty:
        return pd.DataFrame()
    return (
        metrics_df.groupby("strategy", as_index=False)
        .agg(
            tvd_mean=("tvd",     "mean"),
            tvd_median=("tvd",   "median"),
            tvd_max=("tvd",      "max"),
            srmse_mean=("srmse", "mean"),
            srmse_median=("srmse","median"),
            srmse_max=("srmse",  "max"),
            mae_pct_mean=("mae_pct", "mean"),
            mae_pct_median=("mae_pct","median"),
            mae_pct_max=("mae_pct",  "max"),
            n_zone_table_pairs=("zone", "count"),
        )
    )


# ---------------------------------------------------------------------------
# Utility metrics beyond marginals: pMSE and hard-constraint repair rate
# ---------------------------------------------------------------------------
# The marginal metrics above (TVD/SRMSE/MAE%) only say how well each control
# table is reproduced.  Two reviewer-facing metrics are added here:
#
#   * repair rate  (R2 App A.2)
#       Fraction of records that violate a hard logical constraint — a child
#       with an occupation, a non-worker with hours worked, a household both
#       renting and paying a mortgage, etc.  The GAN's raw output is repaired
#       before use, so this measures how much repair each method needs.  A
#       method that enforces the constraints while *constructing* records (the
#       CP feasible pool) should sit near 0%; methods that only fit marginals
#       (SA/IPU) can still emit impossible records.
#
#   * pMSE  (R1 p15 — utility beyond internal goodness-of-fit)
#       A logistic regression is trained to separate synthetic records from
#       real census microdata.  pMSE = mean((p̂ − 0.5)²) on a held-out split,
#       normalised by the null pMSE = c/(1+c)² where c = n_synth/n_real.
#       ratio ≈ 1 → indistinguishable from the census; higher → worse.
#       Reference: Woo et al. (2009), JPC.

# Identifiers, geography, provenance and NA-encoded dwelling metadata.  These
# either hand the classifier a trivial shortcut (a synthetic-only ABSHID) or
# carry coverage differences between the national microdata and the SEQ zone
# populations that are measurement artefacts rather than synthesis quality.
_PMSE_EXCLUDE = {
    "SOURCE", "_SOURCE", "_DUP_COUNT", "_UNIQUENESS", "UNNAMED: 0",
    "ABSHID", "ABSFID", "ABSPID", "AREAENUM", "REGUCP", "REGU1P", "REGU5P",
    "STATE", "STATE_X", "STATE_Y", "STATEUR",
    "UAICP", "UAI1P", "UAI5P",
    "DWTD", "DWIP", "DWTP", "INGDWTD", "LLDD", "TYPP",
    "FAMILY_ID", "HOUSEHOLD_ID", "PERSON_ID",
}

_STRATEGY_ORDER = ["sa_gan", "sa_only", "ipu_gan", "ipu_only",
                   "cp_sa", "cp_ipu", "cp_mip"]


def _upper_frame(df):
    """Return df with upper-cased column names.

    Avoids copying when the columns are already upper-case strings, which
    matters because the census microdata (~1.2M rows) is upper-cased once in
    `_load_microdata` and then reused for every strategy.
    """
    cols = [str(c) for c in df.columns]
    if list(df.columns) == cols and all(c == c.upper() for c in cols):
        return df
    out = df.copy()
    out.columns = [c.upper() for c in cols]
    return out


def _constraint_checks():
    """Hard-constraint definitions shared with gan_train.py / gan_validation.py.

    Imported from constraint_rules (no sklearn/scipy dependency) so the
    benchmark still runs in a minimal environment.
    """
    try:
        from constraint_rules import CONSTRAINT_CHECKS
        return CONSTRAINT_CHECKS
    except Exception as exc:            # pragma: no cover - defensive
        warnings.warn(f"Could not import CONSTRAINT_CHECKS ({exc}); "
                      f"repair rate will be skipped.")
        return {}


def strategy_repair_rate(zones_dict):
    """Per-strategy hard-constraint violation rate.

    Returns
    -------
    summary : dict | None
        n_records, n_violating_records, repair_rate_pct for the whole strategy.
    detail : list[dict]
        One row per violated constraint (descending by count).
    """
    checks = _constraint_checks()
    if not checks or not zones_dict:
        return None, []

    n_total = 0
    n_any = 0
    per_check = {}
    for _zone_id, df in zones_dict.items():
        if df is None or len(df) == 0:
            continue
        d = _upper_frame(df).apply(pd.to_numeric, errors="coerce")
        n = len(d)
        n_total += n
        any_mask = pd.Series(False, index=d.index)
        for desc, fn in checks.items():
            try:
                mask = fn(d)
            except Exception:
                continue
            mask = pd.Series(mask, index=d.index).fillna(False).astype(bool)
            cnt = int(mask.sum())
            if cnt:
                per_check[desc] = per_check.get(desc, 0) + cnt
            any_mask |= mask
        n_any += int(any_mask.sum())

    if n_total == 0:
        return None, []
    summary = {
        "n_records": n_total,
        "n_violating_records": n_any,
        "repair_rate_pct": 100.0 * n_any / n_total,
    }
    detail = [
        {"constraint": k, "n_violations": v,
         "pct_of_records": 100.0 * v / n_total}
        for k, v in sorted(per_check.items(), key=lambda kv: -kv[1])
    ]
    return summary, detail


def _pmse_feature_set(zones_by_strategy, micro_df):
    """Attributes usable by the propensity model, common to *every* strategy.

    Using one shared feature set keeps pMSE comparable across methods: a
    strategy cannot look better simply because it carries more columns.
    """
    if micro_df is None:
        return []
    micro_df = _upper_frame(micro_df)
    micro_cols = set(micro_df.columns)
    common = None
    for zones in zones_by_strategy.values():
        if not zones:
            continue
        sample = next(iter(zones.values()))
        cols = {str(c).upper() for c in sample.columns}
        common = cols if common is None else (common & cols)
    if not common:
        return []

    feats = []
    for c in sorted(common & micro_cols):
        if c in _PMSE_EXCLUDE:
            continue
        if micro_df[c].nunique(dropna=True) < 2:
            continue
        feats.append(c)
    return feats


def _sample_strategy(zones_dict, cols, n_sample=50_000, seed=42):
    """Sample ~n_sample records spread evenly across zones.

    Sampling per zone (rather than concatenating every zone first) keeps peak
    memory flat: with 457 zones we take ~n_sample/457 rows from each.
    """
    if not cols or not zones_dict:
        return pd.DataFrame(columns=cols)
    per_zone = max(1, int(np.ceil(n_sample / max(len(zones_dict), 1))) + 1)
    chunks = []
    for _zone_id, df in zones_dict.items():
        if df is None or len(df) == 0:
            continue
        cmap = {str(c).upper(): c for c in df.columns}
        keep = [cmap[c] for c in cols if c in cmap]
        d = df[keep].copy() if keep else pd.DataFrame(index=df.index)
        d.columns = [str(c).upper() for c in d.columns]
        for c in cols:                      # defensive: keep one feature space
            if c not in d.columns:
                d[c] = np.nan
        d = d[cols]
        if len(d) > per_zone:
            d = d.sample(per_zone, random_state=seed)
        chunks.append(d)
    if not chunks:
        return pd.DataFrame(columns=cols)
    out = pd.concat(chunks, ignore_index=True)
    if len(out) > n_sample:
        out = out.sample(n_sample, random_state=seed)
    return out[cols]


def pmse_propensity(synth_sample, micro_df, cols, n_sample=50_000):
    """Propensity-score MSE of a synthetic sample against census microdata."""
    if micro_df is None or not cols:
        return None
    micro_df = _upper_frame(micro_df)
    synth_sample = _upper_frame(synth_sample)
    cols = [c for c in cols if c in micro_df.columns and c in synth_sample.columns]
    if len(cols) < 2:
        return None

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except Exception as exc:            # pragma: no cover - defensive
        warnings.warn(f"pMSE skipped (sklearn unavailable): {exc}")
        return None

    o = micro_df[cols].apply(pd.to_numeric, errors="coerce").fillna(-9)
    s = synth_sample[cols].apply(pd.to_numeric, errors="coerce").fillna(-9)
    if len(o) == 0 or len(s) == 0:
        return None
    o = o.sample(min(n_sample, len(o)), random_state=42).assign(_label=0)
    s = s.sample(min(n_sample, len(s)), random_state=42).assign(_label=1)

    combined = pd.concat([o, s], ignore_index=True)
    X = combined[cols].to_numpy(dtype=float)
    y = combined["_label"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    p_te = clf.predict_proba(X_te)[:, 1]
    pmse = float(np.mean((p_te - 0.5) ** 2))
    acc = float(clf.score(X_te, y_te))

    c = len(s) / max(len(o), 1)
    null_pmse = c / (1.0 + c) ** 2
    return {
        "pmse": pmse,
        "null_pmse": null_pmse,
        "pmse_ratio": (pmse / null_pmse) if null_pmse > 0 else np.nan,
        "test_accuracy": acc,
        "n_synth": len(s),
        "n_micro": len(o),
        "n_features": len(cols),
    }


def _merge_utility(summary_df, repair_df, pmse_df):
    """Attach repair-rate and pMSE columns to the strategy summary."""
    if summary_df.empty:
        return summary_df
    out = summary_df.copy()
    if repair_df is not None and not repair_df.empty:
        out = out.merge(
            repair_df[["strategy", "n_records", "n_violating_records",
                       "repair_rate_pct"]],
            on="strategy", how="left")
    if pmse_df is not None and not pmse_df.empty:
        out = out.merge(
            pmse_df[["strategy", "pmse", "null_pmse", "pmse_ratio",
                     "test_accuracy", "n_features"]],
            on="strategy", how="left")
    return out


def plot_utility_comparison(repair_df, pmse_df):
    """Two-panel summary: constraint repair rate and pMSE ratio by strategy."""
    strategies = []
    for df in (repair_df, pmse_df):
        if df is not None and not df.empty:
            for s in df["strategy"]:
                if s not in strategies:
                    strategies.append(s)
    if not strategies:
        return
    order = {s: i for i, s in enumerate(_STRATEGY_ORDER)}
    strategies.sort(key=lambda s: order.get(s, 99))
    colors = [_STRATEGY_COLORS.get(s, "#888888") for s in strategies]
    labels = [_STRATEGY_LABELS.get(s, s) for s in strategies]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(strategies) * 2.6), 5))

    ax = axes[0]
    if repair_df is not None and not repair_df.empty:
        r = repair_df.set_index("strategy").reindex(strategies)
        vals = r["repair_rate_pct"].fillna(0).to_numpy(dtype=float)
        ax.bar(range(len(strategies)), vals, color=colors, alpha=0.82)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("records violating ≥1 hard constraint (%)")
    ax.set_title("Hard-constraint violation rate (lower is better)", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    ax = axes[1]
    if pmse_df is not None and not pmse_df.empty:
        p = pmse_df.set_index("strategy").reindex(strategies)
        raw = p["pmse_ratio"].to_numpy(dtype=float)
        vals = np.nan_to_num(raw, nan=0.0)
        ax.bar(range(len(strategies)), vals, color=colors, alpha=0.82)
        for i, v in enumerate(raw):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("pMSE / null pMSE  (1.0 = indistinguishable)")
    ax.set_title("Propensity-score MSE ratio (lower is better)", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "strategy_utility_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved: strategy_utility_comparison.png")


# ---------------------------------------------------------------------------
# Plots
# Consistent palette for all 4 strategies (colour-blind friendly)
_STRATEGY_COLORS = {
    "sa_gan":   "#4C72B0",   # blue
    "sa_only":  "#DD8452",   # orange
    "ipu_gan":  "#55A868",   # green
    "ipu_only": "#C44E52",   # red
    "cp_sa":    "#8172B2",   # purple  (CP feasible pool + SA)
    "cp_ipu":   "#64B5CD",   # cyan    (CP feasible pool + IPU)
    "cp_mip":   "#937860",   # brown   (CP feasible pool + MIP)
    "blended":  "#8172B2",   # purple
}
_STRATEGY_LABELS = {
    "sa_gan":   "SA-GAN",
    "sa_only":  "SA-Only",
    "ipu_gan":  "IPU-GAN",
    "ipu_only": "IPU-Only",
    "cp_sa":    "CP-pool + SA",
    "cp_ipu":   "CP-pool + IPU",
    "cp_mip":   "CP-pool + MIP",
    "blended":  "Blended",
}

# ---------------------------------------------------------------------------

def plot_metric_boxplots(metrics_df):
    if metrics_df.empty:
        return
    strategies = sorted(metrics_df["strategy"].unique())
    colors = [_STRATEGY_COLORS.get(s, "#888888") for s in strategies]
    labels = [_STRATEGY_LABELS.get(s, s) for s in strategies]
    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(strategies) * 4), 5))
    for ax, col, title, cap_pct in zip(
        axes,
        ["tvd", "srmse", "mae_pct"],
        ["TVD (0-1)", "SRMSE", "MAE %"],
        [None, 95, 95],
    ):
        data = []
        for s in strategies:
            vals = metrics_df.loc[metrics_df["strategy"] == s, col].dropna().values
            if cap_pct and len(vals) > 0:
                _cap = float(np.percentile(vals, cap_pct))
                vals = np.clip(vals, 0, _cap)
            data.append(vals)
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=False,
                        showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        # SRMSE and MAE% are extremely heavy-tailed: sparse household tables (near-zero
        # expected counts) blow the GAN-augmented strategies up to SRMSE~240 / MAE~9000%,
        # which on a linear axis flattens the IPU-Only / SA-Only boxes to invisible slivers.
        # A symlog y-axis keeps every strategy's box readable and comparable.
        if col in ("srmse", "mae_pct"):
            ax.set_yscale("symlog", linthresh=1)
        ax.set_title(
            title
            + (f" (clipped at P{cap_pct}" if cap_pct else "")
            + (", log scale)" if col in ("srmse", "mae_pct") else (")" if cap_pct else "")),
            fontsize=11)
        ax.set_ylabel(col)
        ax.set_xlabel("Strategy")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
    plt.suptitle("Benchmark Metric Distribution Across Zones × Tables\n(extreme outliers clipped)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "benchmark_metrics_boxplot.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: benchmark_metrics_boxplot.png")


def plot_sa4_heatmap(sa4_df, metric="tvd_mean"):
    """One heatmap per strategy: SA4 regions (rows) × control tables (columns)."""
    if sa4_df.empty:
        return
    for strat in sorted(sa4_df["strategy"].unique()):
        sub = sa4_df[sa4_df["strategy"] == strat]
        pivot = sub.pivot_table(index="sa4", columns="table", values=metric, aggfunc="mean")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(
            figsize=(max(12, len(pivot.columns) * 0.65),
                     max(5,  len(pivot) * 0.22))
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f"{strat} — SA4 geography × control table  ({metric})")
        plt.colorbar(im, ax=ax, label=metric)
        plt.tight_layout()
        fname = f"sa4_heatmap_{strat}_{metric}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=100)
        plt.close()
        print(f"  Saved: {fname}")


def plot_joint_sa4_heatmap(joint_sa4_df):
    """SA4 regions (rows) × strategies (columns): mean |Δ Cramér's V| vs census microdata.

    Lower (lighter) = the synthetic population for that region/strategy preserves
    pairwise attribute associations closer to the census microdata baseline.
    """
    if joint_sa4_df.empty:
        return
    pivot = (
        joint_sa4_df.dropna(subset=["abs_diff"])
        .pivot_table(index="sa4", columns="strategy", values="abs_diff", aggfunc="mean")
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.6),
                                    max(5, len(pivot) * 0.3)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_STRATEGY_LABELS.get(c, c) for c in pivot.columns],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Joint representation by SA4 region — mean |Cramér's V (synth) − Cramér's V (microdata)|")
    plt.colorbar(im, ax=ax, label="mean |Δ Cramér's V|")
    plt.tight_layout()
    fname = "joint_cramer_sa4_heatmap.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


# SA control-table attribute names (in-sample for the SA fitting step)
_SA_CONTROL_ATTRS = {
    "SEXP", "MSTP", "MTWP", "OCCP", "MRERD", "RNTRD", "VEHRD", "NPRD",
    "RLHP", "HEAP", "HIND", "STRD", "HRSP", "INCP", "AGEP",
}


def plot_joint_scatter(joint_df, ctrl_attrs=None):
    """Cramér's V in synthetic population vs census microdata, one panel per strategy.

    Points are colour-coded:
      * Blue  = pair where BOTH attributes are SA control-table columns (in-sample)
      * Orange = pair where AT LEAST ONE attribute is out-of-sample (GAN-only column)

    This distinguishes between preservation of fitted marginal associations
    (expected to be tighter) and genuine out-of-sample generalization.
    """
    if joint_df.empty:
        return
    if ctrl_attrs is None:
        ctrl_attrs = _SA_CONTROL_ATTRS

    strategies = sorted(joint_df["strategy"].unique())
    fig, axes = plt.subplots(1, len(strategies),
                             figsize=(5 * len(strategies), 5), squeeze=False)
    for ax, strat in zip(axes[0], strategies):
        sub = joint_df[joint_df["strategy"] == strat].dropna(
            subset=["cramers_v_synth", "cramers_v_micro"])
        if sub.empty:
            ax.set_title(f"{strat}  (no data)")
            continue

        # Colour by in-sample vs out-of-sample
        both_in = sub["attr_a"].str.upper().isin(ctrl_attrs) & \
                  sub["attr_b"].str.upper().isin(ctrl_attrs)
        sub_in  = sub[both_in]
        sub_out = sub[~both_in]

        lo = min(sub["cramers_v_micro"].min(), sub["cramers_v_synth"].min()) * 0.95
        hi = max(sub["cramers_v_micro"].max(), sub["cramers_v_synth"].max()) * 1.05

        if not sub_in.empty:
            ax.scatter(sub_in["cramers_v_micro"], sub_in["cramers_v_synth"],
                       alpha=0.7, s=30, color="#4C72B0", label=f"in-sample (n={len(sub_in)})")
        if not sub_out.empty:
            ax.scatter(sub_out["cramers_v_micro"], sub_out["cramers_v_synth"],
                       alpha=0.6, s=20, color="#DD8452", marker="^",
                       label=f"out-of-sample (n={len(sub_out)})")

        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x (perfect)")

        # Pearson r for each group
        def _r(df):
            if len(df) < 3:
                return np.nan
            return float(np.corrcoef(df["cramers_v_micro"], df["cramers_v_synth"])[0, 1])

        r_in  = _r(sub_in)
        r_out = _r(sub_out)
        r_all = _r(sub)
        ax.set_xlabel("Cramér's V  (census microdata)")
        ax.set_ylabel("Cramér's V  (synthetic)")
        title_parts = [strat]
        if not np.isnan(r_all):
            title_parts.append(f"r={r_all:.3f}")
        ax.set_title("  ".join(title_parts))
        ax.legend(fontsize=7)

        # Print per-strategy summary
        print(f"  [{strat}] Cramér's V pairs: {len(sub)}  "
              f"r_all={r_all:.3f}  r_in-sample={r_in:.3f}  r_out-of-sample={r_out:.3f}")

    plt.suptitle(
        "Joint attribute representation: synthetic vs census microdata\n"
        "Blue = SA control attrs (in-sample)  |  Orange▲ = GAN-only attrs (out-of-sample)",
        fontsize=9,
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "joint_cramer_scatter.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: joint_cramer_scatter.png")


def plot_per_table_comparison(metrics_df):
    """
    Grouped bar chart: mean TVD per control table, one bar group per strategy.
    Shows which specific attributes each method fits better/worse.
    Saved as per_table_tvd_comparison.png.
    """
    if metrics_df.empty:
        return
    strategies = sorted(metrics_df["strategy"].unique())
    table_tvd = (
        metrics_df.groupby(["table", "strategy"])["tvd"]
        .mean()
        .unstack("strategy")
        .reindex(columns=strategies)
    )
    table_tvd = table_tvd.sort_values(strategies[0], ascending=False)

    n_tables = len(table_tvd)
    n_strats = len(strategies)
    width = 0.8 / n_strats
    x = np.arange(n_tables)

    fig, ax = plt.subplots(figsize=(max(14, n_tables * 0.55), 5))
    for i, strat in enumerate(strategies):
        vals = table_tvd[strat].values if strat in table_tvd.columns else np.full(n_tables, np.nan)
        bars = ax.bar(
            x + (i - n_strats / 2 + 0.5) * width, vals,
            width=width * 0.9,
            color=_STRATEGY_COLORS.get(strat, "#888888"),
            alpha=0.8,
            label=_STRATEGY_LABELS.get(strat, strat),
        )
    ax.set_xticks(x)
    # Strip the ABS "sa2" table-name prefix for cleaner axis labels (sa2lfsp -> lfsp).
    _labels = [str(t)[3:] if str(t).lower().startswith("sa2") else str(t)
               for t in table_tvd.index]
    ax.set_xticklabels(_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Mean TVD")
    ax.set_title("Per-attribute TVD — strategy comparison")
    ax.legend(fontsize=9)
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "per_table_tvd_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: per_table_tvd_comparison.png")


def write_latex_comparison_table(summary_df, out_path):
    """
    Write a LaTeX booktabs table of strategy-level metrics to `out_path`.
    Columns: Method | n zones | TVD mean | TVD median | SRMSE mean | SRMSE median | MAE% mean
    Best value per column is bolded.
    """
    if summary_df.empty:
        return
    order = ["sa_gan", "sa_only", "ipu_gan", "ipu_only",
             "cp_sa", "cp_ipu", "cp_mip"]
    df = summary_df.copy()
    df["_sort"] = df["strategy"].map({s: i for i, s in enumerate(order)}).fillna(99)
    df = df.sort_values("_sort").drop(columns="_sort")
    df["strategy"] = df["strategy"].map(_STRATEGY_LABELS).fillna(df["strategy"])

    cols_map = {
        "tvd_mean":    "TVD Mean",
        "tvd_median":  "TVD Median",
        "srmse_mean":  "SRMSE Mean",
        "srmse_median":"SRMSE Median",
        "mae_pct_mean":"MAE\\% Mean",
        "repair_rate_pct": "Constraint\\newline Violations (\\%)",
        "pmse_ratio":  "pMSE\\newline Ratio",
    }
    numeric_cols = [c for c in cols_map if c in df.columns]
    # repair rate is a percentage (1 dp), pMSE ratio a dimensionless factor (3 dp)
    fmt_map = {"repair_rate_pct": "{:.1f}", "pmse_ratio": "{:.3f}"}

    lines = [
        "% Auto-generated by benchmark_margins.py — do not edit manually",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Strategy comparison: mean marginal fit metrics across all SA2 zones and control tables.}",
        "  \\label{tab:strategy_comparison}",
        "  \\begin{tabular}{l" + "r" * len(numeric_cols) + "}",
        "    \\toprule",
        "    Method & " + " & ".join(cols_map[c] for c in numeric_cols) + " \\\\",
        "    \\midrule",
    ]
    # Find best (min) value per column for bolding
    best = {c: df[c].min() for c in numeric_cols if c in df.columns}
    for _, row in df.iterrows():
        cells = [row["strategy"]]
        for c in numeric_cols:
            val = row.get(c, float("nan"))
            if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                fmt = fmt_map.get(c, "{:.4f}").format(val)
                if abs(val - best[c]) < 1e-6:
                    fmt = f"\\textbf{{{fmt}}}"
            else:
                fmt = "--"
            cells.append(fmt)
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Saved LaTeX table: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark strategies against census margins")
    _all_strats = ["sa_gan", "sa_only", "ipu_gan", "ipu_only", "blended",
                   "cp_sa", "cp_ipu", "cp_mip"]
    p.add_argument("--strategies", nargs="*", choices=_all_strats,
                   default=["sa_gan", "sa_only", "blended"],
                   help="Which strategies to benchmark")
    p.add_argument("--ipu-gan-dir", default=None,
                   help="Override directory for IPU-GAN zone outputs (default: config.IPU_GAN_RESULTS_DIR)")
    p.add_argument("--ipu-only-dir", default=None,
                   help="Override directory for IPU-Only zone outputs (default: config.IPU_ONLY_RESULTS_DIR)")
    p.add_argument("--cp-sa-dir", default=None,
                   help="Override directory for CP-pool + SA zone outputs (default: config.CP_POOL_RESULTS_DIR)")
    p.add_argument("--cp-ipu-dir", default=None,
                   help="Override directory for CP-pool + IPU zone outputs (default: config.CP_IPU_RESULTS_DIR)")
    p.add_argument("--cp-mip-dir", default=None,
                   help="Override directory for CP-pool + MIP zone outputs (default: config.CP_MIP_RESULTS_DIR)")
    p.add_argument("--microdata", default=config.CENSUS_MICRODATA,
                   help="Path to census microdata seed for joint-representation analysis")
    p.add_argument("--control-level", choices=["sa2", "sa1"], default="sa2",
                   help="Control table geography level for benchmarking")
    p.add_argument("--output-subdir", default=None,
                   help="Optional output subdirectory under outputs/ (default: benchmarking for sa2, benchmarking_sa1 for sa1)")
    p.add_argument("--min-zone-persons", type=int,
                   default=getattr(config, "MIN_REPORT_PERSONS", 500),
                   help="Reporting floor: zones with fewer census residents are "
                        "synthesised and exported, but excluded from the reported "
                        "marginal-fit statistics (default: config.MIN_REPORT_PERSONS)")
    p.add_argument("--all-zones", action="store_true",
                   help="Report every zone regardless of size (disables the "
                        "reporting floor; non-geographic zones stay excluded)")
    args = p.parse_args()

    global OUTPUT_DIR
    if args.output_subdir:
        OUTPUT_DIR = os.path.join(config.OUTPUTS_DIR, args.output_subdir)
    elif args.control_level == "sa1":
        OUTPUT_DIR = os.path.join(config.OUTPUTS_DIR, "benchmarking_sa1")
    else:
        OUTPUT_DIR = os.path.join(config.OUTPUTS_DIR, "benchmarking")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    control_dir = config.SA_CONTROL_DIR if args.control_level == "sa2" else config.SA1_CONTROL_DIR

    print(f"Loading census control tables from: {control_dir}")
    control_tables = load_all_controls(control_dir, zone_level=args.control_level)
    if not control_tables:
        print("[ERROR] No control tables loaded — check SA_CONTROL_DIR in config.py")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 0. Census microdata — needed by both pMSE and the joint analysis,
    #    so load it once up front.
    # ------------------------------------------------------------------
    micro_df = _load_microdata(args.microdata)

    # ------------------------------------------------------------------
    # 1. Per-zone × per-table metrics at SA2 level
    # ------------------------------------------------------------------
    all_metrics = []
    all_zones_dfs = {}

    for strat in args.strategies:
        loaders = {
            "sa_only":  lambda: load_sa_only_zones(),
            "sa_gan":   lambda: load_sa_gan_zones(),
            "ipu_gan":  lambda: load_ipu_gan_zones(args.ipu_gan_dir),
            "ipu_only": lambda: load_ipu_only_zones(args.ipu_only_dir),
            "cp_sa":    lambda: load_cp_sa_zones(args.cp_sa_dir),
            "cp_ipu":   lambda: load_cp_ipu_zones(args.cp_ipu_dir),
            "cp_mip":   lambda: load_cp_mip_zones(args.cp_mip_dir),
            "blended":  lambda: load_blended_zones(),
        }
        zones = loaders[strat]()
        if not zones:
            print(f"  [WARN] No zone files found for strategy '{strat}' — skipping.")
            continue
        all_zones_dfs[strat] = zones
        df = run_strategy(strat, zones, control_tables)
        if not df.empty:
            all_metrics.append(df)

    if not all_metrics:
        print("[ERROR] No metrics produced.  Run SA stages first.")
        sys.exit(1)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_name = f"metrics_{args.control_level}.csv"
    metrics_path = os.path.join(OUTPUT_DIR, metrics_name)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved: {metrics_name}  ({len(metrics_df):,} rows)")

    # ------------------------------------------------------------------
    # 1b. Reporting scope
    # ------------------------------------------------------------------
    # A handful of SA2s have almost no residents (0-80 people: airports,
    # industrial estates, parkland).  They are synthesised in full and stay in
    # every population output -- including the travel-model export -- but their
    # TVD/SRMSE are noise: expected cell counts are far below 1, and SRMSE
    # divides by max(expected, 1) so it explodes.  Only the reported statistics
    # are filtered; nothing is dropped from a run or a results directory.
    floor = 0 if args.all_zones else args.min_zone_persons
    scope = reporting_scope.reporting_scope(
        control_tables, min_persons=floor, zone_level=args.control_level)
    report_metrics_df, scope_exclusions = reporting_scope.filter_reported(
        metrics_df, scope)

    scope.to_csv(os.path.join(OUTPUT_DIR, "zone_reporting_scope.csv"), index=False)
    scope_exclusions.to_csv(
        os.path.join(OUTPUT_DIR, "zone_reporting_exclusions.csv"), index=False)
    report_metrics_df.to_csv(
        os.path.join(OUTPUT_DIR, f"metrics_{args.control_level}_reported.csv"),
        index=False)
    print()
    print(reporting_scope.summarise(scope, metrics_df))
    if len(scope_exclusions):
        print("Excluded from reported statistics (still synthesised and exported):")
        for _, r in scope_exclusions.iterrows():
            _pid = r["intended_persons"]
            _pid_s = "    ?" if pd.isna(_pid) else f"{int(_pid):>5}"
            print(f"  {r['zone']}  persons={_pid_s}  {r['reason']}")

    # ------------------------------------------------------------------
    # 2. SA4 aggregation
    # ------------------------------------------------------------------
    sa4_df = sa4_aggregate(report_metrics_df)
    sa4_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_sa4.csv"), index=False)
    print(f"Saved: metrics_sa4.csv  ({len(sa4_df):,} rows)")

    # ------------------------------------------------------------------
    # 3. Hard-constraint repair rate (R2 App A.2)
    # ------------------------------------------------------------------
    repair_rows = []
    repair_detail_rows = []
    for strat, zones in all_zones_dfs.items():
        summary, detail = strategy_repair_rate(zones)
        if summary is None:
            print(f"  [WARN] repair rate unavailable for '{strat}'")
            continue
        summary["strategy"] = strat
        repair_rows.append(summary)
        for rec in detail:
            rec["strategy"] = strat
            repair_detail_rows.append(rec)
        print(f"  repair rate [{strat}]: {summary['repair_rate_pct']:.2f}% "
              f"({summary['n_violating_records']:,}/{summary['n_records']:,} records)")

    repair_df = pd.DataFrame(repair_rows)
    if not repair_df.empty:
        repair_df = repair_df[["strategy", "n_records", "n_violating_records",
                               "repair_rate_pct"]]
        repair_df.to_csv(os.path.join(OUTPUT_DIR, "repair_rate.csv"), index=False)
        print(f"Saved: repair_rate.csv  ({len(repair_df)} strategies)")
    if repair_detail_rows:
        repair_detail_df = pd.DataFrame(repair_detail_rows)[
            ["strategy", "constraint", "n_violations", "pct_of_records"]]
        repair_detail_df = repair_detail_df.sort_values(
            ["strategy", "n_violations"], ascending=[True, False])
        repair_detail_df.to_csv(
            os.path.join(OUTPUT_DIR, "repair_rate_by_constraint.csv"), index=False)
        print(f"Saved: repair_rate_by_constraint.csv  ({len(repair_detail_df)} rows)")

    # ------------------------------------------------------------------
    # 4. Propensity-score MSE (R1 p15)
    # ------------------------------------------------------------------
    pmse_rows = []
    if micro_df is not None and all_zones_dfs:
        pmse_cols = _pmse_feature_set(all_zones_dfs, micro_df)
        print(f"\npMSE feature set: {len(pmse_cols)} attributes shared by every strategy")
        for strat, zones in all_zones_dfs.items():
            sample = _sample_strategy(zones, pmse_cols)
            res = pmse_propensity(sample, micro_df, pmse_cols)
            if res is None:
                print(f"  [WARN] pMSE unavailable for '{strat}'")
                continue
            res["strategy"] = strat
            pmse_rows.append(res)
            print(f"  pMSE [{strat}]: {res['pmse']:.6f}  "
                  f"ratio={res['pmse_ratio']:.3f}  acc={res['test_accuracy']:.3f}")
    else:
        print("\n[SKIP] pMSE — microdata not available.")

    pmse_df = pd.DataFrame(pmse_rows)
    if not pmse_df.empty:
        pmse_df = pmse_df[["strategy", "pmse", "null_pmse", "pmse_ratio",
                           "test_accuracy", "n_synth", "n_micro", "n_features"]]
        pmse_df.to_csv(os.path.join(OUTPUT_DIR, "pmse.csv"), index=False)
        print(f"Saved: pmse.csv  ({len(pmse_df)} strategies)")

    # ------------------------------------------------------------------
    # 5. Strategy summary table (marginal fit + utility)
    # ------------------------------------------------------------------
    summary_df = _merge_utility(strategy_summary(report_metrics_df), repair_df, pmse_df)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "strategy_summary.csv"), index=False)
    # All-zone summary kept alongside it: equal to the reported numbers when the
    # floor is 0, and otherwise shows exactly what the excluded zones moved.
    _merge_utility(strategy_summary(metrics_df), repair_df, pmse_df).to_csv(
        os.path.join(OUTPUT_DIR, "strategy_summary_all_zones.csv"), index=False)
    print(f"\nStrategy summary (all tables, zones >= {floor} persons):")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    plot_metric_boxplots(report_metrics_df)
    plot_sa4_heatmap(sa4_df, metric="tvd_mean")
    plot_sa4_heatmap(sa4_df, metric="srmse_mean")
    plot_per_table_comparison(report_metrics_df)
    plot_utility_comparison(repair_df, pmse_df)

    # ------------------------------------------------------------------
    # 6b. LaTeX comparison table
    # ------------------------------------------------------------------
    paper_dir = os.path.normpath(os.path.join(HERE, "..", "Population_Synthesiser___GAN_SA"))
    if os.path.isdir(paper_dir):
        write_latex_comparison_table(
            summary_df,
            os.path.join(paper_dir, "strategy_comparison_table.tex"),
        )
    else:
        write_latex_comparison_table(
            summary_df,
            os.path.join(OUTPUT_DIR, "strategy_comparison_table.tex"),
        )

    # ------------------------------------------------------------------
    # 7. Joint representation (Cramér's V vs census microdata)
    # ------------------------------------------------------------------
    # Collect all attribute columns seen across zone population files.
    # Use the union of control-table columns + whatever the zone CSVs carry,
    # so out-of-sample GAN attributes (INCP, EMPP, QALFP …) are also evaluated.
    if all_zones_dfs:
        _sample_zone_df = next(iter(next(iter(all_zones_dfs.values())).values()), None)
    else:
        _sample_zone_df = None

    _ctrl_attr_cols = sorted({
        col
        for ct in control_tables.values()
        for col in ct["attr_cols"]
    })
    if _sample_zone_df is not None:
        _zone_attr_cols = sorted([
            c.upper() for c in _sample_zone_df.columns
            if c.upper() not in {
                "ABSHID", "ABSFID", "ABSPID", "REGUCP", "AREAENUM",
                "FAMILY_ID", "HOUSEHOLD_ID", "PERSON_ID",
            }
        ])
        attr_cols_all = sorted(set(_ctrl_attr_cols) | set(_zone_attr_cols))
    else:
        attr_cols_all = _ctrl_attr_cols

    print(f"  Joint-representation attribute pool: {len(attr_cols_all)} columns")
    print(f"  Control-table attrs (in-sample): {len(_ctrl_attr_cols)}")
    print(f"  Extra GAN-output attrs (out-of-sample): "
          f"{len(set(attr_cols_all) - set(_ctrl_attr_cols))}")

    all_joint = []
    all_joint_sa4 = []
    for strat, zones in all_zones_dfs.items():
        if not zones:
            continue
        synth_all = pd.concat(list(zones.values()), ignore_index=True)
        if micro_df is not None:
            joint = joint_representation(synth_all, micro_df, attr_cols_all)
            for rec in joint:
                rec["strategy"] = strat
            all_joint.extend(joint)

            joint_sa4 = joint_representation_sa4(zones, micro_df, attr_cols_all)
            for rec in joint_sa4:
                rec["strategy"] = strat
            all_joint_sa4.extend(joint_sa4)
        else:
            print(f"  [SKIP] Joint analysis for '{strat}' — microdata not available.")

    if all_joint:
        joint_df = pd.DataFrame(all_joint).sort_values("abs_diff", ascending=False)
        joint_df.to_csv(os.path.join(OUTPUT_DIR, "joint_representation.csv"), index=False)
        print(f"\nSaved: joint_representation.csv  ({len(joint_df):,} pairs)")
        print("Top 10 attribute pairs with largest Cramér's V deviation:")
        top = joint_df.dropna(subset=["abs_diff"]).head(10)
        for _, row in top.iterrows():
            print(f"  [{row['strategy']}] {row['attr_a']} x {row['attr_b']}"
                  f"  synth={row['cramers_v_synth']:.3f}"
                  f"  micro={row['cramers_v_micro']:.3f}"
                  f"  d={row['abs_diff']:.3f}")
        _ctrl_cols_set = {c.upper() for c in _ctrl_attr_cols}
        plot_joint_scatter(joint_df, ctrl_attrs=_ctrl_cols_set)

    if all_joint_sa4:
        joint_sa4_df = pd.DataFrame(all_joint_sa4).sort_values(
            ["strategy", "sa4", "abs_diff"], ascending=[True, True, False])
        joint_sa4_df.to_csv(os.path.join(OUTPUT_DIR, "joint_representation_sa4.csv"), index=False)
        print(f"\nSaved: joint_representation_sa4.csv  "
              f"({len(joint_sa4_df):,} rows across {joint_sa4_df['sa4'].nunique()} SA4 regions)")
        plot_joint_sa4_heatmap(joint_sa4_df)

    # ------------------------------------------------------------------
    # Final checklist
    # ------------------------------------------------------------------
    print(f"\nAll outputs written to: {OUTPUT_DIR}")
    print(
        f"\n[CHECK] {metrics_name}  - inspect which zones and tables have high TVD/SRMSE/MAE%.\n"
        "[CHECK] metrics_sa4.csv  - compare strategies at SA4 level; "
        "report in Table 2 alongside RSSZ.\n"
        "[CHECK] sa4_heatmap_*.png - do high-error SA4 regions correspond to demographically\n"
        "         complex areas?  Mention in the outlier discussion.\n"
        "[CHECK] joint_cramer_scatter.png - are synthetic attribute correlations close to\n"
        "         census?  Report abs_diff > 0.15 pairs in Appendix A.\n"
        "[CHECK] joint_cramer_sa4_heatmap.png / joint_representation_sa4.csv - does the GAN's\n"
        "         joint-distribution benefit hold across all SA4 regions, or concentrate in a few?\n"
        "[CHECK] strategy_summary.csv - add TVD_mean and SRMSE_mean columns to Table 2\n"
        "         (reported set only; strategy_summary_all_zones.csv is the unfiltered\n"
        "         version, and zone_reporting_exclusions.csv lists what the floor dropped).\n"
        "[CHECK] repair_rate.csv / repair_rate_by_constraint.csv - report per-strategy\n"
        "         constraint violation rates (R2 App A.2); CP-pool should be ~0%.\n"
        "[CHECK] pmse.csv - report the normalised pMSE ratio (R1 p15); ≈1.0 means the\n"
        "         synthetic records are indistinguishable from census microdata.\n"
        "[CHECK] strategy_utility_comparison.png - repair rate and pMSE side by side."
    )


if __name__ == "__main__":
    main()

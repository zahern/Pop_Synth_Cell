#!/usr/bin/env python
"""
unified_comparison.py
=====================
Unified comparison of ALL population-synthesis methods on the two axes the
paper needs:

  1. Heterogeneity / cloning  -- how much each method duplicates individuals.
       * within-zone duplication  : per zone, share of records that are exact
                                    duplicates of another record IN THE SAME ZONE.
       * cross-zone cloning       : share of distinct attribute combinations
                                    that appear in MORE THAN ONE zone.
       * novelty                 : share of attribute combinations never seen
                                    in the census microdata.
     Every method is scored on the SAME resolved attribute set (the
     intersection of ATTR_COLS_CANDIDATE with the columns present in every
     method), so the numbers are apples-to-apples.

  2. Joint variables resolved -- how faithfully each method reproduces the
     JOINT (multi-way) structure, not just the marginals.  For every designed
     joint pair (the 2-way control joints from build_joint_controls.py) the
     synthetic joint distribution is compared to the census microdata:
       * joint TVD           : total variation distance of the 2-way joint
                                distribution (synthetic vs microdata).
       * |delta Cramer's V|  : |Cramer's V(synth) - Cramer's V(micro)|.
       * joint_resolved_frac : fraction of pairs with joint TVD <= 0.10.

Outputs (outputs/benchmarking/):
  unified_method_comparison.csv -- one row per method (headline table)
  unified_joint_pairs.csv       -- per method x joint-pair detail
  unified_clone_summary.txt      -- human-readable summary
  unified_summary.png           -- 3-panel summary figure

This script is designed to run on a COMPUTE node via
pbs/stage_unified_comparison.pbs.  It is never run on the login/head node.
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

OUT_DIR = os.path.join(config.OUTPUTS_DIR, "benchmarking")
os.makedirs(OUT_DIR, exist_ok=True)

# Attribute superset for the clone / heterogeneity comparison (identical to
# uniqueness_analysis.py so the new numbers stay comparable with the existing
# per-method figures).
ATTR_COLS_CANDIDATE = [
    "agep", "mstp", "sexp", "heap", "hrsp", "incp", "mtwp",
    "mrerd", "nprd", "occp", "qalfp", "qallp", "rlhp", "hind",
    "rntrd", "strd", "tenlld", "vehrd", "empp", "gngp",
]

# Designed 2-way joint control variables (from build_joint_controls.py).
JOINT_PAIRS = [
    ("sexp", "mstp"), ("sexp", "occp"), ("heap", "occp"), ("mstp", "occp"),
    ("hind", "vehrd"), ("hind", "mrerd"), ("hind", "rntrd"), ("nprd", "vehrd"),
]
JOINT_TVD_THRESHOLD = 0.10

METHOD_DIRS = {
    "sa_gan":   config.SA_GAN_RESULTS_DIR,
    "sa_only":  config.SA_ONLY_RESULTS_DIR,
    "ipu_gan":  config.IPU_GAN_RESULTS_DIR,
    "ipu_only": config.IPU_ONLY_RESULTS_DIR,
    "cp_mip":   config.CP_MIP_RESULTS_DIR,
    "cp_sa":    config.CP_POOL_RESULTS_DIR,
    "cp_ipu":   config.CP_IPU_RESULTS_DIR,
}
METHOD_LABELS = {
    "sa_gan": "SA-GAN", "sa_only": "SA-Only",
    "ipu_gan": "IPU-GAN", "ipu_only": "IPU-Only",
    "cp_mip": "CP-pool + MIP", "cp_sa": "CP-pool + SA",
    "cp_ipu": "CP-pool + IPU",
}
METHOD_COLORS = {
    "sa_gan": "#4C72B0", "sa_only": "#DD8452",
    "ipu_gan": "#55A868", "ipu_only": "#C44E52",
    "cp_mip": "#937860", "cp_sa": "#8172B2", "cp_ipu": "#DA7C30",
}

_POOL_FILE = config.HH_MATCH_OUTPUT  # cleaned_dataset_hhid.csv


# ---------------------------------------------------------------------------
# Method file discovery
# ---------------------------------------------------------------------------
def _zone_id(fname):
    m = re.match(r"^(?:zone_)?(\d{9})(?:\.0)?(?:_|\.csv)", os.path.basename(fname))
    return m.group(1) if m else None


def method_files(method):
    d = METHOD_DIRS.get(method)
    if not d or not os.path.isdir(d):
        return []
    patterns = {
        "cp_sa":    [os.path.join(d, "zone_*.csv")],
        "cp_ipu":   [os.path.join(d, "zone_*.csv")],
        "cp_mip":   [os.path.join(d, "zone_*_synthetic_population.csv")],
    }.get(method, [os.path.join(d, "zone_*_synthetic_population.csv"),
                   os.path.join(d, "zone_*.csv")])
    found = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            base = os.path.basename(f).lower()
            # Never treat a per-control-table audit or a diagnostics CSV as a fitted
            # population: they share the zone_<id> prefix and would be scored as
            # synthetic persons.
            if any(s in base for s in ("_control_balance", "_rssz", "_validation",
                                       "_diagnostics", "_new_population")):
                continue
            z = _zone_id(f)
            if z and z not in found:
                found[z] = f
    return found


def resolve_common_attrs(methods):
    resolved = None
    for m in methods:
        cols = None
        for f in list(method_files(m).values())[:3]:
            try:
                cols = set(c.lower() for c in pd.read_csv(f, nrows=0).columns)
                break
            except Exception:
                continue
        if cols is None:
            continue
        present = set(c for c in ATTR_COLS_CANDIDATE if c in cols)
        resolved = present if resolved is None else (resolved & present)
    return sorted(resolved) if resolved else []


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _cramers_v(x, y):
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    denom = n * (min(ct.shape) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


def _tvd(counts_a, counts_b):
    keys = set(counts_a) | set(counts_b)
    ta = max(sum(counts_a.values()), 1)
    tb = max(sum(counts_b.values()), 1)
    return 0.5 * sum(abs(counts_a.get(k, 0) / ta - counts_b.get(k, 0) / tb) for k in keys)


def _joint_counts(df, a, b):
    grp = df.groupby([a, b]).size()
    return {k: int(v) for k, v in grp.items()}


# ---------------------------------------------------------------------------
# Microdata reference
# ---------------------------------------------------------------------------
def load_micro_reference(common_attrs, want_joint=True):
    """Return (novelty_ref_tuples, joint_ref_counts, micro_full_df).

    novelty_ref_tuples : set of attribute tuples over the MICRODATA-sourced rows
                         (matches uniqueness_analysis.py).
    joint_ref_counts   : {(a,b): count_dict} over ALL rows of the matched pool,
                         used as the microdata joint reference.
    micro_full_df      : the matched pool (upper-cased), for Cramer's V.
    """
    if not os.path.exists(_POOL_FILE):
        print(f"  [WARN] microdata reference not found: {_POOL_FILE}")
        return set(), {}, None
    usecols = lambda c: c.lower() in set(common_attrs) | {a for pr in JOINT_PAIRS for a in pr} | {"source"}
    pool = pd.read_csv(_POOL_FILE, usecols=usecols, low_memory=False)
    pool.columns = [c.lower() for c in pool.columns]

    ref_tuples = set()
    if "source" in pool.columns:
        mask = pool["source"].astype(str).str.lower().eq("microdata")
        ref_tuples = set(map(tuple, pool.loc[mask, common_attrs].astype(str).values))
    else:
        ref_tuples = set(map(tuple, pool[common_attrs].astype(str).values))

    joint_ref = {}
    if want_joint:
        # The joint reference must be REAL records only. Using the whole pool (which
        # is 53.8% GAN-generated) makes the target distribution partly synthetic, so
        # the augmented arms are scored against a reference they helped create while
        # the census-seeded arms are penalised for it.
        ref_rows = pool.loc[mask] if "source" in pool.columns else pool
        for a, b in JOINT_PAIRS:
            if a in pool.columns and b in pool.columns:
                joint_ref[(a, b)] = _joint_counts(ref_rows, a, b)

    micro_full = pool.copy()
    micro_full.columns = [c.upper() for c in micro_full.columns]
    return ref_tuples, joint_ref, micro_full


# ---------------------------------------------------------------------------
# Main per-method analysis
# ---------------------------------------------------------------------------
def analyse_method(method, common_attrs, n_zones, rows_per, seed,
                   novelty_ref, joint_ref, micro_full, zone_files=None):
    zones = zone_files if zone_files is not None else method_files(method)
    if not zones:
        print(f"  {method}: no zone files -- skipped")
        return None
    zids = sorted(zones)
    if len(zids) > n_zones:
        zids = list(np.random.RandomState(seed).choice(zids, n_zones, replace=False))
    zids = sorted(zids)

    common_lower = set(common_attrs)
    within_dup_fracs = []
    tuple_zones = {}
    seen_tuples = set()
    novel_rows = 0
    n_novelty_rows = 0
    n_rows_total = 0

    for zid in zids:
        f = zones[zid]
        try:
            df = pd.read_csv(f, usecols=lambda c: c.lower() in common_lower)
        except Exception as exc:
            print(f"  [warn] {method} zone {zid}: {exc}")
            continue
        df.columns = [c.lower() for c in df.columns]
        if not set(common_attrs) <= set(df.columns) or len(df) < 20:
            continue
        if len(df) > rows_per:
            df = df.sample(n=rows_per, random_state=seed)
        n = len(df)
        if n == 0:
            continue
        n_rows_total += n

        # within-zone duplication
        nuniq = df[common_attrs].drop_duplicates().shape[0]
        within_dup_fracs.append(1.0 - nuniq / n)

        # cross-zone tuple multiplicity
        rows_tuples = list(map(tuple, df[common_attrs].astype(str).values))
        for t in set(rows_tuples):
            tuple_zones.setdefault(t, set()).add(zid)
        seen_tuples.update(rows_tuples)
        # Novelty is counted PER PERSON. Weighting by distinct tuple instead (the
        # previous behaviour) up-weights rare combinations and made the arms
        # incomparable with the person-weighted novelty of Table~9: IPU-GAN scored
        # 87.7% here against 49.9% on the full cohort, because its frequently
        # reused microdata rows are collapsed by the dedupe.
        if novelty_ref:
            novel_rows += sum(1 for t in rows_tuples if t not in novelty_ref)
            n_novelty_rows += len(rows_tuples)

    within_dup = float(np.mean(within_dup_fracs) * 100) if within_dup_fracs else float("nan")
    within_unique = 100.0 - within_dup

    mult = np.array([len(s) for s in tuple_zones.values()])
    cross_clone = float((mult > 1).mean() * 100) if mult.size else float("nan")

    novelty = float("nan")
    if novelty_ref and n_novelty_rows:
        novelty = 100.0 * novel_rows / n_novelty_rows
    if method in ("sa_only", "ipu_only"):
        # Both arms resample the census microdata seed verbatim; scored against the
        # pool's microdata rows they would read ~73% "novel" purely because the pool
        # holds a different subset of the census file (only 50.7% of pool microdata
        # tuples match the seed, as qalfp is recoded). They are 0 by construction.
        novelty = 0.0

    # --- joint resolution (pooled across sampled zones) ---
    # Re-read the sampled zones (pooled) for the joint distributions.
    joint_rows = []
    pooled = []
    for zid in zids:
        try:
            df = pd.read_csv(zones[zid], usecols=lambda c: c.lower() in common_lower)
            df.columns = [c.lower() for c in df.columns]
            if set(common_attrs) <= set(df.columns):
                if len(df) > rows_per:
                    df = df.sample(n=rows_per, random_state=seed)
                pooled.append(df[common_attrs])
        except Exception:
            continue
    if pooled:
        synth = pd.concat(pooled, ignore_index=True)
        for a, b in JOINT_PAIRS:
            if not (a in synth.columns and b in synth.columns):
                continue
            sc = _joint_counts(synth, a, b)
            rc = joint_ref.get((a, b))
            jtvd = _tvd(sc, rc) if rc else float("nan")
            cv_s = _cramers_v(synth[a], synth[b])
            cv_m = float("nan")
            if micro_full is not None and a.upper() in micro_full.columns and b.upper() in micro_full.columns:
                cv_m = _cramers_v(micro_full[a.upper()], micro_full[b.upper()])
            joint_rows.append({
                "method": method, "attr_a": a, "attr_b": b,
                "joint_tvd": jtvd,
                "cramer_synth": cv_s, "cramer_micro": cv_m,
                "abs_cramer_diff": (abs(cv_s - cv_m)
                                    if not (np.isnan(cv_s) or np.isnan(cv_m)) else float("nan")),
            })

    mean_jtvd = float(np.nanmean([r["joint_tvd"] for r in joint_rows])) if joint_rows else float("nan")
    mean_cd = float(np.nanmean([r["abs_cramer_diff"] for r in joint_rows])) if joint_rows else float("nan")
    resolved = [r["joint_tvd"] for r in joint_rows if not np.isnan(r["joint_tvd"])]
    joint_resolved_frac = (float(np.mean([t <= JOINT_TVD_THRESHOLD for t in resolved]) * 100)
                           if resolved else float("nan"))

    return {
        "method": method,
        "n_zones": len(zids),
        "n_rows": n_rows_total,
        "attr_set_size": len(common_attrs),
        "within_zone_dup_pct": within_dup,
        "within_zone_unique_pct": within_unique,
        "cross_zone_clone_pct": cross_clone,
        "novelty_pct": novelty,
        "mean_joint_tvd": mean_jtvd,
        "mean_abs_cramer_diff": mean_cd,
        "n_joint_pairs": len(joint_rows),
        "joint_resolved_frac": joint_resolved_frac,
    }, joint_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(METHOD_DIRS.keys()),
                    choices=list(METHOD_DIRS.keys()))
    ap.add_argument("--zones", type=int, default=80, help="max zones sampled per method")
    ap.add_argument("--rows-per", type=int, default=8000, help="max rows sampled per zone")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("Unified method comparison (clone / heterogeneity + joint resolution)")
    print("=" * 70)

    common_attrs = resolve_common_attrs(args.methods)
    print(f"  Common attribute set ({len(common_attrs)} cols): {common_attrs}")
    if len(common_attrs) < 5:
        print("[ERROR] too few common attributes to compare methods.")
        sys.exit(1)

    novelty_ref, joint_ref, micro_full = load_micro_reference(common_attrs)
    print(f"  microdata reference: {len(novelty_ref):,} microdata tuples; "
          f"{len(joint_ref)} joint pairs available")

    # Cohort matching: every arm is scored on the zones it shares with all the
    # others. Without this, arms covering 454 zones and 18 zones are compared as if
    # coverage were a methodological difference, and the 18-zone CP-pool arm drags
    # its own statistics while the others average over a far larger, easier set.
    files_by_method = {m: method_files(m) for m in args.methods}
    common_ids = None
    for m, fs in files_by_method.items():
        ids = set(fs.keys()) if fs else set()
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = common_ids or set()
    print(f"  Common zone cohort: {len(common_ids)} zones shared by all "
          f"{len(args.methods)} methods")
    for m, fs in files_by_method.items():
        print(f"    {METHOD_LABELS[m]:14s} {len(fs):4d} zones available")

    rows, joint_rows_all = [], []
    for m in args.methods:
        cohort = {z: f for z, f in files_by_method.get(m, {}).items() if z in common_ids}
        res, jr = analyse_method(m, common_attrs, args.zones, args.rows_per, args.seed,
                                 novelty_ref, joint_ref, micro_full, zone_files=cohort)
        if res:
            rows.append(res)
            joint_rows_all.extend(jr)
            print(f"  {METHOD_LABELS[m]:14s} zones={res['n_zones']:3d} rows={res['n_rows']:8,}  "
                  f"within_dup={res['within_zone_dup_pct']:5.1f}%  "
                  f"cross_clone={res['cross_zone_clone_pct']:5.1f}%  "
                  f"novelty={res['novelty_pct']:5.1f}%  "
                  f"jointTVD={res['mean_joint_tvd']:.3f}  "
                  f"joint_resolved={res['joint_resolved_frac']:5.1f}%")

    if not rows:
        print("[ERROR] no methods produced results.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("method")
    out_csv = os.path.join(OUT_DIR, "unified_method_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    if joint_rows_all:
        jdf = pd.DataFrame(joint_rows_all)
        jpath = os.path.join(OUT_DIR, "unified_joint_pairs.csv")
        jdf.to_csv(jpath, index=False)
        print(f"Saved: {jpath}")

    # ---- summary text ----
    lines = ["Unified method comparison", "=" * 70, ""]
    lines.append(f"Common attribute set ({len(common_attrs)} cols): {', '.join(common_attrs)}")
    lines.append(f"Zones/method: {args.zones}   Rows/zone cap: {args.rows_per}")
    lines.append("")
    for _, r in df.iterrows():
        lines.append(
            f"{METHOD_LABELS[r['method']]:14s} "
            f"within_dup={r['within_zone_dup_pct']:5.1f}%  "
            f"cross_clone={r['cross_zone_clone_pct']:5.1f}%  "
            f"novelty={r['novelty_pct']:5.1f}%  "
            f"jointTVD={r['mean_joint_tvd']:.3f}  "
            f"resolved={r['joint_resolved_frac']:5.1f}%")
    lines.append("")
    lines.append("Clone interpretation:")
    lines.append("  within_dup     : % of records that are exact duplicates inside their own zone")
    lines.append("  cross_clone    : % of distinct attribute combos appearing in >1 zone")
    lines.append("  novelty        : % of attribute combos absent from census microdata")
    lines.append(f"  joint_resolved : % of designed joint pairs with joint TVD <= {JOINT_TVD_THRESHOLD}")
    stxt = os.path.join(OUT_DIR, "unified_clone_summary.txt")
    with open(stxt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved: {stxt}")

    # ---- summary figure ----
    order = [m for m in df["method"]]
    labels = [METHOD_LABELS[m] for m in order]
    colors = [METHOD_COLORS[m] for m in order]
    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(order) * 2.2), 4.5))

    def panel(ax, col, title, ylab):
        vals = df.set_index("method").loc[order, col].values
        bars = ax.bar(range(len(order)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=9)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center",
                        va="bottom", fontsize=7)

    panel(axes[0], "within_zone_dup_pct", "(a) Within-zone duplication (%)", "%")
    panel(axes[1], "cross_zone_clone_pct", "(b) Cross-zone cloning (%)", "%")
    panel(axes[2], "mean_joint_tvd", "(c) Mean joint TVD (lower=better)", "TVD")
    fig.suptitle("Heterogeneity (clones) and joint resolution across methods", fontsize=11)
    fig.tight_layout()
    fpath = os.path.join(OUT_DIR, "unified_summary.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")
    print("\nDone.")


if __name__ == "__main__":
    main()

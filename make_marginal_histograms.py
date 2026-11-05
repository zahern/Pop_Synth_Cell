"""make_marginal_histograms.py - synthesised vs true census marginals, per method.

Produces the histogram-comparison figures requested for the paper: for every
synthesis arm, each control-table marginal is plotted against the true census
marginal for the same zone cohort, in the style of a simple overlaid histogram
with a difference statistic in the title (synthpop ages_smcabc example).

What is compared
----------------
For one cohort of SA2 zones (the zones every requested arm completed), and for
each single-attribute control table k:

    truth_k(c)  = sum over zones of the census count for category c
    synth_k(c)  = sum over zones of the matching category count in the arm's
                  synthetic population

Both are normalised to proportions, so the figure shows distributional shape
rather than population size.  TVD = 0.5 * sum_c |p_synth(c) - p_truth(c)|.

Outputs (outputs/paper_figures/)
--------------------------------
  marginal_hist_<arm>.png     4x5 panel grid, one panel per table:
                              grey = census truth, colour = arm
  marginal_hist_overview.png  headline tables, all arms overlaid
  marginal_tvd_by_table.csv   TVD per arm per table, plus the cohort size

Usage
-----
  python make_marginal_histograms.py                 # all arms with data
  python make_marginal_histograms.py --arms cp_sa cp_mip
  python make_marginal_histograms.py --cohort 213    # cap to N shared zones
"""

import argparse
import glob
import os
import re
import sys
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config
from benchmark_margins import load_all_controls, _LABEL_TO_CODE

FIG_DIR = os.path.join(config.OUTPUTS_DIR, "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Arm registry: label -> list of directories to search (recursively)
# ---------------------------------------------------------------------------

def _dir(name):
    return os.path.join(config.OUTPUTS_DIR, name)


ARMS = {
    "sa_only":  [config.SA_ONLY_RESULTS_DIR],
    "ipu_only": [getattr(config, "IPU_ONLY_RESULTS_DIR", _dir("ipu_only_results"))],
    "sa_gan":   [getattr(config, "SA_GAN_RESULTS_DIR", _dir("sa_gan_results"))],
    "ipu_gan":  [getattr(config, "IPU_GAN_RESULTS_DIR", _dir("ipu_gan_results"))],
    # CP-SA writes one sub-directory per chunk, so search recursively.
    "cp_sa":    [getattr(config, "CP_POOL_RESULTS_DIR", _dir("cp_pool_results"))],
    "cp_ipu":   [getattr(config, "CP_IPU_RESULTS_DIR", _dir("cp_ipu_results")),
                 os.path.join(config.OUTPUTS_DIR, "cp_ipu_results_v6")],
    "cp_mip":   [getattr(config, "CP_MIP_RESULTS_DIR", _dir("cp_mip_results")),
                 os.path.join(config.OUTPUTS_DIR, "cp_mip_results_v6")],
}

ARM_COLOURS = {
    "sa_only":  "#1b6ca8",
    "ipu_only": "#c0392b",
    "sa_gan":   "#2e8b57",
    "ipu_gan":  "#8e44ad",
    "cp_sa":    "#d68910",
    "cp_ipu":   "#0e7c7b",
    "cp_mip":   "#34495e",
}

ARM_LABELS = {
    "sa_only":  "SA-Only",
    "ipu_only": "IPU-Only",
    "sa_gan":   "SA-GAN",
    "ipu_gan":  "IPU-GAN",
    "cp_sa":    "CP-pool + SA",
    "cp_ipu":   "CP-pool + IPU",
    "cp_mip":   "CP-MIP",
}

_ZONE_RE = re.compile(r"zone_(\d{9})(?:\.0)?(?:_(?:synthetic|new)_population)?\.csv$")

# Household-level control tables: the census counts dwellings, so these must be
# counted once per household.  Counting them per person row weights every
# household by its size, which biases income/vehicle/dwelling distributions
# toward whatever small households look like.
HH_ATTRS = {"hind", "mrerd", "nprd", "rntrd", "vehrd", "bedrd", "hhcd",
            "tend", "tenlld", "hcfmf"}

# What each control table is, so a panel says what the fit is optimising for
# rather than showing a bare variable name.
ATTR_LABELS = {
    "heap":  "Highest educational attainment",
    "hind":  "Household income (weekly)",
    "hrsp":  "Hours worked",
    "incp":  "Personal income (weekly)",
    "indp":  "Industry of employment",
    "mrerd": "Mortgage repayment (monthly)",
    "mstp":  "Marital status",
    "mtwp":  "Method of travel to work",
    "nprd":  "Dwellings by persons resident",
    "occp":  "Occupation",
    "qalfp": "Field of study",
    "qallp": "Non-school qualification",
    "rlhp":  "Relationship in household",
    "rntrd": "Rent (weekly)",
    "sexp":  "Sex",
    "strd":  "Dwelling structure",
    "stup":  "Educational institution attending",
    "typp":  "Non-private dwelling type",
    "vehrd": "Vehicles per dwelling",
}


def find_zone_files(roots):
    """zone_id -> best population file, searching roots (recursively) in order.

    Recognised names (first match per zone wins):
      zone_<9d>_synthetic_population.csv, zone_<9d>_new_population.csv,
      zone_<9d>.csv
    Earlier roots take precedence so a versioned directory can shadow an old one.
    """
    found = {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for fpath in glob.glob(os.path.join(root, "**", "zone_*.csv"), recursive=True):
            m = _ZONE_RE.search(os.path.basename(fpath))
            if not m:
                continue
            zone = m.group(1)
            if zone in found:
                continue
            # Skip the audit/diagnostic side-files that share the prefix.
            base = os.path.basename(fpath)
            if any(tag in base for tag in ("rssz", "audit", "diagnostic")):
                continue
            found[zone] = fpath
    return found


# ---------------------------------------------------------------------------
# Per-zone marginal aggregation
# ---------------------------------------------------------------------------

def aggregate_zone(path, tables):
    """(table_key, category) -> count for one zone population file."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:                     # pragma: no cover
        warnings.warn(f"could not read {path}: {exc}")
        return None
    df.columns = [str(c).lower() for c in df.columns]
    if "abshid" not in df.columns:
        return None
    out = {}
    for key, ct in tables.items():
        attr_cols = ct["attr_cols"]
        if len(attr_cols) != 1:
            continue                             # marginals only
        col = attr_cols[0]
        if col not in df.columns:
            continue
        if key in HH_ATTRS and "abshid" in df.columns:
            # one observation per dwelling, matching the census universe
            s = df.groupby("abshid")[col].first().dropna()
        else:
            s = df[col].dropna()
        if s.empty:
            continue
        for cat, n in s.value_counts().items():
            out[(key, _norm_cat(cat))] = int(n)
    return out


def _norm_cat(v):
    """Canonical category key: int when numerically integral, else stripped str."""
    if isinstance(v, str):
        v = v.strip()
        try:
            f = float(v)
            return int(f) if f.is_integer() else v
        except ValueError:
            return v
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v


def census_zone_counts(table, zone_id):
    """(table_key, category) -> census count for one zone, decoded to codes."""
    key = table.get("key")
    data = table["data"]
    attr_cols = table["attr_cols"]
    if len(attr_cols) != 1:
        return {}
    zdf = data[data["sa2"].astype(str) == str(zone_id)]
    if zdf.empty:
        return {}
    attr = str(table.get("attr", "")).lower()
    decode = _LABEL_TO_CODE.get(attr)
    out = {}
    for _, row in zdf.iterrows():
        cat = row[attr_cols[0]]
        if decode is not None:
            cat = decode.get(str(cat).strip(), cat)
        k = (key, _norm_cat(cat))
        out[k] = out.get(k, 0) + int(row["count"])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=None,
                    help=f"subset of {sorted(ARMS)}")
    ap.add_argument("--cohort", type=int, default=None,
                    help="cap the shared cohort to the first N zones")
    ap.add_argument("--control-dir", default=None,
                    help="control table dir (default: config.SA_CONTROL_DIR)")
    ap.add_argument("--min-zones", type=int, default=10,
                    help="skip an arm that covers fewer zones than this")
    a = ap.parse_args()

    control_dir = a.control_dir or config.SA_CONTROL_DIR
    print(f"control basis: {control_dir}")
    tables = load_all_controls(control_dir)
    for k, ct in tables.items():
        # Normalise sa2heap -> heap so table ids match the population columns
        # and the headline list below.
        ct["key"] = re.sub(r"^sa[12]", "", str(k).lower())
    single = {ct["key"]: ct for ct in tables.values()
              if len(ct["attr_cols"]) == 1}
    print(f"  {len(single)} single-attribute control tables")

    arms = a.arms or list(ARMS)
    files = {}
    for arm in arms:
        f = find_zone_files(ARMS[arm])
        if len(f) < a.min_zones:
            print(f"  [skip] {arm}: {len(f)} zone files (< {a.min_zones})")
            continue
        files[arm] = f
        print(f"  {arm}: {len(f)} zone files")
    if not files:
        raise SystemExit("no arm has enough zone files")

    # Shared cohort: zones every included arm completed.
    shared = set.intersection(*[set(f) for f in files.values()])
    # Restrict to the paper's 367-zone fitted cohort, so the marginal figures
    # and the tables are computed on the same zone set.
    cohort_path = os.path.join(HERE, "pbs", "zones_371.txt")
    if os.path.exists(cohort_path):
        scope = {ln.strip() for ln in open(cohort_path) if ln.strip()}
        excluded = {"301031014", "304041102", "310021279", "311031315"}
        shared = shared & (scope - excluded)
    shared = sorted(shared)
    if a.cohort:
        shared = shared[: a.cohort]
    print(f"shared cohort: {len(shared)} zones")
    if len(shared) < a.min_zones:
        raise SystemExit("shared cohort too small")

    # --- census truth over the cohort ------------------------------------
    truth = defaultdict(float)
    for zone in shared:
        for key, ct in single.items():
            for k, v in census_zone_counts(ct, zone).items():
                truth[k] += v
    if not truth:
        raise SystemExit("no census truth accumulated (zone ids not in tables?)")

    # --- each arm ---------------------------------------------------------
    rows = []
    profiles = {}
    for arm in files:
        agg = defaultdict(float)
        for i, zone in enumerate(shared, 1):
            got = aggregate_zone(files[arm][zone], single)
            if not got:
                continue
            for k, v in got.items():
                agg[k] += v
            if i % 50 == 0:
                print(f"    {arm}: {i}/{len(shared)} zones", flush=True)
        if not agg:
            print(f"  [skip] {arm}: aggregated nothing")
            continue
        prof, tvds = {}, {}
        for key in single:
            t = {c: v for (k, c), v in truth.items() if k == key}
            s = {c: v for (k, c), v in agg.items() if k == key}
            if not t:
                continue
            tot_t = sum(t.values()) or 1.0
            tot_s = sum(s.values()) or 1.0
            cats = sorted(set(t) | set(s), key=_sort_key)
            pt = {c: t.get(c, 0) / tot_t for c in cats}
            ps = {c: s.get(c, 0) / tot_s for c in cats}
            prof[key] = (cats, pt, ps)
            tvds[key] = 0.5 * sum(abs(ps[c] - pt[c]) for c in cats)
            rows.append({"arm": arm, "arm_label": ARM_LABELS.get(arm, arm),
                         "table": key, "tvd": tvds[key],
                         "n_census": int(tot_t), "n_synth": int(tot_s),
                         "n_zones": len(shared)})
        profiles[arm] = (prof, tvds)
        mean_tvd = sum(tvds.values()) / max(len(tvds), 1)
        print(f"  {arm}: mean TVD over {len(tvds)} tables = {mean_tvd:.4f}")

    if not profiles:
        raise SystemExit("no arm produced a profile")

    # --- per-category shares (for a category-level results table) ----------
    cat_rows = []
    for arm, (prof, tvds) in profiles.items():
        for key, (cats, pt, ps) in prof.items():
            for c in cats:
                cat_rows.append({"arm": arm, "table": key, "category": c,
                                 "census_share": pt.get(c, 0.0),
                                 "synth_share": ps.get(c, 0.0),
                                 "n_zones": len(shared)})
    if cat_rows:
        cat_path = os.path.join(FIG_DIR, "marginal_category_shares.csv")
        pd.DataFrame(cat_rows).to_csv(cat_path, index=False)
        print(f"  wrote {cat_path}")

    # --- per-arm figure ---------------------------------------------------
    for arm, (prof, tvds) in profiles.items():
        _plot_arm(arm, prof, tvds)

    # --- overview figure --------------------------------------------------
    _plot_overview(profiles)

    # --- table ------------------------------------------------------------
    out = pd.DataFrame(rows)
    csv_path = os.path.join(FIG_DIR, "marginal_tvd_by_table.csv")
    out.to_csv(csv_path, index=False)
    summary = (out.groupby(["arm", "arm_label"])["tvd"]
                  .agg(["mean", "median", "max"]).reset_index()
                  .sort_values("mean"))
    sum_path = os.path.join(FIG_DIR, "marginal_tvd_summary.csv")
    summary.to_csv(sum_path, index=False)
    print("\n=== mean TVD over shared tables (lower is better) ===")
    print(summary.to_string(index=False))
    print(f"\nwrote {csv_path}\nwrote {sum_path}")


def _sort_key(c):
    return (0, c, "") if isinstance(c, (int, float)) else (1, 0, str(c))


def _plot_arm(arm, prof, tvds):
    keys = [k for k in prof if not k.startswith("sa2")]
    n = len(keys)
    if n == 0:
        return
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow))
    axes = axes.ravel()
    colour = ARM_COLOURS.get(arm, "#1b6ca8")
    for ax, key in zip(axes, keys):
        cats, pt, ps = prof[key]
        x = range(len(cats))
        ax.bar(x, [pt[c] for c in cats], width=0.9, color="#9e9e9e",
               alpha=0.65, label="census")
        ax.bar(x, [ps[c] for c in cats], width=0.55, color=colour,
               alpha=0.9, label=ARM_LABELS.get(arm, arm))
        ax.set_ylabel(ATTR_LABELS.get(key, key), fontsize=6)
        if len(cats) <= 20:
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(c) for c in cats], fontsize=6, rotation=90)
        else:
            ax.set_xticks([])
        ax.set_yticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=9)
    mean = sum(tvds.values()) / len(tvds)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"marginal_hist_{arm}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


HEADLINE = ["agep", "sexp", "occp", "mstp", "heap", "nprd"]


def _plot_overview(profiles):
    """All arms overlaid on the headline tables, census as grey bars."""
    arms = list(profiles)
    # only tables every arm has
    keys = [k for k in HEADLINE
            if all(k in profiles[a][0] for a in arms)]
    if not keys:
        return
    ncol = 3
    nrow = (len(keys) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.7 * nrow))
    axes = axes.ravel()
    for ax, key in zip(axes, keys):
        cats, pt, _ = profiles[arms[0]][0][key]
        x = list(range(len(cats)))
        ax.bar(x, [pt[c] for c in cats], width=0.95, color="#9e9e9e",
               alpha=0.6, label="census")
        for arm in arms:
            _, _, ps = profiles[arm][0][key]
            ax.step(x, [ps.get(c, 0.0) for c in cats], where="mid",
                    linewidth=1.3, color=ARM_COLOURS.get(arm, "k"),
                    label=ARM_LABELS.get(arm, arm))
        tv = ", ".join(f"{ARM_LABELS.get(a, a)} {profiles[a][1][key]:.2f}"
                       for a in arms)
        ax.set_ylabel(ATTR_LABELS.get(key, key), fontsize=6)
        if len(cats) <= 20:
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(c) for c in cats], fontsize=6, rotation=90)
        else:
            ax.set_xticks([])
        ax.set_yticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[len(keys):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "marginal_hist_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()

"""
paper_figures.py
================
Generates all publication-quality figures for the Population Synthesiser paper
and copies them (plus existing outputs) into the paper LaTeX folder.

Produces:
  1. convergence_all_zones.png   -- SA-Only convergence when iteration logs exist
  2. validation_controls.png     -- Multi-panel predicted-vs-actual for 6
                                    controlled attributes (West End zone)
  3. rssz_by_method.png          -- legacy comparison retained for compatibility;
                                    not used by the constraint-first manuscript
  4. (copies all existing PNG outputs to paper folder)

Usage:
    python paper_figures.py [--paper-dir PATH] [--zone ZONE_CODE]
"""

import os
import re
import sys
import glob
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE       = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR  = os.path.normpath(os.path.join(HERE, "..", "Population_Synthesiser___GAN_SA"))
FIG_OUTDIR = os.path.join(config.OUTPUTS_DIR, "paper_figures")

# Human-readable names for ABS controlled variables
ATTR_LABELS = {
    # Individual controls
    "incp":            "Personal Income (Weekly, INCP)",
    "occp":            "Occupation (OCCP)",
    "heap":            "Highest Educational Attainment (HEAP)",
    "mtwp":            "Method of Travel to Work (MTWP)",
    "sexp":            "Sex (SEXP)",
    "mstp":            "Marital Status (MSTP)",
    "marital_status":  "Marital Status (MSTP)",
    "rlhp":            "Relationship in Household (RLHP)",
    "relationship":    "Relationship in Household (RLHP)",
    "empp":            "Employment (No. of Employees, EMPP)",
    "hrsp":            "Hours Worked per Week (HRSP)",
    "gngp":            "Government / Private Sector (GNGP)",
    "qallp":           "Qualification Level (QALLP)",
    "qalfp":           "Qualification Field (QALFP)",
    "indp":            "Industry of Employment (INDP)",
    "typp":            "Type of Dwelling (TYPP)",
    "stup":            "Student Status (STUP)",
    "siemp":           "Status in Employment (SIEMP)",
    "spip":            "Spouse / Partner Indicator (SPIP)",
    # Household controls
    "nprd":            "Persons per Dwelling (NPRD)",
    "hind":            "Household Income Decile (HIND)",
    "vehrd":           "Motor Vehicles in Household (VEHRD)",
    "mrerd":           "Mortgage Repayments Weekly (MRERD)",
    "rntrd":           "Rent Payments Weekly (RNTRD)",
    "tenlld":          "Tenure and Landlord Type (TENLLD)",
    "strd":            "Dwelling Structure (STRD)",
}

# Short x-axis tick labels for each control attribute
# (categories are ABS numeric codes; these map code->label for key attributes)
CATEGORY_LABELS = {
    "sexp":   {1: "Male", 2: "Female"},
    "nprd":   {1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6+"},
    "hind":   {1:"D1\n(Lowest)", 2:"D2", 3:"D3", 4:"D4", 5:"D5",
               6:"D6", 7:"D7", 8:"D8", 9:"D9", 10:"D10\n(Highest)"},
    "vehrd":  {0:"0 vehicles", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6+"},
    "heap":   {1:"Postgrad", 2:"Grad Dip", 3:"Bachelor", 4:"Adv Dip",
               5:"Cert III/IV", 6:"Cert I/II", 7:"No Qual", 8:"N/A"},
    "incp":   {1:"Nil", 2:"$1–149", 3:"$150–299", 4:"$300–399", 5:"$400–499",
               6:"$500–649", 7:"$650–799", 8:"$800–999", 9:"$1k–1.25k",
               10:"$1.25–1.5k", 11:"$1.5–1.75k", 12:"$1.75–2k",
               13:"$2–3k", 14:"$3k+", 15:"N/S", 16:"N/A"},
    "mtwp":   {1:"Train", 2:"Bus", 3:"Ferry", 4:"Tram/light rail",
               5:"Other pub.", 6:"Taxi", 7:"Car–drv", 8:"Car–pass",
               9:"Truck", 10:"Motorbike", 11:"Bicycle", 12:"Walk",
               13:"Other", 14:"WFH", 15:"N/A"},
    "mrerd":  {1:"$1–149", 2:"$150–299", 3:"$300–449", 4:"$450–599",
               5:"$600–799", 6:"$800–999", 7:"$1k–1.24k", 8:"$1.25k+",
               9:"N/A (renting)", 23:"N/A (other)"},
    "rntrd":  {1:"$1–74", 2:"$75–149", 3:"$150–224", 4:"$225–299",
               5:"$300–374", 6:"$375–449", 7:"$450–549", 8:"$550+",
               9:"N/A (owner)", 27:"N/A (other)"},
    "tenlld": {1:"Own outright", 2:"Mortgage", 3:"Renting–RE", 4:"Renting–Govt",
               5:"Renting–person", 6:"Renting–other", 7:"Other tenure",
               8:"Not stated"},
    "strd":   {1:"Separate house", 2:"Semi-detached", 3:"Flat/apt (3+ st.)",
               4:"Flat/apt (<3 st.)", 5:"Other dwelling", 6:"Not stated"},
}


# ---------------------------------------------------------------------------
# 1. Convergence plot — all SA2 zones
# ---------------------------------------------------------------------------
def plot_convergence(results_dir: str, out_path: str, n_zones: int = 5,
                     strategy_name: str = "SA"):
    """Parse SA logs across all zones and create a two-panel convergence figure.

    Left panel: faint traces for n_zones representative zones (one per RSSZ
    quintile) plus a thick mean trace with ±1 SD shaded band.
    Right panel: histogram of final RSSZ values.
    """
    log_files = sorted(glob.glob(os.path.join(results_dir, "zone_*_log.txt")))
    if not log_files:
        log_files = sorted(glob.glob(os.path.join(results_dir, "*_log.txt")))
    if not log_files:
        log_files = sorted(glob.glob(os.path.join(results_dir, "..", "*.log")))

    patterns = [
        re.compile(r"\[iter\s+(\d+)/\d+\].*?best=\s*([\d.]+)"),
        re.compile(r"Iteration:\s*(\d+),\s*Current Cost:\s*([\d.]+),\s*"
                    r"Best Cost:\s*([\d.]+),\s*Temperature:\s*([\d.]+)"),
        re.compile(r"Iteration\s+(\d+).*?RSSZ:\s*([\d.]+)"),
        re.compile(r"Zone\s+(\d+):.*?RSSZ=([\d.]+).*?best=([\d.]+)"),
    ]

    all_traces = {}
    final_costs = {}

    for lf in log_files:
        zone_id = os.path.basename(lf).replace("_log.txt", "").replace("_log", "").replace("zone_", "")
        its, bc = [], []
        with open(lf, "r") as fh:
            for line in fh:
                for pat in patterns:
                    m = pat.search(line)
                    if m:
                        its.append(int(m.group(1)))
                        cost = float(m.group(2) if len(m.groups()) >= 3 else m.group(2))
                        if len(m.groups()) >= 3:
                            cost = float(m.group(3))
                        bc.append(cost)
                        break
        if its and bc:
            idx = np.argsort(its)
            its = np.array(its)[idx]
            bc = np.array(bc)[idx]
            all_traces[zone_id] = (its, bc)
            final_costs[zone_id] = bc[-1]

    if not all_traces:
        print("  [WARN] No SA log files found in", results_dir)
        return

    final_vals = np.array(list(final_costs.values()))
    n_total = len(all_traces)

    # Pick n_zones representative zones — one per quintile of final RSSZ
    sorted_zones = sorted(all_traces.keys(), key=lambda z: final_costs[z])
    n_show = min(n_zones, n_total)
    if n_total <= n_show:
        sample_zones = sorted_zones
    else:
        indices = [int(round(i * (n_total - 1) / (n_show - 1))) for i in range(n_show)]
        sample_zones = [sorted_zones[i] for i in indices]

    # Build common x-axis from the zone with the most iterations
    ref_zone = max(all_traces, key=lambda z: len(all_traces[z][0]))
    common_x = all_traces[ref_zone][0]

    # Interpolate all zones onto common_x to compute mean ± SD band
    all_interp = []
    for zone_id, (its, bc) in all_traces.items():
        interp = np.interp(common_x, its, bc)
        all_interp.append(interp)
    all_interp = np.array(all_interp)
    mean_trace = all_interp.mean(axis=0)
    std_trace  = all_interp.std(axis=0)

    # Smooth traces
    def _smooth(arr, n=30):
        win = max(1, len(arr) // n)
        return pd.Series(arr).rolling(win, min_periods=1, center=True).mean().values

    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.5], wspace=0.3)
    ax_left = fig.add_subplot(gs[0, 0])

    # Faint representative zone traces (no individual labels)
    rep_color = "#7bafd4"
    _first_rep = True
    for zone_id in sample_zones:
        its, bc = all_traces[zone_id]
        bc_s = _smooth(np.interp(common_x, its, bc))
        lbl = f"Representative zones (n={n_show})" if _first_rep else "_nolegend_"
        ax_left.plot(common_x, bc_s, linewidth=0.9, alpha=0.35, color=rep_color,
                     label=lbl, zorder=3)
        _first_rep = False

    # Median trace + IQR band -- robust to the heavy right tail of per-zone cost,
    # so the central convergence path is not swamped by a few high-cost zones
    # (the previous mean +/- 1 SD band spanned 0-110 and read as pure noise).
    p25 = _smooth(np.percentile(all_interp, 25, axis=0))
    p50 = _smooth(np.percentile(all_interp, 50, axis=0))
    p75 = _smooth(np.percentile(all_interp, 75, axis=0))
    ax_left.fill_between(common_x, p25, p75, color="#2166ac", alpha=0.12, zorder=4,
                         label=f"IQR band ({n_total} zones)")
    ax_left.plot(common_x, p50, linewidth=2.8, color="#2166ac",
                 label=f"Median (final cost {p50[-1]:.1f})", zorder=10)

    ax_left.set_xlabel("SA iteration", fontsize=11)
    # NOTE: the plotted quantity is the raw SA objective (sum of squared Z-scores),
    # NOT the per-table RSSZ (~0.2-2). Labelling it "RSSZ" was wrong.
    ax_left.set_ylabel("SA cost (sum of squared Z-scores)", fontsize=11)
    ax_left.set_title(f"SA convergence ({n_total} zones)",
                      fontsize=11, fontweight="bold")
    ax_left.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax_left.grid(axis="y", linestyle=":", alpha=0.3)
    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    ax_right = fig.add_subplot(gs[0, 1])
    bins = int(max(10, min(40, 2 * len(final_vals) ** (1 / 3))))
    ax_right.hist(final_vals, bins=bins, density=True, alpha=0.6, color="#4C72B0",
                  edgecolor="white", linewidth=0.5)
    if len(final_vals) > 1 and np.std(final_vals) > 1e-15:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(final_vals)
            xk = np.linspace(final_vals.min(), final_vals.max(), 200)
            ax_right.plot(xk, kde(xk), color="#C44E52", linewidth=2)
        except np.linalg.LinAlgError:
            pass

    ax_right.axvline(np.median(final_vals), color="#8C564B", linestyle="-.", linewidth=1.5,
                     label=f"Median = {np.median(final_vals):.3f}")
    ax_right.axvline(np.mean(final_vals), color="#55A868", linestyle="--", linewidth=1.5,
                     label=f"Mean = {np.mean(final_vals):.3f}")
    ax_right.set_xlabel("Final SA cost", fontsize=11)
    ax_right.set_ylabel("Density", fontsize=11)
    ax_right.set_title(f"Final cost ({n_total} zones)", fontsize=11, fontweight="bold")
    ax_right.legend(fontsize=8, framealpha=0.9)
    ax_right.grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle(f"{strategy_name} simulated-annealing convergence",
                 fontsize=12, fontweight="bold")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved convergence plot ({n_total} zones) -> {out_path}")


# ---------------------------------------------------------------------------
# 2. Validation plots — predicted vs actual for controlled attributes
# ---------------------------------------------------------------------------
def _load_sa2_control(attr: str, zone: str):
    """
    Load census control counts for one attribute + SA2 zone.
    Returns a dict {category_code: count} or None.
    """
    path = os.path.join(config.SA_CONTROL_DIR, f"sa2{attr}.csv")
    if not os.path.exists(path):
        return None
    try:
        from CensusRecordImporter import CensusControlTable
        ct = CensusControlTable(path, zone_level="sa2")
        # Rename generic digit column to attr name
        for generic in ("1-digit", "2-digit"):
            if generic in ct.control_table.columns:
                ct.control_table = ct.control_table.rename(columns={generic: attr})
        ct.validate_and_clean()
        df = ct.control_table.copy()
        for generic in ("1-digit", "2-digit"):
            if generic in df.columns:
                df = df.rename(columns={generic: attr})
        # Filter to zone
        zone_col = [c for c in df.columns if c in ("sa2", "sa_num")][0] if any(
            c in df.columns for c in ("sa2", "sa_num")
        ) else df.columns[0]
        df[zone_col] = df[zone_col].astype(str)
        z = df[df[zone_col] == str(zone)]
        if z.empty:
            return None
        attr_col = attr if attr in z.columns else [c for c in z.columns if c not in (zone_col, "count")][0]
        counts = z.groupby(attr_col)["count"].sum()
        return counts.to_dict()
    except Exception as exc:
        warnings.warn(f"  [WARN] Could not load control for {attr} zone {zone}: {exc}")
        return None


def _category_to_code(value, attr):
    """Coerce a control/synthetic-population category value to its numeric
    ABS code, robust to mixed encodings across zone-output files.

    Most values are already numeric codes, but some zone-output files (or
    residual raw census rows) instead carry the original ABS text label
    (e.g. "Overseas visitor") because CensusRecordImporter.DataMappings only
    maps labels it has an explicit entry for and otherwise leaves the raw
    string in place. Cross-reference that same canonical <attr>_map so the
    label is translated to the correct code instead of crashing int().
    Returns None (with a printed WARNING) if no numeric value or canonical
    mapping can be found, so the caller can skip that single category.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        from CensusRecordImporter import DataMappings
        mapping = getattr(DataMappings(), f"{attr}_map", {})
        code = mapping.get(str(value).strip())
        if code is not None:
            return int(code)
    except Exception:
        pass
    print(f"  [WARN] Could not map category value {value!r} for attr '{attr}' "
          "to a numeric code — skipping this category.")
    return None


def plot_validation_controls(zone: str, pop_path: str, out_path: str,
                              attrs_individual=None, attrs_household=None):
    """
    Multi-panel bar chart comparing synthetic population attribute counts
    against census macro-level controls, with human-readable labels.
    """
    if attrs_individual is None:
        attrs_individual = ["incp", "heap", "mtwp", "occp"]
    if attrs_household is None:
        attrs_household = ["nprd", "vehrd"]

    all_attrs = attrs_individual + attrs_household
    pop = pd.read_csv(pop_path)
    pop.columns = [c.lower() for c in pop.columns]

    # Build panels
    panels = []
    for attr in all_attrs:
        census_counts = _load_sa2_control(attr, zone)
        if census_counts is None:
            # fall back to rssz_values_cat.csv expected vs observed if available
            continue
        if attr not in pop.columns:
            continue

        # Household-level control tables (nprd, vehrd, ...) use the ABS
        # "Dwelling Records" counting basis: each household counts ONCE.
        # Every person row carries its household's value, so counting person
        # rows inflates each category by its own household size (NPRD=4 shows
        # ~4x the census count). Dedupe to one row per household before
        # counting; person-level attributes keep the per-row count.
        if attr in attrs_household:
            hh_col = next((c for c in ("abshid", "household_id", "hhid")
                           if c in pop.columns), None)
            if hh_col is not None:
                attr_series = pop.drop_duplicates(subset=hh_col)[attr]
            else:
                print(f"  [WARN] No household-ID column found for household attr "
                      f"'{attr}' zone {zone} — falling back to person-row counts "
                      "(counts will be inflated by household size).")
                attr_series = pop[attr]
        else:
            attr_series = pop[attr]

        # Coerce both sides to numeric category codes via _category_to_code().
        # Zone-output files are not consistently encoded: some (older exports)
        # store the raw ABS text label (e.g. "Overseas visitor") for a category
        # instead of its numeric code, which used to crash a bare int() cast
        # here. Unmappable values are dropped (with a WARNING) rather than
        # aborting the whole figure.
        raw_synth_counts = attr_series.value_counts()
        synth_counts = {}
        for raw_cat, cnt in raw_synth_counts.items():
            code = _category_to_code(raw_cat, attr)
            if code is not None:
                synth_counts[code] = synth_counts.get(code, 0) + cnt

        raw_census_counts = census_counts
        census_counts = {}
        for raw_cat, cnt in raw_census_counts.items():
            code = _category_to_code(raw_cat, attr)
            if code is not None:
                census_counts[code] = census_counts.get(code, 0) + cnt

        if not census_counts and not synth_counts:
            print(f"  [WARN] No numeric categories left for attr '{attr}' zone {zone} — skipping panel")
            continue

        all_cats = sorted(set(census_counts.keys()) | set(synth_counts.keys()))
        census_vals = np.array([census_counts.get(c, 0) for c in all_cats], dtype=float)
        synth_vals  = np.array([synth_counts.get(c, 0) for c in all_cats], dtype=float)

        # X tick labels
        cat_lbl_map = CATEGORY_LABELS.get(attr, {})
        tick_labels = [cat_lbl_map.get(c, str(c)) for c in all_cats]

        panels.append({
            "attr": attr,
            "title": ATTR_LABELS.get(attr, attr.upper()),
            "all_cats": all_cats,
            "census_vals": census_vals,
            "synth_vals": synth_vals,
            "tick_labels": tick_labels,
            "is_household": attr in attrs_household,
        })

    if not panels:
        print(f"  [WARN] No valid panels for zone {zone}")
        return

    ncols = 2
    nrows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.8))
    axes = axes.flatten() if nrows > 1 else np.array(axes).flatten()

    colors = {"census": "#2166ac", "synth": "#d73027"}

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        x = np.arange(len(panel["all_cats"]))
        w = 0.38

        bar_c = ax.bar(x - w / 2, panel["census_vals"], width=w,
                       color=colors["census"], alpha=0.85, label="Census (actual)")
        bar_s = ax.bar(x + w / 2, panel["synth_vals"], width=w,
                       color=colors["synth"],  alpha=0.85, label="Synthetic (SA fit)")

        ax.set_xticks(x)
        ax.set_xticklabels(panel["tick_labels"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Count", fontsize=9)

        group = "Household" if panel["is_household"] else "Individual"
        ax.set_title(f"{group} Control: {panel['title']}", fontsize=9.5, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(axis="y", linestyle=":", alpha=0.35)

        # Annotate RSSZ for this panel
        cat_file = os.path.join(
            config.SA_GAN_RESULTS_DIR, f"{zone}_rssz_values_cat.csv"
        )
        if os.path.exists(cat_file):
            cat_df = pd.read_csv(cat_file)
            matched = cat_df[cat_df["Table"].str.lower().str.contains(panel["attr"])]
            if not matched.empty:
                rssz_val = matched["RSSZ"].iloc[0]
                ax.text(0.98, 0.96, f"RSSZ = {rssz_val:.4f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8.5, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))

        if idx == 0:
            ax.legend(fontsize=8.5, loc="upper left")

    # Hide unused axes
    for idx in range(len(panels), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Controlled Attribute Validation — SA2 Zone {zone} (West End, Brisbane)\n"
        "Comparing SA-fitted synthetic population against ABS 2021 Census macro-level statistics",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved validation plot -> {out_path}")


# ---------------------------------------------------------------------------
# 3. RSSZ comparison with richer metrics
# ---------------------------------------------------------------------------
def _collect_rssz_all_zones():
    """Load per-zone, per-table RSSZ for SA-GAN from outputs/sa_gan_results/."""
    cat_files = glob.glob(os.path.join(config.SA_GAN_RESULTS_DIR, "*_rssz_values_cat.csv"))
    dfs = []
    for f in cat_files:
        d = pd.read_csv(f)
        if not d.empty and "RSSZ" in d.columns:
            d["zone"] = os.path.basename(f).replace("_rssz_values_cat.csv", "")
            dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["RSSZ"] = pd.to_numeric(df["RSSZ"], errors="coerce")
    return df


def _rssz_zone_summary(df):
    """Per-zone mean RSSZ summary."""
    return df.groupby("zone")["RSSZ"].mean()


def plot_rssz_metrics(out_path: str, ipu_gan_data=None, ipu_only_data=None):
    """
    Multi-panel figure:
    Left:  RSSZ distribution (histogram + boxplot) for SA-GAN
    Right: Per-table mean RSSZ bar chart (SA-GAN) with 90th-pct error
    Bottom: Metric summary table (Mean, Median, P90, Max, %<0.15)
    """
    sa_df = _collect_rssz_all_zones()
    if sa_df.empty:
        print("  [WARN] No SA-GAN RSSZ data found.")
        return

    zone_means = _rssz_zone_summary(sa_df)

    # clip for display (outliers above 2.0 are rare blown-up zones)
    clip_hi = 2.0
    zm_clipped = zone_means.clip(upper=clip_hi)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    ax_hist   = fig.add_subplot(gs[0, 0])
    ax_box    = fig.add_subplot(gs[0, 1])
    ax_table  = fig.add_subplot(gs[1, 0])
    ax_perf   = fig.add_subplot(gs[1, 1])

    # -- Panel 1: Histogram of per-zone mean RSSZ (SA-GAN) ----------------
    bins = np.linspace(0, clip_hi, 35)
    n_over = (zone_means > clip_hi).sum()
    ax_hist.hist(zm_clipped, bins=bins, color="#d73027", edgecolor="white",
                 linewidth=0.5, alpha=0.85)
    if n_over > 0:
        ax_hist.text(0.97, 0.93, f"+ {n_over} zones\n> {clip_hi:.0f}",
                     transform=ax_hist.transAxes, ha="right", va="top",
                     fontsize=8.5, color="#333")
    median_val = float(zone_means.median())
    ax_hist.axvline(median_val, color="#2166ac", linewidth=1.8, linestyle="--",
                    label=f"Median = {median_val:.3f}")
    mean_val = float(zone_means.mean())
    ax_hist.axvline(mean_val, color="#4dac26", linewidth=1.8, linestyle="-.",
                    label=f"Mean = {mean_val:.3f}")
    ax_hist.set_xlabel("Mean Zone RSSZ", fontsize=10)
    ax_hist.set_ylabel("Number of Zones", fontsize=10)
    ax_hist.set_title("Distribution of Zone-level Mean RSSZ\n(SA-GAN, all zones)", fontsize=10, fontweight="bold")
    ax_hist.legend(fontsize=8.5)
    ax_hist.grid(axis="y", linestyle=":", alpha=0.4)

    # -- Panel 2: Box plots by method (SA-GAN; IPU-GAN/IPU-Only if provided) --
    datasets = [("SA-GAN", zone_means.values)]
    colors_box = ["#d73027"]
    if ipu_gan_data is not None:
        datasets.append(("IPU-GAN", np.array(ipu_gan_data)))
        colors_box.append("#f4a582")
    if ipu_only_data is not None:
        datasets.append(("IPU-Only\n(Census)", np.array(ipu_only_data)))
        colors_box.append("#92c5de")

    bp = ax_box.boxplot(
        [d[1] for d in datasets],
        labels=[d[0] for d in datasets],
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
        medianprops=dict(color="black", linewidth=2),
        widths=0.5,
    )
    for patch, col in zip(bp["boxes"], colors_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax_box.set_ylabel("Mean Zone RSSZ", fontsize=10)
    ax_box.set_title("Method Comparison\n(Mean Zone RSSZ per Method)", fontsize=10, fontweight="bold")
    ax_box.set_ylim(0, min(clip_hi * 1.2, ax_box.get_ylim()[1]))
    ax_box.grid(axis="y", linestyle=":", alpha=0.4)

    # -- Panel 3: Per-table mean ± P90 bar chart ---------------------------
    tbl_stats = (
        sa_df.groupby("Table")["RSSZ"]
        .agg(mean="mean", p90=lambda x: x.quantile(0.90))
        .sort_values("mean")
    )
    # Clean table names for readability
    tbl_labels = []
    for t in tbl_stats.index:
        short = t.replace("Individual - ", "Ind: ").replace("Household - ", "HH: ")
        # Map to human-readable attr
        for key, label in ATTR_LABELS.items():
            attr_short = label.split("(")[0].strip()
            if key in short.lower():
                short = short.split(":")[0] + ": " + attr_short
                break
        tbl_labels.append(short)

    y_pos = np.arange(len(tbl_stats))
    ax_table.barh(y_pos, tbl_stats["mean"], color="#d73027", alpha=0.8,
                  label="Mean RSSZ")
    ax_table.barh(y_pos, tbl_stats["p90"] - tbl_stats["mean"],
                  left=tbl_stats["mean"], color="#f4a582", alpha=0.6,
                  label="Mean -> 90th pctile")
    ax_table.set_yticks(y_pos)
    ax_table.set_yticklabels(tbl_labels, fontsize=7.5)
    ax_table.set_xlabel("RSSZ", fontsize=9)
    ax_table.set_title("Per-Control Table RSSZ (SA-GAN)\nBar = Mean, Extension = 90th Percentile",
                        fontsize=9.5, fontweight="bold")
    ax_table.legend(fontsize=8, loc="lower right")
    ax_table.grid(axis="x", linestyle=":", alpha=0.35)

    # -- Panel 4: Summary metric table ------------------------------------
    ax_perf.axis("off")
    methods = [("SA-GAN", zone_means.values)]
    if ipu_gan_data is not None:
        methods.append(("IPU-GAN", np.array(ipu_gan_data)))
    if ipu_only_data is not None:
        methods.append(("IPU-Only (Census)", np.array(ipu_only_data)))

    rows = []
    for m_name, m_vals in methods:
        m_vals = m_vals[np.isfinite(m_vals)]
        rows.append([
            m_name,
            f"{np.mean(m_vals):.4f}",
            f"{np.median(m_vals):.4f}",
            f"{np.percentile(m_vals, 90):.4f}",
            f"{np.max(m_vals):.4f}",
            f"{(m_vals < 0.15).mean()*100:.1f}%",
        ])

    col_headers = ["Method", "Mean\nRSSZ", "Median\nRSSZ", "P90\nRSSZ",
                   "Max\nRSSZ", "% Zones\n< 0.15"]
    tbl = ax_perf.table(
        cellText=rows, colLabels=col_headers,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.0)
    # Header row styling
    for j in range(len(col_headers)):
        tbl[(0, j)].set_facecolor("#2166ac")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # SA-GAN row
    for j in range(len(col_headers)):
        tbl[(1, j)].set_facecolor("#fde0d0")

    ax_perf.set_title("Summary Metrics — Control Table Fitness\n"
                      "(lower RSSZ = better match to census margins)",
                      fontsize=9.5, fontweight="bold", pad=12)

    fig.suptitle(
        "SA Synthetic Population: RSSZ Fitness Metrics across All Brisbane Zones",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved RSSZ metrics plot -> {out_path}")

# ---------------------------------------------------------------------------
# 3b. 4-strategy comparison figure (generated after IPU comparison PBS job)
# ---------------------------------------------------------------------------

_STRAT_COLORS = {
    "sa_gan":   "#4C72B0",   # blue
    "sa_only":  "#DD8452",   # orange
    "ipu_gan":  "#55A868",   # green
    "ipu_only": "#C44E52",   # red
}
_STRAT_LABELS = {
    "sa_gan":   "SA-GAN",
    "sa_only":  "SA-Only",
    "ipu_gan":  "IPU-GAN",
    "ipu_only": "IPU-Only",
}


def plot_strategy_comparison(benchmarking_dir, out_path):
    """
    Publication-quality 4-strategy comparison figure.
    Reads:
      {benchmarking_dir}/metrics_sa2.csv         -- per-zone x per-table metrics
      {benchmarking_dir}/strategy_summary.csv    -- per-strategy aggregate
    Panels:
      (A) Grouped boxplot: TVD distributions per strategy
      (B) Grouped boxplot: SRMSE distributions per strategy
      (C) Per-table mean TVD bar chart (top 15 tables), grouped by strategy
      (D) Summary metric table
    Generated after running:
      benchmark_margins.py --strategies sa_gan sa_only ipu_gan ipu_only
    """
    metrics_csv = os.path.join(benchmarking_dir, "metrics_sa2.csv")
    summary_csv = os.path.join(benchmarking_dir, "strategy_summary.csv")

    if not os.path.exists(metrics_csv):
        print(f"  [SKIP] strategy_comparison: {metrics_csv} not found "
              f"(run benchmark_margins.py --strategies sa_gan sa_only ipu_gan ipu_only first)")
        return

    metrics_df = pd.read_csv(metrics_csv)
    strategies = [s for s in ["sa_gan", "sa_only", "ipu_gan", "ipu_only"]
                  if s in metrics_df["strategy"].unique()]
    if not strategies:
        print("  [SKIP] strategy_comparison: no known strategies in metrics CSV")
        return

    colors = [_STRAT_COLORS.get(s, "#888") for s in strategies]
    labels = [_STRAT_LABELS.get(s, s) for s in strategies]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(hspace=0.45, wspace=0.38)
    ax_tvd, ax_srmse, ax_table_bars, ax_summary = axes.flat

    # --- Panel A: TVD boxplot ---
    tvd_data = [metrics_df.loc[metrics_df["strategy"] == s, "tvd"].dropna().values
                for s in strategies]
    bp = ax_tvd.boxplot(tvd_data, labels=labels, patch_artist=True, notch=False,
                        flierprops=dict(marker=".", markersize=2.5, alpha=0.35),
                        medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax_tvd.set_ylabel("TVD (Total Variation Distance)", fontsize=10)
    ax_tvd.set_title("(A) TVD Distribution by Strategy", fontsize=11, fontweight="bold")
    ax_tvd.grid(axis="y", linestyle=":", alpha=0.4)
    ax_tvd.set_ylim(bottom=0)

    # --- Panel B: SRMSE boxplot ---
    srmse_data = [metrics_df.loc[metrics_df["strategy"] == s, "srmse"].dropna().values
                  for s in strategies]
    bp2 = ax_srmse.boxplot(srmse_data, labels=labels, patch_artist=True, notch=False,
                           flierprops=dict(marker=".", markersize=2.5, alpha=0.35),
                           medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax_srmse.set_ylabel("SRMSE (Standardised RMSE)", fontsize=10)
    ax_srmse.set_title("(B) SRMSE Distribution by Strategy", fontsize=11, fontweight="bold")
    ax_srmse.grid(axis="y", linestyle=":", alpha=0.4)
    ax_srmse.set_ylim(bottom=0)

    # --- Panel C: per-table mean TVD grouped bar chart ---
    table_tvd = (
        metrics_df.groupby(["table", "strategy"])["tvd"]
        .mean()
        .unstack("strategy")
        .reindex(columns=strategies)
    )
    table_tvd["_max"] = table_tvd.max(axis=1)
    table_tvd = table_tvd.sort_values("_max", ascending=False).head(15).drop(columns="_max")
    tbl_short = [t.replace("sa2", "").replace("_", " ").upper()[:12] for t in table_tvd.index]
    n_t = len(table_tvd)
    n_s = len(strategies)
    width = 0.8 / n_s
    x = np.arange(n_t)
    for i, strat in enumerate(strategies):
        vals = table_tvd[strat].values if strat in table_tvd.columns else np.full(n_t, np.nan)
        ax_table_bars.bar(
            x + (i - n_s / 2 + 0.5) * width, vals,
            width=width * 0.92,
            color=_STRAT_COLORS.get(strat, "#888"),
            alpha=0.8,
            label=_STRAT_LABELS.get(strat, strat),
        )
    ax_table_bars.set_xticks(x)
    ax_table_bars.set_xticklabels(tbl_short, rotation=50, ha="right", fontsize=8)
    ax_table_bars.set_ylabel("Mean TVD", fontsize=10)
    ax_table_bars.set_title("(C) Per-attribute Mean TVD (Top 15 tables)",
                            fontsize=11, fontweight="bold")
    ax_table_bars.legend(fontsize=8)
    ax_table_bars.axhline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_table_bars.set_ylim(bottom=0)
    ax_table_bars.grid(axis="y", linestyle=":", alpha=0.3)

    # --- Panel D: summary table ---
    ax_summary.axis("off")
    if os.path.exists(summary_csv):
        sum_df = pd.read_csv(summary_csv)
    else:
        sum_df = (
            metrics_df.groupby("strategy", as_index=False)
            .agg(tvd_mean=("tvd", "mean"), tvd_median=("tvd", "median"),
                 tvd_max=("tvd", "max"), srmse_mean=("srmse", "mean"),
                 srmse_median=("srmse", "median"), mae_pct_mean=("mae_pct", "mean"))
        )
    ordered_strats = [s for s in ["sa_gan", "sa_only", "ipu_gan", "ipu_only"]
                      if s in sum_df["strategy"].values]
    sum_df = sum_df.set_index("strategy").reindex(ordered_strats).reset_index()
    sum_df["strategy"] = sum_df["strategy"].map(_STRAT_LABELS).fillna(sum_df["strategy"])
    num_cols = [c for c in ["tvd_mean", "tvd_median", "srmse_mean", "srmse_median", "mae_pct_mean"]
                if c in sum_df.columns]
    col_hdrs = ["Method"] + [c.replace("_", "\n") for c in num_cols]
    rows = []
    for _, row in sum_df.iterrows():
        r = [row["strategy"]]
        for c in num_cols:
            v = row.get(c, float("nan"))
            r.append(f"{v:.4f}" if (isinstance(v, float) and not np.isnan(v)) else "--")
        rows.append(r)
    tbl = ax_summary.table(cellText=rows, colLabels=col_hdrs, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.1)
    for j in range(len(col_hdrs)):
        tbl[(0, j)].set_facecolor("#2c4a8c")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    row_colors = ["#dce6f1", "#f7f7f7", "#d5e8d4", "#ffe6cc"]
    for i in range(1, len(rows) + 1):
        for j in range(len(col_hdrs)):
            tbl[(i, j)].set_facecolor(row_colors[(i - 1) % len(row_colors)])
    ax_summary.set_title("(D) Summary: All-Zone, All-Table Metric Means",
                         fontsize=11, fontweight="bold", pad=10)

    fig.suptitle(
        "Synthetic Population Accuracy: SA-GAN vs SA-Only vs IPU-GAN vs IPU-Only\n"
        "(lower = better fit to census marginals)",
        fontsize=12, fontweight="bold",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved strategy comparison -> {out_path}")




# ---------------------------------------------------------------------------
# 4. Copy all relevant PNGs to paper folder
# ---------------------------------------------------------------------------
PAPER_COPY_MAP = [
    # (source_path, paper_dest_name, description)
    # GAN validation
    (os.path.join(config.GAN_VALIDATION_DIR, "training_loss_detail.png"),
     "gan_training_loss.png",
     "GAN training loss (generator + discriminator over epochs)"),
    (os.path.join(config.GAN_VALIDATION_DIR, "training_loss_overview.png"),
     "gan_training_loss_overview.png",
     "GAN training loss overview (compact version)"),
     (os.path.join(config.GAN_VALIDATION_DIR, "tvd_per_column.png"),
      "tvd_per_column.png",
      "Total Variation Distance per synthetic variable"),
    (os.path.join(config.GAN_VALIDATION_DIR, "cramers_v_heatmap.png"),
     "gan_cramers_v_heatmap_synth.png",
     "Cramér's V inter-variable association (synthetic)"),
    (os.path.join(config.GAN_VALIDATION_DIR, "cramers_v_original.csv"),
     None, None),   # CSV only, skip
    (os.path.join(config.GAN_VALIDATION_DIR, "cramers_v_diff.png"),
     "gan_cramers_v_diff.png",
     "Cramér's V difference: synthetic vs original"),
    (os.path.join(config.GAN_VALIDATION_DIR, "pmse_propensity_histogram.png"),
     "gan_pmse_histogram.png",
     "pMSE propensity score histogram (GAN fidelity)"),
    # Benchmarking
    (os.path.join(config.OUTPUTS_DIR, "benchmarking", "benchmark_metrics_boxplot.png"),
     "benchmark_tvd_srmse.png",
     "Benchmark TVD/SRMSE/MAE% comparison SA-GAN vs SA-Only"),
    (os.path.join(config.OUTPUTS_DIR, "benchmarking", "benchmark_metrics_boxplot.png"),
     "benchmark_metrics_boxplot.png",
     "Benchmark TVD/SRMSE/MAE% boxplot (paper direct reference)"),
    (os.path.join(config.OUTPUTS_DIR, "benchmarking", "joint_cramer_scatter.png"),
     "joint_cramer_scatter.png",
     "Cramér's V scatter: synthetic vs original joint associations"),
    (os.path.join(config.OUTPUTS_DIR, "benchmarking", "sa4_heatmap_sa_gan_tvd_mean.png"),
     "sa4_heatmap_sa_gan_tvd_mean.png",
     "SA4 geographic heatmap: SA-GAN mean TVD"),
    (os.path.join(config.OUTPUTS_DIR, "benchmarking", "sa4_heatmap_sa_only_tvd_mean.png"),
     "sa4_heatmap_sa_only_tvd_mean.png",
     "SA4 geographic heatmap: SA-Only mean TVD"),
    (os.path.join(config.OUTPUTS_DIR, "benchmarking", "uniqueness_summary.png"),
     "uniqueness_summary.png",
     "Population diversity and uniqueness (3-panel)"),
    (os.path.join(config.OUTPUTS_DIR, "benchmarking", "uniqueness_cross_zone.png"),
     "uniqueness_cross_zone.png",
     "Cross-zone uniqueness: SA-GAN vs SA-Only"),
    # RSSZ distribution
    (os.path.join(config.RSSZ_OUTLIER_DIR, "rssz_distribution.png"),
     "rssz_distribution.png",
     "RSSZ distribution across all zones"),
    # Imbalance (IPU vs SA population/household totals)
    (os.path.join(config.OUTPUTS_DIR, "diff.png"),
     "diff.png",
     "% difference in population/household totals: IPU vs SA"),
    (os.path.join(config.OUTPUTS_DIR, "paper_figures", "diff_latest.png"),
     "diff.png",
     "% difference in population/household totals: IPU vs SA (auto-generated)"),
    # Per-table TVD comparison (from benchmark_margins.py)
    (os.path.join(config.OUTPUTS_DIR, "benchmarking", "per_table_tvd_comparison.png"),
     "per_table_tvd_comparison.png",
     "Per-table mean TVD comparison across strategies"),
    # Composite marginals (Figure 10 - pre-SA)
    (os.path.join(config.OUTPUTS_DIR, "paper_figures", "marginal_composite_pre_sa.png"),
     "marginal_composite_pre_sa.png",
     "Composite marginal distributions before SA (6-panel)"),
    (os.path.join(config.OUTPUTS_DIR, "paper_figures", "marginal_nprd_vehrd_pre_sa.png"),
     "marginal_nprd_vehrd_pre_sa.png",
     "NPRD/VEHRD marginal distributions before SA"),
    # Post-SA NPRD/VEHRD (Figure 11)
    (os.path.join(config.OUTPUTS_DIR, "paper_figures", "postsa_nprd_vehrd_latest.png"),
     "postsa_nprd_vehrd.png",
     "Post-SA NPRD/VEHRD marginal fit (auto-regenerated)"),
]


def copy_existing_pngs(paper_dir: str, fig_dir: str):
    """Copy known PNG outputs to the paper folder."""
    os.makedirs(paper_dir, exist_ok=True)
    copied, missing = 0, 0
    for src, dst_name, desc in PAPER_COPY_MAP:
        if dst_name is None:
            continue
        if not os.path.exists(src):
            print(f"  [MISS] {os.path.basename(src)} ({desc or 'no description'})")
            missing += 1
            continue
        dst = os.path.join(paper_dir, dst_name)
        shutil.copy2(src, dst)
        copied += 1
        print(f"  [COPY] {os.path.basename(src)} -> {dst_name}")

    # Copy all joint distribution PNGs (out-of-sample attribute pairs)
    joint_dir = os.path.join(config.GAN_VALIDATION_DIR, "joint_distributions")
    for src in glob.glob(os.path.join(joint_dir, "joint_*.png")):
        dst = os.path.join(paper_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        copied += 1
        print(f"  [COPY] {os.path.basename(src)} -> paper/")

    # Copy selected marginal distribution PNGs (key variables)
    KEY_MARGINALS = ["INCP", "HEAP", "MTWP", "OCCP", "NPRD", "VEHRD",
                     "MRERD", "RNTRD", "AGEP", "SEXP", "MSTP"]
    marg_dir = os.path.join(config.GAN_VALIDATION_DIR, "marginal_distributions")
    for col in KEY_MARGINALS:
        src = os.path.join(marg_dir, f"{col}.png")
        if os.path.exists(src):
            dst = os.path.join(paper_dir, f"marginal_{col.lower()}.png")
            shutil.copy2(src, dst)
            copied += 1

    # Copy generated paper figures
    for f in glob.glob(os.path.join(fig_dir, "*.png")):
        dst = os.path.join(paper_dir, os.path.basename(f))
        shutil.copy2(f, dst)
        copied += 1
        print(f"  [COPY] {os.path.basename(f)} -> paper/")

    print(f"  Total copied: {copied}, missing: {missing}")


# ---------------------------------------------------------------------------
# 5. IPU results loader (when IPU results are available from HPC)
# ---------------------------------------------------------------------------
def load_ipu_zone_means(ipu_results_dir: str):
    """
    Load per-zone mean RSSZ from IPU results directory.
    Expected file: {zone}_rssz_values_cat.csv in ipu_results_dir.
    Returns np.array of per-zone means, or None if dir not found.
    """
    if not ipu_results_dir or not os.path.isdir(ipu_results_dir):
        return None
    cat_files = glob.glob(os.path.join(ipu_results_dir, "*_rssz_values_cat.csv"))
    if not cat_files:
        return None
    dfs = []
    for f in cat_files:
        d = pd.read_csv(f)
        if not d.empty and "RSSZ" in d.columns:
            d["zone"] = os.path.basename(f).replace("_rssz_values_cat.csv", "")
            dfs.append(d)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df["RSSZ"] = pd.to_numeric(df["RSSZ"], errors="coerce")
    return df.groupby("zone")["RSSZ"].mean().values


def plot_constraint_first_fit_diagnostics(out_path: str):
    """Plot final-fit diagnostics for the current raw and CP-pool SA arms."""
    sa_files = sorted(glob.glob(os.path.join(
        config.SA_ONLY_RESULTS_DIR, "sa_only_rssz_summary_chunk*.csv")))
    cp_files = sorted(glob.glob(os.path.join(
        config.CP_POOL_RESULTS_DIR, "cp_pool_rssz_summary_chunk*.csv")))
    if not sa_files or not cp_files:
        print("  [WARN] Current SA/CP summary files unavailable; skipping fit diagnostics")
        return

    def read_summaries(files):
        frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        frame["zone"] = frame["zone"].astype(str)
        return frame.drop_duplicates("zone")

    sa = read_summaries(sa_files)
    cp = read_summaries(cp_files)
    common = sorted(set(sa["zone"]) & set(cp["zone"]))
    if not common:
        print("  [WARN] No common zones in current SA/CP summaries; skipping fit diagnostics")
        return
    sa = sa[sa["zone"].isin(common)]
    cp = cp[cp["zone"].isin(common)]

    person_cols = [c for c in sa.columns
                   if c.startswith("Individual - ") and c in cp.columns]
    household_cols = [c for c in sa.columns
                      if c.startswith("Household - ") and c in cp.columns]
    shared11_names = {
        "heap", "mstp", "mtwp", "occp", "rlhp", "sexp",
        "hind", "mrerd", "nprd", "rntrd", "vehrd",
    }
    table_cols = []
    invalid_shared = []
    for column in person_cols + household_cols:
        sa_values = pd.to_numeric(sa[column], errors="coerce")
        cp_values = pd.to_numeric(cp[column], errors="coerce")
        sa_nonzero = (sa_values.abs() > 1e-12).mean()
        cp_nonzero = (cp_values.abs() > 1e-12).mean()
        table_name = column.rsplit("sa2", 1)[-1].lower()
        if table_name in shared11_names and (sa_nonzero <= 0.05 or cp_nonzero <= 0.05):
            invalid_shared.append(column)
        if sa_nonzero > 0.05 and cp_nonzero > 0.05:
            table_cols.append(column)
        else:
            print(f"  [WARN] Excluding stale/empty residual column: {column}")
    if invalid_shared:
        print("  [WARN] Refusing cross-arm diagnostic: shared-11 residuals are incomplete")
        return
    person_cols = [c for c in person_cols if c in table_cols]
    household_cols = [c for c in household_cols if c in table_cols]
    if not table_cols:
        print("  [WARN] No control-table residual columns; skipping fit diagnostics")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.65]})
    sa_total = sa[table_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    cp_total = cp[table_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    axes[0].boxplot(
        [sa_total, cp_total],
        labels=["Raw + SA", "CP pool + SA"],
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black",
                   "markeredgecolor": "black", "markersize": 4},
        medianprops={"color": "black", "linewidth": 1.2},
    )
    axes[0].set_ylabel("Validated residual sum")
    axes[0].set_title(f"Matched final-fit residuals\n({len(common)} common zones)")
    axes[0].grid(axis="y", alpha=0.25)

    labels = [c.split(" - ", 1)[1].replace("sa2", "").upper()
              for c in table_cols]
    sa_means = [pd.to_numeric(sa[c], errors="coerce").mean() for c in table_cols]
    cp_means = [pd.to_numeric(cp[c], errors="coerce").mean() for c in table_cols]
    x = np.arange(len(labels))
    width = 0.38
    axes[1].bar(x - width / 2, sa_means, width, label="Raw + SA",
                color="#4C78A8")
    axes[1].bar(x + width / 2, cp_means, width, label="CP pool + SA",
                color="#F58518")
    axes[1].set_xticks(x, labels, rotation=55, ha="right")
    axes[1].set_ylabel("Mean final table residual")
    axes[1].set_title("Person and household controls")
    axes[1].axvline(len(person_cols) - 0.5, color="0.5",
                    linewidth=0.8, linestyle="--")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Constraint-first SA fit diagnostics", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01,
             "Final residual summaries; iteration histories were not retained for these runs.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _method_population_path(method_dir: str, zone: str):
    candidates = [
        os.path.join(method_dir, f"zone_{zone}.csv"),
        os.path.join(method_dir, f"zone_{zone}_synthetic_population.csv"),
        os.path.join(method_dir, f"{zone}_synthetic_population.csv"),
    ]
    return next((path for path in candidates if os.path.exists(path)), None)


def plot_joint_pair_lines(joint_csv: str, out_path: str):
    """Plot joint TVD and Cramer's-V error for the active non-GAN arms."""
    if not os.path.exists(joint_csv):
        print(f"  [WARN] Joint-pair results not found: {joint_csv}")
        return
    data = pd.read_csv(joint_csv)
    methods = ["sa_only", "ipu_only", "cp_sa", "cp_ipu"]
    labels = {
        "sa_only": "SA-Only", "ipu_only": "IPU-Only",
        "cp_sa": "CP-pool + SA", "cp_ipu": "CP-pool + IPU",
    }
    data = data[data["method"].isin(methods)].copy()
    if data.empty:
        print("  [WARN] No active non-GAN joint results found")
        return
    data["pair"] = data["attr_a"].str.upper() + r"$\times$" + data["attr_b"].str.upper()
    pairs = data["pair"].drop_duplicates().tolist()
    colors = dict(zip(methods, ["#2166ac", "#d95f02", "#1b9e77",
                                "#7570b3"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.4), sharex=True)
    x = np.arange(len(pairs))
    width = 0.15
    for index, method in enumerate(methods):
        subset = data[data["method"] == method].set_index("pair").reindex(pairs)
        if subset["joint_tvd"].notna().any():
            offset = (index - (len(methods) - 1) / 2) * width
            axes[0].bar(x + offset, subset["joint_tvd"], width=width,
                        label=labels[method], color=colors[method])
            axes[1].bar(x + offset, subset["abs_cramer_diff"], width=width,
                        label=labels[method], color=colors[method])
    axes[0].axhline(0.10, color="0.35", linestyle="--", linewidth=1,
                    label="TVD = 0.10 threshold")
    axes[0].set_ylabel("Joint TVD (lower is better)", fontsize=11)
    axes[1].set_ylabel(r"$|\Delta V|$ (lower is better)", fontsize=11)
    for ax, title in zip(axes, ["Joint marginal distance", "Association error"]):
        ax.set_title(title, fontsize=12)
        # Vertical labels: at 45 degrees an 80 pt pair name projects about
        # 57 pt horizontally, against roughly 41 pt of tick spacing in each
        # panel, so adjacent names overlap.  Upright labels are ~11 pt wide
        # and the bottom margin below holds their full length.
        ax.set_xticks(x, pairs, rotation=90, ha="center", fontsize=9.0)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.grid(axis="y", alpha=0.25)
    # A shared key below the panels.  Inside the left axes it covered the
    # dashed 0.10 threshold line (and sat under the panel title).
    handles, labels = axes[0].get_legend_handles_labels()
    # The key sits under the title rather than below the panels: the vertical
    # pair names now occupy the whole bottom margin.
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.947),
               ncol=len(handles), frameon=False, fontsize=9.0,
               columnspacing=1.2, handletextpad=0.5)
    fig.suptitle("Joint-variable diagnostics across active strategies",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.006,
             "213-zone audit cohort; eight designed joint pairs; no GAN strategies.",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.27, 1, 0.865))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_marginal_difference_lines(zone: str, out_path: str):
    """Plot signed category-share differences for a common zone and five arms."""
    method_dirs = {
        "SA-Only": config.SA_ONLY_RESULTS_DIR,
        "IPU-Only": config.IPU_ONLY_RESULTS_DIR,
        "CP-pool + SA": config.CP_POOL_RESULTS_DIR,
        "CP-pool + IPU": config.CP_IPU_RESULTS_DIR,
        "CP-pool + MIP": config.CP_MIP_RESULTS_DIR,
    }
    attrs = [("heap", False), ("occp", False), ("mtwp", False),
             ("qalfp", False), ("rlhp", False), ("hrsp", False),
             ("hind", True), ("nprd", True), ("vehrd", True)]
    populations = {}
    for label, directory in method_dirs.items():
        path = _method_population_path(directory, zone)
        if path:
            populations[label] = pd.read_csv(path)
    if len(populations) < 3:
        print(f"  [WARN] Too few method populations for marginal differences in {zone}")
        return

    panels = []
    for attr, household in attrs:
        target = _load_sa2_control(attr, zone)
        if target is None:
            continue
        target_counts = {}
        for raw, count in target.items():
            code = _category_to_code(raw, attr)
            if code is not None:
                target_counts[code] = target_counts.get(code, 0) + count
        target_total = sum(target_counts.values())
        if not target_total:
            continue
        categories = sorted(target_counts)
        target_share = {cat: target_counts.get(cat, 0) / target_total
                        for cat in categories}
        differences = {}
        for label, population in populations.items():
            frame = population.copy()
            frame.columns = [str(column).lower() for column in frame.columns]
            if attr not in frame.columns:
                continue
            if household:
                hh_col = next((c for c in ("abshid", "household_id", "hhid")
                               if c in frame.columns), None)
                if hh_col:
                    frame = frame.drop_duplicates(hh_col)
            counts = {}
            for raw, count in frame[attr].value_counts().items():
                code = _category_to_code(raw, attr)
                if code is not None:
                    counts[code] = counts.get(code, 0) + count
            total = sum(counts.values())
            if total:
                differences[label] = [
                    100 * (counts.get(cat, 0) / total - target_share.get(cat, 0))
                    for cat in categories
                ]
        if differences:
            panels.append((attr, household, categories, differences))

    if not panels:
        print(f"  [WARN] No marginal panels available for {zone}")
        return
    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.3),
                             squeeze=False)
    colors = ["#2166ac", "#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    for ax, (attr, household, categories, differences) in zip(axes.flat, panels):
        x = np.arange(len(categories))
        for (label, values), color in zip(differences.items(), colors):
            ax.plot(x, values, marker="o", linewidth=1.5, markersize=3,
                    label=label, color=color)
        ax.axhline(0, color="black", linewidth=0.8)
        tick_map = CATEGORY_LABELS.get(attr, {})
        ax.set_xticks(x, [tick_map.get(cat, str(cat)) for cat in categories],
                      rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Synthetic − target (pp)")
        ax.set_title(("Household" if household else "Individual") +
                     f": {ATTR_LABELS.get(attr, attr.upper())}")
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(f"Category-level marginal differences — SA2 {zone}",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01,
             "Percentage-point difference in category shares; zero means exact marginal agreement.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Generate paper figures and copy to paper folder")
    p.add_argument("--paper-dir",  default=PAPER_DIR,
                   help=f"Target paper LaTeX folder (default: {PAPER_DIR})")
    p.add_argument("--fig-dir",    default=FIG_OUTDIR,
                   help=f"Output directory for generated figures (default: {FIG_OUTDIR})")
    p.add_argument("--zone",       default="301031018",
                   help="West End SA2 zone code (default: 301031018)")
    p.add_argument("--zone-name",  default="West End, Brisbane",
                   help="Human-readable zone name for plot titles")
    p.add_argument("--ipu-gan-dir",  default=None,
                   help="Directory with IPU-GAN zone RSSZ results (optional, from HPC)")
    p.add_argument("--ipu-only-dir", default=None,
                   help="Directory with IPU-Only zone RSSZ results (optional, from HPC)")
    p.add_argument("--copy-only",  action="store_true",
                   help="Only copy existing PNGs to paper folder; skip plot generation")
    args = p.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    zone = args.zone
    zone_dir  = os.path.join(config.SA_ONLY_RESULTS_DIR, f"{zone}_plots_predicted_vs_actual")
    log_path  = os.path.join(config.SA_ONLY_RESULTS_DIR, f"{zone}_log.txt")
    pop_path  = os.path.join(config.SA_ONLY_RESULTS_DIR, f"zone_{zone}_synthetic_population.csv")

    # -- 1. Convergence plot (all SA2 zones) --------------------------------
    if not args.copy_only:
        conv_out = os.path.join(args.fig_dir, "convergence_all_zones.png")
        plot_convergence(config.SA_ONLY_RESULTS_DIR, conv_out,
                         strategy_name="SA-Only")
        if os.path.exists(conv_out):
            shutil.copy2(conv_out, os.path.join(
                args.paper_dir, "cost_vs_iteration_small_markers.png"))
            print("  Updated paper/cost_vs_iteration_small_markers.png")

    # -- 2. Validation plot ------------------------------------------------
    if not args.copy_only and os.path.exists(pop_path):
        val_out = os.path.join(args.fig_dir, "validation_controls.png")
        plot_validation_controls(
            zone=zone,
            pop_path=pop_path,
            out_path=val_out,
            attrs_individual=["incp", "heap", "mtwp", "occp"],
            attrs_household=["nprd", "vehrd"],
        )
    else:
        if not os.path.exists(pop_path):
            print(f"  [SKIP] Population file not found: {pop_path}")

    # -- 3. RSSZ metrics plot (with optional IPU data) ---------------------
    if not args.copy_only:
        ipu_gan_means  = load_ipu_zone_means(args.ipu_gan_dir)
        ipu_only_means = load_ipu_zone_means(args.ipu_only_dir)
        rssz_out = os.path.join(args.fig_dir, "rssz_by_method.png")
        plot_rssz_metrics(rssz_out, ipu_gan_means, ipu_only_means)
        diagnostics_out = os.path.join(
            args.fig_dir, "constraint_first_fit_diagnostics.png")
        plot_constraint_first_fit_diagnostics(diagnostics_out)
        plot_joint_pair_lines(
            os.path.join(config.OUTPUTS_DIR, "benchmarking",
                         "unified_joint_pairs.csv"),
            os.path.join(args.fig_dir, "joint_pair_lines.png"),
        )
        plot_marginal_difference_lines(
            args.zone,
            os.path.join(args.fig_dir,
                         f"marginal_difference_lines_{args.zone}.png"),
        )

    # -- 3b. Generate composite marginals & post-SA validation (NEW) -------
    if not args.copy_only:
        print("\n" + "="*70)
        print("  Generating automated paper marginals...")
        print("="*70)
        try:
            import subprocess
            subprocess.run([sys.executable, os.path.join(HERE, "generate_paper_marginals.py")],
                         check=False, capture_output=False)
        except Exception as e:
            print(f"  [WARN] Could not run generate_paper_marginals.py: {e}")

        print("\n" + "="*70)
        print("  Generating IPU vs SA comparison figure...")
        print("="*70)
        try:
            subprocess.run([sys.executable, os.path.join(HERE, "generate_ipu_sa_comparison.py")],
                         check=False, capture_output=False)
        except Exception as e:
            print(f"  [WARN] Could not run generate_ipu_sa_comparison.py: {e}")

    # -- 4. Copy all PNGs to paper folder ----------------------------------
    print("\n--- Copying figures to paper folder ---")
    copy_existing_pngs(args.paper_dir, args.fig_dir)

    print(f"\nDone. Paper figures in {args.paper_dir}")


if __name__ == "__main__":
    main()

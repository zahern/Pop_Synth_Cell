"""
RSSZ Outlier Analysis
======================
Diagnoses zones with high RSSZ and which control tables drive the poor fit.
Reviewer 2 (Table 2): explain why max RSSZ spikes to 7.52 for SA-GAN.

All paths come from config.py.  Run from synth_pop/:
    python rssz_outlier_analysis.py [--results-dir outputs/sa_gan_results] [--threshold 2.0]
"""

import argparse, glob, os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config

# All dependencies are now local copies within synth_pop/
# (SA_2_level.py, CensusRecordImporter.py copied from sa_run_again_n)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SA_2_level import build_control_dicts, calculate_total_rssz_table
from CensusRecordImporter import CensusControlTable

OUTPUT_DIR = config.RSSZ_OUTLIER_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_control_table_safe(path, key):
    try:
        ct = CensusControlTable(path)
        ct.validate_and_clean()
        zone_col = getattr(ct, "sa_num", "sa2")
        df = ct.control_table.copy()
        # Rename generic ABS column names to the actual attribute name
        attr = re.sub(r'^sa[12]', '', key)   # 'sa2heap' → 'heap'
        for generic in ("1-digit", "2-digit"):
            if generic in df.columns:
                df = df.rename(columns={generic: attr})
        keys = [c for c in df.columns if c not in (zone_col, "count")]
        return {"data": df, "keys": keys}
    except Exception as exc:
        print(f"  WARN control table skipped: {os.path.basename(path)} ({exc})")
        return None


def load_control_tables():
    ind, hh = {}, {}
    skipped = []

    # Preferred layout: .../data_sa2_controls/individual and .../household
    if os.path.isdir(config.SA_CONTROL_IND) and os.path.isdir(config.SA_CONTROL_HH):
        for fname in sorted(os.listdir(config.SA_CONTROL_IND)):
            if not fname.endswith(".csv") or " copy" in fname.lower():
                continue
            key = fname.replace(".csv", "").lower()
            path = os.path.join(config.SA_CONTROL_IND, fname)
            ct = _load_control_table_safe(path, key)
            if ct is not None:
                ind[key] = ct
            else:
                skipped.append(path)
        for fname in sorted(os.listdir(config.SA_CONTROL_HH)):
            if not fname.endswith(".csv") or " copy" in fname.lower():
                continue
            key = fname.replace(".csv", "").lower()
            path = os.path.join(config.SA_CONTROL_HH, fname)
            ct = _load_control_table_safe(path, key)
            if ct is not None:
                hh[key] = ct
            else:
                skipped.append(path)

        if not ind and not hh:
            raise RuntimeError("No valid control tables could be loaded from individual/household directories")

        if skipped:
            pd.DataFrame({"skipped_control_table": skipped}).to_csv(
                os.path.join(OUTPUT_DIR, "skipped_control_tables.csv"), index=False
            )
        return ind, hh

    # Fallback layout: flat .../data_sa2_controls/*.csv
    flat_root = os.path.dirname(config.SA_CONTROL_IND)
    if not os.path.isdir(flat_root):
        raise FileNotFoundError(
            f"SA control directory not found. Checked: {config.SA_CONTROL_IND}, {flat_root}"
        )

    household_markers = {
        "bedrd", "hhcd", "hind", "mrerd", "nprd", "rntrd", "strd", "tend", "vehrd"
    }
    seen_keys = set()
    for fname in sorted(os.listdir(flat_root)):
        if not fname.endswith(".csv"):
            continue
        if " copy" in fname.lower() or fname.startswith("~"):
            skipped.append(os.path.join(flat_root, fname))
            continue

        key = fname.replace(".csv", "").lower()
        if key in seen_keys:
            skipped.append(os.path.join(flat_root, fname))
            continue
        seen_keys.add(key)

        path = os.path.join(flat_root, fname)
        ct = _load_control_table_safe(path, key)
        if ct is None:
            skipped.append(path)
            continue

        # Heuristic split for flat layouts: household-like controls to hh, rest to individual.
        target = hh if any(marker in key for marker in household_markers) else ind
        target[key] = ct

    if not ind and not hh:
        raise RuntimeError(f"No valid control tables could be loaded from {flat_root}")

    if skipped:
        pd.DataFrame({"skipped_control_table": skipped}).to_csv(
            os.path.join(OUTPUT_DIR, "skipped_control_tables.csv"), index=False
        )

    return ind, hh


def zone_rssz(fpath, zone_id, ind_tables, hh_tables):
    try:
        df = pd.read_csv(fpath)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"  WARN zone {zone_id}: CSV unreadable ({type(e).__name__}), skipping")
        return None, None
    if df.empty:
        print(f"  WARN zone {zone_id}: CSV has no rows, skipping")
        return None, None
    ind_d = build_control_dicts(ind_tables, zone_id)
    hh_d  = build_control_dicts(hh_tables,  zone_id)
    try:
        total, per_table = calculate_total_rssz_table(df.to_dict("records"), ind_d, hh_d)
        return total, per_table
    except Exception as e:
        print(f"  WARN zone {zone_id}: {e}")
        return None, None


def build_from_precomputed_rssz(threshold):
    """
    Fallback path: use precomputed SA outputs in SA_DIR/rssz_results/sszk_per_zone.csv.
    Expected columns: Category, Zone, SSZk.
    """
    precomputed = os.path.join(config.SA_DIR, "rssz_results", "sszk_per_zone.csv")
    if not os.path.exists(precomputed):
        raise RuntimeError(
            "Control-table recomputation failed and no precomputed RSSZ file was found at "
            f"{precomputed}"
        )

    raw = pd.read_csv(precomputed)
    needed = {"Category", "Zone", "SSZk"}
    if not needed.issubset(raw.columns):
        raise RuntimeError(
            f"Precomputed RSSZ file missing expected columns {needed}. "
            f"Found: {list(raw.columns)}"
        )

    wide = raw.pivot_table(
        index="Zone", columns="Category", values="SSZk", aggfunc="sum", fill_value=0.0
    ).reset_index()
    wide = wide.rename(columns={"Zone": "zone"})
    value_cols = [c for c in wide.columns if c != "zone"]
    wide["rssz_total"] = wide[value_cols].sum(axis=1)

    rssz_df = wide.sort_values("rssz_total", ascending=False).reset_index(drop=True)
    rssz_df.to_csv(os.path.join(OUTPUT_DIR, "rssz_by_zone.csv"), index=False)

    vals = rssz_df["rssz_total"]
    print(f"\n{'='*60}\nRSSZ summary (precomputed) — {len(vals)} zones\n{'='*60}")
    print(f"  Mean  : {vals.mean():.4f}")
    print(f"  Median: {vals.median():.4f}")
    print(f"  Min   : {vals.min():.4f}")
    print(f"  Max   : {vals.max():.4f}")
    print(f"  Std   : {vals.std():.4f}")

    outliers = rssz_df[rssz_df["rssz_total"] > threshold]
    outliers.to_csv(os.path.join(OUTPUT_DIR, "outlier_zones.csv"), index=False)
    print(f"\n  Outlier zones (>{threshold}): {len(outliers)}")
    for _, row in outliers.iterrows():
        print(f"    Zone {row['zone']}: {row['rssz_total']:.4f}")

    table_cols = [c for c in rssz_df.columns if c not in ("zone", "rssz_total")]
    for _, row in outliers.iterrows():
        pt = {c: row[c] for c in table_cols if pd.notna(row.get(c))}
        plot_zone(row["zone"], pt, threshold)

    if table_cols:
        thresh_per = threshold / max(len(table_cols), 1)
        driver_counts = {c: int((rssz_df[c] > thresh_per).sum()) for c in table_cols}
        driver_df = pd.DataFrame(
            sorted(driver_counts.items(), key=lambda x: -x[1]),
            columns=["control_table", "n_zones_high_rssz"],
        )
        driver_df.to_csv(os.path.join(OUTPUT_DIR, "driver_tables.csv"), index=False)
        print("\nTop control tables driving high RSSZ:")
        print(driver_df.head(10).to_string(index=False))

    from scipy.stats import gaussian_kde
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.05)

    pct99 = vals.quantile(0.99)
    vals_main = vals[vals <= pct99]
    n_outlier = (vals > pct99).sum()
    bins = int(max(15, min(50, 2 * len(vals_main)**(1/3))))
    ax_top.hist(vals_main, bins=bins, density=True, alpha=0.6, color="#4C72B0",
                edgecolor="white", linewidth=0.5, label="Density (trimmed ≤ P99)")
    if len(vals_main) > 1 and np.std(vals_main) > 1e-15:
        try:
            kde = gaussian_kde(vals_main)
            x_kde = np.linspace(vals_main.min(), vals_main.max(), 300)
            ax_top.plot(x_kde, kde(x_kde), color="#C44E52", linewidth=2, label="KDE")
        except np.linalg.LinAlgError:
            pass

    ax_top.axvline(vals.mean(), color="#55A868", linestyle="--", linewidth=2,
                   label=f"Mean = {vals.mean():.3f}")
    ax_top.axvline(vals.median(), color="#8C564B", linestyle="-.", linewidth=1.5,
                   label=f"Median = {vals.median():.3f}")
    ax_top.axvline(vals.quantile(0.95), color="#DD8452", linestyle=":", linewidth=1.5,
                   label=f"P95 = {vals.quantile(0.95):.3f}")
    ax_top.axvline(threshold, color="#E24A33", linestyle="--", linewidth=1,
                   alpha=0.7, label=f"Threshold = {threshold}")
    ax_top.set_ylabel("Density", fontsize=11)
    ax_top.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    ax_top.set_title(f"RSSZ Distribution Across {len(vals)} Zones", fontsize=12, fontweight="bold")
    ax_top.grid(axis="y", linestyle=":", alpha=0.3)
    if n_outlier > 0:
        ax_top.text(0.98, 0.95, f"+ {n_outlier} zones > P99", transform=ax_top.transAxes,
                    ha="right", va="top", fontsize=8, color="#888888")

    ax_bot.boxplot(vals.values, vert=False, widths=0.6,
                   patch_artist=True, showmeans=True,
                   boxprops=dict(facecolor="#DDDDDD", alpha=0.7),
                   meanprops=dict(marker="D", markerfacecolor="#55A868", markersize=6),
                   medianprops=dict(color="#8C564B", linewidth=2),
                   flierprops=dict(marker=".", markersize=2, alpha=0.4))
    ax_bot.set_xlabel("RSSZ  (Relative Sum of Squared Z-scores)", fontsize=11)
    ax_bot.set_yticks([])
    ax_bot.grid(axis="x", linestyle=":", alpha=0.3)

    # Clip the shared x-axis so a handful of high-RSSZ outlier zones don't stretch
    # the whole figure out to ~5 and crush the bulk of the distribution into the
    # left third (the pre-fix figure read as badly skewed for exactly this reason).
    _xupper = max(2.15, float(vals.quantile(0.97)) * 1.15)
    _nbeyond = int((vals > _xupper).sum())
    ax_bot.set_xlim(-0.05, _xupper)
    if _nbeyond:
        ax_top.text(0.99, 0.78,
                    f"+{_nbeyond} zone(s) beyond axis (max {vals.max():.2f})",
                    transform=ax_top.transAxes, ha="right", va="top",
                    fontsize=8, color="#888888")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rssz_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print("\n[CHECK] Used precomputed rssz_results/sszk_per_zone.csv fallback.")


def plot_zone(zone_id, per_table, threshold):
    labels = list(per_table.keys())
    values = [per_table[k] for k in labels]
    thresh_per = threshold / max(len(labels), 1)
    colors = ["crimson" if v > thresh_per else "steelblue" for v in values]
    fig, ax = plt.subplots(figsize=(max(10, len(labels)*0.7), 5))
    ax.bar(range(len(labels)), values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("RSSZ"); ax.set_title(f"Zone {zone_id} — per-table RSSZ (red = high)")
    ax.axhline(np.mean(values), color="orange", linestyle="--",
               label=f"Mean = {np.mean(values):.3f}")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"outlier_zone_{zone_id}.png"), dpi=100)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=config.SA_GAN_RESULTS_DIR)
    p.add_argument("--threshold",   type=float, default=2.0)
    args = p.parse_args()

    # Prefer SA-GAN results; fall back to SA-Only
    result_files = glob.glob(os.path.join(args.results_dir, "zone_*_synthetic_population.csv"))
    if not result_files:
        result_files = glob.glob(os.path.join(config.SA_ONLY_RESULTS_DIR, "zone_*.csv"))
    if not result_files:
        raise FileNotFoundError(f"No result CSVs found in {args.results_dir}")

    print(f"Analysing {len(result_files)} zone files from: {args.results_dir}")
    try:
        ind_tables, hh_tables = load_control_tables()
    except Exception as exc:
        print(f"WARN: Could not load control tables for recomputation ({exc})")
        print("Falling back to precomputed rssz_results/sszk_per_zone.csv")
        build_from_precomputed_rssz(args.threshold)
        return

    records = []
    for fpath in sorted(result_files):
        zone_id = (os.path.basename(fpath)
                   .replace(".csv", "").replace("zone_", "").replace("_synthetic_population", "").replace("_sa_only", ""))
        print(f"  zone {zone_id}...")
        total, per = zone_rssz(fpath, zone_id, ind_tables, hh_tables)
        if total is None: continue
        rec = {"zone": zone_id, "rssz_total": total}
        if per: rec.update(per)
        records.append(rec)

    if not records:
        print("WARN: No zone RSSZ results were computed (all zone files empty or unreadable).")
        print("      This is expected when Stage 3a produced no synthetic population data.")
        print("      Writing empty output files and exiting cleanly.")
        pd.DataFrame(columns=["zone", "rssz_total"]).to_csv(
            os.path.join(OUTPUT_DIR, "rssz_by_zone.csv"), index=False
        )
        pd.DataFrame(columns=["zone", "rssz_total"]).to_csv(
            os.path.join(OUTPUT_DIR, "outlier_zones.csv"), index=False
        )
        return

    rssz_df = pd.DataFrame(records).sort_values("rssz_total", ascending=False)
    rssz_df.to_csv(os.path.join(OUTPUT_DIR, "rssz_by_zone.csv"), index=False)

    vals = rssz_df["rssz_total"]
    print(f"\n{'='*60}\nRSSZ summary — {len(vals)} zones\n{'='*60}")
    print(f"  Mean  : {vals.mean():.4f}")
    print(f"  Median: {vals.median():.4f}")
    print(f"  Min   : {vals.min():.4f}")
    print(f"  Max   : {vals.max():.4f}")
    print(f"  Std   : {vals.std():.4f}")

    outliers = rssz_df[rssz_df["rssz_total"] > args.threshold]
    outliers.to_csv(os.path.join(OUTPUT_DIR, "outlier_zones.csv"), index=False)
    print(f"\n  Outlier zones (>{args.threshold}): {len(outliers)}")
    for _, row in outliers.iterrows():
        print(f"    Zone {row['zone']}: {row['rssz_total']:.4f}")

    table_cols = [c for c in rssz_df.columns if c not in ("zone", "rssz_total")]
    for _, row in outliers.iterrows():
        pt = {c: row[c] for c in table_cols if pd.notna(row.get(c))}
        plot_zone(row["zone"], pt, args.threshold)

    if table_cols:
        thresh_per = args.threshold / max(len(table_cols), 1)
        driver_counts = {c: int((rssz_df[c] > thresh_per).sum())
                         for c in table_cols if c in rssz_df.columns}
        driver_df = pd.DataFrame(
            sorted(driver_counts.items(), key=lambda x: -x[1]),
            columns=["control_table", "n_zones_high_rssz"]
        )
        driver_df.to_csv(os.path.join(OUTPUT_DIR, "driver_tables.csv"), index=False)
        print("\nTop control tables driving high RSSZ:")
        print(driver_df.head(10).to_string(index=False))

    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        gaussian_kde = None
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.05)

    pct99 = vals.quantile(0.99)
    vals_main = vals[vals <= pct99]
    n_outlier = (vals > pct99).sum()
    bins = int(max(15, min(50, 2 * len(vals_main)**(1/3))))
    ax_top.hist(vals_main, bins=bins, density=True, alpha=0.6, color="#4C72B0",
                edgecolor="white", linewidth=0.5, label="Density (trimmed ≤ P99)")
    _kde_ok = (gaussian_kde is not None and len(vals_main) > 1
               and np.std(vals_main) > 1e-15)
    if _kde_ok:
        try:
            kde = gaussian_kde(vals_main)
            x_kde = np.linspace(vals_main.min(), vals_main.max(), 300)
            ax_top.plot(x_kde, kde(x_kde), color="#C44E52", linewidth=2, label="KDE")
        except np.linalg.LinAlgError:
            _kde_ok = False

    ax_top.axvline(vals.mean(), color="#55A868", linestyle="--", linewidth=2,
                   label=f"Mean = {vals.mean():.3f}")
    ax_top.axvline(vals.median(), color="#8C564B", linestyle="-.", linewidth=1.5,
                   label=f"Median = {vals.median():.3f}")
    ax_top.axvline(vals.quantile(0.95), color="#DD8452", linestyle=":", linewidth=1.5,
                   label=f"P95 = {vals.quantile(0.95):.3f}")
    ax_top.axvline(args.threshold, color="#E24A33", linestyle="--", linewidth=1,
                   alpha=0.7, label=f"Threshold = {args.threshold}")
    ax_top.set_ylabel("Density", fontsize=11)
    ax_top.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    ax_top.set_title(f"RSSZ Distribution Across {len(vals)} Zones", fontsize=12, fontweight="bold")
    ax_top.grid(axis="y", linestyle=":", alpha=0.3)
    if n_outlier > 0:
        ax_top.text(0.98, 0.95, f"+ {n_outlier} zones > P99", transform=ax_top.transAxes,
                    ha="right", va="top", fontsize=8, color="#888888")
    elif not _kde_ok:
        ax_top.text(0.98, 0.95, "KDE skipped — all RSSZ values identical", transform=ax_top.transAxes,
                    ha="right", va="top", fontsize=8, color="#888888")

    ax_bot.boxplot(vals.values, vert=False, widths=0.6,
                   patch_artist=True, showmeans=True,
                   boxprops=dict(facecolor="#DDDDDD", alpha=0.7),
                   meanprops=dict(marker="D", markerfacecolor="#55A868", markersize=6),
                   medianprops=dict(color="#8C564B", linewidth=2),
                   flierprops=dict(marker=".", markersize=2, alpha=0.4))
    ax_bot.set_xlabel("RSSZ  (Relative Sum of Squared Z-scores)", fontsize=11)
    ax_bot.set_yticks([])
    ax_bot.grid(axis="x", linestyle=":", alpha=0.3)

    # Clip the shared x-axis so a handful of high-RSSZ outlier zones don't stretch
    # the whole figure out to ~5 and crush the bulk of the distribution into the
    # left third (the pre-fix figure read as badly skewed for exactly this reason).
    _xupper = max(2.15, float(vals.quantile(0.97)) * 1.15)
    _nbeyond = int((vals > _xupper).sum())
    ax_bot.set_xlim(-0.05, _xupper)
    if _nbeyond:
        ax_top.text(0.99, 0.78,
                    f"+{_nbeyond} zone(s) beyond axis (max {vals.max():.2f})",
                    transform=ax_top.transAxes, ha="right", va="top",
                    fontsize=8, color="#888888")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rssz_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print(
        "\n[CHECK] rssz_distribution.png — are outlier zones small (high sampling variance)?")
    print("[CHECK] outlier_zone_*.png   — which control tables fail in outlier zones?")
    print("[CHECK] driver_tables.csv    — flag the worst-fitting tables in the paper discussion.")


if __name__ == "__main__":
    main()

"""
IPU Fit — IPU-GAN + IPU-Only baselines
=======================================
Runs SA_2_level.py --method ipu over every zone twice, mirroring the Stage 5/6
SA comparison so Reviewer 2's "isolate the GAN's contribution" analysis can be
extended from SA (sa_gan vs sa_only) to IPU (ipu_gan vs ipu_only) too:

  Phase 1 (IPU-GAN)  — seed = GAN-enriched microdata   -> outputs/ipu_gan_results/
  Phase 2 (IPU-Only) — seed = original census microdata -> outputs/ipu_only_results/

SA_2_level.py is invoked with --output-dir pointing directly at those folders,
so it writes zone_<zone>_synthetic_population.csv there using the same naming
convention as the SA-GAN and SA-Only results — Stage 8 (benchmark_margins.py)
already loads IPU directories with that convention, with no extra flags required.

This is the local/pipeline equivalent of pbs/ipu_comparison.pbs (which runs
the same loop on the HPC).  Called by run_pipeline.py Stage 6b.

Usage:
    python ipu_fit.py [--zones 301011001 ...] [--max-iter 1000]
"""

import argparse
import glob
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config
from sa_multi_wrapper import get_zones_from_control_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


ZONE_TIMEOUT_SECONDS = 14400  # 4 hours — mirrors sa_multi_wrapper.py's per-zone cap


def run_zone_ipu(zone, seed_file, output_dir, max_iter, control_scope):
    cmd = [
        sys.executable, os.path.join(HERE, "SA_2_level.py"),
        "--zone", str(zone),
        "--method", "ipu",
        "--dictionary_opts", "0",
        "--iterations", str(max_iter),
        "--initial-temp", "100",
        "--cooling-rate", "0.99",
        "--control-scope", control_scope,
        "--output-dir", output_dir,
        "--seed-file", seed_file,
    ]
    logging.info(f"  zone {zone}  seed={os.path.basename(seed_file)} -> {output_dir}")
    try:
        # Without a timeout, a single hung zone blocks subprocess.run forever,
        # and PBS kills the whole chunk job at walltime before any later zone runs.
        result = subprocess.run(cmd, cwd=HERE, timeout=ZONE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        logging.error(f"  [TIMEOUT] zone {zone} exceeded {ZONE_TIMEOUT_SECONDS}s — SA_2_level.py may be hung; skipping")
        return False
    if result.returncode != 0:
        logging.warning(f"  [WARN] zone {zone} failed (rc={result.returncode}); continuing")
    return result.returncode == 0


def collect_rssz(result_dir, out_csv):
    """Summarize per-zone RSSZ from *_rssz_values_cat.csv into per-table breakdown + stats."""
    rows = []
    for fpath in sorted(glob.glob(os.path.join(result_dir, "zone_*_rssz_values_cat.csv"))):
        zone_id = os.path.basename(fpath)
        zone_id = zone_id.replace("zone_", "").replace("_rssz_values_cat.csv", "")
        try:
            cat_df = pd.read_csv(fpath)
            v = pd.to_numeric(cat_df["RSSZ"], errors="coerce").dropna()
        except Exception as exc:
            logging.warning(f"  Could not read {fpath}: {exc}")
            v = pd.Series(dtype=float)
        if len(v) == 0:
            rows.append({"zone": zone_id, "rssz_total": np.nan, "mean_rssz": np.nan,
                         "median_rssz": np.nan, "max_rssz": np.nan, "p90_rssz": np.nan,
                         "n_tables": 0})
        else:
            row = {
                "zone": zone_id,
                "rssz_total": v.sum(),
                "mean_rssz": v.mean(), "median_rssz": v.median(),
                "max_rssz": v.max(), "p90_rssz": v.quantile(0.9),
                "n_tables": int(len(v)),
            }
            row.update(dict(zip(cat_df["Table"], cat_df["RSSZ"])))
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info(f"  Written: {out_csv}  ({len(rows)} zones)")


def main():
    p = argparse.ArgumentParser(description="Run IPU-GAN and/or IPU-Only fitting for all zones")
    p.add_argument("--zones", nargs="*", type=str, default=None,
                   help="Specific SA2 zones to process (default: all from control tables)")
    p.add_argument("--max-iter", type=int, default=config.SA_MAX_ITER,
                   help=f"IPU iterations per zone (default: {config.SA_MAX_ITER})")
    p.add_argument("--control-scope", type=str, default="seq", choices=["seq", "statewide"])
    p.add_argument("--phase", type=str, default="both", choices=["gan", "only", "both"],
                   help="Which seed to fit: 'gan' (IPU-GAN only), 'only' (IPU-Only/census "
                        "only), or 'both' (default — runs both phases sequentially). "
                        "Use 'gan'/'only' to split the fit across two parallel HPC jobs "
                        "(see pbs/stage3c_ipu_gan.pbs / pbs/stage3d_ipu_only.pbs), "
                        "matching the Stage 3a/3b SA-GAN vs SA-Only split.")
    args = p.parse_args()

    zones = [str(z) for z in args.zones] if args.zones else get_zones_from_control_tables(args.control_scope)
    if not zones:
        logging.error("No zones to process. Exiting.")
        sys.exit(1)

    run_gan = args.phase in ("gan", "both")
    run_only = args.phase in ("only", "both")

    if run_gan and not os.path.exists(config.HH_MATCH_OUTPUT):
        logging.error(f"GAN seed not found: {config.HH_MATCH_OUTPUT}. Run Stage 4 first.")
        sys.exit(1)
    if run_only and not os.path.exists(config.CENSUS_MICRODATA):
        logging.error(f"Census seed not found: {config.CENSUS_MICRODATA}.")
        sys.exit(1)

    ipu_gan_dir = config.IPU_GAN_RESULTS_DIR
    ipu_only_dir = config.IPU_ONLY_RESULTS_DIR

    if run_gan:
        os.makedirs(ipu_gan_dir, exist_ok=True)
        logging.info(f"--- IPU-GAN  ({len(zones)} zones, seed = {config.HH_MATCH_OUTPUT}) ---")
        for zone in zones:
            run_zone_ipu(zone, config.HH_MATCH_OUTPUT, ipu_gan_dir, args.max_iter, args.control_scope)
        collect_rssz(ipu_gan_dir, os.path.join(config.OUTPUTS_DIR, "ipu_gan_rssz_summary.csv"))

    if run_only:
        os.makedirs(ipu_only_dir, exist_ok=True)
        logging.info(f"--- IPU-Only ({len(zones)} zones, seed = {config.CENSUS_MICRODATA}) ---")
        for zone in zones:
            run_zone_ipu(zone, config.CENSUS_MICRODATA, ipu_only_dir, args.max_iter, args.control_scope)
        collect_rssz(ipu_only_dir, os.path.join(config.OUTPUTS_DIR, "ipu_only_rssz_summary.csv"))


if __name__ == "__main__":
    main()

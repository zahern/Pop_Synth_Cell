"""Fit the constraint-programming feasible pool with the existing IPU fitter.

This is the CP-IPU ablation: the candidate pool is built by the CP feasibility
procedure, while the zone-level fitter is exactly the same IPU implementation
used by IPU-GAN and IPU-Only.  It therefore isolates candidate-pool construction
from the fitting algorithm.

Seed preparation
----------------
SA_2_level.py requires the seed to carry ``absfid``, ``abshid`` and ``abspid``
(see SA_2_level.py:2704).  The CP pool is person-level but neither
``build_feasible_pool_cp`` nor ``Helper.clean_my_datagan`` emits ``abspid``, so
it is synthesised here and the resulting seed is validated before any zone is
fitted.  Validation is re-run every time the seed is picked up, so a seed left
over from an earlier run cannot be reused silently if it lacks the columns the
fitter needs.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_feasible_pool_cp as cp
import config
from cp_pool_sa import assign_infilled_households, build_pool
from helper_functions import Helper
from ipu_fit import collect_rssz, run_zone_ipu
from sa_multi_wrapper import get_zones_from_control_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CONTROL_ATTRS = [
    "SEXP", "AGEP", "MSTP", "HEAP", "OCCP", "INCP", "LFSP", "EMPP",
    "HRSP", "QALFP", "QALLP", "RLHP", "MTWP", "NPRD", "VEHRD", "MRERD",
    "RNTRD", "STRD", "HIND", "BEDRD", "HHCD",
]

# Columns SA_2_level.py refuses to run without (SA_2_level.py:2704), plus the
# region column it uses to place a zone.  The seed is lowercased by
# clean_my_datagan, so these are compared in lower case.
REQUIRED_SEED_COLUMNS = ("abshid", "absfid", "abspid")
REGION_COLUMN_ALIASES = ("regucp", "areaenum")

# A chunk that cannot get the seed lock waits for whoever holds it, then gives
# up and repairs the seed anyway (writes are atomic, so that is safe — it only
# costs I/O).  Cleaning the 1 GB pool takes minutes, not hours.
SEED_LOCK_TIMEOUT_SECONDS = 3 * 3600
SEED_LOCK_POLL_SECONDS = 15

# Tolerance when checking that a zone's output was written by *this* run rather
# than left behind by an earlier one.
MTIME_SLACK_SECONDS = 5


def read_seed_columns(path):
    """Return the lowercased column names of `path` without loading the data."""
    head = pd.read_csv(path, nrows=0)
    return [str(c).strip().lower() for c in head.columns]


def seed_problems(path):
    """List everything that stops `path` from being usable as an IPU seed."""
    cols = set(read_seed_columns(path))
    missing = [c for c in REQUIRED_SEED_COLUMNS if c not in cols]
    problems = []
    if missing:
        problems.append(f"missing required columns: {missing}")
    if not cols & set(REGION_COLUMN_ALIASES):
        problems.append(
            f"missing region column (need one of {list(REGION_COLUMN_ALIASES)})"
        )
    return problems


def seed_is_valid(path):
    problems = seed_problems(path)
    for problem in problems:
        logging.warning("Seed %s is unusable — %s", path, problem)
    return not problems


def synthesise_person_ids(df):
    """Give every row a unique person id, unless it already has one.

    ``abspid`` identifies a person downstream (it is what household repairs and
    the export stages key on).  The CP pool has no population register behind
    it, so ids are assigned sequentially in pool order, mirroring the GAN-side
    convention in houehold_family_match_c.py (``ABSPID = index + 1``) so the two
    seeds stay interchangeable.
    """
    if "abspid" in df.columns and df["abspid"].notna().all():
        return df
    df = df.copy()
    df["abspid"] = np.arange(1, len(df) + 1, dtype=np.int64)
    return df


def write_csv_atomic(df, path):
    """Write `path` via a private temp file + rename, so readers never see a
    half-written CSV when several chunks prepare the same seed."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def acquire_seed_lock(lock_path, timeout=SEED_LOCK_TIMEOUT_SECONDS):
    """Best-effort exclusive lock; returns True if this process took it."""
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() >= deadline:
                logging.warning(
                    "Seed lock %s still held after %ds — proceeding anyway "
                    "(writes are atomic)", lock_path, timeout,
                )
                return False
            time.sleep(SEED_LOCK_POLL_SECONDS)
        else:
            try:
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
            return True


def ensure_cp_seed(census_path, pool_path, seed=0):
    """Build the CP pool once, failing explicitly on missing inputs."""
    if os.path.exists(pool_path):
        return pool_path
    if not os.path.exists(census_path):
        raise FileNotFoundError(f"CP census input not found: {census_path}")

    census = pd.read_csv(census_path)
    pool, report = build_pool(census, CONTROL_ATTRS, min_count=5,
                              target_per_cell=20, seed=seed)
    write_csv_atomic(pool.drop(columns=["_source"], errors="ignore"), pool_path)
    logging.info("Built CP pool: %s (%s)", pool_path, report)
    return pool_path


def ensure_clean_seed(pool_path, seed_path):
    """Return a seed that SA_2_level.py can actually fit; repair or rebuild it.

    Reusing an existing seed without checking it is what let every CP-IPU chunk
    run to completion on a seed missing ``abspid`` and report success with zero
    zones.  The check below is unconditional.
    """
    if os.path.exists(seed_path) and seed_is_valid(seed_path):
        logging.info("Reusing validated CP seed: %s", seed_path)
        return seed_path

    lock_path = f"{seed_path}.lock"
    got_lock = acquire_seed_lock(lock_path)
    try:
        # Another chunk may have repaired the seed while we waited for the lock.
        if os.path.exists(seed_path) and seed_is_valid(seed_path):
            logging.info("CP seed repaired by a concurrent chunk: %s", seed_path)
            return seed_path

        if os.path.exists(pool_path):
            logging.info("Rebuilding cleaned CP seed from pool: %s", pool_path)
            seed_df = Helper().clean_my_datagan(pool_path)
        elif os.path.exists(seed_path):
            logging.info("Repairing existing CP seed in place: %s", seed_path)
            seed_df = pd.read_csv(seed_path, low_memory=False)
        else:
            raise FileNotFoundError(
                f"Neither the CP pool ({pool_path}) nor a cleaned seed "
                f"({seed_path}) exists; cannot build a seed."
            )

        seed_df.columns = seed_df.columns.str.lower()
        had_person_ids = "abspid" in seed_df.columns and seed_df["abspid"].notna().all()
        seed_df = synthesise_person_ids(seed_df)
        write_csv_atomic(seed_df, seed_path)

        problems = seed_problems(seed_path)
        if problems:
            raise RuntimeError(
                f"Cleaned CP seed {seed_path} is still unusable after repair: "
                f"{'; '.join(problems)}"
            )
        logging.info("Cleaned CP pool seed: %s (%d rows, %d cols, abspid %s)",
                     seed_path, len(seed_df), seed_df.shape[1],
                     "kept" if had_person_ids else "synthesised")
        return seed_path
    finally:
        if got_lock:
            try:
                os.remove(lock_path)
            except OSError:
                pass


def write_zone_summary(result_dir, summary_path, zones):
    """Summarise RSSZ for `zones` only and write it atomically to `summary_path`.

    ``collect_rssz`` sweeps every ``zone_*_rssz_values_cat.csv`` in the directory.
    Chunks share one results directory, so an unfiltered sweep would put another
    chunk's zones in this chunk's summary and make the file's contents depend on
    timing.  Keep only the zones this run actually fitted.
    """
    tmp_summary = os.path.join(result_dir, f".collect_rssz.{os.getpid()}.csv")
    try:
        collect_rssz(result_dir, tmp_summary)
        collected = pd.read_csv(tmp_summary)
    finally:
        if os.path.exists(tmp_summary):
            try:
                os.remove(tmp_summary)
            except OSError:
                pass

    collected["zone"] = collected["zone"].astype(str)
    summary = collected[collected["zone"].isin([str(z) for z in zones])]
    write_csv_atomic(summary, summary_path)
    logging.info("Wrote summary for %d zone(s): %s", len(summary), summary_path)
    return summary


def check_summary(summary_path):
    """Raise unless the summary holds at least one real per-table RSSZ value."""
    if not os.path.exists(summary_path):
        raise RuntimeError(f"CP-IPU summary was not written: {summary_path}")
    try:
        summary = pd.read_csv(summary_path)
    except pd.errors.EmptyDataError:
        # This is the 1-byte summary the failed production run left behind.
        raise RuntimeError(
            f"CP-IPU summary {summary_path} is empty — the fit produced no zones."
        ) from None
    if summary.empty:
        raise RuntimeError(f"CP-IPU summary {summary_path} has no zone rows")
    if "n_tables" not in summary.columns or int(summary["n_tables"].fillna(0).sum()) == 0:
        raise RuntimeError(
            f"CP-IPU summary {summary_path} has {len(summary)} zone row(s) but no "
            f"per-table RSSZ values — the fit produced nothing usable."
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run IPU on the CP-feasible pool")
    parser.add_argument("--census", default=config.LINKED_DATA)
    parser.add_argument("--zones", nargs="*", default=None)
    parser.add_argument("--max-iter", type=int, default=config.SA_MAX_ITER)
    parser.add_argument("--control-scope", choices=["seq", "statewide"], default="statewide")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pool",
        default=os.environ.get("POOL_OVERRIDE", ""),
        help="Household-atomic CP pool override",
    )
    args = parser.parse_args()

    pool_path = args.pool or os.path.join(
        config.CP_POOL_RESULTS_DIR, "cp_feasible_pool.csv"
    )
    ensure_cp_seed(args.census, pool_path, args.seed)
    seed_name = (
        "cp_feasible_pool_expanded_clean.csv"
        if args.pool else "cp_feasible_pool_clean.csv"
    )
    seed_path = os.path.join(config.CP_IPU_RESULTS_DIR, seed_name)
    ensure_clean_seed(pool_path, seed_path)

    zones = ([str(z) for z in args.zones] if args.zones
             else get_zones_from_control_tables(args.control_scope))
    if not zones:
        raise ValueError("No zones available for CP-IPU")

    chunk = os.environ.get("CHUNK_ID", "").strip()
    chunk_label = chunk if chunk else "all"
    logging.info("CP-IPU: fitting %d zone(s) (chunk %s)", len(zones), chunk_label)

    os.makedirs(config.CP_IPU_RESULTS_DIR, exist_ok=True)
    succeeded, failed = [], []
    for zone in zones:
        started = time.time()
        ok = run_zone_ipu(
            zone, seed_path, config.CP_IPU_RESULTS_DIR,
            args.max_iter, args.control_scope,
        )
        # A return code of 0 is not proof of work: check the zone's own output
        # was written by this run, not left behind by an earlier attempt.
        rssz_file = os.path.join(
            config.CP_IPU_RESULTS_DIR, f"zone_{zone}_rssz_values_cat.csv")
        wrote_output = (
            os.path.exists(rssz_file)
            and os.path.getmtime(rssz_file) >= started - MTIME_SLACK_SECONDS
        )
        if ok and wrote_output:
            succeeded.append(zone)
        else:
            failed.append(zone)
            logging.error(
                "zone %s FAILED (fitter rc ok: %s, output written: %s)",
                zone, ok, wrote_output,
            )

    if not succeeded:
        raise RuntimeError(
            f"CP-IPU produced zero usable zones out of {len(zones)} (chunk "
            f"{chunk_label}); refusing to write an empty summary. Check the "
            f"seed {seed_path} and the fitter log above."
        )

    if failed:
        failed_path = os.path.join(
            config.CP_IPU_RESULTS_DIR, f"cp_ipu_failed_zones_chunk{chunk}.txt")
        with open(failed_path, "w") as fh:
            fh.write("\n".join(failed) + "\n")
        logging.error(
            "CP-IPU: %d of %d zone(s) failed (chunk %s): %s — see %s",
            len(failed), len(zones), chunk_label, ", ".join(failed), failed_path,
        )

    # Chunks must not share one summary file.  Write into the run's own results
    # directory: writing to the shared outputs/ root let a large-pool rerun
    # overwrite the current-pool chunk summaries as it finished.
    summary_name = (f"cp_ipu_rssz_summary_chunk{chunk}.csv" if chunk
                    else "cp_ipu_rssz_summary.csv")
    summary_path = os.path.join(config.CP_IPU_RESULTS_DIR, summary_name)
    write_zone_summary(config.CP_IPU_RESULTS_DIR, summary_path, succeeded)
    summary = check_summary(summary_path)

    logging.info(
        "CP-IPU complete (chunk %s): %d/%d zones, mean RSSZ %.4f -> %s",
        chunk_label, len(succeeded), len(zones), summary["rssz_total"].mean(),
        summary_path,
    )


if __name__ == "__main__":
    main()

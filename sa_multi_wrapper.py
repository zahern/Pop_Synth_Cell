"""
SA Multi-Zone Wrapper
======================
Wrapper script to process multiple SA2 zones using SA_2_level.py.
This script is called by run_pipeline.py (Stage 5) with --zones and --max-iter arguments.

Usage:
    python sa_multi_wrapper.py [--zones ZONE1 ZONE2 ...] [--max-iter ITERATIONS]

Arguments:
    --zones: Specific SA2 zone IDs to process (e.g., 301011001 301011002). 
             If omitted, processes all zones from control tables.
    --max-iter: Maximum iterations for SA algorithm (default: 9000)
"""

import argparse
import subprocess
import sys
import os
import logging
import shutil
import pandas as pd
from CensusRecordImporter import CensusControlTable, is_seq_sa2
import config

HERE = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sa_multi_wrapper.log", mode='w')
    ]
)

def get_zones_from_control_tables(control_scope="seq"):
    """Extract zone list from control tables using CensusControlTable parsing."""

    def _normalize_zone_values(series):
        out = (
            series.astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        out = out[out != ""]
        return sorted(set(out.tolist()))

    def _extract_from_file(csv_file):
        try:
            ct = CensusControlTable(csv_file)
            ct.validate_and_clean()
            table = ct.control_table
            if "sa2" in table.columns:
                zones = _normalize_zone_values(table["sa2"].dropna())
                if zones:
                    logging.info(
                        f"Extracted {len(zones)} zones from {csv_file}: {zones[:5]}..."
                    )
                    return zones
        except Exception as exc:
            logging.debug(f"Failed to parse zones from {csv_file}: {exc}")
        return []

    # Legacy SEQ control set used by prior SA-GAN runs.
    seq_individual_files = [
        "sexstringnew.csv",
        "saoccu.csv",
        "vehrd.csv",
        "mtwp.csv",
        "heap.csv",
        "rlhp.csv",
        "nprd.csv",
        "hind.csv",
        "incp.csv",
        "hrsp.csv",
        "mrerd.csv",
        "rntrd.csv",
        "strd.csv",
    ]

    statewide_files = [
        "sa2mstp.csv", "sa2sexp.csv", "sa2occp.csv", "sa2mtwp.csv", "sa2heap.csv",
        "sa2rlhp.csv", "sa2qallp.csv", "sa2qalfp.csv", "sa2incp.csv", "sa2hrsp.csv",
        "sa2indp.csv", "sa2typp.csv", "sa2stup.csv",
        "sa2nprd.csv", "sa2hind.csv", "sa2vehrd.csv", "sa2mrerd.csv", "sa2rntrd.csv", "sa2strd.csv",
    ]

    if control_scope == "statewide":
        control_files = statewide_files
        control_search_dirs = ["data_sa2_controls", ".", "control_tables"]
        # Delegates to CensusRecordImporter.is_seq_sa2, the single source of
        # truth for the SEQ+Tweed Shire study area (357-zone list in
        # sa2_zones_reference_357.csv). Previously this was its own crude
        # heuristic (z[0]=='3' or z[:2]=='11') that matched ANY Queensland SA2
        # (all the way to Cairns/Townsville/Outback) and ANY NSW SA4 in the
        # 11x range (well beyond Tweed Shire) - it happened not to broaden the
        # actual output because the underlying data_sa2_controls/*.csv files
        # were themselves further intersected/filtered elsewhere, but it was
        # never actually enforcing "study area", just "somewhere in QLD or
        # nearby NSW". Fixed 2026-07.
        def _is_study_area_zone(z):
            return is_seq_sa2(z)
    else:
        control_files = seq_individual_files
        control_search_dirs = ["control_tables", ".", "data_sa2_controls"]

    def _candidate_paths(file_name):
        search_roots = [os.getcwd(), HERE]
        for search_dir in control_search_dirs:
            for root in search_roots:
                path = os.path.join(root, search_dir, file_name)
                if os.path.exists(path):
                    yield path

    if control_scope == "statewide":
        # Zone universe = every study-area zone present in the ANCHOR file
        # (sa2sexp.csv -- total person count by sex, the most robust/least
        # likely to be ABS-suppressed table) rather than the intersection
        # across all 18 statewide files. A strict intersection silently
        # drops a zone from EVERY strategy the moment even one household-
        # level table (e.g. sa2nprd/sa2hind/sa2vehrd/sa2mrerd/sa2rntrd/
        # sa2strd) lacks a row for it -- likely small-cell suppression on a
        # sparse zone, not "this zone has no population" (sa2sexp.csv still
        # has it). SA_2_level.py already degrades gracefully per-zone when
        # an individual control table is missing (drops that one table,
        # keeps fitting on the rest), so there's no need to exclude the
        # zone pipeline-wide over a gap in one of eighteen files.
        per_file_zones = []
        anchor_file = "sa2sexp.csv"
        anchor_zones = None
        for csv_file in control_files:
            for path in _candidate_paths(csv_file):
                zones = _extract_from_file(path)
                if zones:
                    filtered = [z for z in zones if _is_study_area_zone(z)]
                    if filtered:
                        per_file_zones.append(set(filtered))
                        if csv_file == anchor_file:
                            anchor_zones = set(filtered)
                    break  # use first path found for this file

        if per_file_zones:
            union_zones = set.union(*per_file_zones)
            result_set = anchor_zones if anchor_zones else union_zones
            result = sorted(result_set)
            intersection = set.intersection(*per_file_zones)
            partial = sorted(result_set - intersection)
            if partial:
                logging.warning(
                    f"Statewide scope: {len(partial)} zone(s) are missing from at least one "
                    f"(but not all) of {len(per_file_zones)} control files -- kept in the zone "
                    f"universe from {anchor_file}, SA_2_level.py will drop only the specific "
                    f"missing table(s) for those zones: {partial}"
                )
            logging.info(
                f"Statewide scope: {len(result)} study-area zones from {anchor_file} "
                f"(Brisbane 3xxxxxxxx + Tweed Heads 11xxxxxxxx + Toowoomba 317xxxxxx)"
            )
            return result

        # Fallback: data_sa2_controls — any sa2*.csv file
        controls_dir = "data_sa2_controls"
        if os.path.isdir(controls_dir):
            for name in sorted(os.listdir(controls_dir)):
                if not name.lower().endswith(".csv"):
                    continue
                path = os.path.join(controls_dir, name)
                raw = _extract_from_file(path)
                filtered = [z for z in raw if _is_study_area_zone(z)]
                if filtered:
                    logging.info(f"Statewide fallback: {len(filtered)} study-area zones from {path}")
                    return filtered
    else:
        for csv_file in control_files:
            for path in _candidate_paths(csv_file):
                zones = _extract_from_file(path)
                if zones:
                    return zones

        # Fallback: data_sa2_controls — filter to SEQ zones only (SA4 301-315, Queensland)
        controls_dir = "data_sa2_controls"
        if os.path.isdir(controls_dir):
            for name in sorted(os.listdir(controls_dir)):
                if not name.lower().endswith(".csv"):
                    continue
                path = os.path.join(controls_dir, name)
                raw = _extract_from_file(path)
                seq_zones = [z for z in raw if len(z) == 9 and z[:2] == "30"]
                if seq_zones:
                    logging.info(f"Filtered to {len(seq_zones)} SEQ zones from {path}")
                    return seq_zones

    logging.warning("Could not extract zones from control tables")
    return []


def collect_zone_outputs(zone, src_dir, dst_dir):
    """Copy per-zone SA outputs from SA_2_level default folder into canonical output folder."""
    copied = 0
    if not os.path.isdir(src_dir):
        return copied

    for name in os.listdir(src_dir):
        if not (name.startswith(f"zone_{zone}_") or name.startswith(f"{zone}_")):
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        try:
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied += 1
            elif os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
                copied += 1
        except (FileNotFoundError, OSError, shutil.Error) as e:
            logging.warning(f"Could not copy {name}: {e}")
            continue
    return copied


def collect_new_outputs(before_names, src_dir, dst_dir):
    """Copy outputs that appeared after a zone run into the canonical output folder."""
    copied = 0
    if not os.path.isdir(src_dir):
        return 0, []

    current_names = set(os.listdir(src_dir))
    new_names = sorted(current_names - before_names)
    for name in new_names:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        try:
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied += 1
            elif os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
                copied += 1
        except (FileNotFoundError, OSError, shutil.Error) as e:
            logging.warning(f"Could not copy {name}: {e}")
            continue
    return copied, new_names


def read_zone_rssz(canonical_out, zone):
    """Read per-zone RSSZ value from SA_2_level outputs copied into canonical output folder."""
    zone = str(zone)
    rssz_csv = os.path.join(canonical_out, f"zone_{zone}_rssz_value.csv")
    rssz_txt = os.path.join(canonical_out, f"zone_{zone}_rssz_value.txt")

    if os.path.exists(rssz_csv):
        try:
            df = pd.read_csv(rssz_csv)
            if "rssz_value" in df.columns and not df.empty:
                return float(df["rssz_value"].iloc[0])
        except Exception as exc:
            logging.warning(f"Could not parse RSSZ CSV for zone {zone}: {exc}")

    if os.path.exists(rssz_txt):
        try:
            text = open(rssz_txt, "r", encoding="utf-8").read().strip()
            # Expected: "RSSZ for the new synthetic population: <value>"
            value = text.split(":")[-1].strip()
            return float(value)
        except Exception as exc:
            logging.warning(f"Could not parse RSSZ TXT for zone {zone}: {exc}")

    return None


def read_rssz_from_names(canonical_out, names):
    """Read the first RSSZ artifact among a set of newly created filenames."""
    for name in names:
        if not (name.endswith("_rssz_value.csv") or name.endswith("_rssz_value.txt")):
            continue
        path = os.path.join(canonical_out, name)
        if name.endswith("_rssz_value.csv"):
            try:
                df = pd.read_csv(path)
                if "rssz_value" in df.columns and not df.empty:
                    raw_zone = name.rsplit("_rssz_value.csv", 1)[0]
                    zone_id = raw_zone.replace("zone_", "")
                    return zone_id, float(df["rssz_value"].iloc[0])
            except Exception as exc:
                logging.warning(f"Could not parse RSSZ CSV {name}: {exc}")
        else:
            try:
                text = open(path, "r", encoding="utf-8").read().strip()
                value = text.split(":")[-1].strip()
                raw_zone = name.rsplit("_rssz_value.txt", 1)[0]
                zone_id = raw_zone.replace("zone_", "")
                return zone_id, float(value)
            except Exception as exc:
                logging.warning(f"Could not parse RSSZ TXT {name}: {exc}")
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Run SA_2_level.py over multiple zones.")
    parser.add_argument("--zones", nargs="*", type=str, default=None,
                        help="Specific SA2 zones to run. Default: all detected zones.")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="Max SA iterations per zone.")
    parser.add_argument(
        "--control-scope",
        type=str,
        default=os.environ.get("SA_CONTROL_SCOPE", "statewide"),
        choices=["seq", "statewide"],
        help="Control-table geography scope for zone discovery and SA_2_level loading.",
    )
    args = parser.parse_args()

    # Build master zone list exactly the way SA_2_level indexes zones.
    all_zones = get_zones_from_control_tables(args.control_scope)
    if not all_zones:
        logging.error("Could not determine zone universe from control tables. Exiting.")
        sys.exit(1)

    # Determine zones to process and map each requested zone to the proper SA_2_level index.
    if args.zones:
        requested = [str(z) for z in args.zones]
        zones = [z for z in requested if z in all_zones]
        missing = [z for z in requested if z not in all_zones]
        if missing:
            for bad_zone in missing:
                logging.warning(
                    f"Zone {bad_zone}: not found in control tables (no overlapping zone code "
                    "between microdata and SA2 control tables) — skipping this zone only."
                )
            logging.warning(
                f"Skipped {len(missing)}/{len(requested)} requested zone(s) with no control-table "
                f"overlap: {missing}. Continuing with the remaining {len(zones)} zone(s)."
            )
        logging.info(f"Processing user-specified zones: {zones}")
    else:
        zones = all_zones
        logging.info(f"Auto-detected zones: {zones}")

    if not zones:
        logging.warning("Zone list is empty after filtering. Exiting.")
        sys.exit(1)

    if not os.path.exists("cleaned_dataset_hhid.csv"):
        logging.error("cleaned_dataset_hhid.csv not found. Ensure Stage 4 completed successfully.")
        sys.exit(1)

    # Canonical output directory used by downstream scripts.
    canonical_out = config.SA_GAN_RESULTS_DIR
    os.makedirs(canonical_out, exist_ok=True)

    # When this process is one of several parallel PBS chunk jobs (CHUNK_ID is
    # set by stage3a_sa_gan.pbs), give the shared summary CSVs a per-chunk
    # filename so concurrent chunks don't overwrite each other's rows via a
    # plain to_csv() on the same path. Stage 4 (stage4_analysis.pbs) globs and
    # aggregates all per-chunk files back into the canonical filenames below.
    _chunk_id = os.environ.get("CHUNK_ID", "").strip()

    def _chunked_name(base_name):
        if not _chunk_id:
            return base_name
        stem, ext = os.path.splitext(base_name)
        return f"{stem}_chunk{_chunk_id}{ext}"

    # SA_2_level writes outputs to zones_sa_0 when method=sa,dictionary_opts=0 (scratch dir).
    sa2_level_out = "zones_sa_0"
    os.makedirs(sa2_level_out, exist_ok=True)

    # Process each zone
    failed_zones = []
    successful_zones = []
    rssz_rows = []

    for run_idx, zone in enumerate(zones):
        logging.info(f"\n{'='*70}")
        logging.info(f"Processing zone {run_idx+1}/{len(zones)}: {zone}")
        logging.info(f"{'='*70}")

        if os.path.isdir(sa2_level_out):
            for name in os.listdir(sa2_level_out):
                path = os.path.join(sa2_level_out, name)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except (FileNotFoundError, OSError) as e:
                    # Ignore cleanup errors; not critical
                    logging.warning(f"Could not clean {path}: {e}")
        else:
            os.makedirs(sa2_level_out, exist_ok=True)

        # Call SA_2_level.py for this zone
        cmd = [
            sys.executable,
            "SA_2_level.py",
            "--zone", str(zone),
            "--method", "sa",
            "--dictionary_opts", "0",
            "--iterations", str(args.max_iter),
            "--control-scope", args.control_scope,
        ]
        # Optional aligned/completed seed (stage3a_sa_gan.pbs sets SA_SEED_FILE
        # after prepare_sa_seed.py). Unset means SA_2_level's own default.
        _seed_override = os.environ.get("SA_SEED_FILE", "").strip()
        if _seed_override:
            cmd += ["--seed-file", _seed_override]

        before_names = set()

        try:
            logging.info(f"Running: {' '.join(cmd)}")
            # Per-zone timeout: 4 hours (14,400 seconds).  Some zones cause
            # SA_2_level.py to hang (e.g. zone 301031019 took 16.5h before
            # being killed by PBS exit-143).  If a single zone exceeds this
            # limit, we kill SA_2_level.py, log a warning, and continue to
            # the next zone so the chunk job still produces partial results.
            ZONE_TIMEOUT_SECONDS = 14400  # 4 hours
            result = subprocess.run(cmd, check=True, capture_output=False,
                                    timeout=ZONE_TIMEOUT_SECONDS)

            copied, new_names = collect_new_outputs(before_names, sa2_level_out, canonical_out)
            if copied == 0:
                logging.warning(
                    f"Zone {zone} ran but no outputs were copied from {sa2_level_out}; "
                    "check SA_2_level logs for that zone."
                )

            # Write a plain zone_{id}.csv alias (matches sa_only_results naming convention)
            # so benchmark scripts can treat both result dirs identically.
            src_pop = os.path.join(canonical_out, f"zone_{zone}_synthetic_population.csv")
            dst_pop = os.path.join(canonical_out, f"zone_{zone}.csv")
            if os.path.exists(src_pop) and not os.path.exists(dst_pop):
                shutil.copy2(src_pop, dst_pop)

            zone_name, zone_rssz = read_rssz_from_names(canonical_out, new_names)
            if zone_rssz is None:
                zone_rssz = read_zone_rssz(canonical_out, zone)
                zone_name = str(zone)
            if zone_rssz is not None:
                rssz_rows.append({"zone": str(zone_name), "rssz_value": zone_rssz})
            else:
                logging.warning(f"Zone {zone} produced no readable RSSZ value artifact")

            if result.returncode == 0:
                successful_zones.append(zone)
                logging.info(f"Zone {zone} completed successfully (copied {copied} output artifacts)")
            else:
                failed_zones.append(zone)
                logging.error(f"Zone {zone} failed with return code {result.returncode}")
        except subprocess.CalledProcessError as e:
            failed_zones.append(zone)
            logging.error(f"Zone {zone} failed: {e}")
        except subprocess.TimeoutExpired as e:
            failed_zones.append(zone)
            logging.error(f"Zone {zone} TIMEOUT after {ZONE_TIMEOUT_SECONDS}s — SA_2_level.py may be hung; skipping")
        except Exception as e:
            failed_zones.append(zone)
            logging.error(f"Zone {zone} encountered error: {e}")

    # Summary
    logging.info("\n" + "=" * 70)
    logging.info("SA MULTI-ZONE PROCESSING SUMMARY")
    logging.info("=" * 70)
    logging.info(f"Total zones: {len(zones)}")
    logging.info(f"Successful: {len(successful_zones)}")
    logging.info(f"Failed: {len(failed_zones)}")

    if successful_zones:
        logging.info(f"Completed zones: {', '.join(successful_zones[:10])}")
        if len(successful_zones) > 10:
            logging.info(f"  ... and {len(successful_zones) - 10} more")

    if failed_zones:
        logging.error(f"Failed zones: {', '.join(failed_zones)}")
        logging.info(f"Check logs and outputs in {canonical_out}/ and {sa2_level_out}/")

    rssz_summary_path = os.path.join(canonical_out, _chunked_name("stage3a_rssz_summary.csv"))
    if rssz_rows:
        rssz_df = pd.DataFrame(rssz_rows).sort_values("zone")
        rssz_df.to_csv(rssz_summary_path, index=False)
        logging.info(f"Wrote Stage 3a RSSZ summary: {rssz_summary_path} ({len(rssz_df)} zones)")

    sa_gan_rssz_summary_path = os.path.join(canonical_out, _chunked_name("sa_gan_rssz_summary.csv"))
    sa_gan_summary_rows = []
    for zone in successful_zones:
        zone_rssz_cat = os.path.join(canonical_out, f"zone_{zone}_rssz_values_cat.csv")
        if not os.path.exists(zone_rssz_cat):
            continue
        try:
            cat_df = pd.read_csv(zone_rssz_cat)
            row = {"zone": str(zone), "rssz_total": cat_df["RSSZ"].sum()}
            row.update(dict(zip(cat_df["Table"], cat_df["RSSZ"])))
            sa_gan_summary_rows.append(row)
        except Exception:
            pass
    if sa_gan_summary_rows:
        pd.DataFrame(sa_gan_summary_rows).to_csv(sa_gan_rssz_summary_path, index=False)
        logging.info(f"Wrote SA-GAN per-table RSSZ summary: {sa_gan_rssz_summary_path} ({len(sa_gan_summary_rows)} zones)")
    elif rssz_rows:
        pd.DataFrame(rssz_rows).sort_values("zone").to_csv(sa_gan_rssz_summary_path, index=False)
        logging.info(f"Wrote SA-GAN RSSZ summary (basic): {sa_gan_rssz_summary_path} ({len(rssz_rows)} zones)")

    # ------------------------------------------------------------------
    # Assemble per-zone synthetic populations into a single file so that
    # Stage 4 (household_repairs_e.py) can run household consistency
    # repair on the SA-fitted output.
    # ------------------------------------------------------------------
    pop_parts = []
    for zone in successful_zones:
        zone_pop_path = os.path.join(canonical_out, f"zone_{zone}_synthetic_population.csv")
        if not os.path.exists(zone_pop_path):
            logging.warning(f"Zone population CSV missing for zone {zone}: {zone_pop_path}")
            continue
        try:
            zdf = pd.read_csv(zone_pop_path)
            zdf["_sa_zone"] = str(zone)
            pop_parts.append(zdf)
        except Exception as exc:
            logging.warning(f"Could not read zone population for {zone}: {exc}")

    if pop_parts:
        assembled = pd.concat(pop_parts, ignore_index=True)
        # SA_2_level writes lowercase columns (via clean_my_datagan); uppercase them
        # so Stage 4's repair_household_consistency can find ABSHID, ABSFID, ABSPID.
        assembled.columns = [c.upper() if c != "_sa_zone" else c for c in assembled.columns]
        if "REGUCP" in assembled.columns and "AREAENUM" not in assembled.columns:
            assembled = assembled.rename(columns={"REGUCP": "AREAENUM"})
        # Ensure ABS ID columns exist (SA used lowercase; now uppercase)
        if "ABSHID" not in assembled.columns and "HOUSEHOLD_ID" in assembled.columns:
            assembled = assembled.rename(columns={"HOUSEHOLD_ID": "ABSHID"})
        if "ABSFID" not in assembled.columns and "FAMILY_ID" in assembled.columns:
            assembled = assembled.rename(columns={"FAMILY_ID": "ABSFID"})
        if "ABSPID" not in assembled.columns:
            assembled["ABSPID"] = range(1, len(assembled) + 1)
        # Drop internal tracking column
        assembled = assembled.drop(columns=["_sa_zone"], errors="ignore")
        import config as _cfg
        os.makedirs(_cfg.HH_MATCH_OUT_DIR, exist_ok=True)
        sa_pop_path = os.path.join(_cfg.HH_MATCH_OUT_DIR, "hhid_fid_fixed.csv")
        assembled.to_csv(sa_pop_path, index=False)
        logging.info(
            f"Assembled {len(assembled):,} records from {len(pop_parts)} zones "
            f"into {sa_pop_path} — ready for Stage 4 household repair."
        )
    else:
        logging.warning(
            "No per-zone population CSVs found; skipping assembly for Stage 4."
        )

    if failed_zones:
        if not successful_zones:
            logging.error("Stage 5 failed: every zone failed. Check logs above.")
            sys.exit(1)
        else:
            logging.warning(
                f"Stage 5 completed with {len(failed_zones)} failed zone(s) out of {len(zones)}. "
                "Pipeline will continue with the zones that succeeded."
            )
    if not rssz_rows:
        logging.warning("Stage 5 completed without any RSSZ artifacts; treating as partial success.")
        sys.exit(0)

    logging.info("Stage 5 completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()

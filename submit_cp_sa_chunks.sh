#!/bin/bash
# =============================================================================
# Submit CP-SA baseline chunks over the CP feasible pool (one qsub per chunk).
#
# Why (Oct-2026): the single stage_cp_sa_only.pbs job covered all ~371 zones in
# one 24h slot and was SIGKILLed at the walltime limit (job 26593125,
# Exit_status -29) after 21 zones -- roughly 17 days of serial walltime needed.
# This splits the same work across parallel chunks.
#
# Unlike the MIP chunker, this one DERIVES the chunk list at submit time rather
# than reading a pre-baked directory.  pbs/zone_chunks/ has been frozen at
# 15/07/2026 and holds only chunk_0..chunk_19 for the old 354-zone scope, which
# is exactly how the "ERROR: Chunk file not found ... chunk_20.txt" jobs in
# sa_c20/ipug_c20 (28 Jul - 5 Aug) died.  Nothing here reads that directory.
#
# Zone derivation uses pbs/zones_from_controls.py, which imports
# sa_only_baseline.load_control_tables - the same canonical path used by
# rechunk_mip.sh and the SA stages themselves.
#
# Usage:
#   ./submit_cp_sa_chunks.sh                 # all zones from the control tables
#   ./submit_cp_sa_chunks.sh 112031250 301011001   # only these zone IDs
#   ZONE_FILE=zones.txt ./submit_cp_sa_chunks.sh   # explicit list, one per line
#   EXCLUDE=done.txt ./submit_cp_sa_chunks.sh      # skip already-completed zones
#   DEP=26593084.aqua ./submit_cp_sa_chunks.sh     # hold afterok a pool rebuild
#
# Env:
#   CHUNK_SIZE   zones per chunk (default 20)
#   SA_ITER      anneal iterations (default 1000)
#   DEP          optional job ID(s) -> chunks held afterok them
#   WALLTIME     per-chunk walltime (default 48:00:00)
#   MEM_GB       per-chunk memory (default 64)
#   CONTROL_DIR  control-table dir used to derive the zone list
#   ZONE_FILE    optional explicit zone list file (one zone per line)
#   EXCLUDE      optional file of already-completed zones to skip
#   DRY_RUN=1    write the chunk files and print what would be submitted, but
#                do not call qsub
#   POOL_OVERRIDE optional household-atomic expanded pool passed to every chunk
# =============================================================================
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

SYNTH_POP="${SYNTH_POP:-/home/ahernz/Paper_Plan/synth_pop}"
CONTROL_DIR="${CONTROL_DIR:-$SYNTH_POP/data_sa2_controls}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
SA_ITER="${SA_ITER:-1000}"
WALLTIME="${WALLTIME:-48:00:00}"
MEM_GB="${MEM_GB:-64}"
CPU_ENV_NAME="${CPU_ENV_NAME:-syn_env}"
CPU_ENV_PREFIX="${CPU_ENV_PREFIX:-/home/ahernz/miniconda3/envs}"
CENSUS_DIR="${CENSUS_DIR:-/home/ahernz/Population_Synth}"
POOL_OVERRIDE="${POOL_OVERRIDE:-}"
# Optional cleaned-pool pickle shared by every chunk (see cp_pool_sa.py): without
# it each chunk re-reads and re-cleans the whole pool.
CP_POOL_CLEAN_CACHE="${CP_POOL_CLEAN_CACHE:-}"
QSUB_QUEUE="${QSUB_QUEUE:-cpu_batch_exec}"
PYTHON_BIN="${PYTHON_BIN:-${CPU_ENV_PREFIX}/${CPU_ENV_NAME}/bin/python}"

CHUNK_OUT="${CHUNK_OUT:-pbs_logs/cp_sa_chunks}"
LOG_DIR="${LOG_DIR:-pbs_logs}"

if [[ -n "${DEP:-}" ]]; then
  _deps="afterok:$(echo "$DEP" | tr ' ' ':')"
  DEPFLAG=(-W "depend=$_deps")
else
  DEPFLAG=()
fi

if [[ ! -f "$CENSUS_DIR/LINKED_DATA_national.csv" ]]; then
  echo "ERROR: $CENSUS_DIR/LINKED_DATA_national.csv missing - CP-SA cannot run." >&2
  exit 1
fi

# --- Build the zone list -----------------------------------------------------
ZONES=""
if [[ $# -gt 0 ]]; then
  ZONES="$*"
elif [[ -n "${ZONE_FILE:-}" ]]; then
  echo "Using ZONE_FILE=$ZONE_FILE"
  ZONES="$(tr '\n' ' ' < "$ZONE_FILE")"
elif [[ -f "${ZONE_FILE_DEFAULT:-$SYNTH_POP/pbs/zones_371.txt}" ]]; then
  # The study scope, not every SA2 that carries control data: the control
  # directory holds 457 zones, while the paper fits 371 (368 with a census
  # population). Deriving from the tables would silently refit 86 zones that no
  # reported statistic covers.
  ZONE_FILE="${ZONE_FILE_DEFAULT:-$SYNTH_POP/pbs/zones_371.txt}"
  echo "Using study scope $ZONE_FILE"
  ZONES="$(tr '\n' ' ' < "$ZONE_FILE")"
else
  # Derive zones from the control tables via the same code path the SA stages
  # use.  NOT from outputs/feasible_pool.csv: that file is person-level
  # (ABSHID, ABSFID, AREAENUM, REGUCP, ...) and has no zone column at all.
  echo "Deriving zone list from control tables in $CONTROL_DIR ..."
  ZONES_RAW="$(CONTROL_DIR="$CONTROL_DIR" "$PYTHON_BIN" "$SYNTH_POP/pbs/zones_from_controls.py" ${EXCLUDE:+--exclude-file "$EXCLUDE"} | tr '\n' ' ')" || true
  ZONES="${ZONES_RAW}"
  if [[ -z "${ZONES// /}" ]]; then
    echo "ERROR: no zone ids found in $CONTROL_DIR (see messages above)." >&2
    echo "       Pass zones as arguments, or set ZONE_FILE=/path/to/zones.txt." >&2
    exit 1
  fi
fi

read -r -a ZONE_ARR <<< "$ZONES"
N_ZONES=${#ZONE_ARR[@]}
if [[ "$N_ZONES" -eq 0 ]]; then
  echo "ERROR: zone list is empty - refusing to submit." >&2
  exit 1
fi

mkdir -p "$CHUNK_OUT" "$LOG_DIR"
echo "Zones: $N_ZONES   chunk size: $CHUNK_SIZE   => $(( (N_ZONES + CHUNK_SIZE - 1) / CHUNK_SIZE )) chunks"

SUBMITTED=0
DRY="${DRY_RUN:-0}"
for ((i = 0; i < N_ZONES; i += CHUNK_SIZE)); do
  n=$(( i / CHUNK_SIZE ))
  chunk_zones="${ZONE_ARR[@]:i:CHUNK_SIZE}"
  cnt=$(echo "$chunk_zones" | wc -w)
  # shellcheck disable=SC2086  # deliberate: one zone id per line
  printf '%s\n' ${chunk_zones} > "${CHUNK_OUT}/cp_sa_chunk${n}.txt"
  if [[ "$DRY" == "1" ]]; then
    echo "  [dry] chunk ${n} (${cnt} zones): ${chunk_zones}"
    SUBMITTED=$((SUBMITTED + 1))
    continue
  fi
  JID=$(qsub "${DEPFLAG[@]}" -q "$QSUB_QUEUE" -N "synth_cp_sa_c${n}" \
      -l "select=1:ncpus=4:mem=${MEM_GB}GB,walltime=${WALLTIME}" \
      -v "SYNTH_POP=${SYNTH_POP},CENSUS_DIR=${CENSUS_DIR},CHUNK_ID=${n},ZONES=${chunk_zones},SA_ITER=${SA_ITER},CPU_ENV_NAME=${CPU_ENV_NAME},CPU_ENV_PREFIX=${CPU_ENV_PREFIX},POOL_OVERRIDE=${POOL_OVERRIDE},CP_POOL_CLEAN_CACHE=${CP_POOL_CLEAN_CACHE}" \
      -o "${LOG_DIR}/cp_sa_chunk${n}.log" -e "${LOG_DIR}/cp_sa_chunk${n}.err" \
      pbs/stage_cp_sa_chunk.pbs)
  if [[ -z "$JID" ]]; then
    echo "ERROR: qsub returned nothing for chunk ${n} - stopping." >&2
    echo "       ${SUBMITTED} chunk(s) were already submitted and are still queued." >&2
    exit 1
  fi
  echo "  chunk ${n} (${cnt} zones) -> ${JID}"
  SUBMITTED=$((SUBMITTED + 1))
done

echo ""
if [[ "$DRY" == "1" ]]; then
  echo "DRY RUN - nothing submitted. ${SUBMITTED} chunks would be created."
  echo "Re-run without DRY_RUN=1 to submit."
else
  echo "Submitted ${SUBMITTED} CP-SA chunks."
fi
echo "Logs    : ${LOG_DIR}/cp_sa_chunk<N>.log"
echo "Markers : ${SYNTH_POP}/outputs/cp_pool_results/.done/cp_sa_chunk<N>.done"
echo "Summaries: ${SYNTH_POP}/outputs/cp_pool_results/cp_pool_rssz_summary_chunk<N>.csv"
echo "Check   : ls ${SYNTH_POP}/outputs/cp_pool_results/.done/ | wc -l   (expect ${SUBMITTED})"
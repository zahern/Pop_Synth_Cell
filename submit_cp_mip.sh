#!/bin/bash
# Submit the CP-MIP integer-count fit across the paper's 20 zone chunks, one
# PBS job per chunk.  Run this on the HPC LOGIN node: it only calls qsub -- the
# python fit runs on compute nodes via pbs/stage_cp_mip_chunk.pbs.
#
# No --max-candidates cap: the fitter deduplicates exact household copies and
# then uses every distinct type in the zone's SA4 slice.  MAX_CANDIDATES is an
# optional emergency cap only; leave it empty.
#
# Usage:
#   bash submit_cp_mip.sh                         # 20 chunks, rep=6, lambda=1
#   REP_MAX=8 LAMBDA_JOINT=5 bash submit_cp_mip.sh
#   CHUNK_FIRST=0 CHUNK_LAST=0 bash submit_cp_mip.sh   # one chunk
#   AUTO_SETTING=1 SWEEP_BEST=.../best_setting.txt \
#       DEP=afterok:123456 bash submit_cp_mip.sh       # use sweep winner
#
# Monitor:  qstat -u "$USER"
# Results:  outputs/cp_mip_results/
set -euo pipefail
SYNTH_POP="${SYNTH_POP:-/home/ahernz/Paper_Plan/synth_pop}"
RPID="${RPID:-FPAXJW8067}"
POOL="${POOL:-$SYNTH_POP/outputs/feasible_pool.csv}"
RESULTS_ROOT="${RESULTS_ROOT:-$SYNTH_POP/outputs/cp_mip_results}"
LOG_DIR="${LOG_DIR:-$SYNTH_POP/pbs_logs/cp_mip_fitters}"
CHUNK_DIR="${CHUNK_DIR:-$SYNTH_POP/pbs/zone_chunks}"
CHUNK_FIRST="${CHUNK_FIRST:-0}"
CHUNK_LAST="${CHUNK_LAST:-19}"
REP_MAX="${REP_MAX:-6}"
TIME_LIMIT="${TIME_LIMIT:-120}"
LAMBDA_JOINT="${LAMBDA_JOINT:-1.0}"
MAX_CANDIDATES="${MAX_CANDIDATES:-}"
AUTO_SETTING="${AUTO_SETTING:-0}"
SWEEP_BEST="${SWEEP_BEST:-$SYNTH_POP/outputs/cp_mip_sweep/best_setting.txt}"
DEP="${DEP:-}"
WALLTIME="${WALLTIME:-06:00:00}"
NCPUS="${NCPUS:-2}"
MEM="${MEM:-24GB}"
mkdir -p "$LOG_DIR" "$RESULTS_ROOT"

[[ -f "$POOL" ]] || { echo "ERROR: pool not found: $POOL" >&2; exit 1; }
if [[ "$AUTO_SETTING" == "1" && ! -s "$SWEEP_BEST" ]]; then
  echo "WARNING: AUTO_SETTING=1 but $SWEEP_BEST is missing; jobs will fall back" >&2
  echo "         to REP_MAX=$REP_MAX LAMBDA_JOINT=$LAMBDA_JOINT at run time." >&2
fi

DEP_ARGS=()
[[ -n "$DEP" ]] && DEP_ARGS=(-W "depend=$DEP")
STAMP="$(date +%Y%m%d_%H%M%S)"
SUBMITTED="$LOG_DIR/submitted_${STAMP}.txt"
: > "$SUBMITTED"

n=0
for i in $(seq "$CHUNK_FIRST" "$CHUNK_LAST"); do
  chunk="$CHUNK_DIR/chunk_${i}.txt"
  [[ -s "$chunk" ]] || { echo "  chunk $i: no zone file, skipped"; continue; }
  jid=$(qsub "${DEP_ARGS[@]}" -P "$RPID" -q cpu_batch_exec \
    -N "cp_mip_c${i}" \
    -l "walltime=${WALLTIME},select=1:ncpus=${NCPUS}:mem=${MEM}" \
    -v "SYNTH_POP=$SYNTH_POP,CHUNK_ID=$i,ZONES_FILE=$chunk,POOL_OVERRIDE=$POOL,RESULTS_DIR=$RESULTS_ROOT,REP_MAX=$REP_MAX,TIME_LIMIT=$TIME_LIMIT,LAMBDA_JOINT=$LAMBDA_JOINT,MAX_CANDIDATES=$MAX_CANDIDATES,AUTO_SETTING=$AUTO_SETTING,SWEEP_BEST=$SWEEP_BEST" \
    -o "$LOG_DIR/cp_mip_${i}.log" -e "$LOG_DIR/cp_mip_${i}.err" \
    "$SYNTH_POP/pbs/stage_cp_mip_chunk.pbs")
  echo "$i $jid" >> "$SUBMITTED"
  echo "  chunk $i -> $jid"
  n=$((n+1))
done

echo "submitted $n CP-MIP chunk job(s) -> $RESULTS_ROOT"
echo "  rep_max=$REP_MAX lambda_joint=$LAMBDA_JOINT max_candidates=${MAX_CANDIDATES:-none}"
echo "  job ids: $SUBMITTED"
if [[ "$AUTO_SETTING" == "1" ]]; then
  echo "  auto setting: each job reads the sweep winner from $SWEEP_BEST at run time"
fi
if [[ -n "$DEP" ]]; then
  echo "  dependency: $DEP"
fi
echo "After they finish, score with: bash submit_cp_mip_metrics.sh"

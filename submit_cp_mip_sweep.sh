#!/bin/bash
# Hyperparameter sweep for the CP-MIP integer-count fit on ONE zone chunk.
#
# Grid: rep_max in {4,6,8,10} x lambda_joint in {1,5,10,20} = 16 PBS jobs.
# Each job runs the same chunk through cp_pool_mip.py at one setting.  A final
# scoring job (dependent on all 16) computes the held-out RSSZ on the seven
# tables never used by the fitter, compares every setting with the base SA and
# IPU arms on the same zones, and writes the winner to best_setting.txt.
#
# Run on the login node: this script only calls qsub.
#
#   bash submit_cp_mip_sweep.sh
#   SWEEP_CHUNK=3 bash submit_cp_mip_sweep.sh
#   REP_MAX_GRID="4 6 8 10 12" LAMBDA_GRID="1 5 10 20 40" bash submit_cp_mip_sweep.sh
set -euo pipefail
SYNTH_POP="${SYNTH_POP:-/home/ahernz/Paper_Plan/synth_pop}"
RPID="${RPID:-FPAXJW8067}"
POOL="${POOL:-$SYNTH_POP/outputs/feasible_pool.csv}"
SWEEP_ROOT="${SWEEP_ROOT:-$SYNTH_POP/outputs/cp_mip_sweep}"
LOG_DIR="${LOG_DIR:-$SYNTH_POP/pbs_logs/cp_mip_sweep}"
SWEEP_CHUNK="${SWEEP_CHUNK:-0}"
ZONES_FILE="${ZONES_FILE:-$SYNTH_POP/pbs/zone_chunks/chunk_${SWEEP_CHUNK}.txt}"
REP_MAX_GRID="${REP_MAX_GRID:-4 6 8 10}"
LAMBDA_GRID="${LAMBDA_GRID:-1 5 10 20}"
TIME_LIMIT="${TIME_LIMIT:-120}"
MAX_CANDIDATES="${MAX_CANDIDATES:-}"
WALLTIME="${WALLTIME:-06:00:00}"
NCPUS="${NCPUS:-2}"
MEM="${MEM:-24GB}"
mkdir -p "$LOG_DIR" "$SWEEP_ROOT"

[[ -f "$POOL" ]] || { echo "ERROR: pool not found: $POOL" >&2; exit 1; }
[[ -s "$ZONES_FILE" ]] || { echo "ERROR: zone file not found: $ZONES_FILE" >&2; exit 1; }

echo "Sweep on chunk ${SWEEP_CHUNK}: $(wc -l < "$ZONES_FILE") zones"
JOB_IDS=()
for R in $REP_MAX_GRID; do
  for L in $LAMBDA_GRID; do
    tag="rep${R}_lam${L}"
    outdir="$SWEEP_ROOT/$tag"
    mkdir -p "$outdir"
    jid=$(qsub -P "$RPID" -q cpu_batch_exec \
      -N "cp_mip_sw_${tag}" \
      -l "walltime=${WALLTIME},select=1:ncpus=${NCPUS}:mem=${MEM}" \
      -v "SYNTH_POP=$SYNTH_POP,CHUNK_ID=$tag,ZONES_FILE=$ZONES_FILE,POOL_OVERRIDE=$POOL,RESULTS_DIR=$outdir,REP_MAX=$R,TIME_LIMIT=$TIME_LIMIT,LAMBDA_JOINT=$L,AUTO_SETTING=0,MAX_CANDIDATES=$MAX_CANDIDATES" \
      -o "$LOG_DIR/sweep_${tag}.log" -e "$LOG_DIR/sweep_${tag}.err" \
      "$SYNTH_POP/pbs/stage_cp_mip_chunk.pbs")
    JOB_IDS+=("$jid")
    echo "  $tag -> $jid"
  done
done

DEP="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
SCORE_JOB=$(qsub -P "$RPID" -q cpu_batch_exec \
  -N "cp_mip_sweep_score" \
  -l "walltime=04:00:00,select=1:ncpus=2:mem=32GB" \
  -W "depend=$DEP" \
  -v "SYNTH_POP=$SYNTH_POP,SWEEP_ROOT=$SWEEP_ROOT,ZONES_FILE=$ZONES_FILE,OUT=$SWEEP_ROOT/sweep_summary.csv,BEST=$SWEEP_ROOT/best_setting.txt" \
  -o "$LOG_DIR/sweep_score.log" -e "$LOG_DIR/sweep_score.err" \
  "$SYNTH_POP/pbs/score_cp_mip_sweep.pbs")
echo "  scoring job -> $SCORE_JOB  (after all 16 fits)"
echo "$SCORE_JOB" > "$SWEEP_ROOT/score_job_id.txt"
echo
echo "Winner will be written to $SWEEP_ROOT/best_setting.txt"
echo "Then launch all zones with:"
echo "  AUTO_SETTING=1 DEP=afterok:$SCORE_JOB bash $SYNTH_POP/submit_cp_mip.sh"

#!/bin/bash
set -euo pipefail
SYNTH_POP="${SYNTH_POP:-/home/ahernz/Paper_Plan/synth_pop}"
QSUB_QUEUE="${QSUB_QUEUE:-cpu_batch_exec}"
RPID="${RPID:-${PBS_RPID:-}}"
PROJ=()
[[ -n "$RPID" ]] && PROJ=(-P "$RPID")
EXPANDED_JOB="${EXPANDED_JOB:-27396325.aqua}"
POOL="${POOL:-$SYNTH_POP/outputs/feasible_pool_expanded.csv}"
RESULTS_ROOT="${RESULTS_ROOT:-$SYNTH_POP/outputs/cp_pool_results}"
LOG_DIR="${LOG_DIR:-$SYNTH_POP/pbs_logs/non_gan_fitters_v3}"
mkdir -p "$LOG_DIR"

for ((i=0; i<20; i++)); do
  chunk="$SYNTH_POP/pbs/zone_chunks/chunk_${i}.txt"
  [[ -s "$chunk" ]] || continue
  zones="$(tr '\n' ' ' < "$chunk")"
  qsub "${PROJ[@]}" -q "$QSUB_QUEUE" -W "depend=afterok:$EXPANDED_JOB" \
    -N "cp_sa_exp_v3_c${i}" \
    -l "walltime=48:00:00,select=1:ncpus=1:mem=32GB" \
    -v "SYNTH_POP=$SYNTH_POP,CHUNK_ID=$i,ZONES=$zones,SA_ITER=1000,POOL_OVERRIDE=$POOL,RESULTS_DIR=$RESULTS_ROOT/expanded_v3_chunk_${i}" \
    -o "$LOG_DIR/cp_sa_${i}.log" -e "$LOG_DIR/cp_sa_${i}.err" \
    "$SYNTH_POP/pbs/stage_cp_sa_chunk.pbs"
done

qsub "${PROJ[@]}" -q "$QSUB_QUEUE" -W "depend=afterok:$EXPANDED_JOB" \
  -N cp_mip_expanded_v3 -l "walltime=48:00:00,select=1:ncpus=4:mem=64GB" \
  -v "SYNTH_POP=$SYNTH_POP,POOL_OVERRIDE=$POOL,SA_ITER=1000" \
  -o "$LOG_DIR/cp_mip.log" -e "$LOG_DIR/cp_mip.err" \
  "$SYNTH_POP/pbs/stage_cp_mip.pbs"

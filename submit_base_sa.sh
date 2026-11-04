#!/bin/bash
set -euo pipefail
SYNTH_POP="${SYNTH_POP:-/home/ahernz/Paper_Plan/synth_pop}"
RPID="${RPID:-FPAXJW8067}"
POOL="${POOL:-$SYNTH_POP/outputs/feasible_pool.csv}"
RESULTS_ROOT="${RESULTS_ROOT:-$SYNTH_POP/outputs/cp_pool_results_base}"
LOG_DIR="${LOG_DIR:-$SYNTH_POP/pbs_logs/base_fitters}"
mkdir -p "$LOG_DIR" "$RESULTS_ROOT"
for i in $(seq 0 19); do
  chunk="$SYNTH_POP/pbs/zone_chunks/chunk_${i}.txt"
  [[ -s "$chunk" ]] || continue
  zones="$(tr '\n' ' ' < "$chunk")"
  qsub -P "$RPID" -q cpu_batch_exec \
    -N "cp_sa_base_c${i}" \
    -l "walltime=48:00:00,select=1:ncpus=1:mem=32GB" \
    -v "SYNTH_POP=$SYNTH_POP,CHUNK_ID=$i,ZONES=$zones,SA_ITER=1000,POOL_OVERRIDE=$POOL,RESULTS_DIR=$RESULTS_ROOT/expanded_v3_chunk_${i}" \
    -o "$LOG_DIR/cp_sa_${i}.log" -e "$LOG_DIR/cp_sa_${i}.err" \
    "$SYNTH_POP/pbs/stage_cp_sa_chunk.pbs"
done

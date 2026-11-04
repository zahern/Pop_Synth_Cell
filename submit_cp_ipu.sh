#!/bin/bash
# Submit CP-IPU using the same zone chunks as the main paper experiments.
set -euo pipefail
SYNTH_POP="${SYNTH_POP:-$HOME/Paper_Plan/synth_pop}"
N_CHUNKS="${N_CHUNKS:-20}"
SA_ITER="${SA_ITER:-1000}"
POOL_OVERRIDE="${POOL_OVERRIDE:-}"
RESULTS_DIR="${RESULTS_DIR:-}"
DEP="${DEP:-}"
QSUB_QUEUE="${QSUB_QUEUE:-cpu_batch_exec}"
RPID="${RPID:-${PBS_RPID:-}}"
PROJ=()
[[ -n "$RPID" ]] && PROJ=(-P "$RPID")
cd "$SYNTH_POP"

DEPFLAG=()
if [[ -n "$DEP" ]]; then
  DEPFLAG=(-W "depend=afterok:$(echo "$DEP" | tr ' ' ':')")
fi

for ((i=0; i<N_CHUNKS; i++)); do
  chunk="$SYNTH_POP/pbs/zone_chunks/chunk_${i}.txt"
  [[ -s "$chunk" ]] || continue
  qsub "${PROJ[@]}" "${DEPFLAG[@]}" -q "$QSUB_QUEUE" -N "cp_ipu_c${i}" \
    -l "walltime=48:00:00,select=1:ncpus=1:mem=32GB" \
    -v "SYNTH_POP=$SYNTH_POP,CHUNK_ID=$i,SA_ITER=$SA_ITER,SA_CONTROL_SCOPE=statewide,POOL_OVERRIDE=$POOL_OVERRIDE,RESULTS_DIR=$RESULTS_DIR" \
    "$SYNTH_POP/pbs/stage_cp_ipu.pbs"
done

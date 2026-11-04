#!/bin/bash
# Submit the full CP-MIP metric suite after the chunk fits have finished.
# Run on the login node: only qsub happens here.
#
#   DEP=afterok:123456 bash submit_cp_mip_metrics.sh
#   DEP=afterok:123456,789012 bash submit_cp_mip_metrics.sh   # if PBS accepts it
set -euo pipefail
SYNTH_POP="${SYNTH_POP:-/home/ahernz/Paper_Plan/synth_pop}"
RPID="${RPID:-FPAXJW8067}"
RESULTS_DIR="${RESULTS_DIR:-$SYNTH_POP/outputs/cp_mip_results}"
LOG_DIR="${LOG_DIR:-$SYNTH_POP/pbs_logs/cp_mip_fitters}"
DEP="${DEP:-}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

DEP_ARGS=()
[[ -n "$DEP" ]] && DEP_ARGS=(-W "depend=$DEP")
JID=$(qsub "${DEP_ARGS[@]}" -P "$RPID" -q cpu_batch_exec \
  -N "cp_mip_metrics" \
  -l "walltime=12:00:00,select=1:ncpus=4:mem=48GB" \
  -v "SYNTH_POP=$SYNTH_POP,CP_MIP_RESULTS_DIR=$RESULTS_DIR" \
  -o "$LOG_DIR/cp_mip_metrics.log" -e "$LOG_DIR/cp_mip_metrics.err" \
  "$SYNTH_POP/pbs/score_cp_mip_full.pbs")
echo "metrics job -> $JID"
[[ -n "$DEP" ]] && echo "  dependency: $DEP"

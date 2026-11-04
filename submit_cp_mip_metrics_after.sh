#!/bin/bash
# Submit the CP-MIP metric suite after the most recent set of chunk jobs.
set -euo pipefail
cd "$(dirname "$0")"
F="$(ls -t pbs_logs/cp_mip_fitters/submitted_*.txt | head -1)"
DEP="afterok:$(awk '{print $2}' "$F" | paste -sd: -)"
echo "chunk file: $F ($(wc -l < "$F") jobs)"
echo "dependency: $DEP"
DEP="$DEP" bash submit_cp_mip_metrics.sh

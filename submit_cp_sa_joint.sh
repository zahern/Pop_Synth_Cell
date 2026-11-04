#!/bin/bash
# submit_cp_sa_joint.sh -- joint-aware CP-pool+SA refit.
# Re-fits every requested zone with the joint control tables active (the
# stage_cp_sa_joint.pbs wrapper loads SA_TABLE_WEIGHTS from joint_weights.json).
#
# Run on the login node (no Python):
#     MODE=dry bash submit_cp_sa_joint.sh          # plan only
#     MODE=go  bash submit_cp_sa_joint.sh          # submit
# Override: NEWID_START=200 CHUNK_N=20
set -uo pipefail
cd ~/Paper_Plan/synth_pop || exit 1
MODE="${MODE:-dry}"
NEWID_START="${NEWID_START:-200}"
CHUNK_N="${CHUNK_N:-20}"

SYNTH_POP=~/Paper_Plan/synth_pop
CH="$SYNTH_POP/pbs_logs/cp_sa_chunks"
RS="$SYNTH_POP/pbs_logs/resubmit"
PROJECT=FPAXJW8067
NCPUS=4; MEM_GB=64; WALLTIME=48:00:00; SA_ITER=1000
ENV_NAME=syn_env; ENV_PREFIX=/home/ahernz/miniconda3/envs
CENSUS_DIR=/home/ahernz/Population_Synth

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# every requested zone
cat "$CH"/cp_sa_chunk*.txt 2>/dev/null | grep -E '^[0-9]+$' | sort -u > "$TMP/req.txt"
NREQ=$(grep -cE '^[0-9]+$' "$TMP/req.txt" || true)
echo "requested zones: $NREQ  (chunk size $CHUNK_N, NEWID_START $NEWID_START)"

split -l "$CHUNK_N" "$TMP/req.txt" "$TMP/part_"
NEWID=$NEWID_START
for p in "$TMP"/part_*; do
  zlist=$(paste -sd' ' "$p")
  nz=$(wc -l < "$p")
  cp "$p" "$RS/rs_chunk${NEWID}.txt"
  if [ "$MODE" = "go" ]; then
    JID=$(qsub -q cpu_batch_exec -P "$PROJECT" \
      -N "synth_cp_sa_joint${NEWID}" \
      -l "select=1:ncpus=${NCPUS}:mem=${MEM_GB}GB,walltime=${WALLTIME}" \
      -v "SYNTH_POP=${SYNTH_POP},CENSUS_DIR=${CENSUS_DIR},SA_ITER=${SA_ITER},CPU_ENV_NAME=${ENV_NAME},CPU_ENV_PREFIX=${ENV_PREFIX},CHUNK_ID=${NEWID},ZONES=${zlist},RESUBMIT_OF=joint" \
      -o "$RS/rs_chunk${NEWID}.log" -e "$RS/rs_chunk${NEWID}.err" \
      "$SYNTH_POP/pbs/stage_cp_sa_joint.pbs")
    echo "  rs_chunk${NEWID}: $nz zones  [$JID]"
  else
    echo "  [dry] rs_chunk${NEWID}: $nz zones"
  fi
  NEWID=$((NEWID+1))
done
echo "done (last id $((NEWID-1)))."

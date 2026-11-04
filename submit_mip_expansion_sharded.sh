#!/bin/bash
# Submit the pool expansion as zone shards, then merge, then optionally fits.
#
# Usage:
#   bash submit_mip_expansion_sharded.sh
#   N_SHARDS=20 SUBMIT_FITS=1 bash submit_mip_expansion_sharded.sh
#   ATTACH_HHTS=1 SUBMIT_FITS=1 bash submit_mip_expansion_sharded.sh
#
# Shards run on one CPU each; merge runs after all shards succeed.  With
# ATTACH_HHTS=1 the merged pool is first augmented with observed BIKES/EBIKES/
# ESCOOTER (and licence) values from the HHTS-fused census.  With SUBMIT_FITS=1
# the SA/MIP/IPU chunks are released after that stage, using the same scripts as
# the serial chain.
set -euo pipefail

SYNTH_POP="${SYNTH_POP:-/home/ahernz/Paper_Plan/synth_pop}"
QSUB_QUEUE="${QSUB_QUEUE:-cpu_batch_exec}"
RPID="${RPID:-${PBS_RPID:-}}"
PROJ=()
[[ -n "$RPID" ]] && PROJ=(-P "$RPID")
N_SHARDS="${N_SHARDS:-20}"
POOL="${POOL:-$SYNTH_POP/outputs/feasible_pool.csv}"
OUT="${OUT:-$SYNTH_POP/outputs/feasible_pool_expanded.csv}"
SHARD_DIR="${SHARD_DIR:-$SYNTH_POP/outputs/expansion_shards}"
LOG_DIR="${LOG_DIR:-$SYNTH_POP/pbs_logs/expansion_shards}"
SHARD_WALLTIME="${SHARD_WALLTIME:-12:00:00}"
SHARD_MEM="${SHARD_MEM:-16GB}"
DONORS_PER_CELL="${DONORS_PER_CELL:-4}"
SHELLS_PER_CELL="${SHELLS_PER_CELL:-8}"
INCLUDE_PARTIAL="${INCLUDE_PARTIAL:-0}"
SIZE_AWARE="${SIZE_AWARE:-0}"
COVERAGE_ONLY="${COVERAGE_ONLY:-0}"
LAMBDA_COMPOSITION="${LAMBDA_COMPOSITION:-}"
MAX_SIZE7_SHARE="${MAX_SIZE7_SHARE:-}"
MAX_MTWP17_SHARE="${MAX_MTWP17_SHARE:-}"
MIN_ADMITTED_HOUSEHOLDS="${MIN_ADMITTED_HOUSEHOLDS:-}"
CENSUS_CSV="${CENSUS_CSV:-/home/ahernz/Population_Synth/LINKED_DATA_national.csv}"
HHTS_CENSUS="${HHTS_CENSUS:-/home/ahernz/Population_Synth/LINKED_DATA_national_hhts.csv}"
ATTACH_HHTS="${ATTACH_HHTS:-0}"
EXPANDED_HHTS="${EXPANDED_HHTS:-$SYNTH_POP/outputs/feasible_pool_expanded_hhts.csv}"
CPU_ENV_NAME="${CPU_ENV_NAME:-syn_env}"
CPU_ENV_PREFIX="${CPU_ENV_PREFIX:-/home/ahernz/miniconda3/envs}"
PYTHON_BIN="${CPU_ENV_PREFIX}/${CPU_ENV_NAME}/bin/python"

[[ -s "$POOL" ]] || { echo "ERROR: base pool not found: $POOL" >&2; exit 1; }
mkdir -p "$SHARD_DIR" "$LOG_DIR"
rm -f "$SHARD_DIR"/shard_*.csv "$SHARD_DIR"/shard_*.report.json

# Build the union of the paper's zone chunks.  Zones with no control targets
# are skipped by the expansion audit, so a slightly wider list is harmless.
ZONE_UNION="$SHARD_DIR/zones.txt"
cat "$SYNTH_POP"/pbs/zone_chunks/chunk_*.txt | awk 'NF' | sort -u > "$ZONE_UNION"
N_ZONES="$(wc -l < "$ZONE_UNION")"
[[ "$N_ZONES" -gt 0 ]] || { echo "ERROR: no zones found" >&2; exit 1; }

# Disjoint, deterministic zone shards.  The old shell round-robin produced
# duplicate shard pairs; this helper writes each zone to exactly one shard.
"$PYTHON_BIN" "$SYNTH_POP/make_zone_shards.py" \
  --chunks "$SYNTH_POP/pbs/zone_chunks" --n "$N_SHARDS" --out "$SHARD_DIR"
N_ZONES="$(wc -l < "$ZONE_UNION")"
[[ "$N_ZONES" -gt 0 ]] || { echo "ERROR: no zones found" >&2; exit 1; }

echo "Submitting $N_SHARDS expansion shards over $N_ZONES zones"
SHARD_JOBS=()
for ((i=0; i<N_SHARDS; i++)); do
  out="$SHARD_DIR/shard_${i}.csv"
  rep="$SHARD_DIR/shard_${i}.report.json"
  jid=$(qsub "${PROJ[@]}" -q "$QSUB_QUEUE" \
    -N "mip_exp_shard_${i}" \
    -l "walltime=${SHARD_WALLTIME},select=1:ncpus=1:mem=${SHARD_MEM}" \
    -v "SYNTH_POP=$SYNTH_POP,SHARD_ID=$i,ZONES_FILE=$SHARD_DIR/shard_${i}.zones,POOL=$POOL,OUT=$out,REPORT=$rep,ID_OFFSET=$((i*10000000)),CENSUS_CSV=$CENSUS_CSV,CPU_ENV_NAME=$CPU_ENV_NAME,CPU_ENV_PREFIX=$CPU_ENV_PREFIX,EXPAND_DONORS_PER_CELL=$DONORS_PER_CELL,EXPAND_SHELLS_PER_CELL=$SHELLS_PER_CELL,EXPAND_INCLUDE_PARTIAL=$INCLUDE_PARTIAL,EXPAND_SIZE_AWARE=$SIZE_AWARE,EXPAND_COVERAGE_ONLY=$COVERAGE_ONLY,EXPAND_LAMBDA_COMPOSITION=$LAMBDA_COMPOSITION,EXPAND_MAX_SIZE7_SHARE=$MAX_SIZE7_SHARE,EXPAND_MAX_MTWP17_SHARE=$MAX_MTWP17_SHARE,EXPAND_MIN_ADMITTED_HOUSEHOLDS=$MIN_ADMITTED_HOUSEHOLDS" \
    -o "$LOG_DIR/shard_${i}.log" -e "$LOG_DIR/shard_${i}.err" \
    "$SYNTH_POP/pbs/stage_mip_expand_shard.pbs")
  echo "  shard $i -> $jid"
  SHARD_JOBS+=("$jid")
done

DEP="afterok:$(IFS=:; echo "${SHARD_JOBS[*]}")"
MERGE_JOB=$(qsub "${PROJ[@]}" -q "$QSUB_QUEUE" \
  -N "mip_expand_merge" \
  -l "walltime=06:00:00,select=1:ncpus=1:mem=${SHARD_MEM}" \
  -W "depend=$DEP" \
  -v "SYNTH_POP=$SYNTH_POP,BASE_POOL=$POOL,SHARD_DIR=$SHARD_DIR,OUT=$OUT,CPU_ENV_NAME=$CPU_ENV_NAME,CPU_ENV_PREFIX=$CPU_ENV_PREFIX" \
  -o "$LOG_DIR/merge.log" -e "$LOG_DIR/merge.err" \
  "$SYNTH_POP/pbs/merge_expansion_shards.pbs")
echo "  merge -> $MERGE_JOB"

FIT_JOB="$MERGE_JOB"
FIT_POOL="$OUT"
if [[ "$ATTACH_HHTS" == "1" ]]; then
  [[ -s "$HHTS_CENSUS" ]] || { echo "ERROR: HHTS census not found: $HHTS_CENSUS" >&2; exit 1; }
  ATTACH_JOB=$(qsub "${PROJ[@]}" -q "$QSUB_QUEUE" \
    -N "hhts_attach" \
    -l "walltime=06:00:00,select=1:ncpus=1:mem=${SHARD_MEM}" \
    -W "depend=afterok:$MERGE_JOB" \
    -v "SYNTH_POP=$SYNTH_POP,POOL=$OUT,HHTS_CENSUS=$HHTS_CENSUS,OUT=$EXPANDED_HHTS,REPORT=$EXPANDED_HHTS.attach_report.json,CPU_ENV_NAME=$CPU_ENV_NAME,CPU_ENV_PREFIX=$CPU_ENV_PREFIX" \
    -o "$LOG_DIR/attach.log" -e "$LOG_DIR/attach.err" \
    "$SYNTH_POP/pbs/attach_hhts_to_pool.pbs")
  echo "  HHTS attach -> $ATTACH_JOB"
  FIT_JOB="$ATTACH_JOB"
  FIT_POOL="$EXPANDED_HHTS"
fi

if [[ "${SUBMIT_FITS:-0}" == "1" ]]; then
  EXPANDED_JOB="$FIT_JOB" POOL="$FIT_POOL" bash "$SYNTH_POP/resubmit_expanded_fitters.sh"
  DEP="$FIT_JOB" POOL_OVERRIDE="$FIT_POOL" N_CHUNKS=20 \
    bash "$SYNTH_POP/submit_cp_ipu.sh"
fi

echo "Done. Merge output: $OUT"
echo "      fit pool: $FIT_POOL"
echo "Check: qstat -u $USER"

#!/usr/bin/env bash
# Submit one SynRBL sbatch job per chunk file.
#
# Usage (from repo root):
#   bash submit_synrbl_chunks.sh \
#     --chunks-dir results/SynRBL/uspto_uncurated_CanonicalReaction_processed_chunks/ \
#     --output-dir results/SynRBL/uspto_uncurated_CanonicalReaction_results/
#
#   bash submit_synrbl_chunks.sh \
#     --chunks-dir results/SynRBL/uspto_uncurated_InputReaction_processed_chunks/ \
#     --output-dir results/SynRBL/uspto_uncurated_InputReaction_results/
#
# Each chunk is submitted as an independent job whose output CSV lands in
# <output-dir>/<chunk_stem>_results.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHUNKS_DIR=""
OUTPUT_DIR=""
SYNRBL_SIF=""
DRY_RUN=0

usage() {
  cat <<EOF
Usage: bash submit_synrbl_chunks.sh [options]

Options:
  --chunks-dir DIR    Directory containing chunk CSV files (required)
  --output-dir DIR    Directory where result CSVs will be written (required)
  --script-dir DIR    Repo root on the cluster (default: ${SCRIPT_DIR})
  --sif PATH          SynRBL SIF (default: <script-dir>/models/SynRBL/synrbl_env.sif)
  --dry-run           Print sbatch commands without submitting
  -h, --help          Show this help
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunks-dir)  CHUNKS_DIR="$2";  shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
    --script-dir)  SCRIPT_DIR="$2";  shift 2 ;;
    --sif)         SYNRBL_SIF="$2";  shift 2 ;;
    --dry-run)     DRY_RUN=1;        shift   ;;
    -h|--help)     usage ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$CHUNKS_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Error: --chunks-dir and --output-dir are required." >&2
  exit 1
fi

# Resolve to absolute paths so sbatch jobs work regardless of working directory.
CHUNKS_DIR="$(cd "$CHUNKS_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/models/SynRBL/run_synrbl.sbatch"
SYNRBL_SIF="${SYNRBL_SIF:-${SCRIPT_DIR}/models/SynRBL/synrbl_env.sif}"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "SBATCH script not found: $SBATCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$SYNRBL_SIF" ]]; then
  echo "SynRBL SIF not found: $SYNRBL_SIF" >&2
  exit 1
fi

shopt -s nullglob
chunks=("${CHUNKS_DIR}"/*.csv)

if [[ ${#chunks[@]} -eq 0 ]]; then
  echo "No CSV files found in ${CHUNKS_DIR}" >&2
  exit 1
fi

echo "Found ${#chunks[@]} chunk(s) in ${CHUNKS_DIR}"
echo "Results will be written to ${OUTPUT_DIR}"
echo ""

for chunk in "${chunks[@]}"; do
  stem="$(basename "${chunk}" .csv)"
  output_csv="${OUTPUT_DIR}/${stem}_results.csv"

  cmd=(
    sbatch
    --job-name="SynRBL_${stem}"
    --export="ALL,CHUNK_INPUT=${chunk},CHUNK_OUTPUT=${output_csv},SCRIPT_DIR=${SCRIPT_DIR},SYNRBL_SIF=${SYNRBL_SIF}"
    --output="${OUTPUT_DIR}/${stem}_%j.log"
    --error="${OUTPUT_DIR}/${stem}_%j.err"
    "$SBATCH_SCRIPT"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] ${cmd[*]}"
  else
    "${cmd[@]}"
    echo "Submitted: ${stem}"
  fi
done

echo ""
echo "Done."

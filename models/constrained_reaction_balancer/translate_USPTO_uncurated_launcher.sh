#!/bin/bash
# Splits USPTO_uncurated/prepared/test_src.txt into chunks and submits a
# SLURM job array (one GPU job per chunk) using translate_USPTO_uncurated_chunk.sh.
#
# Usage (from models/constrained_reaction_balancer/):
#   bash translate_USPTO_uncurated_launcher.sh [chunk_size]
#
# Arguments:
#   chunk_size   Lines per chunk (default: 20000)
#
# Outputs:
#   USPTO_uncurated/chunks/test_src_chunk_NNNN    — tokenised source chunks
#   USPTO_uncurated/chunks/test_index_chunk_NNNN  — corresponding index slices
#   results/USPTO_uncurated/chunk_NNNN/           — per-chunk translation output
#   results/USPTO_uncurated/logs/                 — SLURM stdout/stderr logs

set -euo pipefail

CHUNK_SIZE="${1:-20000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNCURATED_DIR="$SCRIPT_DIR/USPTO_uncurated"
PREPARED_DIR="$UNCURATED_DIR/prepared"
CHUNKS_DIR="$UNCURATED_DIR/chunks"
RESULTS_DIR="$SCRIPT_DIR/results/USPTO_uncurated"
LOGS_DIR="$RESULTS_DIR/logs"

SRC_FILE="$PREPARED_DIR/test_src.txt"
IDX_FILE="$PREPARED_DIR/test_index.csv"

# --------------------------------------------------------------------------- #
# Validate inputs
# --------------------------------------------------------------------------- #
if [ ! -f "$SRC_FILE" ]; then
    echo "ERROR: Source file not found: $SRC_FILE"
    echo "Run USPTO_uncurated/prepare_testset.py first."
    exit 1
fi

TOTAL=$(wc -l < "$SRC_FILE")
echo "Total input reactions : $TOTAL"
echo "Chunk size            : $CHUNK_SIZE"

# --------------------------------------------------------------------------- #
# (Re)create chunk directory — remove old chunks to avoid stale data
# --------------------------------------------------------------------------- #
if [ -d "$CHUNKS_DIR" ]; then
    echo "Removing existing chunks directory: $CHUNKS_DIR"
    rm -rf "$CHUNKS_DIR"
fi
mkdir -p "$CHUNKS_DIR"
mkdir -p "$LOGS_DIR"

# --------------------------------------------------------------------------- #
# Split tokenised source
# --------------------------------------------------------------------------- #
split -l "$CHUNK_SIZE" -d --suffix-length=4 "$SRC_FILE" "$CHUNKS_DIR/test_src_chunk_"
N_SRC=$(ls "$CHUNKS_DIR"/test_src_chunk_* | wc -l)
echo "Created $N_SRC source chunks"

# --------------------------------------------------------------------------- #
# Split index CSV (keep header on every slice for easy reassembly)
# --------------------------------------------------------------------------- #
if [ -f "$IDX_FILE" ]; then
    IDX_TOTAL=$(wc -l < "$IDX_FILE")
    IDX_ROWS=$((IDX_TOTAL - 1))
    echo "Index rows (excluding header): $IDX_ROWS"
    if [ "$IDX_ROWS" -le 0 ]; then
        echo "ERROR: $IDX_FILE contains no data rows (only header)."
        exit 1
    fi

    IDX_HEADER=$(head -1 "$IDX_FILE")
    tail -n +2 "$IDX_FILE" | split -l "$CHUNK_SIZE" -d --suffix-length=4 - "$CHUNKS_DIR/test_index_chunk_"
    if ! ls "$CHUNKS_DIR"/test_index_chunk_* >/dev/null 2>&1; then
        echo "ERROR: No index chunks were produced from $IDX_FILE"
        exit 1
    fi
    for f in "$CHUNKS_DIR"/test_index_chunk_*; do
        { echo "$IDX_HEADER"; cat "$f"; } > "${f}.tmp" && mv "${f}.tmp" "$f"
    done
    echo "Created $(ls "$CHUNKS_DIR"/test_index_chunk_* | wc -l) index chunks"
else
    echo "WARNING: Index file not found, skipping index split: $IDX_FILE"
fi

# --------------------------------------------------------------------------- #
# Submit SLURM job array
# --------------------------------------------------------------------------- #
NCHUNKS=$(ls "$CHUNKS_DIR"/test_src_chunk_* | wc -l)
if [ "$NCHUNKS" -eq 0 ]; then
    echo "ERROR: No chunks created — nothing to submit."
    exit 1
fi

ARRAY_END=$((NCHUNKS - 1))
CHUNK_SCRIPT="$SCRIPT_DIR/translate_USPTO_uncurated_chunk.sh"

if [ ! -f "$CHUNK_SCRIPT" ]; then
    echo "ERROR: Chunk script not found: $CHUNK_SCRIPT"
    exit 1
fi

echo "Submitting SLURM array job: 0-$ARRAY_END  ($NCHUNKS tasks)"
sbatch --chdir="$SCRIPT_DIR" --array=0-"$ARRAY_END" "$CHUNK_SCRIPT"

#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --output=results/USPTO_uncurated/logs/chunk_%a_%j.out
#SBATCH --error=results/USPTO_uncurated/logs/chunk_%a_%j.err

## short <= 4h | Each 20k chunk at beam_size=5 takes ~2-4h on a single GPU.

set -euo pipefail

module use /opt/insy/modulefiles
module load cuda/11.8

export APPTAINER_NAME="CRB_env.sif"

echo -n "This script is running on "
hostname
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

checkpointName="model_step_50000"
beamSize=5
nBest=5

SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [[ -d "$SCRIPT_DIR/models/constrained_reaction_balancer" ]]; then
    SCRIPT_DIR="$SCRIPT_DIR/models/constrained_reaction_balancer"
fi
CHUNKS_DIR="$SCRIPT_DIR/USPTO_uncurated/chunks"
RESULTS_DIR="$SCRIPT_DIR/results/USPTO_uncurated"

if [ ! -f "$SCRIPT_DIR/$APPTAINER_NAME" ]; then
    echo "ERROR: Missing apptainer image: $SCRIPT_DIR/$APPTAINER_NAME"
    exit 1
fi
if [ ! -f "$SCRIPT_DIR/translate.py" ]; then
    echo "ERROR: Missing translate.py in $SCRIPT_DIR"
    exit 1
fi

# Zero-padded chunk index matching the split output names
CHUNK_ID=$(printf "%04d" "$SLURM_ARRAY_TASK_ID")
SRC_CHUNK="$CHUNKS_DIR/test_src_chunk_$CHUNK_ID"

if [ ! -f "$SRC_CHUNK" ]; then
    echo "ERROR: Missing chunk file: $SRC_CHUNK"
    exit 1
fi

modelPath="models/checkpoints_pretrained/Stereo_separated/random_seed42_rep0/${checkpointName}"
if [ ! -f "$SCRIPT_DIR/$modelPath.pt" ]; then
    echo "ERROR: Missing model checkpoint: $SCRIPT_DIR/$modelPath.pt"
    exit 1
fi

chunkResultsDir="$RESULTS_DIR/chunk_$CHUNK_ID"
mkdir -p "$chunkResultsDir"
outputPath="$chunkResultsDir/beamSize${beamSize}_nBest${nBest}.txt"

echo "Chunk:       $CHUNK_ID"
echo "Source:      $SRC_CHUNK"
echo "Model:       $SCRIPT_DIR/$modelPath.pt"
echo "Output:      $outputPath"
echo "Beam size:   $beamSize  |  n-best: $nBest"

apptainer exec --nv "$SCRIPT_DIR/$APPTAINER_NAME" python "$SCRIPT_DIR/translate.py" \
    -model "$SCRIPT_DIR/$modelPath.pt" \
    -src "$SRC_CHUNK" \
    -output "$outputPath" \
    -constrained_atom_balance \
    -constrained_charge_balance \
    -batch_size 64 \
    -replace_unk \
    -max_length 200 \
    -fast \
    -beam_size "$beamSize" \
    -n_best "$nBest" \
    -log_probs

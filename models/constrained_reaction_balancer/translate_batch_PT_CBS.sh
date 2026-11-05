#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --mail-type=END

## 10k steps takes roughly an hour | short<=4h | medium<=36h | long=7days

# Load modules
module use /opt/insy/modulefiles
module load cuda/11.8

export APPTAINER_NAME="CRB_env.sif"

# Debug info
echo -n "This script is running on "
hostname

# Set variables
checkpointName="_step_50000"
beamSize=5
nBest=5

# Split/run configuration (override with environment variables when submitting jobs).
splitType="cluster"
splitSeed="42"
splitRepetition="1"
splitSuffix="${splitType}_seed${splitSeed}_rep${splitRepetition}"

dataPathBase="Template_curated/benchmark/prepared"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

legacyModelPath="models/checkpoints_pretrained/Stereo_separated$checkpointName"
splitModelPath="models/checkpoints_pretrained/Stereo_separated/${splitSuffix}/model${checkpointName}"
if [ -f "$SCRIPT_DIR/$splitModelPath.pt" ]; then
    modelPath="$splitModelPath"
else
    modelPath="$legacyModelPath"
fi

srcSplitPath="$dataPathBase/$splitSuffix/valid_src.txt"
srcLegacyPath="$dataPathBase/valid_src.txt"
if [ -f "$SCRIPT_DIR/$srcSplitPath" ]; then
    srcTestPath="$srcSplitPath"
else
    srcTestPath="$srcLegacyPath"
fi

outputDir="results/valid_results/CBS_PT/${splitSuffix}"
mkdir -p "$SCRIPT_DIR/$outputDir"
outputPath="$outputDir/beamSize${beamSize}_nBest${nBest}.txt"

if [ ! -f "$SCRIPT_DIR/$srcTestPath" ]; then
    echo "Missing source file: $SCRIPT_DIR/$srcTestPath"
    exit 1
fi
if [ ! -f "$SCRIPT_DIR/$modelPath.pt" ]; then
    echo "Missing model checkpoint: $SCRIPT_DIR/$modelPath.pt"
    exit 1
fi

echo "Running constrained translation with split suffix: ${splitSuffix}"
echo "Source file: $SCRIPT_DIR/$srcTestPath"
echo "Model file: $SCRIPT_DIR/$modelPath.pt"
echo "Output file: $SCRIPT_DIR/$outputPath"

# Run job with apptainer environment
apptainer exec --nv $SCRIPT_DIR/$APPTAINER_NAME python $SCRIPT_DIR/translate.py \
                    -model $SCRIPT_DIR/$modelPath.pt \
                    -src $SCRIPT_DIR/$srcTestPath \
                    -output $SCRIPT_DIR/$outputPath -constrained_atom_balance -constrained_charge_balance \
                    -batch_size 64 -replace_unk -max_length 200 -fast -beam_size $beamSize -n_best $nBest

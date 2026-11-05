#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
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

#Take random split trained model
splitType="random"
splitSeed="42"
splitRepetition="1"
splitSuffix="${splitType}_seed${splitSeed}_rep${splitRepetition}"

dataPathBase="SynRXN_complex/prepared"

# sbatch executes a spool copy; resolve from the submission directory instead.
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [[ -d "$SCRIPT_DIR/models/constrained_reaction_balancer" ]]; then
    SCRIPT_DIR="$SCRIPT_DIR/models/constrained_reaction_balancer"
fi
if [[ ! -f "$SCRIPT_DIR/translate.py" ]]; then
    echo "Cannot locate CRB code; submit from the repository root or set SCRIPT_DIR." >&2
    exit 1
fi

legacyModelPath="models/checkpoints_pretrained/Stereo_separated$checkpointName"
splitModelPath="models/checkpoints_pretrained/Stereo_separated/${splitSuffix}/model${checkpointName}"
if [ -f "$SCRIPT_DIR/$splitModelPath.pt" ]; then
    modelPath="$splitModelPath"
else
    modelPath="$legacyModelPath"
fi

srcSplitPath="$dataPathBase/$splitSuffix/test_src.txt"
srcLegacyPath="$dataPathBase/test_src.txt"
if [ -f "$SCRIPT_DIR/$srcSplitPath" ]; then
    srcTestPath="$srcSplitPath"
else
    srcTestPath="$srcLegacyPath"
fi

outputDir="results/SynRXN/BS_PT/${splitSuffix}"
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
                    -output $SCRIPT_DIR/$outputPath \
                    -batch_size 64 -replace_unk -max_length 200 -fast -beam_size $beamSize -n_best $nBest

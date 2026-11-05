#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=03:00:00
#SBATCH --mail-type=END

## 10k steps takes roughly an hour | short<=4h | medium<=36h | long=7days

module use /opt/insy/modulefiles
module load cuda/11.8

export APPTAINER_NAME="CRB_env.sif"

echo -n "This script is running on "
hostname

checkpointName="_step_50000"
beamSize=5
nBest=5

# Arguments:
#   random <seed> <repetition>
#   group  <repetition>
#   extreme_ood <repetition>
splitType="$1"

if [ "$splitType" = "random" ]; then
    splitSeed="$2"
    splitRepetition="$3"

    if [ -z "$splitSeed" ] || [ -z "$splitRepetition" ]; then
        echo "Usage:"
        echo "  sbatch translate.sbatch random <splitSeed> <splitRepetition>"
        echo "  sbatch translate.sbatch group <splitRepetition>"
        echo "  sbatch translate.sbatch extreme_ood <splitRepetition>"
        exit 1
    fi

    splitSuffix="${splitType}_seed${splitSeed}_rep${splitRepetition}"

elif [ "$splitType" = "group" ]; then
    splitRepetition="$2"

    if [ -z "$splitRepetition" ]; then
        echo "Usage:"
        echo "  sbatch translate.sbatch random <splitSeed> <splitRepetition>"
        echo "  sbatch translate.sbatch group <splitRepetition>"
        echo "  sbatch translate.sbatch extreme_ood <splitRepetition>"
        exit 1
    fi

    splitSuffix="${splitType}_rep${splitRepetition}"

elif [ "$splitType" = "extreme_ood" ]; then
    splitRepetition="$2"

    if [ -z "$splitRepetition" ]; then
        echo "Usage:"
        echo "  sbatch translate.sbatch random <splitSeed> <splitRepetition>"
        echo "  sbatch translate.sbatch group <splitRepetition>"
        echo "  sbatch translate.sbatch extreme_ood <splitRepetition>"
        exit 1
    fi

    splitSuffix="${splitType}_rep${splitRepetition}"

else
    echo "Invalid splitType: $splitType"
    echo "Allowed values: random, group, extreme_ood"
    exit 1
fi

dataPathBase="Template_curated/benchmark/prepared"

# sbatch executes a spool copy; resolve from the submission directory instead.
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [[ -d "$SCRIPT_DIR/models/constrained_reaction_balancer" ]]; then
    SCRIPT_DIR="$SCRIPT_DIR/models/constrained_reaction_balancer"
fi
if [[ ! -f "$SCRIPT_DIR/translate.py" ]]; then
    echo "Cannot locate CRB code; submit from the repository root or set SCRIPT_DIR." >&2
    exit 1
fi

legacyModelPath="models/checkpoints_pretrained/Stereo_separated${checkpointName}"
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

outputDir="$SCRIPT_DIR/../../results/main/predictions/rb/${splitType}/${splitSuffix}"
mkdir -p "$outputDir"
outputPath="$outputDir/beamSize${beamSize}_nBest${nBest}.txt"

if [ ! -f "$SCRIPT_DIR/$srcTestPath" ]; then
    echo "Missing source file: $SCRIPT_DIR/$srcTestPath"
    exit 1
fi
if [ ! -f "$SCRIPT_DIR/$modelPath.pt" ]; then
    echo "Missing model checkpoint: $SCRIPT_DIR/$modelPath.pt"
    exit 1
fi

echo "Split type: $splitType"
if [ "$splitType" = "random" ]; then
    echo "Split seed: $splitSeed"
fi
echo "Split repetition: $splitRepetition"
echo "Running constrained translation with split suffix: ${splitSuffix}"
echo "Source file: $SCRIPT_DIR/$srcTestPath"
echo "Model file: $SCRIPT_DIR/$modelPath.pt"
echo "Output file: $outputPath"

apptainer exec --nv "$SCRIPT_DIR/$APPTAINER_NAME" python "$SCRIPT_DIR/translate.py" \
    -model "$SCRIPT_DIR/$modelPath.pt" \
    -src "$SCRIPT_DIR/$srcTestPath" \
    -output "$outputPath" \
    -batch_size 64 \
    -replace_unk \
    -max_length 200 \
    -fast \
    -beam_size "$beamSize" \
    -n_best "$nBest" \
    -log_probs

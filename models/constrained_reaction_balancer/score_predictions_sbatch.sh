#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:15:00
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/11.8

APPTAINER_IMAGE="CRB_env.sif"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
HOST_REPO_ROOT="${HOST_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -d "$HOST_REPO_ROOT/models/constrained_reaction_balancer" ]]; then
    HOST_REPO_ROOT="$(cd "$HOST_REPO_ROOT/../.." && pwd)"
fi
CONTAINER_REPO_ROOT="/repo"
HOST_CRB_DIR="${HOST_REPO_ROOT}/models/constrained_reaction_balancer"

echo -n "This script is running on "
hostname
echo "Submit dir: $SCRIPT_DIR"

beamSize=5
nBest=5
accuracyMetric="equivalence"
scoringMetric="${SCORING_METRIC:-v2}"

splitType="$1"
experimentType="$2"

if [ -z "$splitType" ] || [ -z "$experimentType" ]; then
    echo "Usage:"
    echo "  sbatch score_predictions.sbatch random <experiment_type> <seed> <rep> [accuracy_metric]"
    echo "  sbatch score_predictions.sbatch group  <experiment_type> <rep> [accuracy_metric]"
    exit 1
fi

if [ "$splitType" = "random" ]; then
    splitSeed="$3"
    splitRepetition="$4"

    if [ -z "$splitSeed" ] || [ -z "$splitRepetition" ]; then
        echo "Usage: sbatch score_predictions.sbatch random <experiment_type> <seed> <rep> [accuracy_metric]"
        exit 1
    fi
    if [ -n "$5" ]; then
        accuracyMetric="$5"
    fi

    splitSuffix="${splitType}_seed${splitSeed}_rep${splitRepetition}"

elif [ "$splitType" = "group" ]; then
    splitRepetition="$3"

    if [ -z "$splitRepetition" ]; then
        echo "Usage: sbatch score_predictions.sbatch group <experiment_type> <rep> [accuracy_metric]"
        exit 1
    fi
    if [ -n "$4" ]; then
        accuracyMetric="$4"
    fi

    splitSuffix="${splitType}_rep${splitRepetition}"

elif [ "$splitType" = "extreme_ood" ]; then
    splitRepetition="$3"

    if [ -z "$splitRepetition" ]; then
        echo "Usage: sbatch score_predictions.sbatch extreme_ood <experiment_type> <rep> [accuracy_metric]"
        exit 1
    fi
    if [ -n "$4" ]; then
        accuracyMetric="$4"
    fi

    splitSuffix="${splitType}_rep${splitRepetition}"

else
    echo "Invalid splitType: $splitType"
    exit 1
fi

if [ "$experimentType" = "BS_PT" ]; then
    modelDir="rb"
elif [ "$experimentType" = "CBS_PT" ]; then
    modelDir="crb"
else
    echo "Unsupported experiment type: $experimentType" >&2
    exit 1
fi
predictionsPath="results/main/predictions/${modelDir}/${splitType}/${splitSuffix}/beamSize${beamSize}_nBest${nBest}.txt"
top1Path="results/main/predictions/${modelDir}/${splitType}/${splitSuffix}/top1_predictions_with_stats.txt"

targetsPath="Template_curated/benchmark/prepared/${splitSuffix}/test_tgt.txt"
inputPath="Template_curated/benchmark/prepared/${splitSuffix}/test_src.txt"

# This is required if reactions with unknown tokens should be excluded
checkpointName="_step_50000.pt"
splitModelPath="models/checkpoints_pretrained/Stereo_separated/${splitSuffix}/model${checkpointName}"

# Checks
if [ ! -f "$HOST_REPO_ROOT/$predictionsPath" ]; then
    echo "Missing predictions file: $HOST_REPO_ROOT/$predictionsPath"
    exit 1
fi

if [ ! -f "$HOST_CRB_DIR/$targetsPath" ]; then
    echo "Missing targets file: $HOST_CRB_DIR/$targetsPath"
    exit 1
fi

if [ ! -f "$HOST_CRB_DIR/$inputPath" ]; then
    echo "Missing input file: $HOST_CRB_DIR/$inputPath"
    exit 1
fi

if [ ! -f "$HOST_CRB_DIR/$APPTAINER_IMAGE" ]; then
    echo "Missing apptainer image: $HOST_CRB_DIR/$APPTAINER_IMAGE"
    exit 1
fi

echo "Experiment type: $experimentType"
echo "Split type: $splitType"
if [ "$splitType" = "random" ]; then
    echo "Split seed: $splitSeed"
fi
echo "Split repetition: $splitRepetition"
echo "Split suffix: $splitSuffix"
echo "Predictions: $HOST_REPO_ROOT/$predictionsPath"
echo "Top1 output: $HOST_REPO_ROOT/$top1Path"
echo "Accuracy metric: $accuracyMetric"
echo "Bind mount: $HOST_REPO_ROOT -> $CONTAINER_REPO_ROOT"

apptainer exec \
    --bind "$HOST_REPO_ROOT:$CONTAINER_REPO_ROOT" \
    "$HOST_CRB_DIR/$APPTAINER_IMAGE" \
    python "$CONTAINER_REPO_ROOT/models/constrained_reaction_balancer/score_predictions.py" \
    --scoring-metric "$scoringMetric" \
    --output-dir "$CONTAINER_REPO_ROOT/results/local/model_scores/$scoringMetric/$modelDir/$splitSuffix" \
    -beam_size "$beamSize" \
    -invalid_smiles \
    -predictions "$CONTAINER_REPO_ROOT/$predictionsPath" \
    -targets "$CONTAINER_REPO_ROOT/models/constrained_reaction_balancer/$targetsPath" \
    -write_top1 "$CONTAINER_REPO_ROOT/$top1Path" \
    -inputs "$CONTAINER_REPO_ROOT/models/constrained_reaction_balancer/$inputPath" \
    --accuracy-metric "$accuracyMetric" \
    --exclude-unknown-tokens \
    --vocab-path "$CONTAINER_REPO_ROOT/models/constrained_reaction_balancer/$splitModelPath"

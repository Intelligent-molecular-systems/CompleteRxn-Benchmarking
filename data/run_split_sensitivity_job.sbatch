#!/bin/bash
# Run one split-sensitivity configuration.
#
# Usage (normally called by submit_split_sensitivity_jobs.sh):
#   sbatch data/run_split_sensitivity_job.sbatch group 0.55 100
#   sbatch data/run_split_sensitivity_job.sbatch extreme 0.55 100

#SBATCH --job-name=split_sensitivity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=short
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/split_sensitivity_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: sbatch $0 <group|extreme> <threshold> <knn_k>" >&2
    exit 2
fi

STRATEGY="$1"
THRESHOLD="$2"
KNN_K="$3"

if [[ "$STRATEGY" != "group" && "$STRATEGY" != "extreme" ]]; then
    echo "ERROR: strategy must be 'group' or 'extreme', got '$STRATEGY'" >&2
    exit 2
fi
if [[ ! "$THRESHOLD" =~ ^0\.[0-9]+$ ]]; then
    echo "ERROR: threshold must look like 0.55, got '$THRESHOLD'" >&2
    exit 2
fi
if [[ ! "$KNN_K" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: knn_k must be a positive integer, got '$KNN_K'" >&2
    exit 2
fi

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
BENCH="$PROJECT_ROOT/data/Template_curated/benchmark"
SIF="$PROJECT_ROOT/infra/data_processing_splits/drfp_data_env.sif"
PYTHON=(apptainer exec "$SIF" python)

INFILE="$BENCH/benchmark_final.csv"
FPS="$BENCH/benchmark_final_fps_CanonicalReaction.npy"
COL="CanonicalReaction"
FOLD_COL="ood_fold_RI"
GROUP_COL="leakage_group_id_RI"

for required_file in "$INFILE" "$FPS" "$SIF"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: $required_file" >&2
        exit 1
    fi
done

# A launcher can set a run-specific output root. The default is stable and is
# useful for individual manual jobs.
OUTPUT_ROOT="${SENSITIVITY_OUTPUT_ROOT:-$PROJECT_ROOT/sensitivity_sweeps}"
TAG="t${THRESHOLD/./}"
OUTDIR="$OUTPUT_ROOT/$STRATEGY"
mkdir -p "$OUTDIR"

if [[ "$STRATEGY" == "group" ]]; then
    SPLIT_SCRIPT="$PROJECT_ROOT/data/Template_curated/knngraph_group_splits.py"
    SPLIT="$OUTDIR/benchmark_final_group_splits_RI_${TAG}_k${KNN_K}.csv"
    GROUP_SUMMARY="$OUTDIR/benchmark_final_group_summary_RI_${TAG}_k${KNN_K}.csv"
    META=""
else
    SPLIT_SCRIPT="$PROJECT_ROOT/data/Template_curated/knngraph_extreme_ood_splits.py"
    SPLIT="$OUTDIR/benchmark_final_ood_splits_RI_${TAG}_k${KNN_K}.csv"
    GROUP_SUMMARY="$OUTDIR/benchmark_final_ood_group_summary_RI_${TAG}_k${KNN_K}.csv"
    META="$OUTDIR/benchmark_final_ood_splits_RI_${TAG}_k${KNN_K}_meta.json"
fi

METRICS="${SPLIT%.csv}_fold_metrics.csv"
PER_REACTION="${SPLIT%.csv}_per_reaction_metrics.csv"
PLOT="${SPLIT%.csv}_fold_metrics_tanimoto_distributions.png"

echo "====================================================================="
echo "Strategy           : $STRATEGY"
echo "Leakage threshold  : $THRESHOLD"
echo "kNN candidates (k) : $KNN_K"
echo "Output root        : $OUTPUT_ROOT"
echo "Job                : ${SLURM_JOB_ID}"
echo "====================================================================="

split_args=(
    --infile "$INFILE"
    --column "$COL"
    --outfile "$SPLIT"
    --group_summary_out "$GROUP_SUMMARY"
    --leakage_tanimoto "$THRESHOLD"
    --knn_k "$KNN_K"
)
if [[ "$STRATEGY" == "extreme" ]]; then
    split_args+=(--meta_out "$META")
fi

# A run-specific output directory is normally fresh. These checks also make a
# timed-out job safely resumable without overwriting complete artifacts.
split_complete=false
if [[ -f "$SPLIT" && -f "$GROUP_SUMMARY" ]]; then
    if [[ "$STRATEGY" == "group" || -f "$META" ]]; then
        split_complete=true
    fi
fi

if [[ "$split_complete" == true ]]; then
    echo "[skip] complete split artifacts already exist"
else
    "${PYTHON[@]}" "$SPLIT_SCRIPT" "${split_args[@]}"
fi

if [[ -f "$METRICS" && -f "$PER_REACTION" && -f "$PLOT" ]]; then
    echo "[skip] complete analysis artifacts already exist"
else
    "${PYTHON[@]}" "$PROJECT_ROOT/data/Template_curated/analyze_ood_splits.py" \
        --infile "$SPLIT" \
        --fps_file "$FPS" \
        --column "$COL" \
        --fold_column "$FOLD_COL" \
        --group_column "$GROUP_COL" \
        --outfile "$METRICS" \
        --per_reaction_out "$PER_REACTION"
fi

echo "Done: $STRATEGY, threshold=$THRESHOLD, k=$KNN_K"

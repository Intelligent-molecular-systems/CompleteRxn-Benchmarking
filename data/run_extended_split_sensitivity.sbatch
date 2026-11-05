#!/bin/bash
# Extended split sensitivity sweep (one factor at a time).
# Tasks 0-1 add threshold 0.50 at k=100; tasks 2-5 vary k at threshold 0.55.
# Submit from the repository root after running: mkdir -p logs

#SBATCH --job-name=split_sensitivity
#SBATCH --array=0-5%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=short
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/split_sensitivity_%A_%a.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Slurm copies batch scripts into /var/spool/slurmd before execution, so
# BASH_SOURCE points at the spool copy rather than the repository.  Resolve the
# project from the directory in which `sbatch` was invoked instead.
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

case "${SLURM_ARRAY_TASK_ID}" in
    0) STRATEGY="group"   ; THRESHOLD="0.50"; KNN_K="100" ;;
    1) STRATEGY="extreme" ; THRESHOLD="0.50"; KNN_K="100" ;;
    2) STRATEGY="group"   ; THRESHOLD="0.55"; KNN_K="50"  ;;
    3) STRATEGY="extreme" ; THRESHOLD="0.55"; KNN_K="50"  ;;
    4) STRATEGY="group"   ; THRESHOLD="0.55"; KNN_K="200" ;;
    5) STRATEGY="extreme" ; THRESHOLD="0.55"; KNN_K="200" ;;
    *) echo "ERROR: unsupported array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

TAG="t${THRESHOLD/./}"
OUTDIR="$BENCH/sensitivity_sweeps/$STRATEGY"
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
PLOT="${SPLIT%.csv}_fold_metrics_tanimoto_distributions.png"

echo "====================================================================="
echo "Strategy           : $STRATEGY"
echo "Leakage threshold  : $THRESHOLD"
echo "kNN candidates (k) : $KNN_K"
echo "Output             : $SPLIT"
echo "Job / array task   : ${SLURM_JOB_ID} / ${SLURM_ARRAY_TASK_ID}"
echo "====================================================================="

split_complete=false
if [[ -f "$SPLIT" && -f "$GROUP_SUMMARY" ]]; then
    if [[ "$STRATEGY" == "group" || -f "$META" ]]; then
        split_complete=true
    fi
fi

if [[ "$split_complete" == true ]]; then
    echo "[skip] split artifacts already exist"
else
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
    "${PYTHON[@]}" "$SPLIT_SCRIPT" "${split_args[@]}"
fi

if [[ -f "$METRICS" && -f "$PLOT" ]]; then
    echo "[skip] fold analysis already exists"
else
    "${PYTHON[@]}" "$PROJECT_ROOT/data/Template_curated/analyze_ood_splits.py" \
        --infile "$SPLIT" \
        --fps_file "$FPS" \
        --column "$COL" \
        --fold_column "$FOLD_COL" \
        --group_column "$GROUP_COL"
fi

echo "Done: $STRATEGY, threshold=$THRESHOLD, k=$KNN_K"

#!/bin/bash
# =============================================================================
# Sensitivity sweep — single Slurm job
#
# Loops over 4 leakage_tanimoto thresholds, runs the chosen split script,
# then runs the analysis. Skips any step whose output already exists.
#
# Usage:
#   sbatch data/run_sensitivity_sweep.sbatch
#
# Adjust SCRIPT_NAME and COL below before submitting.
# =============================================================================

#SBATCH --job-name=sensitivity_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=short
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/sensitivity_sweep_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# =============================================================================
# *** Adjust these two variables before submitting ***
#
# SCRIPT_NAME — one of:
#   knngraph_ood_identification
#   knngraph_group_splits
#   knngraph_extreme_ood_splits
#
# COL — one of:
#   CanonicalReaction
#   InputReaction_canonical
# =============================================================================

SCRIPT_NAME="knngraph_ood_identification"
COL="CanonicalReaction"

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF="$SCRIPT_DIR/infra/data_processing_splits/drfp_data_env.sif"
RUN="apptainer exec $SIF python"

BENCH="$SCRIPT_DIR/data/Template_curated/benchmark"
INFILE="$BENCH/benchmark_final.csv"
FPS_RI="$BENCH/benchmark_final_fps_CanonicalReaction.npy"
FPS_noRI="$BENCH/benchmark_final_fps_InputReaction_canonical.npy"

THRESHOLDS=(0.55 0.60 0.65 0.70)

# =============================================================================
# Derive variables from SCRIPT_NAME and COL
# =============================================================================

if [[ "$COL" == "CanonicalReaction" ]]; then
    FPS="$FPS_RI"
    FOLD_COL="ood_fold_RI"
    GROUP_COL="leakage_group_id_RI"
    SUFFIX="RI"
else
    FPS="$FPS_noRI"
    FOLD_COL="ood_fold_noRI"
    GROUP_COL="leakage_group_id_noRI"
    SUFFIX="noRI"
fi

case "$SCRIPT_NAME" in
    (knngraph_ood_identification)
        SCRIPT_PATH="$SCRIPT_DIR/data/${SCRIPT_NAME}.py"
        OUTDIR="$BENCH/ood_folds"
        FILE_PREFIX="benchmark_final_ood_splits"
        ;;
    (knngraph_group_splits)
        SCRIPT_PATH="$SCRIPT_DIR/data/Template_curated/${SCRIPT_NAME}.py"
        OUTDIR="$BENCH/rxn_groups"
        FILE_PREFIX="benchmark_final_group_splits"
        ;;
    (knngraph_extreme_ood_splits)
        SCRIPT_PATH="$SCRIPT_DIR/data/Template_curated/${SCRIPT_NAME}.py"
        OUTDIR="$BENCH/ood_folds_missing_stats"
        FILE_PREFIX="benchmark_final_ood_splits"
        ;;
    (*)
        echo "ERROR: unknown SCRIPT_NAME '$SCRIPT_NAME'" >&2
        exit 1
        ;;
esac

mkdir -p "$OUTDIR" "$SCRIPT_DIR/logs"

# =============================================================================
# Run
# =============================================================================

echo "====================================================================="
echo " Script : $SCRIPT_NAME"
echo " Column : $COL"
echo " Host   : $(hostname)   Job: $SLURM_JOB_ID"
echo "====================================================================="

for T in "${THRESHOLDS[@]}"; do

    TAG="t${T/./}"
    SPLIT="$OUTDIR/${FILE_PREFIX}_${SUFFIX}_${TAG}.csv"
    PLOT="${SPLIT%.csv}_fold_metrics_tanimoto_distributions.png"

    echo ""
    echo "  threshold=$T  →  $(basename "$SPLIT")"

    # Step 1: split
    if [[ -f "$SPLIT" ]]; then
        echo "    [skip] split already exists"
    else
        $RUN "$SCRIPT_PATH" \
            --infile            "$INFILE" \
            --column            "$COL"   \
            --leakage_tanimoto  "$T"
    fi

    # Step 2: analysis
    if [[ -f "$PLOT" ]]; then
        echo "    [skip] analysis already exists"
    else
        echo "    -> running analysis"
        $RUN "$SCRIPT_DIR/data/Template_curated/analyze_ood_splits.py" \
            --infile        "$SPLIT"     \
            --fps_file      "$FPS"       \
            --column        "$COL"       \
            --fold_column   "$FOLD_COL"  \
            --group_column  "$GROUP_COL"
    fi

done

echo ""
echo "Done."

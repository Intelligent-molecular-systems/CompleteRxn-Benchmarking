#!/usr/bin/env bash
# Sensitivity sweep over leakage_tanimoto thresholds for both split strategies
# and both reaction columns (CanonicalReaction / InputReaction_canonical).
#
# Usage:
#   conda activate drfp_clustering_env
#   bash data/run_sensitivity_sweep.sh
#
# Outputs:
#   data/Template_curated/benchmark/ood_folds/       <- knngraph_ood_identification splits
#   data/Template_curated/benchmark/rxn_groups/ <- knngraph_group_splits splits
# Each folder also contains the fold_metrics and per_reaction_metrics CSVs.

set -euo pipefail

# Helper: run split script only if output CSV missing; run analysis only if plot missing.
run_split()  {
    local SPLIT="$1"; shift
    if [[ -f "$SPLIT" ]]; then
        echo "  [skip] split already exists: $SPLIT"
    else
        "$@"
    fi
}

run_analysis() {
    local SPLIT="$1"; shift
    local PLOT="${SPLIT%.csv}_fold_metrics_tanimoto_distributions.png"
    if [[ -f "$PLOT" ]]; then
        echo "  [skip] analysis already exists: $PLOT"
    else
        "$@"
    fi
}

INFILE="data/Template_curated/benchmark/benchmark_final.csv"
FPS_RI="data/Template_curated/benchmark/benchmark_final_fps_CanonicalReaction.npy"
FPS_noRI="data/Template_curated/benchmark/benchmark_final_fps_InputReaction_canonical.npy"

THRESHOLDS=(0.55 0.60 0.65 0.70)
RXN_COLUMNS=(CanonicalReaction InputReaction_canonical)

# ---------------------------------------------------------------------------
# 1. OOD identification splits  (knngraph_ood_identification.py)
# ---------------------------------------------------------------------------
echo "============================================================"
echo " OOD identification splits"
echo "============================================================"
for COL in "${RXN_COLUMNS[@]}"; do
    FPS=$([ "$COL" = "CanonicalReaction" ] && echo "$FPS_RI" || echo "$FPS_noRI")
    FOLD_COL=$([ "$COL" = "CanonicalReaction" ] && echo "ood_fold_RI" || echo "ood_fold_noRI")
    GROUP_COL=$([ "$COL" = "CanonicalReaction" ] && echo "leakage_group_id_RI" || echo "leakage_group_id_noRI")
    SUFFIX=$([ "$COL" = "CanonicalReaction" ] && echo "RI" || echo "noRI")

    for T in "${THRESHOLDS[@]}"; do
        TAG=$(python3 -c "print(f't{int(round($T*100)):03d}')")
        SPLIT="data/Template_curated/benchmark/ood_folds/benchmark_final_ood_splits_${SUFFIX}_${TAG}.csv"

        echo ""
        echo "--- OOD | $COL | leakage_tanimoto=$T ---"
        run_split "$SPLIT" \
            python3 data/knngraph_ood_identification.py \
                --infile   "$INFILE" \
                --column   "$COL" \
                --leakage_tanimoto "$T"

        echo "  -> Analyzing $SPLIT"
        run_analysis "$SPLIT" \
            python3 data/Template_curated/analyze_ood_splits.py \
                --infile      "$SPLIT" \
                --fps_file    "$FPS" \
                --column      "$COL" \
                --fold_column "$FOLD_COL" \
                --group_column "$GROUP_COL"
    done
done

# ---------------------------------------------------------------------------
# 2. Reaction group splits  (knngraph_group_splits.py)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Reaction group splits"
echo "============================================================"
for COL in "${RXN_COLUMNS[@]}"; do
    FPS=$([ "$COL" = "CanonicalReaction" ] && echo "$FPS_RI" || echo "$FPS_noRI")
    FOLD_COL=$([ "$COL" = "CanonicalReaction" ] && echo "ood_fold_RI" || echo "ood_fold_noRI")
    GROUP_COL=$([ "$COL" = "CanonicalReaction" ] && echo "leakage_group_id_RI" || echo "leakage_group_id_noRI")
    SUFFIX=$([ "$COL" = "CanonicalReaction" ] && echo "RI" || echo "noRI")

    for T in "${THRESHOLDS[@]}"; do
        TAG=$(python3 -c "print(f't{int(round($T*100)):03d}')")
        SPLIT="data/Template_curated/benchmark/rxn_groups/benchmark_final_group_splits_${SUFFIX}_${TAG}.csv"

        echo ""
        echo "--- Group | $COL | leakage_tanimoto=$T ---"
        run_split "$SPLIT" \
            python3 data/Template_curated/knngraph_group_splits.py \
                --infile   "$INFILE" \
                --column   "$COL" \
                --leakage_tanimoto "$T"

        echo "  -> Analyzing $SPLIT"
        run_analysis "$SPLIT" \
            python3 data/Template_curated/analyze_ood_splits.py \
                --infile      "$SPLIT" \
                --fps_file    "$FPS" \
                --column      "$COL" \
                --fold_column "$FOLD_COL" \
                --group_column "$GROUP_COL"
    done
done

# ---------------------------------------------------------------------------
# 3. OOD + missing-stats splits  (knngraph_extreme_ood_splits.py)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " OOD + missing-stats splits"
echo "============================================================"
for COL in "${RXN_COLUMNS[@]}"; do
    FPS=$([ "$COL" = "CanonicalReaction" ] && echo "$FPS_RI" || echo "$FPS_noRI")
    FOLD_COL=$([ "$COL" = "CanonicalReaction" ] && echo "ood_fold_RI" || echo "ood_fold_noRI")
    GROUP_COL=$([ "$COL" = "CanonicalReaction" ] && echo "leakage_group_id_RI" || echo "leakage_group_id_noRI")
    SUFFIX=$([ "$COL" = "CanonicalReaction" ] && echo "RI" || echo "noRI")

    for T in "${THRESHOLDS[@]}"; do
        TAG=$(python3 -c "print(f't{int(round($T*100)):03d}')")
        SPLIT="data/Template_curated/benchmark/ood_folds_missing_stats/benchmark_final_ood_splits_${SUFFIX}_${TAG}.csv"

        echo ""
        echo "--- MissingStats | $COL | leakage_tanimoto=$T ---"
        run_split "$SPLIT" \
            python3 data/Template_curated/knngraph_extreme_ood_splits.py \
                --infile   "$INFILE" \
                --column   "$COL" \
                --leakage_tanimoto "$T"

        echo "  -> Analyzing $SPLIT"
        run_analysis "$SPLIT" \
            python3 data/Template_curated/analyze_ood_splits.py \
                --infile      "$SPLIT" \
                --fps_file    "$FPS" \
                --column      "$COL" \
                --fold_column "$FOLD_COL" \
                --group_column "$GROUP_COL"
    done
done

echo ""
echo "Done. All splits and metrics saved under:"
echo "  data/Template_curated/benchmark/ood_folds/"
echo "  data/Template_curated/benchmark/rxn_groups/"
echo "  data/Template_curated/benchmark/ood_folds_missing_stats/"

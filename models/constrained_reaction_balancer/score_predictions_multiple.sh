#!/bin/bash
cd "$(dirname "$0")" || exit 1

SBATCH_SCRIPT="score_predictions_sbatch.sh"

# CHANGE THIS MANUALLY
experimentType="BS_PT"
accuracyMetric="deep_pragmatic_equivalence"

# -------------------------
# RANDOM SPLITS
# -------------------------
seeds=(42 43 44 45 46)

for i in "${!seeds[@]}"; do
    seed="${seeds[$i]}"
    rep="$i"
    echo "Submitting random scoring: $experimentType seed=$seed rep=$rep metric=$accuracyMetric"
    sbatch "$SBATCH_SCRIPT" random "$experimentType" "$seed" "$rep" "$accuracyMetric"
done

# -------------------------
# GROUP SPLITS
# -------------------------
for rep in 0 1 2 3 4; do
    echo "Submitting group scoring: $experimentType rep=$rep metric=$accuracyMetric"
    sbatch "$SBATCH_SCRIPT" group "$experimentType" "$rep" "$accuracyMetric"
done

# -------------------------
# Extreme ood
# -------------------------
for rep in 0 1 2 3 4; do
    echo "Submitting extreme_ood scoring: $experimentType rep=$rep metric=$accuracyMetric"
    sbatch "$SBATCH_SCRIPT" extreme_ood "$experimentType" "$rep" "$accuracyMetric"
done

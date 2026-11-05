#!/usr/bin/env bash
# Submit the complete split sensitivity study as separate Slurm jobs.
#
# Design:
#   - threshold sweep: t={0.50,0.55,0.60,0.65,0.70} at k=100
#   - k sweep: k={50,100,200} at t=0.55
#   - both group and extreme-OOD strategies
# The shared baseline (t=0.55, k=100) is submitted only once per strategy.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

WORKER="data/run_split_sensitivity_job.sbatch"
THRESHOLDS=(0.50 0.55 0.60 0.65 0.70)
K_VALUES=(50 200)
RUN_LABEL="${1:-rerun_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="$PROJECT_ROOT/sensitivity_sweeps/$RUN_LABEL"

mkdir -p logs "$OUTPUT_ROOT"

submit_pair() {
    local threshold="$1"
    local knn_k="$2"
    local group_result group_job extreme_result

    group_result=$(sbatch \
        --job-name="sens_g_t${threshold/./}_k${knn_k}" \
        --export="ALL,SENSITIVITY_OUTPUT_ROOT=$OUTPUT_ROOT" \
        "$WORKER" group "$threshold" "$knn_k")
    group_job="${group_result##* }"
    echo "$group_result  group t=$threshold k=$knn_k"

    # The extreme job uses the same k-specific caches. Waiting for the group
    # job prevents concurrent creation of those cache files.
    extreme_result=$(sbatch \
        --job-name="sens_x_t${threshold/./}_k${knn_k}" \
        --dependency="afterok:$group_job" \
        --export="ALL,SENSITIVITY_OUTPUT_ROOT=$OUTPUT_ROOT" \
        "$WORKER" extreme "$threshold" "$knn_k")
    echo "$extreme_result  extreme t=$threshold k=$knn_k (after $group_job)"
}

echo "Submitting sensitivity sweep"
echo "Run label  : $RUN_LABEL"
echo "Output root: $OUTPUT_ROOT"

# Threshold sweep at the paper k.
for threshold in "${THRESHOLDS[@]}"; do
    submit_pair "$threshold" 100
done

# Additional k values at the paper threshold. k=100 was submitted above.
for knn_k in "${K_VALUES[@]}"; do
    submit_pair 0.55 "$knn_k"
done

echo
echo "Submitted 14 separate jobs (7 group + 7 dependent extreme)."
echo "Monitor with: squeue -u \"$USER\""
echo "Results will be written under: $OUTPUT_ROOT"

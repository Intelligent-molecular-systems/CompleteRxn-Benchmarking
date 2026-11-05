#!/bin/bash
cd "$(dirname "$0")" || exit 1

#SBATCH_SCRIPT="translate_batch_PT_CBS_parametrized.sh"
SBATCH_SCRIPT="translate_batch_PT_BS_parametrized.sh"


# Random splits
#seeds=(42 43 44 45 46)
#for i in "${!seeds[@]}"; do
#    seed="${seeds[$i]}"
#    rep="$i"
#    echo "Submitting random split: seed=$seed rep=$rep"
#    sbatch "$SBATCH_SCRIPT" random "$seed" "$rep"
#done

# Group splits
#for rep in 0 1 2 3 4; do
#    echo "Submitting group split: rep=$rep"
#    sbatch "$SBATCH_SCRIPT" group "$rep"
#done

# Extreme OOD splits
for rep in 0 1 2 3 4; do
    echo "Submitting extreme ood split: rep=$rep"
    sbatch "$SBATCH_SCRIPT" extreme_ood "$rep"
done

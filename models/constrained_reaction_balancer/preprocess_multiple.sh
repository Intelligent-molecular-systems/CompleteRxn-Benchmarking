#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:15:00
#SBATCH --mail-type=END
#SBATCH --array=0-4

module use /opt/insy/modulefiles
module load cuda/11.8

SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [[ -d "$SCRIPT_DIR/models/constrained_reaction_balancer" ]]; then
    SCRIPT_DIR="$SCRIPT_DIR/models/constrained_reaction_balancer"
fi
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$SCRIPT_DIR/CRB_env.sif}"

echo -n "This script is running on "
hostname

# Configure the split mode here: group | random | extreme_ood
SPLIT_MODE="extreme_ood"

splitSeeds=(42 43 44 45 46)
splitRepetition=${SLURM_ARRAY_TASK_ID}

case "$SPLIT_MODE" in
    group|extreme_ood)
        splitType="$SPLIT_MODE"
        splitSeed=""
        ;;
    random)
        splitType="random"
    splitSeed=${splitSeeds[$splitRepetition]}
        ;;
    *)
        echo "ERROR: Unsupported SPLIT_MODE='$SPLIT_MODE'. Use one of: group, random, extreme_ood" >&2
        exit 1
        ;;
esac


echo "SplitType: $splitType"
echo "Repetition: $splitRepetition"
if [[ -n "$splitSeed" ]]; then
    echo "Seed: $splitSeed"
fi

cmd=(
    apptainer exec "$APPTAINER_IMAGE" python "$SCRIPT_DIR/preprocess.py"
    --prepared-dir "$SCRIPT_DIR/Template_curated/benchmark/prepared"
    --split-type "$splitType"
    --split-repetition "$splitRepetition"
    --save-data-root "$SCRIPT_DIR/Template_curated/benchmark/preprocessed"
    --save-data-prefix data
    -src_seq_length 1000
    -tgt_seq_length 1000
    -src_vocab_size 1000
    -tgt_vocab_size 1000
    -share_vocab
)

if [[ -n "$splitSeed" ]]; then
    cmd+=(--split-seed "$splitSeed")
fi

"${cmd[@]}"

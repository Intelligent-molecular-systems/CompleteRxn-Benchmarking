#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=6:00:00
#SBATCH --mail-type=END
#SBATCH --array=0-4

## 10k steps takes roughly an hour | short<=4h | medium<=36h | long=7days

module use /opt/insy/modulefiles
module load cuda/11.8

SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [[ -d "$SCRIPT_DIR/models/constrained_reaction_balancer" ]]; then
    SCRIPT_DIR="$SCRIPT_DIR/models/constrained_reaction_balancer"
fi
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$SCRIPT_DIR/CRB_env.sif}"

echo -n "This script is running on "
hostname

splitType="extreme_ood"
splitRepetition=${SLURM_ARRAY_TASK_ID}

trainSeed=42

splitSuffix="${splitType}_rep${splitRepetition}"
dataPath="Template_curated/benchmark/preprocessed/data_${splitSuffix}"
savePath="models/checkpoints_pretrained/Stereo_separated/${splitSuffix}/model"

trainFromModel="STEREO_separated_augm_model_average_20.pt"
trainSteps=50000
validSteps=5000

mkdir -p "$SCRIPT_DIR/$(dirname "$savePath")"

if [ ! -f "$SCRIPT_DIR/${dataPath}.train.0.pt" ] && [ ! -f "$SCRIPT_DIR/${dataPath}.train.pt" ]; then
    echo "Could not find preprocessed training data for suffix '${splitSuffix}'."
    echo "Expected one of:"
    echo "  - $SCRIPT_DIR/${dataPath}.train.0.pt"
    echo "  - $SCRIPT_DIR/${dataPath}.train.pt"
    exit 1
fi

echo "SplitType: $splitType"
echo "Repetition: $splitRepetition"
echo "Running training with split suffix: ${splitSuffix}"
echo "Data prefix: $SCRIPT_DIR/$dataPath"
echo "Checkpoint prefix: $SCRIPT_DIR/$savePath"
echo "Training seed: ${trainSeed}"
echo "Container: $APPTAINER_IMAGE"

apptainer exec --nv "$APPTAINER_IMAGE" python "$SCRIPT_DIR/train.py" \
    -data "$SCRIPT_DIR/$dataPath" \
    -save_model "$SCRIPT_DIR/$savePath" \
    -seed "$trainSeed" \
    -gpu_ranks 0 \
    -save_checkpoint_steps 10000 \
    -keep_checkpoint 3 \
    -train_steps "$trainSteps" \
    -param_init 0 \
    -param_init_glorot \
    -max_generator_batches 32 \
    -valid_steps "$validSteps" \
    -batch_size 4096 \
    -batch_type tokens \
    -normalization tokens \
    -max_grad_norm 0 \
    -accum_count 4 \
    -optim adam \
    -adam_beta1 0.9 \
    -adam_beta2 0.998 \
    -decay_method noam \
    -warmup_steps 8000 \
    -learning_rate 2 \
    -label_smoothing 0.0 \
    -report_every 1000 \
    -layers 4 \
    -rnn_size 256 \
    -word_vec_size 256 \
    -encoder_type transformer \
    -decoder_type transformer \
    -dropout 0.1 \
    -position_encoding \
    -share_embeddings \
    -global_attention general \
    -global_attention_function softmax \
    -self_attn_type scaled-dot \
    -heads 8 \
    -transformer_ff 2048 \
    -train_from "$SCRIPT_DIR/models/$trainFromModel" \
    -valid_batch_size 16 \
    -gpu_verbose_level 0
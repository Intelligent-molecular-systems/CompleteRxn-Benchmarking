#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=6:00:00
#SBATCH --mail-type=END

## 10k steps takes roughly an hour | short<=4h | medium<=36h | long=7days

# Load modules
module use /opt/insy/modulefiles
module load cuda/11.8

export APPTAINER_NAME="CRB_env.sif"

# Debug info
echo -n "This script is running on "
hostname

# Set variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Split/run configuration (override with environment variables when submitting jobs).
# Example:
#   sbatch --export=ALL,SPLIT_TYPE=random,SPLIT_SEED=42,SPLIT_REPETITION=1 train_model_apptainer_from_pretrained.sh
splitType="cluster"
splitRepetition="1"
trainSeed="42"

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

echo "Running training with split suffix: ${splitSuffix}"
echo "Data prefix: $SCRIPT_DIR/$dataPath"
echo "Checkpoint prefix: $SCRIPT_DIR/$savePath"
echo "Training seed: ${trainSeed}"

# Run job with apptainer environment
#srun 
apptainer exec --nv $SCRIPT_DIR/$APPTAINER_NAME python $SCRIPT_DIR/train.py -data $SCRIPT_DIR/$dataPath \
                   -save_model $SCRIPT_DIR/$savePath \
                   -seed $trainSeed -gpu_ranks 0 -save_checkpoint_steps 5000 -keep_checkpoint 5 \
                   -train_steps $trainSteps -param_init 0  -param_init_glorot -max_generator_batches 32 \
                   -valid_steps $validSteps -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
                   -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                   -learning_rate 2 -label_smoothing 0.0 -report_every 1000 \
                   -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                   -dropout 0.1 -position_encoding -share_embeddings \
                   -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                   -heads 8 -transformer_ff 2048 \
                   -train_from $SCRIPT_DIR/models/$trainFromModel \
                   -valid_batch_size 16 -gpu_verbose_level 0
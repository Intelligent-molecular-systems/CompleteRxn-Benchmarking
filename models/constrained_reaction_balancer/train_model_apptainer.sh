#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
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
dataPath="Template_curated/benchmark/preprocessed/"
savePath="models/checkpoints/"
trainFromModel="STEREO_mixed_augm_model_average_20.pt"
trainSteps=100000


# Run job with apptainer environment
#srun 
apptainer exec --nv $SCRIPT_DIR/$APPTAINER_NAME python $SCRIPT_DIR/train.py -data $SCRIPT_DIR/$dataPath \
                   -save_model $SCRIPT_DIR/$savePath \
                   -seed 42 -gpu_ranks 0 -save_checkpoint_steps 5000 -keep_checkpoint 20 \
                   -train_steps $trainSteps -param_init 0  -param_init_glorot -max_generator_batches 32 \
                   -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
                   -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                   -learning_rate 2 -label_smoothing 0.0 -report_every 1000 \
                   -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                   -dropout 0.1 -position_encoding -share_embeddings \
                   -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                   -heads 8 -transformer_ff 2048 \
                   -valid_batch_size 16 -gpu_verbose_level 0

#!/bin/sh
#SBATCH --job-name=SynRBL_script
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=short
#SBATCH --mem=4GB
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL

 
# Debug info
echo -n "This script is running on "
hostname

# Repository and container paths can be overridden at submission time.
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SYNRBL_SIF="${SYNRBL_SIF:-${SCRIPT_DIR}/models/SynRBL/synrbl_env.sif}"

# Allow chunk-based submission to override input/output paths via env vars.
INPUT_CSV="${CHUNK_INPUT:?CHUNK_INPUT must identify one input chunk}"
OUTPUT_CSV="${CHUNK_OUTPUT:?CHUNK_OUTPUT must identify its result CSV}"

test -f "$SYNRBL_SIF"
test -f "$INPUT_CSV"
mkdir -p "$(dirname "$OUTPUT_CSV")"

# Run job with apptainer environment
srun apptainer exec "$SYNRBL_SIF" \
  python "$SCRIPT_DIR/models/SynRBL/predictor.py" "$INPUT_CSV" "$OUTPUT_CSV"

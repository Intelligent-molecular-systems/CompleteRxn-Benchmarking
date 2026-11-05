# Constrained Reaction Balancer (CRB / RB)

A pretrained Molecular Transformer, fine-tuned for reaction completion. The same fine-tuned model is used in two ways:

- **CRB** (Constrained Reaction Balancer): beam search constrained so that predictions are balanced for atoms and charge.
- **RB** (Reaction Balancer): plain beam search without those constraints.

Scripts and folder names use `CBS_PT` for CRB (constrained beam search, from the pretrained model) and `BS_PT` for RB (beam search, from the pretrained model). Saved predictions use `crb` and `rb`.

Both variants are trained from the same [IBM Molecular Transformer](https://ibm.ent.box.com/v/MolecularTransformerModels) pretrained checkpoint (`STEREO_separated_augm_model_average_20.pt`) and evaluated on all three splits (random, group, extreme OOD), each with 5 repetitions.

This folder includes code derived from [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer) and [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). Their MIT and BSD notices are retained in [LICENSE.upstream.md](LICENSE.upstream.md). They continue to apply to that code. Benchmark-specific additions include atom/charge masking, constrained decoding and reaction scoring. See `onmt/utils/masking.py`, `onmt/translate/translator.py`, `onmt/translate/beam.py`, `onmt/opts.py`, `onmt/utils/loss.py`, and the preparation, training and inference scripts.

Training and prediction were run on a SLURM cluster with GPUs. Scripts with `#SBATCH` headers are cluster jobs. The `run_multiple_translations.sh` and `score_predictions_multiple.sh` scripts submit several jobs at once.

---

## Environment setup

**Option A — Conda:**

```bash
conda env create -f infra/constrained_reaction_balancer/environment.yml
conda activate crb-env
```

**Option B — Apptainer (cluster / exact reproducibility):**

```bash
apptainer build models/constrained_reaction_balancer/CRB_env.sif \
  infra/constrained_reaction_balancer/apptainer_env.def
```

---

## Workflow

Submit the cluster commands below from the repository root unless a command says otherwise. The batch scripts use the submission directory to locate the repository. Set `SCRIPT_DIR` or `HOST_REPO_ROOT` when submitting from another location. The training and preprocessing scripts use `models/constrained_reaction_balancer/CRB_env.sif` by default and accept `APPTAINER_IMAGE` to use a different image.

### Step 1 - Prepare tokenized input files

Create the text files the model reads for training, validation, and testing:

```bash
python models/constrained_reaction_balancer/prepare_train_val_test.py \
  --split-type random --seed 42 --repetition 0
```

Repeat for all split types and repetitions, or use the sbatch array:

```bash
# Cluster: runs 5 array jobs (reps 0-4) for the chosen split type
# Edit SPLIT_MODE inside the script before submitting
sbatch models/constrained_reaction_balancer/preprocess_multiple.sh
```

### Step 2 - Preprocess for OpenNMT

Convert those text files into the `.pt` files used for training:

```bash
apptainer exec models/constrained_reaction_balancer/CRB_env.sif \
  python models/constrained_reaction_balancer/preprocess.py \
  --prepared-dir models/constrained_reaction_balancer/Template_curated/benchmark/prepared \
  --split-type random --split-seed 42 --split-repetition 0 \
  --save-data-root models/constrained_reaction_balancer/Template_curated/benchmark/preprocessed \
  --save-data-prefix data -share_vocab
```

The sbatch array script does this for all repetitions of one split type at once.

### Step 3 - Train

Continue training from the pretrained model (50,000 steps, about six hours per run on one GPU):

```bash
# Cluster: edit split variables at the top of the script, then
sbatch models/constrained_reaction_balancer/train_model_apptainer_from_pretrained_random.sh
# or _group.sh / _extreme_ood.sh for other split types
```

Checkpoints are saved every 10,000 steps under `models/checkpoints_pretrained/Stereo_separated/<repetition>/` inside this folder, where `<repetition>` is a folder name such as `random_seed42_rep0` or `group_rep3`. The paper uses the checkpoint at step 50,000.

| Training script | Use |
|---|---|
| `train_model_apptainer_from_pretrained_{random,group,extreme_ood}.sh` | Fine-tune from the pretrained model for one split type (used for the paper) |
| `train_model_apptainer_from_pretrained.sh` | Same, with the split chosen through `SPLIT_TYPE`, `SPLIT_SEED` and `SPLIT_REPETITION` |
| `train_model_apptainer_from_scratch_{random,group}.sh` | Train without the pretrained model (not used for the paper results) |
| `train_model_apptainer.sh` | Older training script without per-repetition data paths |

### Step 4 - Inference (translate)

Generate five candidate completions for each test reaction:

```bash
# Cluster: submit all repetitions for one split type
# Edit SBATCH_SCRIPT at the top to choose CBS_PT or BS_PT
bash models/constrained_reaction_balancer/run_multiple_translations.sh
```

Test predictions are written to `results/main/predictions/<rb|crb>/<split_type>/<repetition>/`.

| Translation script | Use |
|---|---|
| `translate_batch_PT_{CBS,BS}_parametrized.sh` | Benchmark test sets. Takes the split type and repetition (random splits also need a seed). Can also be run locally with Apptainer. |
| `translate_batch_PT_{CBS,BS}_synrxn.sh` | SynRXN Complex evaluation |
| `translate_USPTO_uncurated_launcher.sh` | Uncurated USPTO (see below) |
| `translate_batch_PT_{CBS,BS}.sh` | Validation sets only; writes under `results/valid_results/` in this folder |

### Step 5 - Score predictions

Compare each prediction with the reference reaction:

```bash
# Cluster: submit all split types and repetitions for one experiment
# Edit experimentType at the top of the script
bash models/constrained_reaction_balancer/score_predictions_multiple.sh
```

This writes a `beamSize5_nBest5_summary.txt` into each repetition folder.

### Step 6 - Main paper results

The `beamSize5_nBest5_summary.txt` files contain rounded values, so don't average them to build the main results table. Use the main scoring script instead:

```bash
bash analysis/reproduce_main_results.sh --rescore --output-root /tmp/completerxn-rescore
```

This rescores the saved predictions on the filtered test set and writes the main results table (Table 2) to `/tmp/completerxn-rescore/main/scores/main_table_filtered.{md,tex,csv}`. SynRBL is scored on the same reactions. Neither model is rerun. The files it needs are listed in the [prediction inventory](../../results/main/predictions/README.md); see also the [analysis index](../../analysis/README.md).

---

## Uncurated USPTO predictions

To generate CRB predictions for USPTO reactions outside the curated benchmark, split the input into smaller files and submit one cluster job per file. These reactions have no reference, so they are evaluated by balance and agreement rather than accuracy.

```bash
# Prepare tokenized source and index files
python models/constrained_reaction_balancer/USPTO_uncurated/prepare_testset.py

# Split into chunks and submit (from models/constrained_reaction_balancer/)
cd models/constrained_reaction_balancer
bash translate_USPTO_uncurated_launcher.sh
```

Results land in `models/constrained_reaction_balancer/results/USPTO_uncurated/chunk_NNNN/`. Merge them afterwards by running `analysis/uncurated/merge_crb_uncurated_chunks.py` from the repository root. The uncurated job uses the checkpoint fine-tuned on the random split, repetition 0 (`random_seed42_rep0`). The SynRXN scripts set their own checkpoint; keep that setting when reproducing the saved SynRXN predictions.

`score_predictions.py` and the scoring batch scripts report equivalence accuracy by default. To reproduce historical v1 results, add `--scoring-metric v1` and use a separate output directory. See [Historical results](../../analysis/README.md#historical-results).

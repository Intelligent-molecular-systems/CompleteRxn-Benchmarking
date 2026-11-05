# SynRBL

SynRBL completes reactions by finding molecular structures shared by the input and product sides. See the [SynRBL project](https://github.com/TieuLongPhan/SynRBL) for the method and software.

Generating predictions takes substantial CPU time and was done on a SLURM cluster. The scoring and summary steps can run locally.

---

## Environment Setup

**Option A — Conda:**

```bash
conda create --name synrbl-env python=3.11
conda activate synrbl-env
conda install tqdm
pip install synrbl==1.0.5
```

**Option B — Apptainer:**

```bash
apptainer build models/SynRBL/synrbl_env.sif infra/synrbl/apptainer_env.def
```

---

## Workflow

Submit the cluster commands from the repository root. The chunk launcher finds the repository from its own location and passes that path to each job. Use `--script-dir` if the cluster sees it at a different path.

### Step 1 — Preprocess

Prepare a reaction column without a separate reagent field. The main benchmark run uses `InputReaction_harmonized`, the same inputs as RB and CRB, without reagent identification. File names mark this run as `noRI`. The second command prepares the run with reagent identification (`RI`), which is only used for the comparison in the appendix (Table C1):

```bash
# Main benchmark (noRI): same inputs as RB and CRB
python models/SynRBL/preprocess.py \
  data/Template_curated/benchmark/benchmark_final.csv \
  results/SynRBL/benchmark_final_processed.csv \
  --reaction-col InputReaction_harmonized

# Reagent-identification comparison (RI): canonicalized original USPTO reaction
python models/SynRBL/preprocess.py \
  data/Template_curated/benchmark/benchmark_final.csv \
  results/SynRBL/benchmark_final_processed_canonicalOriginalReaction.csv \
  --reaction-col CanonicalOriginalReaction
```

### Step 2 — Run predictor

SynRBL attempts to complete each input reaction. The cluster job script needs input and output paths. For a large CSV, first split it into chunks, then submit one job per chunk:

```bash
python models/SynRBL/split_processed_into_chunks.py \
  results/SynRBL/benchmark_final_processed.csv \
  results/SynRBL/benchmark_final_processed_chunks/

bash models/SynRBL/submit_synrbl_chunks.sh \
  --chunks-dir results/SynRBL/benchmark_final_processed_chunks/ \
  --output-dir results/SynRBL/benchmark_final_results_chunks/ \
  --script-dir "$PWD"
```

Each output CSV contains the completed (or unchanged) reaction in `Reaction`, plus SynRBL's `solved`, `solved_by`, and `confidence` fields. The chunk launcher does not merge its outputs: combine the chunk CSVs in their original input order into one file before Step 3.

To reproduce the published scores without rerunning SynRBL, skip to Step 4 with the saved evaluated predictions listed in the [prediction inventory](../../results/main/predictions/README.md).

### Step 3 — Evaluate

Compare the completed predictions with the references for all splits and repetitions:

```bash
bash models/SynRBL/run_eval_all_splits.sh \
  --input-csv results/main/predictions/synrbl/benchmark_final_results_noRI_complete.csv \
  --output-dir results/main/predictions/synrbl/eval_all_splits_noRI_complete
```

This evaluates the full results CSV once, then summarizes the scores by split.

The evaluator uses `rxn_flower` as SynRBL's reference. SynRBL does not keep stereochemistry, so comparing it with a reference that does would count otherwise matching reactions as wrong. RB and CRB do keep stereochemistry and use `TargetReaction_with_stereo`. Both references describe the same molecules; they differ only in stereochemical annotation.

### Step 4 — Main paper results

The main results table reads the evaluated SynRBL predictions from `results/main/predictions/synrbl/eval_all_splits_noRI_complete/full_results.csv` (the output of Step 3). To recalculate the table:

```bash
bash analysis/reproduce_main_results.sh --rescore --output-root /tmp/completerxn-rescore
```

SynRBL is scored on the same filtered test set as RB and CRB. SynRBL must report `solved=True`, and its reaction must pass an independent atom and charge balance check, before it can count as correct. The balance column reports that independent check. The example above writes scores to `/tmp/completerxn-rescore/main/scores/`. The saved paper scores are in `results/main/scores/`. See the [analysis index](../../analysis/README.md).

---

## Uncurated USPTO predictions

Preprocess the uncurated data, then submit chunk jobs:

```bash
# Preprocess
python models/SynRBL/preprocess.py \
  data/USPTO_uncurated/uspto_uncurated_processed.csv \
  results/SynRBL/uspto_uncurated_CanonicalReaction_processed.csv \
  --reaction-col CanonicalOriginalReaction

# Split into chunks (optional helper — adapt path as needed)
python models/SynRBL/split_processed_into_chunks.py \
  results/SynRBL/uspto_uncurated_CanonicalReaction_processed.csv \
  results/SynRBL/uspto_uncurated_CanonicalReaction_processed_chunks/

# Cluster: submit one job per chunk
bash models/SynRBL/submit_synrbl_chunks.sh \
  --chunks-dir results/SynRBL/uspto_uncurated_CanonicalReaction_processed_chunks/ \
  --output-dir results/SynRBL/uspto_uncurated_CanonicalReaction_results/
```

`run_eval_all_splits.sh` and `evaluate.py` report equivalence accuracy by default. To reproduce historical v1 results, add `--scoring-metric v1` and use a separate output directory. See [Historical results](../../analysis/README.md#historical-results).

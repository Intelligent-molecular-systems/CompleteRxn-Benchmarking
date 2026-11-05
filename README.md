# CompleteRXN benchmark

The CompleteRXN benchmark tests whether a model is able to add missing molecules to an incomplete chemical reaction to make the reaction atom-balanced. The benchmark compares three baseline models: Reaction Balancer (RB), Constrained Reaction Balancer (CRB), and SynRBL on random, group, and extreme out-of-distribution (OOD) splits. Each split has five repetitions. The terms used throughout this repository are explained under [Terms](#terms).

The underlying [dataset is on Hugging Face](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-benchmark), including the exact splits used in our [paper (arXiv Version)](https://arxiv.org/abs/2605.00222). This repository contains the benchmark tools, model code, paper analyses, and compact results. A [fine-tuned CRB checkpoint](https://huggingface.co/Intelligent-molecular-systems/completeRXN-CRB) and the [files for reproducing the paper](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-reproduction) (saved predictions and software containers) are also on Hugging Face.

## Get started

The benchmark package needs Python 3.10 or later and installs RDKit 2025.9.5 itself. The SynRBL example needs Python 3.11 or later. We recommend a conda environment with Python 3.11, which covers both:

```bash
conda create -n completerxn python=3.11 -y
conda activate completerxn
python -m pip install -e .
python examples/quickstart.py
```

Without conda, any Python 3.10 or later works for everything except the SynRBL example. Check your version with `python3 --version`, then replace the first two lines with:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Model training and the paper analyses use their own environments, described in the model and analysis READMEs.

The quickstart downloads the benchmark and random-split assignments from Hugging Face, runs a dummy model on ten test reactions, scores its predictions, and saves reaction drawings. The dummy model returns each input unchanged. It demonstrates the workflow rather than useful prediction accuracy. The first run downloads about 706 MB, including the full benchmark table, the requested split, and its retained test IDs. No Hugging Face login is needed. Later runs reuse the checked local files. Outputs go to a new temporary directory.

## Use your own model

Your model needs a `predict(inputs)` method that returns one completed reaction SMILES for each input, in the same order. It receives incomplete reactions without their references.

```python
from completerxn import DataLoader

class MyModel:
    def predict(self, inputs):
        # Replace this with your own prediction code.
        return inputs

loader = DataLoader()
train, val, test = loader.load(split="random", rep=0)
sample = test[:10]
predictions = sample.predict(MyModel(), batch_size=64)
result = sample.evaluate(predictions)
print(result.summary)
result.save("my_run")

print(result.failure_summary())
result.plot_failures().save("failure_counts.svg")
if result.failures:
    result.plot(result.failures[0]["reaction_id"]).save("example.svg")
```

Use `split="group"` or `split="extreme_ood"` for the other splits, and `rep=0` through `4` for the five repetitions. `DataLoader(data_dir="data/Template_curated/benchmark")` reads local files. `DataLoader(cache_dir="/path/to/cache")` changes where downloaded files are stored.

`test[:100]` selects the first 100 examples. `test.evaluate({reaction_id: completed_reaction, ...})` scores predictions you already have. You can also return `Completion(reaction, solved=True, confidence=0.9)` from `completerxn.predictors` when your model supplies a solved flag or confidence score.

Failure summaries distinguish missing, unsolved, invalid, unbalanced, and reference-mismatching predictions. These are automated labels, not chemical judgments. To compare difficulty groups, use `result.failure_summary(group_by="MissingNrCarbons")`. Group summaries include all observations in the group.

### Benchmark on all test sets

To compare your model with the paper, evaluate it on the filtered test set of all three splits and five repetitions, and report the mean and sample standard deviation per split. If your model needs no training, such as SynRBL, `loader.test()` loads only the test set. That is about three times faster than `loader.load()`:

```python
import statistics
from completerxn import DataLoader

class MyModel:
    def predict(self, inputs):
        # Replace this with your own prediction code.
        return inputs

model = MyModel()
loader = DataLoader()
scores = {}
for split in ["random", "group", "extreme_ood"]:
    for rep in range(5):
        test = loader.test(split=split, rep=rep, population="filtered")
        result = test.evaluate(test.predict(model))
        result.save(f"my_model/{split}_rep{rep}")
        scores.setdefault(split, []).append(result.summary["equivalence_pct"])

for split, values in scores.items():
    print(f"{split}: {statistics.mean(values):.1f} ± {statistics.stdev(values):.1f} %")
```

Loading all 15 test sets takes about two minutes; prediction time depends on your model. Each run is saved in its own folder with predictions, scores, failures and a `provenance.json` record. The paper's results for comparison are in [main_table_filtered.csv](results/main/scores/main_table_filtered.csv).

For a complete example, see [run_synrbl_benchmark.py](examples/run_synrbl_benchmark.py). It runs this loop with SynRBL, one of the paper's baselines (`pip install synrbl==1.0.5`); `--splits` and `--reps` select a subset. Save your model class in `models/`, import it in your script, and load the test sets from HF to predict.

If your model learns or derives anything from data, such as reaction rules or templates, train it separately for each repetition, using only that repetition's training set: `train, val, test = loader.load(split=split, rep=rep, population="filtered")`. The repetitions share reactions, so a test reaction of one repetition can be a training reaction of another.

## Run the small examples

Each example uses ten test reactions from the random split, repetition 0, downloaded from Hugging Face. It saves predictions, scores, and drawings. Add `--limit 0` for the full test set or `--data-dir PATH` for local data. Choose a new output directory for each run.

- **Dummy model:** [run_dummy.py](examples/run_dummy.py) returns the inputs unchanged.

  ```bash
  python examples/run_dummy.py --output-dir /tmp/completerxn-dummy
  ```

- **SynRBL:** [run_synrbl.py](examples/run_synrbl.py) uses the installed SynRBL package. It requires Python 3.11 or later, which the recommended conda setup above provides.

  ```bash
  python -m pip install synrbl==1.0.5
  python examples/run_synrbl.py --output-dir /tmp/completerxn-synrbl
  ```

- **CRB:** [run_crb.py](examples/run_crb.py) downloads the checkpoint fine-tuned on the random split, repetition 0 (`random_seed42_rep0`), and completes ten test reactions. The model needs older software, which is provided as an [Apptainer](https://apptainer.org/) container, so first [install Apptainer](https://apptainer.org/docs/admin/main/installation.html) (`apptainer --version` checks that it works). Then run from the repository root with your Python environment activated:

  ```bash
  # Download the CRB container (4.5 GB, once) to models/constrained_reaction_balancer/CRB_env.sif
  hf download Intelligent-molecular-systems/completeRXN-reproduction \
    models/constrained_reaction_balancer/CRB_env.sif \
    --repo-type dataset --local-dir .

  MODEL_REVISION=$(python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Intelligent-molecular-systems/completeRXN-CRB').sha)")

  python examples/run_crb.py \
    --model-repo Intelligent-molecular-systems/completeRXN-CRB \
    --model-file random_seed42_rep0/model_step_50000.pt \
    --model-revision "$MODEL_REVISION" \
    --container models/constrained_reaction_balancer/CRB_env.sif \
    --limit 10 \
    --output-dir results/local/crb_example_10
  ```

  Results are saved in `results/local/crb_example_10`; open `reaction.svg` to see the first completed reaction. Inference runs on the CPU; add `--gpu 0` to use a GPU. You can also build the container yourself, as described in the [RB/CRB instructions](models/constrained_reaction_balancer/README.md#environment-setup).

In Python, the same models are available as `SynRBLPredictor` and `CRBPredictor.from_pretrained` in `completerxn.predictors`. Leave `batch_size` unset for CRB so its translator starts only once.

## Prediction format

Predictions must contain the **full completed reaction**, not just the missing molecules. Saved CSVs use one row per reaction ID:

```csv
reaction_id,prediction,solved,confidence
example_id,CCO>>CCO,true,0.9
```

| Field | Meaning |
|---|---|
| `reaction_id` | Required. The benchmark's `Unique_identifier`. Each ID must occur at most once. |
| `prediction` | Required. The completed `reactants>agents>products` reaction SMILES. |
| `solved` | Optional. True/false or 1/0, with true as the default. False prevents a correct label. |
| `confidence` | Optional. A finite number from 0 to 1. Leave it empty if unavailable. It does not change accuracy. |

Rows can be in any order, because scoring matches predictions to references by ID. Missing predictions count as incorrect; extra or duplicate IDs raise an error.

## Scoring and test sets

Scoring reports equivalence accuracy (see [Terms](#terms)); only valid predictions balanced for atoms and charge can count as correct. Stereochemistry is ignored by default; use `test.evaluate(predictions, stereo=True)` to include it. `equivalence_lenient=True` adds extra comparison rules for sensitivity checks; it is not used for the paper results.

The default reference column is `TargetReaction_with_stereo`; select another with `DataLoader(reference_column="TargetReaction")`.

The loader returns the **full test set** by default. `population="filtered"` selects the paper's filtered test set and downloads its ID lists automatically. For local data, place them in `eligibility/<split>_rep<rep>.csv` or set `eligibility_dir`.

`result.save()` writes predictions, scores, failures and a `provenance.json` record of the inputs and settings, and never overwrites an existing folder. The loader uses a fixed revision of the [benchmark dataset](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-benchmark) (`7994bbef`) and checks every downloaded file.

## Command-line tools

If your model doesn't run in Python, or runs on another machine, use the command line instead. It works with plain CSV files:

```bash
# 1. Write a test set to a CSV file (downloads the data on first use).
completerxn data export --split group --rep 0 --population filtered --output test.csv

# 2. Run your own model on the `input` column of test.csv and write
#    predictions.csv in the prediction format above.

# 3. Score the predictions.
completerxn score --references test.csv --predictions predictions.csv --output-dir scores
```

`test.csv` has the columns `reaction_id`, `input` and `reference`. The reference is included for scoring, so your model should read only `input`. Step 3 prints the summary and saves per-reaction results in `scores/`. `completerxn data download` only fetches the files, for example to prepare an offline machine.

## Use the models on your own reactions

The models also can attempt to complete reactions outside the benchmark (e.g. the uncurated USPTO reactions or your own set of reactions). Put your incomplete reactions in a CSV file with the columns `reaction_id` and `input`, with reagents and solvents on the reactant side (`reactants.reagents>>products`), as in [custom_reactions.csv](examples/custom_reactions.csv). Then run:

```bash
python examples/run_custom_reactions.py --model synrbl --input path/to/custom_reactions.csv --output path/to/custom_predictions.csv
```

`--model` can be `synrbl` (needs `pip install synrbl==1.0.5`) or `crb` (needs the CRB container, see [Run the small examples](#run-the-small-examples)). The script prints each completed reaction, says whether it is balanced, and writes `predictions.csv` in the [prediction format](#prediction-format). If you also have references, add them as a `reference` column and score the predictions with `completerxn score --references my_reactions.csv --predictions predictions.csv --output-dir scores`.

Without references, balance is the only automatic check. CRB works best on USPTO-like chemistry and on inputs that only use tokens from its vocabulary. SynRBL removes stereochemistry from the whole reaction.

## Reproduce the paper

The [analysis index](analysis/README.md) maps each table and figure to its script and output files. Paper results are in `results/main/` and `results/additional/`. Results computed with the earlier v1 metric, together with the expert audit, are kept in `results/archive/metric_v1/`.

Git includes the code, compact result CSVs, and selected figures. Large datasets, predictions, checkpoints, and containers are separate downloads. Restore them at the paths described in [data/README.md](data/README.md) and [the prediction inventory](results/main/predictions/README.md).

```bash
# Rebuild the main tables, figures, and agreement using saved scores.
bash analysis/reproduce_main_results.sh

# Recalculate scores into a new directory.
bash analysis/reproduce_main_results.sh --rescore --output-root /tmp/completerxn-rescore

# Rebuild only the main results table (Table 2), using Python's standard library.
python3 analysis/build_main_table.py
```

To rebuild the dataset, see [data/Template_curated/README.md](data/Template_curated/README.md). For training and new predictions, see the [RB/CRB instructions](models/constrained_reaction_balancer/README.md) and [SynRBL instructions](models/SynRBL/README.md).

## Repository layout

```text
src/completerxn/        # Installable package, CLI, and equivalence metric
  legacy/               # v1 metric, kept for historical results
examples/               # Benchmark examples and completing your own reactions
tests/                  # Chemical-rule and workflow tests
data/                   # Data preparation code and local data
models/                 # Model code: RB/CRB, SynRBL and a simple rule-based example model (not in the paper)
analysis/               # Paper analyses and their central index
results/
  main/predictions/     # Shared model predictions
  main/scores/          # Main paper scores
  main/figures/         # Main paper figures
  additional/           # Appendix results
  archive/metric_v1/    # Historical v1 results and the expert audit
  local/                # Ignored experiments and temporary outputs
infra/                  # Environment files and container recipes
hf_cards/               # README cards of the three Hugging Face repositories
utils/                  # Helpers for the analysis scripts; new code should import completerxn
```

## Tests

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests
```

The tests use small examples and need no downloads or model weights.

## Terms

| Term | Meaning |
|---|---|
| **Split** | How reactions are divided into train, validation and test sets: `random`, `group` or `extreme_ood` (extreme out-of-distribution). |
| **Repetition** (`rep`) | Each split is generated five times (`rep=0` to `4`). Folders name repetitions as `random_seed42_rep0` … `random_seed46_rep4`, `group_rep0` … and `extreme_ood_rep0` …. Some scripts call a repetition a 'fold'. |
| **Input** | The incomplete reaction given to a model. By default this is `InputReaction_harmonized`: the original `InputReaction` with extra copies of a molecule removed where the input contains more copies than the reference (at least one copy is kept). All three models in the paper use it. |
| **Reference** | The complete, balanced reaction a prediction is compared with. By default this is `TargetReaction_with_stereo`. |
| **Equivalence match** | A prediction counts as correct if it is valid, balanced for atoms and charge, and represents the same reaction as the reference once equivalent ways of writing the molecules are normalized. **Equivalence accuracy** is the percentage of test reactions with an equivalence match. Run records identify this metric as `v2`. |
| **v1 metric** | The earlier version of the equivalence metric. It is kept only to reproduce historical results. See [Historical results](analysis/README.md#historical-results). |
| **Full and filtered test sets** | The full test set contains every test reaction of a split. The filtered test set, used in the paper, excludes reactions containing tokens outside the pretrained Molecular Transformer's vocabulary. The IDs it keeps are listed in `eligibility/<split>_rep<rep>.csv`. |
| **RB and CRB** | Reaction Balancer and Constrained Reaction Balancer: the same fine-tuned Molecular Transformer, decoded with plain beam search (RB) or with beam search constrained to atom and charge balance (CRB). Model scripts and folders call these `BS_PT` and `CBS_PT`. |
| **Reagent identification (RI)** | Moving molecules that are unchanged by the reaction into the reagent field. The SynRBL comparison runs with (`RI`) and without (`noRI`) this step. The main SynRBL results are `noRI`. |
| **Uncurated USPTO** | USPTO reactions outside the benchmark. They have no reference, so analyses report balance and agreement between models instead of accuracy. |
| **Agreement** | Whether CRB and SynRBL predict the same completed reaction. Some folders call this *consensus*. |

## Citation

If you use CompleteRXN, please cite:

```bibtex
@article{vogel2026completerxn,
  title={CompleteRXN: Toward Completing Open Chemical Reaction Databases},
  author={Vogel, Gabriel and Noordsij, Minouk and Pidko, Evgeny and Weber, Jana M.},
  journal={Advances in Neural Information Processing Systems},
  volume={39},
  year={2026}
}
```

### Other work to cite

Please also cite the work underlying the components you use:

- **Dataset:** [FlowER](https://doi.org/10.6084/m9.figshare.28359407) and the original USPTO data sources.
- **SynRBL:** [SynRBL](https://github.com/TieuLongPhan/SynRBL) when using its integration or predictions.
- **RB/CRB:** the original [Reaction Balancer](https://www.sciencedirect.com/science/chapter/bookseries/abs/pii/B978044328824150524X) and [Molecular Transformer](https://doi.org/10.1021/acscentsci.9b00576) papers when using these models.

References and source details are provided in the [dataset card](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-benchmark) and the [model documentation](models/README.md).

## Acknowledgements

This work was supported by the Mainstreaming Open Science Fund of the Open Science Community Delft (OSCD). We thank the chemistry annotators for the discussions and their careful curation work.

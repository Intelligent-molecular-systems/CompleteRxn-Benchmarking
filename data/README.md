# Data

```
data/
├── Template_curated/   # Curated benchmark dataset (FlowER + USPTO mapping)
├── USPTO_uncurated/    # Uncurated USPTO reactions without references (prediction only)
└── SynRXN_complex/     # External SynRXN evaluation dataset
```

## Template_curated

The main benchmark pairs [FlowER](https://doi.org/10.6084/m9.figshare.28359407) reactions with their USPTO records. See [Template_curated/README.md](Template_curated/README.md) to build the dataset and its train/test splits.

**Output:** `data/Template_curated/benchmark/benchmark_final.csv`

You don't need to rebuild the dataset to use the benchmark. The `completerxn` loader downloads the benchmark table, the requested split and its retained test IDs automatically. To work from local files instead, download them from the [Hugging Face dataset](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-benchmark), place them under `data/Template_curated/benchmark/`, and use `DataLoader(data_dir="data/Template_curated/benchmark")`.

Dataset construction and reaction fingerprint calculations run in the data-processing environment described under [Environment setup](#environment-setup).

### Hosted files

The [benchmark dataset](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-benchmark) contains `benchmark_final.csv`, the 15 split files (3 splits × 5 repetitions), and the lists of test IDs in the filtered test set under `eligibility/`. The loader uses a fixed revision of this repository and checks every downloaded file.

The [reproduction repository](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-reproduction) contains saved predictions, additional analysis inputs, and containers. Restore these files to the paths listed there and in [the prediction inventory](../results/main/predictions/README.md). They are not needed for the dummy-model quickstart.

### Rebuild from source records

The `raw/` folder on HF also contains the two source files needed to rebuild the dataset. Download them into an empty staging directory, then move them to the source paths expected by the processing scripts:

```bash
hf download Intelligent-molecular-systems/completeRXN-benchmark \
  raw/master_dataset_mapped.csv \
  raw/flower_full_single_reactions_with_main_products.tsv \
  --repo-type dataset \
  --local-dir /tmp/completerxn-source-download
mkdir -p data/Template_curated/USPTO_FlowER_raw
mv -n /tmp/completerxn-source-download/raw/master_dataset_mapped.csv \
  data/Template_curated/USPTO_FlowER_raw/
mv -n /tmp/completerxn-source-download/raw/flower_full_single_reactions_with_main_products.tsv \
  data/Template_curated/USPTO_FlowER_raw/
```

## USPTO_uncurated

This set contains USPTO reactions outside the curated benchmark. They have no validated reference completion, so this analysis measures properties such as balance and model agreement rather than accuracy against a known answer.

**Prepare the input file:**

```bash
python data/USPTO_uncurated/prepare_uncurated_input.py
```

The script reads `data/Template_curated/USPTO_FlowER_raw/master_dataset_mapped.csv`, removes reactions already in `benchmark_final.csv`, moves recorded reagents to the reactant side, and writes:

- `data/USPTO_uncurated/uspto_uncurated_processed.csv`

That file is the input for both the CRB and SynRBL inference pipelines described in their respective model READMEs.

---

## SynRXN_complex

The external evaluation uses SynRXN's [Complex dataset](https://github.com/TieuLongPhan/SynRXN/blob/166b2deef4981ca60bfc61cc6d138f943adf9830/Data/rbl/complex.csv.gz). The included `SynRXN_complex/complex.csv.gz` is an unchanged, byte-identical copy at commit `166b2deef4981ca60bfc61cc6d138f943adf9830` (1,748 records). Data are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as specified in [SynRXN's data license](https://github.com/TieuLongPhan/SynRXN/blob/main/Data/LICENSE). Please cite Phan, Nguyen Song and Stadler, [SynRXN, Scientific Data 13, 625 (2026)](https://doi.org/10.1038/s41597-026-07260-w), and the Complex source references listed in [SynRXN's data records](https://github.com/TieuLongPhan/SynRXN/blob/main/doc/data_records.rst).

`complex_processed.csv` is the exact 1,746-row input used for the paper: we excluded `complex_163` (empty reference) and `complex_51` (overlap with CompleteRXN), retained row order and original reaction/reference strings, and added reagent-identification metadata and the missing-species target. The preparation helpers are in this folder. The processed file is included so reproducing the scores does not require the original preparation notebook.

The saved CRB predictions contain five candidates per record, in the same order, at `results/main/predictions/crb/synrxn/beamSize5_nBest5.txt`. The paper scores were calculated with the CRB container (RDKit 2020.09.1). To recalculate them in a new folder, run from the repository root:

```bash
apptainer exec models/constrained_reaction_balancer/CRB_env.sif \
  python analysis/synrxn_analysis/analyze.py \
  --output-dir /tmp/completerxn-synrxn
```

Scores computed with the historical v1 metric, and the 12-case review sampled from them, are in `results/archive/metric_v1/additional/synrxn_analysis/`. That run used the SynRBL container (RDKit 2025.09.4) with `PYTHONHASHSEED=0`. To repeat it, add `--metric v1` and use a new output folder. See the [analysis notes](../analysis/NOTES.md#synrxn-case-review) for how the review relates to the equivalence metric.

For new inference, prepare tokenized inputs with `models/constrained_reaction_balancer/SynRXN_complex/prepare_testset.py`, then use `translate_batch_PT_CBS_synrxn.sh` with the trained checkpoint and GPU environment. Saved predictions allow score reproduction without either.

| Included file | SHA-256 |
| --- | --- |
| Source archive | `a2a2d887a65806fbd27596147f6209213cb3eedfbafd428370ec68586827db4c` |
| Processed input | `dfb19e6514ede44bd70c044639d69b1737fc4afae019d6f00956cad2e6fe7893` |
| Saved predictions | `fcb4c6806d79b5fb728299238b074fa6acba7dcaac97c35c8a4055100a4cdd92` |

## Environment setup

One conda environment covers data processing and split generation:

```bash
conda env create -f infra/data_processing_splits/environment.yml
conda activate data-env
```

### Apptainer container (alternative)

The data processing steps can also be run inside an Apptainer container. The analysis scripts expect it at this path:

```bash
apptainer build infra/data_processing_splits/drfp_data_env.sif infra/data_processing_splits/apptainer_env.def
apptainer exec --bind "$PWD":"$PWD" --pwd "$PWD" infra/data_processing_splits/drfp_data_env.sif \
  bash data/Template_curated/run_data_processing.sh
```

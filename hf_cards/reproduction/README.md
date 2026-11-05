---
license: mit
pretty_name: CompleteRXN reproduction files
tags:
- chemistry
- reaction-completion
- reproducibility
configs:
- config_name: default
  data_files:
  - split: files
    path: manifest.csv
---

# CompleteRXN reproduction files

Files for reproducing the results of the [CompleteRXN paper](https://arxiv.org/abs/2605.00222) without rerunning the models: saved model predictions, inputs for the additional analyses, and the software environments used by the scripts. The code, result tables and figures are in the [CompleteRxn-Benchmarking](https://github.com/Intelligent-molecular-systems/CompleteRxn-Benchmarking) repository.

You don't need these files to use the benchmark itself. The benchmark data and the train, validation and test splits are in the [benchmark dataset](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-benchmark), and a fine-tuned CRB model is in the [model repository](https://huggingface.co/Intelligent-molecular-systems/completeRXN-CRB).

## Using these files

Download the files you need into a checkout of the code repository, keeping their paths unchanged. For example, `models/constrained_reaction_balancer/CRB_env.sif` goes to that same path in your checkout. The [analysis index](https://github.com/Intelligent-molecular-systems/CompleteRxn-Benchmarking/blob/main/analysis/README.md) lists which script and files belong to each paper result, and the [prediction inventory](https://github.com/Intelligent-molecular-systems/CompleteRxn-Benchmarking/blob/main/results/main/predictions/README.md) lists the files needed for rescoring.

`manifest.csv` lists every file with its path, size, description and SHA-256 checksum, so you can check your downloads. The dataset viewer shows this list.

## Where to find the files

Folders follow the layout of the code repository. `...` marks folders left out for readability.

```text
data/
├── Template_curated/
│   ├── benchmark/                   # Inputs for additional benchmark analyses
│   └── sensitivity_sweeps/extreme/  # Inputs for the OOD split sensitivity analysis
├── USPTO_1k_TPL/                    # Reactions used for the template analysis
└── USPTO_uncurated/                 # Processed uncurated USPTO reactions
models/
├── constrained_reaction_balancer/
│   ├── CRB_env.sif                  # RB/CRB software environment
│   └── Template_curated/benchmark/prepared/
│       └── .../                     # Test inputs and references for each repetition
└── SynRBL/synrbl_env.sif            # SynRBL software environment
infra/data_processing_splits/
└── drfp_data_env.sif                # Data-processing and analysis environment
results/
├── main/predictions/
│   ├── rb/                          # RB benchmark predictions
│   ├── crb/                         # CRB benchmark and uncurated predictions
│   └── synrbl/                      # SynRBL benchmark predictions
├── SynRBL/                          # SynRBL predictions with reagent identification, and uncurated predictions
├── additional/                      # Saved inputs for additional analyses
└── archive/metric_v1/               # Agreement ID lists from the earlier v1 metric
```

**RB and CRB.** Predictions are organized by split (`random`, `group` or `extreme_ood`) and repetition. For example, `results/main/predictions/crb/random/random_seed42_rep0/` holds the CRB predictions for the random split, repetition 0. Each folder contains:

- `beamSize5_nBest5.txt`: five candidate completions per reaction,
- `beamSize5_nBest5.txt_log_probs`: the log probability of each candidate,
- `top1_predictions_with_stats.txt`: the top-ranked prediction per reaction with summary statistics.

**SynRBL.** The main benchmark predictions are in `results/main/predictions/synrbl/eval_all_splits_noRI_complete/full_results.csv`. They use the same inputs as RB and CRB, without reagent identification (`noRI`). The comparison run with reagent identification, which moves molecules unchanged by the reaction into the reagent field, is in `results/SynRBL/eval_all_splits_CanonicalReaction/full_results.csv`.

**Uncurated USPTO.** USPTO reactions outside the benchmark, without references. SynRBL predictions are split across CSV files in `results/SynRBL/uspto_uncurated_CanonicalReaction_results/`; the CRB predictions are in `results/main/predictions/crb/uncurated/`.

**Additional analyses.** `results/additional/` contains saved inputs for the comparison of benchmark and uncurated reactions, the bias analysis and the missing-molecule analysis.

## Sizes

| Part | Contents | Size |
|---|---|---:|
| Main benchmark | Benchmark predictions and prepared test inputs and references | 496 MB |
| Additional analyses | Inputs and predictions for the additional analyses | 3.49 GB |
| Containers | Three Apptainer software environments (`.sif`) | 8.04 GB |

The `.sif` files are [Apptainer](https://apptainer.org/) containers. Run a script inside one with `apptainer exec <container>.sif python <script>`. The build recipes are in `infra/` in the code repository.

## License

The files created for this project (predictions, prepared inputs, analysis inputs and `manifest.csv`) are released under the **MIT License**, like the code repository. Keep the copyright and license notice when you redistribute them. The reaction data they contain derive from the sources listed in the [benchmark dataset card](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-benchmark), which keep their own terms.

**The containers are not covered by the MIT License.** They bundle third-party software, each part under its own license, including the conda-forge Miniforge base image, Python, PyTorch, RDKit, SynRBL and, in the RB/CRB container, NVIDIA CUDA and cuDNN libraries and Intel MKL. Those licenses apply when you use or redistribute the containers.

## Citation

If you use these files, please cite:

```bibtex
@article{vogel2026completerxn,
  title={CompleteRXN: Toward Completing Open Chemical Reaction Databases},
  author={Vogel, Gabriel and Noordsij, Minouk and Pidko, Evgeny and Weber, Jana M.},
  journal={Advances in Neural Information Processing Systems},
  volume={39},
  year={2026}
}
```

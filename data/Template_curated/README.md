# Building the curated benchmark

CompleteRXN links reactions from the [FlowER dataset](https://doi.org/10.6084/m9.figshare.28359407) to their USPTO records. The scripts in this folder build the paired benchmark and its random, group, and extreme out-of-distribution (OOD) splits.

You can download the finished `benchmark_final.csv` and split files from the [CompleteRXN dataset](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-benchmark). Rebuilding them from source requires `data/Template_curated/USPTO_FlowER_raw/master_dataset_mapped.csv` and `data/Template_curated/USPTO_FlowER_raw/flower_full_single_reactions_with_main_products.tsv`. The download command is in [data/README.md](../README.md).

From the repository root, run:

```bash
bash data/Template_curated/run_data_processing.sh
bash data/Template_curated/run_generate_all_splits.sh
```

The first script creates `data/Template_curated/benchmark/benchmark_final.csv`. It runs the following steps in order:

1. `map_uspto_flower.py` matches USPTO and FlowER records. The one-to-one matches form the starting set for this benchmark. The script also saves many-to-one and one-to-many matches for inspection.
2. `reagent_identification_flower.py` moves unchanged molecules from the reactant and product sides to the reagent field in FlowER reactions.
3. `add_flower_reactions_to_uspto.py` joins the matched USPTO and FlowER reactions.
4. `benchmark_columns.py` prepares the input and target columns.
5. `add_stereochemistry_flower.py` transfers stereochemical information from the USPTO records to the matched FlowER targets where the match is clear.
6. `extract_input_target_diff.py` creates the final input and target difference used for prediction. Its checks remove cases where the source cannot be reconciled with the target.

Intermediate files go to `data/Template_curated/processed/` and `data/Template_curated/benchmark/`. The second script constructs the group and extreme OOD assignments, then writes the five train/validation/test splits for each split type under `benchmark/random_splits/`, `benchmark/group_splits/`, and `benchmark/extreme_ood_splits/`. The group and OOD builders use the reaction fingerprint similarity threshold of 0.55 used in the paper.

These split files list reaction identifiers and split labels. The CRB/RB preparation script uses them to create the model's actual input and target text files. See [the CRB/RB README](../../models/constrained_reaction_balancer/README.md) for that next step.

---
license: mit
task_categories:
- other
language:
- en
tags:
- chemistry
- reactions
- SMILES
- reaction-completion
- benchmarking
- USPTO
pretty_name: CompleteRXN Benchmark
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: data
    path: benchmark_final.csv
dataset_info:
  features:
  - name: Unique_identifier
    dtype: string
  - name: OriginalReaction
    dtype: string
  - name: PatentNumber
    dtype: string
  - name: ParagraphNum
    dtype: string
  - name: Year
    dtype: int64
  - name: TextMinedYield
    dtype: string
  - name: CalculatedYield
    dtype: string
  - name: ReactionSmiles
    dtype: string
  - name: RewrittenReaction
    dtype: string
  - name: MovedMetadata
    dtype: string
  - name: MultiProduct
    dtype: int64
  - name: CanonicalReaction
    dtype: string
  - name: IsBalanced
    dtype: bool
  - name: Balance
    dtype: string
  - name: MissingMw
    dtype: float64
  - name: MissingNrAtoms
    dtype: int64
  - name: MissingNrCarbons
    dtype: int64
  - name: rxn_class
    dtype: string
  - name: InvalidReason
    dtype: string
  - name: IsIn
    dtype: string
  - name: 1kTPLLabel
    dtype: string
  - name: FlowER_ID
    dtype: string
  - name: CanonicalOriginalReaction
    dtype: string
  - name: FlowER_ID_parsed
    dtype: string
  - name: rxn_flower
    dtype: string
  - name: rxn_reagents_identified_flower
    dtype: string
  - name: moved_metadata
    dtype: string
  - name: InputReaction
    dtype: string
  - name: TargetReaction
    dtype: string
  - name: InputReaction_canonical
    dtype: string
  - name: TargetReaction_canonical
    dtype: string
  - name: TargetReaction_with_stereo
    dtype: string
  - name: has_unresolved_stereo
    dtype: bool
  - name: n_unresolved_stereo
    dtype: int64
  - name: InputReaction_harmonized
    dtype: string
  - name: input_target_diff_harmonized
    dtype: string
  - name: input_harmonized_modified
    dtype: bool
  - name: source_not_in_target
    dtype: bool
---

# CompleteRXN Benchmark

**206,423 reactions** for **reaction completion**: given a USPTO reaction that is missing molecules, so its atoms don't balance, predict the missing molecules so that the completed reaction is balanced for atoms and charge. The references come from [FlowER](https://doi.org/10.6084/m9.figshare.28359407) mechanistic reactions matched to their USPTO records.

Three splits test how well models generalize to increasingly unfamiliar reactions: **random**, **group** out-of-distribution and **extreme** out-of-distribution. Each split has 5 repetitions.

- **Code, models and paper analyses:** [CompleteRxn-Benchmarking](https://github.com/Intelligent-molecular-systems/CompleteRxn-Benchmarking)
- **Paper:** [CompleteRXN: Toward Completing Open Chemical Reaction Databases](https://arxiv.org/abs/2605.00222)
- **Fine-tuned CRB model:** [completeRXN-CRB](https://huggingface.co/Intelligent-molecular-systems/completeRXN-CRB)
- **Saved predictions and environments for reproducing the paper:** [completeRXN-reproduction](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-reproduction)

## Quick start

The `completerxn` package from the code repository downloads and checks these files for you:

```python
from completerxn import DataLoader

train, val, test = DataLoader().load(split="random", rep=0, population="filtered")
predictions = test.predict(my_model)   # my_model.predict(inputs) returns completed reactions
print(test.evaluate(predictions).summary)
```

Leave out `population="filtered"` to evaluate on the full test set. See the code repository's README for the prediction format, command-line tools and examples.

## Files

```text
benchmark_final.csv     206,423 reactions with metadata; Unique_identifier links all files
random_splits/          5 files, split column random_split
group_splits/           5 files, split column group_split
extreme_ood_splits/     5 files, split column extreme_ood
eligibility/            IDs of the filtered test set, one file per split and repetition
raw/                    source files for rebuilding the dataset
  master_dataset_mapped.csv
  flower_full_single_reactions_with_main_products.tsv
```

In every split file, the split column is `0` for train, `1` for validation and `2` for test.

The paper evaluates on the **filtered test set**: test reactions whose input or reference contains tokens outside the pretrained Molecular Transformer's vocabulary are excluded, so that all models are compared on the same reactions. `eligibility/<split>_rep<rep>.csv` lists the reactions kept, and `eligibility/coverage.csv` gives the number kept per split and repetition.

## Main columns in `benchmark_final.csv`

| Column | Type | Description |
|---|---|---|
| `Unique_identifier` | string | Reaction ID (e.g. `USPTO_US03930836_0`), used to join the split and eligibility files |
| `InputReaction` | string | Incomplete reaction given to a model, as `reactants>reagents>products` SMILES |
| `InputReaction_harmonized` | string | **Default model input**, used by all models in the paper: `InputReaction` with extra copies of a molecule removed where the input contains more copies than the reference (at least one copy is kept) |
| `TargetReaction` | string | **Reference**: complete reaction from FlowER, balanced for atoms and charge |
| `TargetReaction_with_stereo` | string | **Default reference**: `TargetReaction` with stereochemistry restored from the USPTO record, since FlowER does not include it. Reactions whose stereochemistry could not be restored unambiguously are excluded from the dataset, so this column is always filled |
| `input_target_diff_harmonized` | string | The molecules missing from `InputReaction_harmonized`, i.e. what a model has to add |
| `MissingNrAtoms` | int | Number of heavy atoms missing from the product side (the main difficulty measure in the paper) |
| `MissingNrCarbons` | int | Number of carbon atoms missing from the product side |
| `MissingMw` | float | Molecular weight of the missing molecules |
| `Balance` | string | Type and count of the missing atoms |
| `rxn_class` | string | Reaction class (1–10, Schneider et al.), predicted by a classifier trained on the two USPTO 50k datasets |
| `1kTPLLabel` | string | Reaction template label from USPTO 1k TPL, which groups reactions by the 1,000 most common templates. Empty if the reaction is not in that set |
| `OriginalReaction` | string | Original USPTO reaction SMILES before curation (Lowe, 2017) |
| `CanonicalOriginalReaction` | string | `OriginalReaction` in canonical SMILES, without atom mapping |
| `PatentNumber`, `ParagraphNum`, `Year` | string, string, int | Source patent, paragraph and year |
| `FlowER_ID` | string | ID of the matched FlowER reaction |

### Other columns

- `has_unresolved_stereo`, `n_unresolved_stereo`: whether, and for how many molecules, stereochemistry could not be restored. Kept from the processing pipeline. In this release they are always `False` and `0`, because the 147 affected reactions were excluded. A further 47 reactions were excluded because their input, including reagents and stereochemistry, is not fully contained in the matched FlowER reaction.
- `input_harmonized_modified`: whether `InputReaction_harmonized` differs from `InputReaction`.
- `source_not_in_target`: whether the input contains molecules that do not occur in the reference.
- `rxn_flower`, `rxn_reagents_identified_flower`, `moved_metadata`: the matched FlowER reaction, the same reaction with unchanged molecules moved to the reagent field, and a record of which molecules were moved.
- `InputReaction_canonical`, `TargetReaction_canonical`: canonical SMILES versions of the input and reference.
- `TextMinedYield`, `CalculatedYield`, `ReactionSmiles`: fields from the original USPTO extraction.
- `RewrittenReaction`, `MovedMetadata`, `MultiProduct`, `CanonicalReaction`, `IsBalanced`, `InvalidReason`, `IsIn`, `FlowER_ID_parsed`: intermediate fields from curation and from matching USPTO to FlowER, kept for traceability.

## Task and metric

Given `InputReaction_harmonized`, predict the **complete reaction**, including the molecules that were missing.

The metric is **equivalence accuracy**: the percentage of test reactions whose prediction is an equivalence match with the reference. A prediction is an equivalence match if it is valid, balanced for atoms and charge, and represents the same reaction as the reference once equivalent ways of writing the molecules are normalized. Stereochemistry is ignored by default. The rules are implemented in `src/completerxn/equivalence.py` in the code repository.

## Rebuilding the dataset

Download the two files in [`raw/`](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-benchmark/tree/main/raw), place them in `data/Template_curated/USPTO_FlowER_raw/` in a checkout of the code repository, and run `bash data/Template_curated/run_data_processing.sh`. See `data/Template_curated/README.md` there for the individual steps.

## Source data

`master_dataset_mapped.csv` matches the USPTO extraction by Lowe (2017) to USPTO STEREO, USPTO 1k TPL and FlowER by canonical SMILES (column `IsIn`). The two USPTO 50k datasets are not part of the released files: they were used only to train the classifier behind `rxn_class` and for the analysis of incompleteness per reaction class in the paper.

| Dataset | Reference | Download | License |
|---|---|---|---|
| USPTO (grants, 1976–2016) | Lowe (2017) | [figshare](https://doi.org/10.6084/m9.figshare.5104873) | CC0 |
| USPTO 50k (2015) | Schneider et al., *JCIM* 2015 | [rxn4chemistry/rxnfp](https://github.com/rxn4chemistry/rxnfp/blob/master/data/schneider50k.tsv) | MIT (rxnfp repository) |
| USPTO 50k (2016) | Schneider et al., *JCIM* 2016 | [ACS supporting information](https://doi.org/10.1021/acs.jcim.6b00564.s002) | CC BY-NC 4.0 ([ACS figshare](https://acs.figshare.com/articles/dataset/What_s_What_The_Nearly_Definitive_Guide_to_Reaction_Role_Assignment/4295660)) |
| USPTO STEREO | Schwaller et al., *Chem. Sci.* 2018 | [IBM Box](https://ibm.ent.box.com/v/ReactionSeq2SeqDataset) | No license stated (subset of Lowe, 2017) |
| USPTO 1k TPL | Schwaller et al., *Nat. Mach. Intell.* 2021 | [IBM Box](https://ibm.ent.box.com/v/MappingChemicalReactions/folder/124192222443) | No license stated (subset of Lowe, 2017) |

These datasets are derived from USPTO patent data.

`flower_full_single_reactions_with_main_products.tsv` is a preprocessed extract of the FlowER single-step mechanistic reactions: Joung et al., *Nature* 645, 115–123 (2025), [figshare](https://doi.org/10.6084/m9.figshare.28359407) (MIT).

## License

This dataset is released under the **MIT License**, like the code repository. Keep the copyright and license notice when you redistribute it.

It is derived from the sources above, which keep their own terms. The references (`TargetReaction`, `TargetReaction_with_stereo`, `rxn_flower` and derived columns) are derived from FlowER, released under the MIT License by its authors; keep their notice as well. The USPTO extraction by Lowe (2017), the source of all reaction SMILES, is CC0. USPTO STEREO and USPTO 1k TPL state no license; from them, this dataset uses only membership (`IsIn`) and the template labels in `1kTPLLabel`. No records of the two USPTO 50k datasets are included; `rxn_class` holds predictions of our own classifier trained on them.

## Citation

If you use this dataset, please cite:

```bibtex
@article{vogel2026completerxn,
  title={CompleteRXN: Toward Completing Open Chemical Reaction Databases},
  author={Vogel, Gabriel and Noordsij, Minouk and Pidko, Evgeny and Weber, Jana M.},
  journal={Advances in Neural Information Processing Systems},
  volume={39},
  year={2026}
}
```

Please also cite FlowER (Joung et al., 2025) and the USPTO sources above that are relevant to your use.

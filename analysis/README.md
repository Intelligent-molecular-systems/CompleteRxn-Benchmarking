# Paper analyses

This folder holds the scripts behind every table and figure in the paper. Main paper scores and figures are in `results/main/`, appendix results are in `results/additional/`, and saved model predictions are in `results/main/predictions/`. All commands report equivalence accuracy by default (see [Terms](../README.md#terms)). Results from the earlier v1 metric are described under [Historical results](#historical-results).

Run commands from the repository root. Scoring uses the CRB container (`models/constrained_reaction_balancer/CRB_env.sif`). Figures and reaction fingerprints use the data-processing container (`infra/data_processing_splits/drfp_data_env.sif`, built as described in [data/README.md](../data/README.md#environment-setup)).

```bash
bash analysis/reproduce_main_results.sh                 # main paper results from saved scores
python3 analysis/build_main_table.py                    # main results table (Table 2), no extra packages

# Recalculate scores into a new directory, leaving the published scores untouched:
bash analysis/reproduce_main_results.sh --rescore --output-root /tmp/completerxn-rescore
```

`reproduce_main_results.sh` rebuilds the main results table, the template statistics, the missing-atoms figure (Figure 3), the outcome figure (Figure C1) and the agreement results. It reuses saved agreement results only when the metric, stereochemistry setting and input scores match.

Each scoring run writes a `provenance.json` file that records the metric version, the RDKit version and checksums of the input files, so you can tell which code and inputs produced a result. Existing run records are never overwritten.

## Find an analysis

| Paper topic | Script | Output |
|---|---|---|
| Main comparison (Table 2) | [score_benchmark.py](score_benchmark.py), [build_main_table.py](build_main_table.py) | `main/scores/` |
| Accuracy by number of missing atoms/carbons (Figure 3) | [plot_crb_synrbl_top2_by_missing_atoms.py](figures/plot_crb_synrbl_top2_by_missing_atoms.py) | `main/figures/` |
| Outcome fractions (Figure C1) | [plot_synrbl_vs_crb_outcome_fractions.py](figures/plot_synrbl_vs_crb_outcome_fractions.py) | `main/figures/` |
| Accuracy and confidence plots | [plot_accuracy_merged_by_missing_atoms.py](figures/plot_accuracy_merged_by_missing_atoms.py), [plot_confidence_by_category.py](figures/plot_confidence_by_category.py) | `main/figures/` |
| Performance by reaction template | [analyze_random_split_template_label_performance.py](templates/analyze_random_split_template_label_performance.py), [analyze_template_error_concentration_by_model.py](templates/analyze_template_error_concentration_by_model.py) | `main/figures/`, `additional/templates/` |
| Split construction and quality | [plot_ood_fold_quality.py](figures/plot_ood_fold_quality.py) | Local figures only |
| Sensitivity of the OOD splits to their construction | [analyze.py](split_sensitivity/analyze.py) | `additional/split_sensitivity/` |
| Synthetic vs. observed missing molecules, target selection bias | [analyze.py](bias_analysis/analyze.py) | `additional/bias_analysis/` |
| Categories of missing molecules | [classify_missing_species.py](missing_species/classify_missing_species.py), [analyze_missing_sides.py](missing_species/analyze_missing_sides.py) | `additional/missing_species_results/` |
| Accuracy per missing-molecule category | [build_category_accuracy_table.py](missing_species/build_category_accuracy_table.py) | `additional/missing_species_results/` |
| Paired statistical tests, RB vs. CRB | [test_rb_vs_crb.py](statistical_tests/test_rb_vs_crb.py) | `additional/statistical_tests/` |
| SynRBL with vs. without reagent identification (Table C1) | [build_table_c1.py](ri_comparison/build_table_c1.py) | `additional/ri_comparison/` |
| CRB errors caused by stereochemistry alone | [crb_stereo_only_errors.py](stereo_errors/crb_stereo_only_errors.py) | `additional/stereo_errors/` |
| CRB–SynRBL agreement on the benchmark | [recompute_agreement.py](consensus_audit/recompute_agreement.py) | `additional/consensus_audit/` |
| Results on the full (unfiltered) test set | [run.py](unfiltered_evaluation/run.py), [build_appendix_table.py](unfiltered_evaluation/build_appendix_table.py) | `additional/unfiltered_evaluation/` |
| CRB–SynRBL agreement on uncurated USPTO | [recompute.py](uncurated_stereo_free_agreement/recompute.py) | `additional/uncurated_stereo_free_agreement/` |
| Similarity of benchmark and uncurated reactions | [analyze_similarity.py](benchmark_vs_uncurated/analyze_similarity.py) | `additional/benchmark_vs_uncurated/` |
| Failure analysis on uncurated USPTO | [analyze.py](uncurated_failure_modes/analyze.py) | `additional/uncurated_failure_modes/` |
| SynRXN Complex evaluation | [analyze.py](synrxn_analysis/analyze.py) | `additional/synrxn_analysis/` |
| Expert audit of the metric | [analyze_assessments.py](equivalence_metric_audit/analyze_assessments.py) | `archive/metric_v1/additional/equivalence_metric_audit/` |
| Expert-audit cases, reaction figures and LaTeX appendix | [export_cases.py](expert_audits/export_cases.py), then [build_appendix.py](expert_audits/build_appendix.py) | `additional/expert_audits/` |

Output paths are relative to `results/`. Use a new output directory when repeating an analysis. Table builders accept `--scores-dir` to choose which saved scores to read. Details on how individual analyses aggregate their numbers are in [NOTES.md](NOTES.md).

## Reproduce all results

Git contains the compact result CSVs, run records and selected figures. Large per-prediction tables and generated Markdown/LaTeX reports are not tracked. For a complete fresh run, first place the data, predictions and containers at the paths listed in [data/README.md](../data/README.md) and [the prediction inventory](../results/main/predictions/README.md), then run:

```bash
# Use a new directory. Existing results are never overwritten.
export RUN="$PWD/results/local/reproduction"
chem() { apptainer exec models/constrained_reaction_balancer/CRB_env.sif python "$@"; }
analyze() { apptainer exec infra/data_processing_splits/drfp_data_env.sif python "$@"; }

bash analysis/reproduce_main_results.sh --rescore --output-root "$RUN"

chem analysis/ri_comparison/build_table_c1.py \
  --scores-dir "$RUN/main/scores/per_reaction" --output-dir "$RUN/additional/ri_comparison"
chem analysis/synrxn_analysis/analyze.py \
  --output-dir "$RUN/additional/synrxn_analysis"

# Category assignments do not depend on the metric and are shared between runs.
# If they are missing, first run: analyze analysis/missing_species/classify_missing_species.py
analyze analysis/missing_species/build_category_accuracy_table.py \
  --scores-dir "$RUN/main/scores/per_reaction" --output-dir "$RUN/additional/missing_species_results"
analyze analysis/statistical_tests/test_rb_vs_crb.py \
  --outcomes "$RUN/additional/missing_species_results/uid_model_outcomes.csv" \
  --output-dir "$RUN/additional/statistical_tests"
analyze analysis/split_sensitivity/analyze.py \
  --scores-dir "$RUN/main/scores/per_reaction" --output-dir "$RUN/additional/split_sensitivity"

chem analysis/unfiltered_evaluation/run.py \
  --output-dir "$RUN/additional/unfiltered_evaluation"
chem analysis/unfiltered_evaluation/build_appendix_table.py \
  --scores-dir "$RUN/additional/unfiltered_evaluation"
analyze analysis/uncurated_stereo_free_agreement/recompute.py \
  --output-dir "$RUN/additional/uncurated_stereo_free_agreement"
analyze analysis/uncurated_failure_modes/analyze.py \
  --consensus "$RUN/additional/uncurated_stereo_free_agreement/summary.csv" \
  --output-dir "$RUN/additional/uncurated_failure_modes"

# Draw the reviewed SynRXN cases whose prediction still differs from the reference.
analyze analysis/synrxn_analysis/plot_ambiguous_GT.py \
  --data "$RUN/additional/synrxn_analysis/per_reaction.csv" \
  --output-dir "$RUN/additional/synrxn_analysis/review"
```

Composition and similarity results do not depend on the metric. Rescore with the container recorded in each run's `provenance.json`, because a different RDKit version can change individual labels.

## Equivalence metric

The metric rewrites each reaction into a standard form, without changing its atoms or charge, and then compares it with the reference. Its rules are listed in [equivalence_rules.tex](equivalence_metric_audit/equivalence_rules.tex) and implemented in [equivalence.py](../src/completerxn/equivalence.py). `--metric equivalence_lenient` adds extra comparison rules for exploring how sensitive the results are to the metric. It is off by default, not used for paper results, and writes under the ignored `results/local/`. The category and similarity helpers in `utils/chemistry.py` do not affect accuracy.

## Historical results

The earlier v1 metric is kept only to reproduce historical results. Its code is [equivalence_v1.py](../src/completerxn/legacy/equivalence_v1.py) and its results are in [results/archive/metric_v1/](../results/archive/metric_v1/README.md). Both metrics score the same saved predictions.

```bash
# Rebuild the v1 main results table under results/local/reproduced_v1/.
bash analysis/reproduce_main_results.sh --metric v1

# Recalculate v1 scores into a new directory.
bash analysis/reproduce_main_results.sh --metric v1 --rescore --output-root /tmp/completerxn-v1
```

Analysis scripts that take `--metric` accept `v1`. Plotting scripts read v1 scores when `COMPLETERXN_RESULTS_ROOT=$PWD/results/archive/metric_v1` is set; give them a separate output directory so the archive stays unchanged.

The expert audit of the metric and the SynRXN case review were both sampled using v1 and are kept unchanged. How their judgments compare with the equivalence metric is described in [NOTES.md](NOTES.md#expert-audit-of-the-metric).

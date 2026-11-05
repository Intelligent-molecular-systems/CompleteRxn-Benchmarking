# Analysis notes

Details for checking or extending individual analyses. For which script produces which result, see the [analysis index](README.md).

## How numbers are aggregated

- **Main and sensitivity tables** report the mean and sample standard deviation of the five repetition percentages, calculated from unrounded scores.
- **Statistical tests** pool observations across the five repetitions. This is a different aggregation from the tables, so their numbers are not directly comparable.
- **Split sensitivity** uses two different test sets. Do not mix them:
  - `split_sweep.csv` comes from rerunning the split construction (7,549 OOD test reactions at its baseline setting).
  - `burden_sensitivity.csv` uses the exact paper split (7,457 test reactions before filtering, 6,158 evaluated).
- **Accuracy per missing-molecule category** joins the selected scores to the category assignments by reaction ID. It never uses accuracy flags stored in the raw prediction files.

## Stereochemistry in the CRB evaluation

RB and CRB are scored against the reference with stereochemistry, SynRBL against the reference without it, because SynRBL removes stereochemistry during input standardization. [crb_stereo_only_errors.py](stereo_errors/crb_stereo_only_errors.py) checks how much the stricter evaluation affects CRB by rescoring its balanced but incorrect top-1 predictions without stereochemistry.

Only 76 of CRB's 3,536 incorrect predictions (2.1%) match the reference when stereochemistry is ignored: 30 on the random split, 27 on the group split and 19 on the extreme OOD split, pooled over five repetitions. Ignoring stereochemistry would raise CRB's accuracy by 0.03 percentage points on the random and group splits and by 0.31 on the extreme OOD split. In 52 of the 76 cases, the prediction contains more stereochemistry annotations than the reference, 22 times on a reagent restored unchanged on the product side. In 17 it contains fewer, and in 7 it assigns stereochemistry differently. The cases are listed in `results/additional/stereo_errors/cases.csv`.

## RDKit versions

The category and statistical analyses summarize existing scores and do not recompute matches, so the RDKit version in their run records is that of the analysis container, not of scoring. The saved full-test and uncurated results were scored with a newer RDKit than the filtered benchmark results. Check the recorded versions before comparing individual labels between them.

## v1 metric details

- Older run records call the v1 metric `legacy`. Scripts still accept `legacy` as another name for `v1`.
- Imports through `utils/metrics.py` still use the v1 implementation. New code should import `completerxn`.
- Reproducing v1 results requires `PYTHONHASHSEED=0` (set automatically by `reproduce_main_results.sh`).
- The historical sensitivity and failure CSVs in the archive were copied from saved v1 reports. Their `export_provenance.json` files identify the rounded values copied.
- Helper scripts used only for historical cases and reports are in `legacy/` subfolders.

## Expert audit of the metric

A chemistry expert reviewed a fixed sample of 20 SynRBL predictions that were solved and balanced but that v1 marked as not matching the reference. The sample was drawn from the full test set of the random split, repetition 0 (see [select_synrbl_audit.py](equivalence_metric_audit/select_synrbl_audit.py)). 18 of the 20 cases are in the filtered test set; audit IDs 13 and 17 are not.

The expert judged 4 cases equivalent to the reference, 8 reference preferred, 3 SynRBL preferred, 3 both plausible and 2 uncertain. The equivalence metric accepts all four equivalent cases (IDs 4, 10, 11 and 19). This sample informed the metric's development, so it is not an independent test of the equivalence metric or an estimate of its error rate. The sample must not be redrawn after a metric change.

`analyze_assessments.py` reads the released answer key, the blinded sample and the expert annotations, and writes two CSVs. `--llm PATH` adds the separate LLM assessment; `--scores PATH` joins one saved full-test SynRBL score file.

## SynRXN case review

This is a separate review of 12 SynRXN cases, unrelated to the expert audit above. The cases were sampled (seed 73) from 146 balanced CRB predictions that v1 marked as not matching: 9 with molecules missing on the product side and 3 with molecules missing on both sides. The review found 8 plausible alternatives, 2 where CRB was preferred and 2 where the SynRXN reference was preferred.

With the equivalence metric, 5 of the 8 plausible alternatives match the reference. The remaining 7 are 3 plausible alternatives, 2 CRB preferred and 2 SynRXN preferred. The judgments and sample are unchanged; current labels are in `results/additional/synrxn_analysis/review/audit_status.csv`.

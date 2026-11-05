# Historical v1 results

This folder keeps the results computed with the earlier v1 version of the equivalence metric, together with the expert audit annotations. It exists only so that historical results can be reproduced. Current results use equivalence accuracy (identified as `v2` in run records) and are in `results/main/` and `results/additional/`. Older run records call v1 `legacy`.

The v1 implementation is [equivalence_v1.py](../../../src/completerxn/legacy/equivalence_v1.py) and is unchanged. Both metrics score the same predictions in `results/main/predictions/`. Git includes the compact score archive and selected figures. Large prediction files and generated reports are separate downloads or local files.

The score files and run records here are unchanged. Some were moved during a repository reorganization; [path_relocations.csv](../path_relocations.csv) lists their old and new locations. File hashes in the run records describe the files at the time of the run, before that move.

Run `bash analysis/reproduce_main_results.sh --metric v1` to rebuild the v1 main results table in `results/local/reproduced_v1/` without touching this folder. To recalculate v1 scores, add `--rescore --output-root PATH` with a new output folder. V1 requires `PYTHONHASHSEED=0` (set by the script). Use the RDKit version recorded for the original run, because another version may produce different labels.

The expert audit was sampled using v1. The equivalence metric accepts all four cases the expert judged equivalent. The audit is kept here unchanged as part of the original analysis. See the [analysis notes](../../../analysis/NOTES.md#expert-audit-of-the-metric) for details.

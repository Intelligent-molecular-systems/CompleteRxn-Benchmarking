#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
usage() {
  echo 'Usage: bash analysis/reproduce_main_results.sh [--metric v2|v1|equivalence_lenient] [--rescore] [--output-root PATH]'
}
metric=v2
rescore=false
release=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rescore) rescore=true; shift ;;
    --metric|--output-root)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { usage >&2; exit 2; }
      if [[ "$1" == --metric ]]; then metric="$2"; else release="$2"; fi
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
# Fix the process seed before imports, including in the historical container.
export PYTHONHASHSEED=0 APPTAINERENV_PYTHONHASHSEED=0
metric=$(python3 -c 'from utils.result_paths import metric_name; import sys; print(metric_name(sys.argv[1]))' "$metric")
release=$(python3 -c 'from utils.result_paths import results_root; from pathlib import Path; import sys; print(Path(sys.argv[2]).resolve() if sys.argv[2] else results_root(sys.argv[1]))' "$metric" "$release")
scores="$release/main/scores"
agreement="$release/additional/consensus_audit"
# The historical release is read-only. Rebuild its display table locally;
# full historical rescoring must explicitly choose a new release directory.
archive=$(python3 -c 'from utils.result_paths import results_root; print(results_root("v1"))')
if [[ "$metric" == v1 && "$release" == "$archive" ]] && ! $rescore; then
  python3 -c 'from utils.result_paths import verify_score_release; import sys; verify_score_release(sys.argv[1],"v1")' "$scores"
  python3 analysis/build_main_table.py --scores-dir "$scores" \
    --output-dir results/local/reproduced_v1/main/scores
  echo "Frozen v1 figures and agreement remain in $archive; full reruns require --rescore --output-root PATH."
  exit 0
fi
if $rescore; then
  if [[ -d "$scores" ]] && [[ -n "$(ls -A "$scores")" ]]; then
    echo "Score directory is not empty: $scores. Choose a new --output-root for --rescore." >&2
    exit 2
  fi
  apptainer exec models/constrained_reaction_balancer/CRB_env.sif \
    python analysis/score_benchmark.py --metric "$metric" --output-dir "$scores"
else
  python3 -c 'from utils.result_paths import verify_score_release; import sys; verify_score_release(sys.argv[1],sys.argv[2])' "$scores" "$metric"
fi
# Restore the compact archived score rows and rebuild the summaries first.
python3 analysis/refresh_main_results.py --scores-dir "$scores"
if $rescore || [[ ! -f "$scores/main_table_auxiliary_per_repetition.csv" ]]; then
  apptainer exec models/constrained_reaction_balancer/CRB_env.sif \
    python analysis/build_main_table.py --scores-dir "$scores" --recompute-auxiliary
fi
python3 analysis/build_main_table.py --scores-dir "$scores"
export COMPLETERXN_RESULTS_ROOT="$release" APPTAINERENV_COMPLETERXN_RESULTS_ROOT="$release"
MPLBACKEND=Agg apptainer exec infra/data_processing_splits/drfp_data_env.sif \
  python analysis/figures/plot_crb_synrbl_top2_by_missing_atoms.py
MPLBACKEND=Agg apptainer exec infra/data_processing_splits/drfp_data_env.sif \
  python analysis/figures/plot_synrbl_vs_crb_outcome_fractions.py
if [[ -f "$agreement/provenance.json" ]]; then
  python3 -c 'from utils.result_paths import verify_cached_agreement; import sys; print("Using completed agreement:", verify_cached_agreement(*sys.argv[1:]))' "$agreement" "$scores" "$metric"
else
  apptainer exec models/constrained_reaction_balancer/CRB_env.sif \
    python analysis/consensus_audit/recompute_agreement.py --metric "$metric" \
      --scores-dir "$scores/per_reaction" --output-dir "$agreement"
fi

#!/usr/bin/env bash
set -euo pipefail

# Evaluate SynRBL predictions once on full dataset, then aggregate stats by split.
# 
# Workflow:
# 1. Run evaluate.py ONCE on full benchmark_results.csv with --add-all-metrics
#    → full_results.csv (all reactions + metric columns + similarity)
# 2. Run aggregate_split_stats.py to compute stats per split type/replica
#    → split_aggregate_summary.csv
#
# Example:
#   bash models/SynRBL/run_eval_all_splits.sh \
#     --input-csv results/main/predictions/synrbl/benchmark_final_results_noRI_complete.csv \
#     --output-dir results/main/predictions/synrbl/eval_all_splits_noRI_complete

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INPUT_CSV="${ROOT_DIR}/results/main/predictions/synrbl/benchmark_final_results_noRI_complete.csv"
EVAL_SCRIPT="${ROOT_DIR}/models/SynRBL/evaluate.py"
AGGREGATE_SCRIPT="${ROOT_DIR}/models/SynRBL/aggregate_split_stats.py"
BENCHMARK_DIR="${ROOT_DIR}/data/Template_curated/benchmark"
BENCHMARK_FINAL="${ROOT_DIR}/data/Template_curated/benchmark/benchmark_final.csv"
OUTPUT_DIR=""
SCORING_METRIC="${SCORING_METRIC:-v2}"
PYTHON_BIN="python3"
ML_RESULTS_DIR="${ROOT_DIR}/results/main/predictions"
ML_EXPERIMENT="BS_PT"

SIMILARITY_METHOD="ecfp"
MIN_CONFIDENCE="0.5"
REAGENTS_IDENTIFIED="0"
ACCURACY_METRIC="deep_pragmatic_equivalence"
METRICS=(
  "exact_match"
  "proton_shuffle_match"
  "all_normalizations_match"
  "deep_pragmatic_equivalence_match"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scoring-metric) SCORING_METRIC="$2"; shift 2 ;;
    --input-csv) INPUT_CSV="$2"; shift 2 ;;
    --eval-script) EVAL_SCRIPT="$2"; shift 2 ;;
    --aggregate-script) AGGREGATE_SCRIPT="$2"; shift 2 ;;
    --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
    --benchmark-final) BENCHMARK_FINAL="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --ml-results-dir) ML_RESULTS_DIR="$2"; shift 2 ;;
    --ml-experiment) ML_EXPERIMENT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --similarity-method) SIMILARITY_METHOD="$2"; shift 2 ;;
    --min-confidence) MIN_CONFIDENCE="$2"; shift 2 ;;
    --reagents-identified) REAGENTS_IDENTIFIED="$2"; shift 2 ;;
    --accuracy-metric) ACCURACY_METRIC="$2"; shift 2 ;;
    --metrics)
      IFS=',' read -r -a METRICS <<< "$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_eval_all_splits.sh [options]
  --scoring-metric v2|v1      v2 by default; write scores separately from predictions

Options:
  --input-csv PATH            SynRBL prediction CSV (default: results/main/predictions/synrbl/benchmark_final_results_noRI_complete.csv)
  --eval-script PATH          Path to evaluate.py
  --aggregate-script PATH     Path to aggregate_split_stats.py
  --benchmark-dir PATH        Benchmark split root (contains random_splits/group_splits/extreme_ood_splits)
  --benchmark-final PATH      benchmark_final.csv for ML original_index mapping
  --output-dir PATH           Output directory (will store full_results.csv + split_aggregate_summary.csv)
  --ml-results-dir PATH       Root dir of ML fold outputs (top1_predictions_with_stats.txt)
  --ml-experiment NAME        ML experiment to match (BS_PT or CBS_PT; default: BS_PT)
  --python CMD                Python executable (default: python3)
  --similarity-method NAME    ecfp|set_match
  --min-confidence FLOAT      Min confidence used for self-reported mcs stats (GT accuracy metrics are not confidence-masked)
  --reagents-identified INT   0 or 1
  --accuracy-metric NAME      exact|proton_shuffle|all_normalizations|pragmatic_equivalence|deep_pragmatic_equivalence
  --metrics LIST              Comma-separated metric columns for split aggregation
            default: exact_match,proton_shuffle_match,all_normalizations_match,deep_pragmatic_equivalence_match
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

case "$SCORING_METRIC" in
  legacy) SCORING_METRIC=v1 ;;
  v1|v2|equivalence_lenient) ;;
  *) echo 'Unknown scoring metric' >&2; exit 2 ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/local/model_scores/$SCORING_METRIC/synrbl}"
if [[ "$SCORING_METRIC" != v1 ]]; then
  METRICS=(exact_match equivalent_match)
  if [[ "$ACCURACY_METRIC" == deep_pragmatic_equivalence ]]; then ACCURACY_METRIC=equivalence; fi
fi

resolve_file() {
  local raw="$1"
  local candidate
  if [[ "$raw" = /* ]]; then
    [[ -f "$raw" ]] && { echo "$raw"; return 0; }
    return 1
  fi

  # 1) as provided (relative to current cwd)
  candidate="$raw"
  [[ -f "$candidate" ]] && { echo "$(cd "$(dirname "$candidate")" && pwd)/$(basename "$candidate")"; return 0; }

  # 2) relative to repo root
  candidate="$ROOT_DIR/$raw"
  [[ -f "$candidate" ]] && { echo "$candidate"; return 0; }

  # 3) convenience: relative to results/SynRBL
  candidate="$ROOT_DIR/results/SynRBL/$raw"
  [[ -f "$candidate" ]] && { echo "$candidate"; return 0; }

  return 1
}

resolve_dir() {
  local raw="$1"
  if [[ "$raw" = /* ]]; then
    echo "$raw"
  else
    echo "$ROOT_DIR/$raw"
  fi
}

if ! INPUT_CSV="$(resolve_file "$INPUT_CSV")"; then
  echo "Input CSV not found: $INPUT_CSV" >&2
  echo "Tried: as-provided, repo-relative, and results/SynRBL-relative paths." >&2
  exit 1
fi

if ! EVAL_SCRIPT="$(resolve_file "$EVAL_SCRIPT")"; then
  echo "evaluate.py not found: $EVAL_SCRIPT" >&2
  exit 1
fi

if ! AGGREGATE_SCRIPT="$(resolve_file "$AGGREGATE_SCRIPT")"; then
  echo "aggregate_split_stats.py not found: $AGGREGATE_SCRIPT" >&2
  exit 1
fi

BENCHMARK_DIR="$(resolve_dir "$BENCHMARK_DIR")"
OUTPUT_DIR="$(resolve_dir "$OUTPUT_DIR")"
ML_RESULTS_DIR="$(resolve_dir "$ML_RESULTS_DIR")"

if ! BENCHMARK_FINAL="$(resolve_file "$BENCHMARK_FINAL")"; then
  echo "benchmark_final.csv not found: $BENCHMARK_FINAL" >&2
  exit 1
fi

for d in "$BENCHMARK_DIR"/random_splits "$BENCHMARK_DIR"/group_splits "$BENCHMARK_DIR"/extreme_ood_splits; do
  if [[ ! -d "$d" ]]; then
    echo "Split directory missing: $d" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"

FULL_RESULTS_CSV="${OUTPUT_DIR}/full_results.csv"
SUMMARY_CSV_UNFILTERED="${OUTPUT_DIR}/split_aggregate_summary_unfiltered.csv"
SUMMARY_CSV_MLFILTERED="${OUTPUT_DIR}/split_aggregate_summary_mlfiltered.csv"

echo "SynRBL Full-Dataset Evaluation + Split Aggregation"
echo "Input CSV:           $INPUT_CSV"
echo "Benchmark dir:       $BENCHMARK_DIR"
echo "Benchmark final:     $BENCHMARK_FINAL"
echo "ML results dir:      $ML_RESULTS_DIR"
echo "ML experiment:       $ML_EXPERIMENT"
echo "Full results CSV:    $FULL_RESULTS_CSV"
echo "Summary CSV (raw):   $SUMMARY_CSV_UNFILTERED"
echo "Summary CSV (ML):    $SUMMARY_CSV_MLFILTERED"
echo "Accuracy metric:     $ACCURACY_METRIC"
echo "Aggregate metrics:   ${METRICS[*]}"
echo "Min confidence:      $MIN_CONFIDENCE (self-reported stats only)"
echo ""

# Step 1: Evaluate once on full dataset with all metrics
echo "Step 1: Running evaluation on full dataset (--add-all-metrics)..."
"$PYTHON_BIN" "$EVAL_SCRIPT" "$INPUT_CSV" \
  --scoring-metric "$SCORING_METRIC" \
  --add-all-metrics \
  --accuracy-metric "$ACCURACY_METRIC" \
  --similarity-method "$SIMILARITY_METHOD" \
  --min-confidence "$MIN_CONFIDENCE" \
  --reagents-identified "$REAGENTS_IDENTIFIED" \
  --output-csv "$FULL_RESULTS_CSV"

echo "✓ Evaluation complete. Full results saved to: $FULL_RESULTS_CSV"
echo ""

# Step 2a: Aggregate stats by split (unfiltered)
echo "Step 2a: Aggregating stats by split type/replica (unfiltered)..."
"$PYTHON_BIN" "$AGGREGATE_SCRIPT" "$FULL_RESULTS_CSV" \
  --benchmark-dir "$BENCHMARK_DIR" \
  --metrics "${METRICS[@]}" \
  --min-confidence "$MIN_CONFIDENCE" \
  --output-csv "$SUMMARY_CSV_UNFILTERED"

echo "✓ Unfiltered aggregation complete: $SUMMARY_CSV_UNFILTERED"
echo ""

# Step 2b: Aggregate stats by split (ML top1 filtered)
echo "Step 2b: Aggregating stats with ML top1 row filtering..."
"$PYTHON_BIN" "$AGGREGATE_SCRIPT" "$FULL_RESULTS_CSV" \
  --benchmark-dir "$BENCHMARK_DIR" \
  --metrics "${METRICS[@]}" \
  --min-confidence "$MIN_CONFIDENCE" \
  --filter-to-ml-top1 \
  --ml-results-dir "$ML_RESULTS_DIR" \
  --ml-experiment "$ML_EXPERIMENT" \
  --benchmark-final "$BENCHMARK_FINAL" \
  --output-csv "$SUMMARY_CSV_MLFILTERED"

echo "✓ ML-filtered aggregation complete: $SUMMARY_CSV_MLFILTERED"
echo ""
echo "Done! Outputs:"
echo "  - $FULL_RESULTS_CSV (all reactions with metric columns)"
echo "  - $SUMMARY_CSV_UNFILTERED (per-split stats, no ML row filter)"
echo "  - $SUMMARY_CSV_MLFILTERED (per-split stats, ML top1 row universe)"


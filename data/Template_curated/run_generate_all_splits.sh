#!/usr/bin/env bash
set -euo pipefail

# One-call split generation for Template_curated benchmark data.
# Produces:
# - source fold assignments: rxn_groups/, ood_folds_missing_stats/
# - canonical split indices: random_splits/, group_splits/, extreme_ood_splits/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BENCH_CSV="data/Template_curated/benchmark/benchmark_final.csv"

if [[ ! -f "$BENCH_CSV" ]]; then
  echo "ERROR: Missing benchmark CSV: $BENCH_CSV" >&2
  echo "Run data/Template_curated/run_data_processing.sh first." >&2
  exit 1
fi

echo "[1/5] Building group fold assignments (rxn_groups/)"
python data/Template_curated/knngraph_group_splits.py \
  --infile "$BENCH_CSV" \
  --column CanonicalReaction \
  --leakage_tanimoto 0.55

echo "[2/5] Building extreme OOD fold assignments (ood_folds_missing_stats/)"
python data/Template_curated/knngraph_extreme_ood_splits.py \
  --infile "$BENCH_CSV" \
  --column CanonicalReaction \
  --leakage_tanimoto 0.55

echo "[3/5] Generating random canonical split indices (random_splits/)"
python data/Template_curated/data_split.py --split-type random

echo "[4/5] Generating group canonical split indices (group_splits/)"
python data/Template_curated/data_split.py --split-type group

echo "[5/5] Generating extreme OOD canonical split indices (extreme_ood_splits/)"
python data/Template_curated/data_split.py --split-type extreme_ood

echo "Done. Split outputs are under data/Template_curated/benchmark/."

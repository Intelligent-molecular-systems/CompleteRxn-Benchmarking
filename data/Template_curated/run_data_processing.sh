#!/usr/bin/env bash
set -euo pipefail

# One-call runner for Template_curated data processing steps 1-6.
# Run from anywhere; script switches to repository root automatically.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[1/6] Mapping USPTO to FlowER subsets"
python data/Template_curated/map_uspto_flower.py \
  --uspto data/Template_curated/USPTO_FlowER_raw/master_dataset_mapped.csv \
  --flower data/Template_curated/USPTO_FlowER_raw/flower_full_single_reactions_with_main_products.tsv \
  --output data/Template_curated/processed

echo "[2/6] Identifying reagents in FlowER reactions"
python data/Template_curated/reagent_identification_flower.py \
  --input data/Template_curated/USPTO_FlowER_raw/flower_full_single_reactions_with_main_products.tsv \
  --output data/Template_curated/processed/flower_reagents_identified.tsv

echo "[3/6] Joining FlowER reactions into USPTO 1:1 subset"
python data/Template_curated/add_flower_reactions_to_uspto.py \
  --uspto data/Template_curated/processed/uspto_flower_one_to_one.csv \
  --flower data/Template_curated/processed/flower_reagents_identified.tsv \
  --output data/Template_curated/processed/one_to_one_final.csv

echo "[4/6] Creating benchmark columns"
python data/Template_curated/benchmark_columns.py

echo "[5/6] Restoring stereochemistry"
python data/Template_curated/add_stereochemistry_flower.py

echo "[6/6] Extracting harmonized input-target differences"
python data/Template_curated/extract_input_target_diff.py \
  --input-csv data/Template_curated/benchmark/benchmark_stereo_resolved.csv \
  --output-csv data/Template_curated/benchmark/benchmark_final.csv \
  --attempt-src-multiplicity-fix \
  --exclude-source-not-in-target

echo "Done. Produced benchmark file: data/Template_curated/benchmark/benchmark_final.csv"

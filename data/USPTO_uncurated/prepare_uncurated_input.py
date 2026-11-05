#!/usr/bin/env python3
"""Prepare USPTO reactions outside the benchmark for model prediction.

Source:
  data/Template_curated/USPTO_FlowER_raw/master_dataset_mapped.csv

Filtering:
  Keep only rows whose Unique_identifier is NOT in
  data/Template_curated/benchmark/benchmark_final.csv

Transformation:
  Create InputReaction by moving reagents to reactant side:
    reactants>reagents>products  ->  reactants.reagents>>products

Output:
  data/USPTO_uncurated/uspto_uncurated_processed.csv
"""

import argparse
import os
import pandas as pd


def move_reagents_to_reactants(reaction: str) -> str:
    if not isinstance(reaction, str):
        return ""
    reaction = reaction.strip()
    if not reaction:
        return ""

    parts = reaction.split('>')
    if len(parts) != 3:
        return ""

    reactants, reagents, products = parts
    left_parts = [p for p in (reactants, reagents) if p]
    left = '.'.join(left_parts)
    return f"{left}>>{products}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare uncurated USPTO input-only dataset.")
    parser.add_argument(
        "--master-csv",
        default="data/Template_curated/USPTO_FlowER_raw/master_dataset_mapped.csv",
        help="Path to master_dataset_mapped.csv",
    )
    parser.add_argument(
        "--benchmark-csv",
        default="data/Template_curated/benchmark/benchmark_final.csv",
        help="Path to benchmark_final.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="data/USPTO_uncurated/uspto_uncurated_processed.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--id-col",
        default="Unique_identifier",
        help="Identifier column shared by both input CSVs",
    )
    parser.add_argument(
        "--reaction-col",
        default="CanonicalReaction",
        help="Reaction column in master CSV (expects reactants>reagents>products format)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Chunk size for streaming master CSV",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on kept rows for quick tests (0 means no cap)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.master_csv):
        raise FileNotFoundError(f"Master CSV not found: {args.master_csv}")
    if not os.path.exists(args.benchmark_csv):
        raise FileNotFoundError(f"Benchmark CSV not found: {args.benchmark_csv}")

    benchmark_ids = set(
        pd.read_csv(args.benchmark_csv, usecols=[args.id_col], low_memory=False)[args.id_col]
        .dropna()
        .astype(str)
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)

    usecols = [args.id_col, args.reaction_col]
    header_written = False
    seen_ids = set()

    total_rows = 0
    total_kept = 0
    total_filtered_benchmark = 0
    total_invalid_reaction = 0
    total_duplicate_ids = 0

    for chunk in pd.read_csv(args.master_csv, usecols=usecols, chunksize=args.chunksize, low_memory=False):
        total_rows += len(chunk)

        chunk[args.id_col] = chunk[args.id_col].astype(str)
        chunk[args.reaction_col] = chunk[args.reaction_col].astype(str)

        in_benchmark = chunk[args.id_col].isin(benchmark_ids)
        total_filtered_benchmark += int(in_benchmark.sum())
        chunk = chunk.loc[~in_benchmark].copy()

        # Keep first occurrence per identifier globally.
        dedup_keep = ~chunk[args.id_col].isin(seen_ids)
        total_duplicate_ids += int((~dedup_keep).sum())
        chunk = chunk.loc[dedup_keep].copy()
        seen_ids.update(chunk[args.id_col].tolist())

        chunk["InputReaction"] = chunk[args.reaction_col].apply(move_reagents_to_reactants)
        invalid_mask = chunk["InputReaction"].eq("")
        total_invalid_reaction += int(invalid_mask.sum())
        chunk = chunk.loc[~invalid_mask].copy()

        out_chunk = pd.DataFrame(
            {
                args.id_col: chunk[args.id_col].values,
                "CanonicalReaction": chunk[args.reaction_col].values,
                "InputReaction": chunk["InputReaction"].values,
                # Keep this alias to match existing conventions in other pipelines.
                "InputReaction_harmonized": chunk["InputReaction"].values,
            }
        )

        if args.max_rows > 0:
            remaining = args.max_rows - total_kept
            if remaining <= 0:
                break
            out_chunk = out_chunk.iloc[:remaining]

        if len(out_chunk) == 0:
            continue

        out_chunk.to_csv(args.output_csv, mode='a', header=not header_written, index=False)
        header_written = True
        total_kept += len(out_chunk)

        if args.max_rows > 0 and total_kept >= args.max_rows:
            break

    print(f"Total rows read from master: {total_rows}")
    print(f"Rows filtered (in benchmark): {total_filtered_benchmark}")
    print(f"Rows filtered (duplicate IDs): {total_duplicate_ids}")
    print(f"Rows filtered (invalid reaction format): {total_invalid_reaction}")
    print(f"Rows written: {total_kept}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()

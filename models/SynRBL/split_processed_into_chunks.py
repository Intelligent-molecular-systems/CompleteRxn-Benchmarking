#!/usr/bin/env python3
"""Split a CSV into chunks of N rows.

Usage:
  python models/SynRBL/split_processed_into_chunks.py \
    results/SynRBL/uspto_uncurated_InputReaction_processed.csv \
    results/SynRBL/uspto_uncurated_chunks/

Output files are named <stem>_chunk_0000.csv, _chunk_0001.csv, etc.
"""

import argparse
import math
import os

import pandas as pd


def split_csv(input_file: str, output_dir: str, chunk_size: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    total = sum(1 for _ in open(input_file)) - 1  # minus header
    n_chunks = math.ceil(total / chunk_size)
    pad = len(str(n_chunks - 1))
    stem = os.path.splitext(os.path.basename(input_file))[0]

    chunk_idx = 0
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False):
        out_path = os.path.join(output_dir, f"{stem}_chunk_{chunk_idx:0{pad}d}.csv")
        chunk.to_csv(out_path, index=False)
        print(f"  Wrote {len(chunk):>7} rows → {out_path}")
        chunk_idx += 1

    print(f"\nDone. {chunk_idx} chunks written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a CSV into fixed-size chunks.")
    parser.add_argument("input_file", help="Path to input CSV.")
    parser.add_argument("output_dir", help="Directory where chunk files will be written.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=40_000,
        help="Rows per chunk (default: 40000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_csv(args.input_file, args.output_dir, args.chunk_size)
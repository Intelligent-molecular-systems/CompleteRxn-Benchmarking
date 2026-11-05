#!/usr/bin/env python3
"""Finish the incomplete SynRBL benchmark run without repeating completed rows.

``prepare`` validates the existing output as a positional prefix of the full
processed input, then writes only the remaining rows as small independent CSV
chunks. ``merge`` validates every returned result against its input chunk,
repairs SynRBL-dropped UIDs by position, and creates a new complete result.

Examples, from the repository root::

    python models/SynRBL/resume_synrbl.py prepare
    python models/SynRBL/resume_synrbl.py merge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
UID = "Unique_identifier"
REFERENCE_COLUMNS = [UID, "rxn_flower", "rxn_reagents_identified_flower"]
CHUNK_PREFIX = "benchmark_final_noRI_remaining_"


def repo_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_uids(csv_path: Path) -> pd.Series:
    return pd.read_csv(csv_path, usecols=[UID], low_memory=False)[UID]


def validate_prefix(source_uids: pd.Series, partial_uids: pd.Series) -> int:
    resume_row = len(partial_uids)
    if resume_row > len(source_uids):
        raise ValueError(
            f"Partial output has {resume_row:,} rows but input has only "
            f"{len(source_uids):,}"
        )
    if source_uids.isna().any() or source_uids.astype(str).duplicated().any():
        raise ValueError("Processed input UIDs must be non-missing and unique")
    known = partial_uids.notna()
    observed = partial_uids.loc[known].astype(str).to_numpy()
    expected = source_uids.iloc[:resume_row].loc[known].astype(str).to_numpy()
    mismatches = int((observed != expected).sum())
    if mismatches:
        raise ValueError(
            f"Partial output is not a positional prefix: {mismatches:,} "
            "known UIDs differ"
        )
    return resume_row


def prepare(args: argparse.Namespace) -> None:
    source_path = repo_path(args.input)
    partial_path = repo_path(args.partial_results)
    chunks_dir = repo_path(args.chunks_dir)
    manifest_path = chunks_dir / "resume_manifest.json"

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if chunks_dir.exists() and any(chunks_dir.iterdir()):
        raise FileExistsError(
            f"Chunk directory is not empty: {chunks_dir}\n"
            "Use a new --chunks-dir or move the previous contents first."
        )

    source_uids = read_uids(source_path)
    partial_uids = read_uids(partial_path)
    resume_row = validate_prefix(source_uids, partial_uids)
    remaining = len(source_uids) - resume_row
    if remaining == 0:
        raise ValueError("The existing result already covers the complete input")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    skip_rows = range(1, resume_row + 1)  # preserve CSV header at line zero
    reader = pd.read_csv(
        source_path,
        skiprows=skip_rows,
        chunksize=args.chunk_size,
        low_memory=False,
    )
    manifest_chunks = []
    written = 0
    with tqdm(total=remaining, unit="row", desc="Writing remaining chunks") as bar:
        for number, chunk in enumerate(reader):
            chunk_path = chunks_dir / f"{CHUNK_PREFIX}{number:03d}.csv"
            chunk.to_csv(chunk_path, index=False)
            count = len(chunk)
            manifest_chunks.append(
                {
                    "input_file": chunk_path.name,
                    "result_file": f"{chunk_path.stem}_results.csv",
                    "rows": count,
                    "first_uid": str(chunk[UID].iloc[0]),
                    "last_uid": str(chunk[UID].iloc[-1]),
                }
            )
            written += count
            bar.update(count)

    if written != remaining:
        raise RuntimeError(f"Wrote {written:,} rows; expected {remaining:,}")
    manifest = {
        "source": str(source_path),
        "partial_results": str(partial_path),
        "source_rows": len(source_uids),
        "resume_row": resume_row,
        "remaining_rows": remaining,
        "chunk_size": args.chunk_size,
        "chunks": manifest_chunks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Validated prefix rows: {resume_row:,}")
    print(f"Blank UIDs in prefix:  {int(partial_uids.isna().sum()):,}")
    print(f"Remaining rows:         {remaining:,}")
    print(f"Created chunks:         {len(manifest_chunks)}")
    print(f"Chunks directory:       {chunks_dir}")
    print(f"Manifest:               {manifest_path}")


def restore_metadata(
    output: pd.DataFrame, source: pd.DataFrame, label: str
) -> pd.DataFrame:
    output = output.reset_index(drop=True).copy()
    source = source.reset_index(drop=True)
    if len(output) != len(source):
        raise ValueError(
            f"{label}: result has {len(output):,} rows; expected {len(source):,}"
        )
    known = output[UID].notna()
    observed = output.loc[known, UID].astype(str).to_numpy()
    expected = source.loc[known, UID].astype(str).to_numpy()
    mismatches = int((observed != expected).sum())
    if mismatches:
        raise ValueError(f"{label}: {mismatches:,} positional UID mismatches")
    for column in REFERENCE_COLUMNS:
        output[column] = source[column].to_numpy()
    return output


def merge(args: argparse.Namespace) -> None:
    source_path = repo_path(args.input)
    partial_path = repo_path(args.partial_results)
    chunks_dir = repo_path(args.chunks_dir)
    results_dir = repo_path(args.results_dir)
    output_path = repo_path(args.output)
    manifest_path = chunks_dir / "resume_manifest.json"

    if output_path.exists():
        raise FileExistsError(
            f"Complete output already exists: {output_path}\n"
            "Move it first if a replacement is intentional."
        )
    manifest = json.loads(manifest_path.read_text())
    source_uids = read_uids(source_path)
    partial = pd.read_csv(partial_path, low_memory=False)
    resume_row = validate_prefix(source_uids, partial[UID])
    if resume_row != int(manifest["resume_row"]):
        raise ValueError("Partial result length differs from the prepare manifest")

    prefix_source = pd.read_csv(
        source_path,
        usecols=REFERENCE_COLUMNS,
        nrows=resume_row,
        low_memory=False,
    )
    frames = [restore_metadata(partial, prefix_source, "existing prefix")]

    for item in tqdm(manifest["chunks"], desc="Validating result chunks"):
        input_path = chunks_dir / item["input_file"]
        result_path = results_dir / item["result_file"]
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing result chunk: {result_path}")
        chunk_source = pd.read_csv(
            input_path, usecols=REFERENCE_COLUMNS, low_memory=False
        )
        chunk_result = pd.read_csv(result_path, low_memory=False)
        frames.append(
            restore_metadata(chunk_result, chunk_source, result_path.name)
        )

    complete = pd.concat(frames, ignore_index=True)
    if len(complete) != len(source_uids):
        raise ValueError(
            f"Merged result has {len(complete):,} rows; expected "
            f"{len(source_uids):,}"
        )
    if complete[UID].isna().any() or not complete[UID].astype(str).is_unique:
        raise ValueError("Merged result UIDs are missing or duplicated")
    if not complete[UID].astype(str).equals(source_uids.astype(str)):
        raise ValueError("Merged UID order differs from the full processed input")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(output_path, index=False)
    print(f"Complete rows: {len(complete):,}")
    print(f"Unique UIDs:   {complete[UID].nunique():,}")
    print(f"Wrote:         {output_path}")


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        default="results/SynRBL/benchmark_final_processed.csv",
    )
    parser.add_argument(
        "--partial-results",
        default="results/SynRBL/benchmark_final_results.csv",
    )
    parser.add_argument(
        "--chunks-dir",
        default="results/SynRBL/benchmark_final_noRI_remaining_chunks",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    common_arguments(prepare_parser)
    prepare_parser.add_argument("--chunk-size", type=int, default=4096)
    prepare_parser.set_defaults(function=prepare)

    merge_parser = subparsers.add_parser("merge")
    common_arguments(merge_parser)
    merge_parser.add_argument(
        "--results-dir",
        default="results/SynRBL/benchmark_final_noRI_remaining_results",
    )
    merge_parser.add_argument(
        "--output",
        default="results/main/predictions/synrbl/benchmark_final_results_noRI_complete.csv",
    )
    merge_parser.set_defaults(function=merge)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

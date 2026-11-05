#!/usr/bin/env python3
"""Combine the saved CRB predictions for uncurated USPTO chunks.

Assumptions:
- Prediction chunks live in --crb-results-dir/chunk_XXXX/.
- For chunk IDs >= --retry-start-chunk, source lines come from retry src files
  in --retry-src-dir and are mapped by offset:
    retry chunk 0000 -> global chunk {retry_start_chunk}
- For chunk IDs < --retry-start-chunk, source lines come from --index-src-dir.

Outputs:
- merged_topn.txt
- merged_topn_log_probs.txt (optional)
- merged_src.txt
- merge_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _chunk_id_from_name(name: str) -> int | None:
    m = re.match(r"chunk_(\d+)$", name)
    if not m:
        return None
    return int(m.group(1))


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


@dataclass
class ChunkRecord:
    chunk_id: int
    pred_path: str
    logp_path: str
    src_path: str
    pred_lines: int
    logp_lines: int
    src_lines: int
    src_lines_used: int
    src_tail_dropped: int
    n_samples: int
    source_kind: str


def _build_retry_src_map(retry_src_dir: str, retry_start_chunk: int) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for name in sorted(os.listdir(retry_src_dir)):
        m = re.match(r"test_src_chunk_(\d+)$", name)
        if not m:
            continue
        retry_idx = int(m.group(1))
        global_chunk = retry_start_chunk + retry_idx
        mapping[global_chunk] = os.path.join(retry_src_dir, name)
    return mapping


def gather_chunk_records(
    crb_results_dir: str,
    index_src_dir: str,
    retry_src_dir: str,
    retry_start_chunk: int,
    n_best: int,
) -> List[ChunkRecord]:
    retry_src_map = _build_retry_src_map(retry_src_dir, retry_start_chunk)

    records: List[ChunkRecord] = []
    for name in sorted(os.listdir(crb_results_dir)):
        chunk_id = _chunk_id_from_name(name)
        if chunk_id is None:
            continue

        pred_path = os.path.join(crb_results_dir, name, "beamSize5_nBest5.txt")
        logp_path = os.path.join(crb_results_dir, name, "beamSize5_nBest5.txt_log_probs")
        if not (os.path.isfile(pred_path) and os.path.isfile(logp_path)):
            continue

        if chunk_id in retry_src_map:
            src_path = retry_src_map[chunk_id]
            source_kind = "retry"
        else:
            src_path = os.path.join(index_src_dir, f"test_src_chunk_{chunk_id:04d}")
            source_kind = "base"

        if not os.path.isfile(src_path):
            raise FileNotFoundError(
                f"Missing src file for chunk_{chunk_id:04d}: {src_path}"
            )

        pred_lines = sum(1 for _ in open(pred_path, "r", encoding="utf-8"))
        logp_lines = sum(1 for _ in open(logp_path, "r", encoding="utf-8"))
        src_lines = sum(1 for _ in open(src_path, "r", encoding="utf-8"))

        if pred_lines != logp_lines:
            raise ValueError(
                f"chunk_{chunk_id:04d}: pred/logp line mismatch "
                f"({pred_lines} vs {logp_lines})"
            )
        if pred_lines % n_best != 0:
            raise ValueError(
                f"chunk_{chunk_id:04d}: pred lines {pred_lines} not divisible by n_best={n_best}"
            )

        n_samples = pred_lines // n_best
        if n_samples > src_lines:
            raise ValueError(
                f"chunk_{chunk_id:04d}: n_samples={n_samples} exceeds src_lines={src_lines}"
            )
        src_lines_used = n_samples
        src_tail_dropped = src_lines - src_lines_used

        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                pred_path=pred_path,
                logp_path=logp_path,
                src_path=src_path,
                pred_lines=pred_lines,
                logp_lines=logp_lines,
                src_lines=src_lines,
                src_lines_used=src_lines_used,
                src_tail_dropped=src_tail_dropped,
                n_samples=n_samples,
                source_kind=source_kind,
            )
        )

    if not records:
        raise FileNotFoundError(f"No usable chunk predictions found in {crb_results_dir}")

    records.sort(key=lambda x: x.chunk_id)
    return records


def write_merged_outputs(
    records: List[ChunkRecord],
    out_pred: str,
    out_logp: str,
    out_src: str,
    out_manifest: str,
) -> Tuple[int, int]:
    os.makedirs(os.path.dirname(out_pred), exist_ok=True)

    total_pred_lines = 0
    total_src_lines = 0

    with open(out_pred, "w", encoding="utf-8") as pred_out, \
        open(out_logp, "w", encoding="utf-8") as logp_out, \
        open(out_src, "w", encoding="utf-8") as src_out:

        for rec in records:
            for line in _read_lines(rec.pred_path):
                pred_out.write(line + "\n")
            for line in _read_lines(rec.logp_path):
                logp_out.write(line + "\n")
            src_lines = _read_lines(rec.src_path)
            for line in src_lines[: rec.src_lines_used]:
                src_out.write(line + "\n")

            total_pred_lines += rec.pred_lines
            total_src_lines += rec.src_lines_used

    with open(out_manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "chunk_id",
                "source_kind",
                "pred_path",
                "logp_path",
                "src_path",
                "pred_lines",
                "logp_lines",
                "src_lines",
                "src_lines_used",
                "src_tail_dropped",
                "n_samples",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    f"{rec.chunk_id:04d}",
                    rec.source_kind,
                    rec.pred_path,
                    rec.logp_path,
                    rec.src_path,
                    rec.pred_lines,
                    rec.logp_lines,
                    rec.src_lines,
                    rec.src_lines_used,
                    rec.src_tail_dropped,
                    rec.n_samples,
                ]
            )

    return total_pred_lines, total_src_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CRB chunk predictions into one top-n file and one src file."
    )
    parser.add_argument(
        "--crb-results-dir",
        default="models/constrained_reaction_balancer/results/USPTO_uncurated",
        help="Directory containing chunk_XXXX with beamSize5_nBest5 outputs.",
    )
    parser.add_argument(
        "--index-src-dir",
        default="models/constrained_reaction_balancer/USPTO_uncurated/chunks",
        help="Directory containing base test_src_chunk_XXXX files.",
    )
    parser.add_argument(
        "--retry-src-dir",
        default="results/main/predictions/crb/uncurated/tail_retry/chunks_tail_retry",
        help="Directory containing retry test_src_chunk_XXXX files.",
    )
    parser.add_argument(
        "--retry-start-chunk",
        type=int,
        default=44,
        help="Global chunk ID where retry src chunks start (retry 0000 maps to this).",
    )
    parser.add_argument(
        "--n-best",
        type=int,
        default=5,
        help="Beam size used to validate pred/src alignment.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/main/predictions/crb/uncurated/merged",
        help="Output directory for merged files and manifest.",
    )
    parser.add_argument(
        "--pred-file-name",
        default="beamSize5_nBest5_all.txt",
        help="Merged predictions (top-n tokenized) filename.",
    )
    parser.add_argument(
        "--logp-file-name",
        default="beamSize5_nBest5_all.txt_log_probs",
        help="Merged log-probabilities filename.",
    )
    parser.add_argument(
        "--src-file-name",
        default="test_src_all.txt",
        help="Merged tokenized src filename.",
    )
    parser.add_argument(
        "--manifest-file-name",
        default="merge_manifest.csv",
        help="CSV containing per-chunk merge bookkeeping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = gather_chunk_records(
        crb_results_dir=args.crb_results_dir,
        index_src_dir=args.index_src_dir,
        retry_src_dir=args.retry_src_dir,
        retry_start_chunk=args.retry_start_chunk,
        n_best=args.n_best,
    )

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_pred = os.path.join(out_dir, args.pred_file_name)
    out_logp = os.path.join(out_dir, args.logp_file_name)
    out_src = os.path.join(out_dir, args.src_file_name)
    out_manifest = os.path.join(out_dir, args.manifest_file_name)

    total_pred_lines, total_src_lines = write_merged_outputs(
        records=records,
        out_pred=out_pred,
        out_logp=out_logp,
        out_src=out_src,
        out_manifest=out_manifest,
    )
    total_src_dropped = sum(r.src_tail_dropped for r in records)

    print(f"Merged chunks: {len(records)}")
    print(f"Chunk range: {records[0].chunk_id:04d}..{records[-1].chunk_id:04d}")
    print(f"Total pred lines: {total_pred_lines:,}")
    print(f"Total src lines: {total_src_lines:,}")
    print(f"Total src tail rows dropped: {total_src_dropped:,}")
    print(f"Expected pred/src ratio (n_best): {args.n_best}")
    print(f"Output predictions: {out_pred}")
    print(f"Output log_probs:   {out_logp}")
    print(f"Output src:         {out_src}")
    print(f"Manifest:           {out_manifest}")


if __name__ == "__main__":
    main()

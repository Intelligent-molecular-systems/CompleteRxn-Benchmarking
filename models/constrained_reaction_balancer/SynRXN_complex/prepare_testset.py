#!/usr/bin/env python3
"""Prepare and preprocess a single SynRXN test set for CRB model testing.

This script is a simplified, test-only variant of:
- models/constrained_reaction_balancer/prepare_train_val_test.py
- models/constrained_reaction_balancer/preprocess.py

It uses:
- source column: rxn
- target column: input_target_diff

Outputs are written to:
- <base_output_dir>/prepared
- <base_output_dir>/preprocessed
"""

import argparse
import os
import re
from typing import Tuple

import pandas as pd


def smi_tokenizer(smi: str) -> str:
    """Tokenize a SMILES molecule or reaction string."""
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens), f"Tokenization failed for: {smi}"
    return " ".join(tokens)


def _clean_pair_dataframe(df: pd.DataFrame, src_col: str, tgt_col: str) -> pd.DataFrame:
    """Keep rows with non-empty source and target strings only."""
    out = df[[src_col, tgt_col]].copy()
    out[src_col] = out[src_col].astype(str).str.strip()
    out[tgt_col] = out[tgt_col].astype(str).str.strip()

    # Remove rows that are NaN-coerced strings or empty.
    invalid = {
        "",
        "nan",
        "none",
        "null",
    }
    mask = ~out[src_col].str.lower().isin(invalid) & ~out[tgt_col].str.lower().isin(invalid)
    return out.loc[mask].reset_index(drop=True)


def _write_lines(path: str, series: pd.Series) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    series.to_csv(path, index=False, header=False)


def prepare_and_preprocess(
    input_csv: str,
    base_output_dir: str,
    src_col: str = "rxn",
    tgt_col: str = "input_target_diff",
) -> str:
    """Create prepared test files for SynRXN CRB testing."""
    df = pd.read_csv(input_csv)

    missing = {src_col, tgt_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean_df = _clean_pair_dataframe(df, src_col=src_col, tgt_col=tgt_col)
    if clean_df.empty:
        raise ValueError("No valid rows left after filtering empty/missing source-target pairs.")

    prepared_dir = os.path.join(base_output_dir, "prepared")
    os.makedirs(prepared_dir, exist_ok=True)

    # Prepared: raw, untokenized test files.
    prepared_test_src = os.path.join(prepared_dir, "test_src_raw.txt")
    prepared_test_tgt = os.path.join(prepared_dir, "test_tgt_raw.txt")
    _write_lines(prepared_test_src, clean_df[src_col])
    _write_lines(prepared_test_tgt, clean_df[tgt_col])

    # Prepared tokenized files for model testing.
    tokenized_src = clean_df[src_col].apply(smi_tokenizer)
    tokenized_tgt = clean_df[tgt_col].apply(smi_tokenizer)

    # Keep only test-set outputs for this single-set SynRXN pipeline.
    _write_lines(os.path.join(prepared_dir, "test_src.txt"), tokenized_src)
    _write_lines(os.path.join(prepared_dir, "test_tgt.txt"), tokenized_tgt)

    # Remove stale validation files from earlier runs if present.
    for stale_name in ("valid_src.txt", "valid_tgt.txt"):
        stale_path = os.path.join(prepared_dir, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    # Keep a filtered csv for traceability.
    clean_df.to_csv(os.path.join(prepared_dir, "test_filtered.csv"), index=False)

    print(f"Input rows: {len(df)}")
    print(f"Valid rows kept: {len(clean_df)}")
    print(f"Prepared dir: {prepared_dir}")
    print(f"Wrote tokenized source for testing: {os.path.join(prepared_dir, 'test_src.txt')}")

    return prepared_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and preprocess SynRXN single test set for CRB.")
    parser.add_argument(
        "--input-csv",
        default="data/SynRXN_complex/complex_processed.csv",
        help="Path to processed SynRXN csv.",
    )
    parser.add_argument(
        "--base-output-dir",
        default="models/constrained_reaction_balancer/SynRXN_complex",
        help="Base output directory containing prepared/ and preprocessed/.",
    )
    parser.add_argument("--src-col", default="rxn", help="Source column name.")
    parser.add_argument("--tgt-col", default="input_target_diff", help="Target column name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_and_preprocess(
        input_csv=args.input_csv,
        base_output_dir=args.base_output_dir,
        src_col=args.src_col,
        tgt_col=args.tgt_col,
    )


if __name__ == "__main__":
    main()

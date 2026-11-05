#!/usr/bin/env python3
"""Prepare USPTO reactions without reference completions for CRB prediction.

This script creates test-only source files (no targets):
- <base_output_dir>/prepared/test_src_raw.txt
- <base_output_dir>/prepared/test_src.txt
- <base_output_dir>/prepared/test_index.csv

It mirrors the lightweight preparation style used for SynRXN test sets,
but only requires a reaction input column.
"""

import argparse
import os
import re
import subprocess
import sys
import pandas as pd


def smi_tokenizer(smi: str) -> str:
    #pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    if smi != "".join(tokens):
        raise ValueError(f"Tokenization failed for: {smi}")
    return " ".join(tokens)



def safe_smi_tokenizer(smi: str):
    try:
        return smi_tokenizer(smi)
    except Exception:
        return None


def choose_reaction_col(df: pd.DataFrame, requested: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested reaction column not found: {requested}")
        return requested

    candidates = [
        "InputReaction_harmonized",
        "InputReaction",
        "InputReaction_canonical",
        "CanonicalReaction",
        "ReactionSmiles",
        "rxn",
        "reaction",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "No reaction column found. Pass --reaction-col explicitly. "
        f"Columns available: {list(df.columns)}"
    )


def clean_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    invalid = {"", "nan", "none", "null"}
    return out[~out.str.lower().isin(invalid)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare no-GT USPTO test source files for CRB inference.")
    parser.add_argument(
        "--input-csv",
        default="data/USPTO_uncurated/uspto_uncurated_processed.csv",
        help="Input CSV with reaction column (typically produced by prepare_uncurated_input.py).",
    )
    parser.add_argument(
        "--base-output-dir",
        default="models/constrained_reaction_balancer/USPTO_uncurated",
        help="Base output directory that will contain prepared/.",
    )
    parser.add_argument(
        "--reaction-col",
        default="",
        help="Reaction source column. If omitted, auto-detected.",
    )
    parser.add_argument(
        "--id-col",
        default="Unique_identifier",
        help="Optional identifier column to keep in test_index.csv.",
    )
    parser.add_argument(
        "--run-upstream-prep",
        action="store_true",
        help="Run data/USPTO_uncurated/prepare_uncurated_input.py first.",
    )
    parser.add_argument(
        "--master-csv",
        default="data/Template_curated/USPTO_FlowER_raw/master_dataset_mapped.csv",
        help="Source master CSV for upstream prep.",
    )
    parser.add_argument(
        "--benchmark-csv",
        default="data/Template_curated/benchmark/benchmark_final.csv",
        help="Benchmark CSV used to exclude already-benchmarked IDs in upstream prep.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Chunk size for upstream prep.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row cap for upstream prep (0 means no cap).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))

    if args.run_upstream_prep:
        upstream_script = os.path.join(repo_root, "data", "USPTO_uncurated", "prepare_uncurated_input.py")
        if not os.path.exists(upstream_script):
            raise FileNotFoundError(f"Upstream prep script not found: {upstream_script}")

        cmd = [
            sys.executable,
            upstream_script,
            "--master-csv", args.master_csv,
            "--benchmark-csv", args.benchmark_csv,
            "--output-csv", args.input_csv,
            "--chunksize", str(args.chunksize),
        ]
        if args.max_rows > 0:
            cmd.extend(["--max-rows", str(args.max_rows)])

        print("Running upstream uncurated prep:")
        print("  " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(
            f"Input CSV not found: {args.input_csv}. "
            "Use --run-upstream-prep to create it first."
        )

    df = pd.read_csv(args.input_csv, low_memory=False)
    rxn_col = choose_reaction_col(df, args.reaction_col)

    cleaned = clean_series(df[rxn_col])
    kept_idx = cleaned.index

    out_df = pd.DataFrame({
        "original_index": kept_idx,
        "input_reaction": cleaned.values,
    })
    if args.id_col in df.columns:
        out_df[args.id_col] = df.loc[kept_idx, args.id_col].values

    prepared_dir = os.path.join(args.base_output_dir, "prepared")
    os.makedirs(prepared_dir, exist_ok=True)

    raw_src_path = os.path.join(prepared_dir, "test_src_raw.txt")
    tok_src_path = os.path.join(prepared_dir, "test_src.txt")
    idx_path = os.path.join(prepared_dir, "test_index.csv")

    out_df["tokenized_input_reaction"] = out_df["input_reaction"].apply(safe_smi_tokenizer)
    failed_tok = int(out_df["tokenized_input_reaction"].isna().sum())
    if failed_tok > 0:
        failed_rows = out_df[out_df["tokenized_input_reaction"].isna()]
        failed_rows.to_csv(os.path.join(prepared_dir, "failed_tokenization.csv"), index=False)
        out_df = out_df.loc[out_df["tokenized_input_reaction"].notna()].copy()

    out_df["input_reaction"].to_csv(raw_src_path, index=False, header=False)
    out_df["tokenized_input_reaction"].to_csv(tok_src_path, index=False, header=False)
    out_df.drop(columns=["tokenized_input_reaction"]).to_csv(idx_path, index=False)

    print(f"Loaded rows: {len(df)}")
    print(f"Kept rows after cleaning: {len(cleaned)}")
    if failed_tok > 0:
        print(f"Dropped untokenizable rows: {failed_tok}")
    print(f"Final rows written: {len(out_df)}")
    print(f"Reaction column: {rxn_col}")
    print(f"Wrote: {raw_src_path}")
    print(f"Wrote: {tok_src_path}")
    print(f"Wrote: {idx_path}")


if __name__ == "__main__":
    main()

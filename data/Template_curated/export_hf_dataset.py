"""Export the selected columns used by the original Hugging Face dataset.

Write benchmark_hf.csv beside the input. This reduced export omits columns
needed by some paper analyses, including InputReaction_harmonized.
Run from the repository root with: python data/Template_curated/export_hf_dataset.py
"""

import argparse
import pandas as pd
from pathlib import Path

KEEP_COLUMNS = [
    "Unique_identifier",
    "OriginalReaction",
    "InputReaction",
    "TargetReaction",
    "TargetReaction_with_stereo",
    "has_unresolved_stereo",
    "CanonicalOriginalReaction",
    "CanonicalReaction",
    "MissingNrAtoms",
    "MissingNrCarbons",
    "MissingMw",
    "IsBalanced",
    "Balance",
    "rxn_class",
    "1kTPLLabel",
    "FlowER_ID",
    "PatentNumber",
    "ParagraphNum",
    "Year",
]

DEFAULT_INPUT = "data/Template_curated/benchmark/benchmark_final.csv"
DEFAULT_OUTPUT = "data/Template_curated/benchmark/benchmark_hf.csv"


def main():
    parser = argparse.ArgumentParser(description="Export simplified HF dataset CSV.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)

    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found in input: {missing}")

    df_out = df[KEEP_COLUMNS]
    print(f"Kept {len(KEEP_COLUMNS)} of {len(df.columns)} columns ({len(df_out):,} rows).")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()

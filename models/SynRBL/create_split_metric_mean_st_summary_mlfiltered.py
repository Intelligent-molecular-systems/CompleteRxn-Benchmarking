#!/usr/bin/env python3
"""Build split-level mean/std accuracy summary from mlfiltered aggregate stats.

Default input:
  results/SynRBL/eval_all_splits/split_aggregate_summary_mlfiltered.csv

Default output:
  results/SynRBL/eval_all_splits/split_metric_mean_st_summary_mlfiltered.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_MAP = {
    "exact_match": "exact_acc",
    "proton_shuffle_match": "proton_shuffle_acc",
    "all_normalizations_match": "all_normalizations_acc",
    "deep_pragmatic_equivalence_match": "deep_pragmatic_acc",
}


def build_summary(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    # Backward compatibility: older aggregate CSVs expose solved percentage as
    # success_rate (or derivable from solved_cnt/reaction_cnt) rather than solved_pct.
    if "solved_pct" not in df.columns:
        if "success_rate" in df.columns:
            df["solved_pct"] = pd.to_numeric(df["success_rate"], errors="coerce")
        elif {"solved_cnt", "reaction_cnt"}.issubset(df.columns):
            solved_cnt = pd.to_numeric(df["solved_cnt"], errors="coerce")
            reaction_cnt = pd.to_numeric(df["reaction_cnt"], errors="coerce")
            df["solved_pct"] = solved_cnt.div(reaction_cnt.where(reaction_cnt != 0))

    required = {"split_type", "metric", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing)}")

    # Keep only the metrics requested by user.
    work = df[df["metric"].isin(METRIC_MAP.keys())].copy()
    if work.empty:
        raise ValueError("No rows found for requested metrics in input CSV.")

    # Rename metric names for compact output labels.
    work["metric"] = work["metric"].map(METRIC_MAP)

    # Number of reps: use unique rep count when available, otherwise row count.
    if "rep" in work.columns:
        grouped = work.groupby(["split_type", "metric"], dropna=False)
        summary = grouped.agg(
            mean=("accuracy", "mean"),
            std=("accuracy", "std"),
            n_reps=("rep", pd.Series.nunique),
        ).reset_index()
    else:
        grouped = work.groupby(["split_type", "metric"], dropna=False)
        summary = grouped.agg(
            mean=("accuracy", "mean"),
            std=("accuracy", "std"),
            n_reps=("accuracy", "size"),
        ).reset_index()

    summary["mean_pct"] = summary["mean"] * 100.0
    summary["std_pct"] = summary["std"] * 100.0

    summary = summary[
        ["split_type", "metric", "mean", "std", "n_reps", "mean_pct", "std_pct"]
    ]

    def _append_optional_metric(
        summary_df: pd.DataFrame,
        source_df: pd.DataFrame,
        value_col: str,
        metric_label: str,
    ) -> pd.DataFrame:
        if value_col not in source_df.columns:
            return summary_df

        # One value per (split_type, rep) - deduplicate before aggregating.
        group_cols = ["split_type", "rep"] if "rep" in source_df.columns else ["split_type"]
        metric_df = source_df[group_cols + [value_col]].drop_duplicates(subset=group_cols)

        if "rep" in metric_df.columns:
            metric_grouped = metric_df.groupby("split_type", dropna=False)
            metric_summary = metric_grouped.agg(
                mean=(value_col, "mean"),
                std=(value_col, "std"),
                n_reps=("rep", pd.Series.nunique),
            ).reset_index()
        else:
            metric_grouped = metric_df.groupby("split_type", dropna=False)
            metric_summary = metric_grouped.agg(
                mean=(value_col, "mean"),
                std=(value_col, "std"),
                n_reps=(value_col, "size"),
            ).reset_index()

        metric_summary["metric"] = metric_label
        metric_summary["mean_pct"] = metric_summary["mean"] * 100.0
        metric_summary["std_pct"] = metric_summary["std"] * 100.0
        metric_summary = metric_summary[
            ["split_type", "metric", "mean", "std", "n_reps", "mean_pct", "std_pct"]
        ]
        return pd.concat([summary_df, metric_summary], ignore_index=True)

    # Append SynRBL self-reported accuracy and solved percentage when available.
    summary = _append_optional_metric(
        summary_df=summary,
        source_df=df,
        value_col="SynRBL_self_reported_acc",
        metric_label="synrbl_self_reported_acc",
    )
    summary = _append_optional_metric(
        summary_df=summary,
        source_df=df,
        value_col="solved_pct",
        metric_label="solved_pct",
    )

    summary = summary.sort_values(["split_type", "metric"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create split_metric_mean_std_summary_mlfiltered.csv from split_aggregate_summary_mlfiltered.csv"
    )
    parser.add_argument(
        "--input-csv",
        default="results/SynRBL/eval_all_splits/split_aggregate_summary_mlfiltered.csv",
        help="Path to split_aggregate_summary_mlfiltered.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="results/SynRBL/eval_all_splits/split_metric_mean_std_summary_mlfiltered.csv",
        help="Path to output summary CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    summary = build_summary(input_csv=input_csv, output_csv=output_csv)

    print(f"Wrote {len(summary)} rows to {output_csv}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

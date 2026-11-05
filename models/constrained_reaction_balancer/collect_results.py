#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
from pathlib import Path
import pandas as pd


METRIC_KEYS = [
    "exact",
    "proton_shuffle",
    "all_normalizations",
    "deep_pragmatic_equivalence",
]


def _parse_pct(val):
    """Convert percentage-like strings (e.g. '89.2%') to float."""
    return float(str(val).strip().rstrip("%"))


def parse_summary_file(path):
    row = {
        "summary_path": path,
        "experiment_type": "",
        "split_type": "",
        "seed": "",
        "rep": "",
        "excluded_count": "",
        "excluded_pct": "",
        "remaining_count": "",
    }

    # Parse either the current benchmark layout or older model-local runs.
    norm_path = os.path.normpath(path)
    parts = norm_path.split(os.sep)

    if len(parts) >= 5 and parts[-4] in {"rb", "crb"} and parts[-3] in {"random", "group", "extreme_ood"}:
        row["experiment_type"] = {"rb": "BS_PT", "crb": "CBS_PT"}[parts[-4]]
        split_suffix = parts[-2]
    # Older layout: results / experimentType / splitSuffix / filename
    else:
        split_suffix = parts[-2] if len(parts) >= 4 else ""
        if len(parts) >= 4:
            row["experiment_type"] = parts[-3]

    if len(parts) >= 4:
        m_random = re.match(r"random_seed(\d+)_rep(\d+)", split_suffix)
        m_group = re.match(r"group_rep(\d+)", split_suffix)
        m_extreme_ood = re.match(r"extreme_ood_rep(\d+)", split_suffix)

        if m_random:
            row["split_type"] = "random"
            row["seed"] = m_random.group(1)
            row["rep"] = m_random.group(2)
        elif m_group:
            row["split_type"] = "group"
            row["rep"] = m_group.group(1)
        elif m_extreme_ood:
            row["split_type"] = "extreme_ood"
            row["rep"] = m_extreme_ood.group(1)
        else:
            row["split_type"] = split_suffix

    # Prepare top-k columns
    for k in range(1, 6):
        row[f"top{k}_acc"] = ""
        row[f"top{k}_exact_acc"] = ""
        row[f"top{k}_proton_shuffle_acc"] = ""
        row[f"top{k}_all_normalizations_acc"] = ""
        row[f"top{k}_deep_pragmatic_equivalence_acc"] = ""
        row[f"top{k}_invalid_smiles_pct"] = ""
        row[f"top{k}_balanced_pct"] = ""

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    table_headers = None
    for line in lines:
        # Exclusion line
        # Example:
        # Excluded 7321 reactions with unknown tokens (14.64%). 42679 reactions remain for evaluation.
        m_excl = re.match(
            r"Excluded\s+(\d+)\s+reactions\s+with\s+unknown\s+tokens\s+\(([\d.]+)%\)\.\s+(\d+)\s+reactions\s+remain\s+for\s+evaluation\.",
            line
        )
        if m_excl:
            row["excluded_count"] = int(m_excl.group(1))
            row["excluded_pct"] = float(m_excl.group(2))
            row["remaining_count"] = int(m_excl.group(3))
            continue

        # Top-k line variants
        # Top-1: 78.3%
        # Top-1: 78.3% || Balanced 65.2%
        # Top-1: 78.3% || Invalid SMILES 1.23%
        # Top-1: 78.3% || Invalid SMILES 1.23% || Balanced 65.2%
        m_top = re.match(r"Top-(\d+):\s+([\d.]+)%", line)
        if m_top:
            k = int(m_top.group(1))
            if 1 <= k <= 5:
                row[f"top{k}_acc"] = float(m_top.group(2))

                m_invalid = re.search(r"Invalid SMILES\s+([\d.]+)%", line)
                if m_invalid:
                    row[f"top{k}_invalid_smiles_pct"] = float(m_invalid.group(1))

                m_bal = re.search(r"Balanced\s+([\d.]+)%", line)
                if m_bal:
                    row[f"top{k}_balanced_pct"] = float(m_bal.group(1))

        # New tabular format header, e.g.:
        # k  | exact | proton_shuffle | all_normalizations | ... | invalid_smiles | balanced_beam1
        if line.startswith("k") and "|" in line:
            table_headers = [part.strip() for part in line.split("|")]
            continue

        # New tabular format rows: parse dynamically based on header columns.
        if table_headers is not None and re.match(r"^\d+\s*\|", line):
            parts = [part.strip() for part in line.split("|")]
            if len(parts) == len(table_headers):
                k = int(parts[0])
                if 1 <= k <= 5:
                    values = dict(zip(table_headers[1:], parts[1:]))

                    if "exact" in values:
                        row[f"top{k}_exact_acc"] = _parse_pct(values["exact"])
                        row[f"top{k}_acc"] = row[f"top{k}_exact_acc"]
                    if "proton_shuffle" in values:
                        row[f"top{k}_proton_shuffle_acc"] = _parse_pct(values["proton_shuffle"])
                    if "all_normalizations" in values:
                        row[f"top{k}_all_normalizations_acc"] = _parse_pct(values["all_normalizations"])
                    if "deep_pragmatic_equivalence" in values:
                        row[f"top{k}_deep_pragmatic_equivalence_acc"] = _parse_pct(values["deep_pragmatic_equivalence"])
                    if "invalid_smiles" in values:
                        row[f"top{k}_invalid_smiles_pct"] = _parse_pct(values["invalid_smiles"])
                    if "balanced_beam1" in values:
                        row[f"top{k}_balanced_pct"] = _parse_pct(values["balanced_beam1"])

    return row


def main():
    print("Legacy rounded-log collector: use analysis/score_benchmark.py and analysis/build_main_table.py for manuscript results.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "results/main/predictions"),
        help="Root results directory to search"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/legacy/compiled_summary.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/legacy/compiled_summary_table.csv",
        help="Output CSV file for the summary-ready aggregate table"
    )
    args = parser.parse_args()

    summary_files = []
    for root, _, files in os.walk(args.results_dir):
        for fname in files:
            if fname.endswith("_summary.txt"):
                summary_files.append(os.path.join(root, fname))

    if not summary_files:
        raise FileNotFoundError(f"No *_summary.txt files found under: {args.results_dir}")

    rows = [parse_summary_file(path) for path in sorted(summary_files)]

    fieldnames = [
        "experiment_type",
        "split_type",
        "seed",
        "rep",
        "excluded_count",
        "excluded_pct",
        "remaining_count",
        "top1_acc",
        "top1_exact_acc",
        "top1_proton_shuffle_acc",
        "top1_all_normalizations_acc",
        "top1_deep_pragmatic_equivalence_acc",
        "top1_invalid_smiles_pct",
        "top1_balanced_pct",
        "top2_acc",
        "top2_exact_acc",
        "top2_proton_shuffle_acc",
        "top2_all_normalizations_acc",
        "top2_deep_pragmatic_equivalence_acc",
        "top2_invalid_smiles_pct",
        "top2_balanced_pct",
        "top3_acc",
        "top3_exact_acc",
        "top3_proton_shuffle_acc",
        "top3_all_normalizations_acc",
        "top3_deep_pragmatic_equivalence_acc",
        "top3_invalid_smiles_pct",
        "top3_balanced_pct",
        "top4_acc",
        "top4_exact_acc",
        "top4_proton_shuffle_acc",
        "top4_all_normalizations_acc",
        "top4_deep_pragmatic_equivalence_acc",
        "top4_invalid_smiles_pct",
        "top4_balanced_pct",
        "top5_acc",
        "top5_exact_acc",
        "top5_proton_shuffle_acc",
        "top5_all_normalizations_acc",
        "top5_deep_pragmatic_equivalence_acc",
        "top5_invalid_smiles_pct",
        "top5_balanced_pct",
        "summary_path",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")

    # Read the file that was just written
    df = pd.read_csv(args.output)

    # Metrics to summarize
    metrics = [
        "excluded_pct",
        "top1_exact_acc",
        "top1_proton_shuffle_acc",
        "top1_all_normalizations_acc",
        "top1_deep_pragmatic_equivalence_acc",
        "top5_exact_acc",
        "top5_proton_shuffle_acc",
        "top5_all_normalizations_acc",
        "top5_deep_pragmatic_equivalence_acc",
        "top1_invalid_smiles_pct",
        "top1_balanced_pct",
    ]

    # Keep only numeric columns that actually exist
    metrics = [m for m in metrics if m in df.columns]

    # Aggregate mean and std by experiment + split
    agg = df.groupby(["experiment_type", "split_type"], dropna=False)[metrics].agg(["mean", "std"]).reset_index()

    # Flatten multi-index columns
    agg.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in agg.columns.to_flat_index()
    ]

    # Optional nicer ordering
    split_order = {"random": 0, "group": 1}
    agg["split_order"] = agg["split_type"].map(split_order).fillna(99)
    agg = agg.sort_values(["experiment_type", "split_order"]).drop(columns=["split_order"])

    # Build a nicer presentation table
    summary_rows = []
    for _, row in agg.iterrows():
        out = {
            "Experiment": row["experiment_type"],
            "Split": row["split_type"],
        }

        if "excluded_pct_mean" in row.index:
            out["Excluded %"] = f"{row['excluded_pct_mean']:.2f} ± {row['excluded_pct_std']:.2f}"
        if "top1_exact_acc_mean" in row.index:
            out["Top-1 Exact %"] = f"{row['top1_exact_acc_mean']:.2f} ± {row['top1_exact_acc_std']:.2f}"
        if "top1_proton_shuffle_acc_mean" in row.index:
            out["Top-1 Proton Shuffle %"] = (
                f"{row['top1_proton_shuffle_acc_mean']:.2f} ± "
                f"{row['top1_proton_shuffle_acc_std']:.2f}"
            )
        if "top1_all_normalizations_acc_mean" in row.index:
            out["Top-1 All Normalizations %"] = (
                f"{row['top1_all_normalizations_acc_mean']:.2f} ± "
                f"{row['top1_all_normalizations_acc_std']:.2f}"
            )
        if "top1_deep_pragmatic_equivalence_acc_mean" in row.index:
            out["Top-1 Deep Pragmatic %"] = (
                f"{row['top1_deep_pragmatic_equivalence_acc_mean']:.2f} ± "
                f"{row['top1_deep_pragmatic_equivalence_acc_std']:.2f}"
            )
        if "top5_exact_acc_mean" in row.index:
            out["Top-5 Exact %"] = f"{row['top5_exact_acc_mean']:.2f} ± {row['top5_exact_acc_std']:.2f}"
        if "top5_proton_shuffle_acc_mean" in row.index:
            out["Top-5 Proton Shuffle %"] = (
                f"{row['top5_proton_shuffle_acc_mean']:.2f} ± "
                f"{row['top5_proton_shuffle_acc_std']:.2f}"
            )
        if "top5_all_normalizations_acc_mean" in row.index:
            out["Top-5 All Normalizations %"] = (
                f"{row['top5_all_normalizations_acc_mean']:.2f} ± "
                f"{row['top5_all_normalizations_acc_std']:.2f}"
            )
        if "top5_deep_pragmatic_equivalence_acc_mean" in row.index:
            out["Top-5 Deep Pragmatic %"] = (
                f"{row['top5_deep_pragmatic_equivalence_acc_mean']:.2f} ± "
                f"{row['top5_deep_pragmatic_equivalence_acc_std']:.2f}"
            )
        if "top1_invalid_smiles_pct_mean" in row.index:
            out["Invalid SMILES %"] = (
                f"{row['top1_invalid_smiles_pct_mean']:.2f} ± "
                f"{row['top1_invalid_smiles_pct_std']:.2f}"
            )
        if "top1_balanced_pct_mean" in row.index:
            out["Balanced % (Top-1)"] = (
                f"{row['top1_balanced_pct_mean']:.2f} ± "
                f"{row['top1_balanced_pct_std']:.2f}"
            )

        summary_rows.append(out)

    summary_df = pd.DataFrame(summary_rows)

    # Save nice aggregate table
    summary_df.to_csv(args.summary_output, index=False)


if __name__ == "__main__":
    main()

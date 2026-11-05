#!/usr/bin/env python3
"""Count whether missing molecules occur on the reactant or product side.

Run from the repository root:

    apptainer exec models/constrained_reaction_balancer/CRB_env.sif \
      python analysis/missing_species/analyze_missing_sides.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SIDE_ORDER = ["Product only", "Both sides", "Pre-product only", "Neither"]


def path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        default="results/additional/missing_species_results/reaction_categories.csv",
    )
    parser.add_argument(
        "--output-dir", default="results/additional/missing_species_results"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(
        path(args.categories),
        usecols=[
            "Unique_identifier",
            "missing_category",
            "n_missing",
            "missing_preproduct",
            "missing_products",
        ],
        low_memory=False,
    )
    pre = data["missing_preproduct"].fillna("").ne("")
    product = data["missing_products"].fillna("").ne("")
    data["missing_side"] = "Neither"
    data.loc[pre & ~product, "missing_side"] = "Pre-product only"
    data.loc[~pre & product, "missing_side"] = "Product only"
    data.loc[pre & product, "missing_side"] = "Both sides"

    summary = (
        data.groupby("missing_side", sort=False)
        .agg(
            reactions=("Unique_identifier", "nunique"),
            total_missing_molecules=("n_missing", "sum"),
            mean_missing_molecules=("n_missing", "mean"),
            median_missing_molecules=("n_missing", "median"),
        )
        .reindex(SIDE_ORDER)
        .dropna(subset=["reactions"])
        .reset_index()
    )
    summary["fraction_pct"] = 100 * summary.reactions / len(data)
    summary = summary[
        [
            "missing_side",
            "reactions",
            "fraction_pct",
            "total_missing_molecules",
            "mean_missing_molecules",
            "median_missing_molecules",
        ]
    ]
    cross = pd.crosstab(data.missing_side, data.missing_category).reindex(
        SIDE_ORDER, fill_value=0
    )
    cross = cross.loc[cross.sum(axis=1).gt(0)].reset_index()

    output_dir = path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "missing_side_summary.csv", index=False)
    cross.to_csv(output_dir / "missing_side_by_category.csv", index=False)

    lines = [
        "# Location of missing molecules",
        "",
        (
            "`Pre-product` denotes missing target reactants after matching against "
            "known USPTO reactants and reagents; target reagents are excluded. "
            "Products are compared separately."
        ),
        "",
        "| Location | Reactions | Fraction | Missing molecules | Mean/reaction | Median/reaction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.missing_side} | {int(row.reactions):,} | "
            f"{row.fraction_pct:.2f}% | {int(row.total_missing_molecules):,} | "
            f"{row.mean_missing_molecules:.2f} | "
            f"{row.median_missing_molecules:.1f} |"
        )
    (output_dir / "missing_side_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Wrote side analysis to {output_dir}")


if __name__ == "__main__":
    main()

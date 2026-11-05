#!/usr/bin/env python3
"""Compare RB and CRB accuracy by split and number of missing carbon atoms.

Use the category outcomes produced by build_category_accuracy_table.py.
The paired test keeps repeated predictions of the same reaction together.
It tests CRB-minus-RB differences by reversing their signs, with a normal
approximation for more than 20 nonzero reaction-level differences.
Holm correction accounts for the multiple comparisons in the output.
"""

from itertools import product
from math import erfc, sqrt
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from utils.result_paths import METRICS, metric_name, results_root, verify_score_release
from utils.scoring import provenance
OUT = ROOT / "results/additional/statistical_tests"
OUTCOMES = ROOT / "results/additional/missing_species_results/uid_model_outcomes.csv"
BENCHMARK = ROOT / "data/Template_curated/benchmark/benchmark_final.csv"
METRIC = "deep_pragmatic_correct"
BINS = (
    [(str(i), i, i) for i in range(9)]
    + [("9-10", 9, 10), ("11-13", 11, 13), ("≥14", 14, np.inf)]
)


def paired_test(frame: pd.DataFrame) -> dict:
    """Test the paired accuracy difference, grouping repeated observations by reaction ID."""
    delta = frame["CRB"].astype(int) - frame["RB"].astype(int)
    by_uid = delta.groupby(frame["Unique_identifier"]).sum()
    effects = by_uid[by_uid.ne(0)].to_numpy(dtype=float)
    observed, variance = float(delta.sum()), float(np.square(effects).sum())
    if not variance:
        p_value, se = 1.0, 0.0
    elif len(effects) <= 20:  # exact sign-flip distribution
        null = np.fromiter(
            (np.dot(signs, effects) for signs in product((-1, 1), repeat=len(effects))),
            dtype=float,
        )
        p_value = float(np.mean(np.abs(null) >= abs(observed)))
        se = sqrt(variance) / len(frame)
    else:  # normal approximation to the clustered sign-flip distribution
        se = sqrt(variance) / len(frame)
        p_value = erfc(abs(observed) / sqrt(2.0 * variance))

    delta_pp = 100.0 * observed / len(frame)
    return {
        "paired_n": len(frame),
        "unique_uids": frame["Unique_identifier"].nunique(),
        "rb_accuracy_pct": 100.0 * frame["RB"].mean(),
        "crb_accuracy_pct": 100.0 * frame["CRB"].mean(),
        "delta_pp": delta_pp,
        "ci95_low_pp": delta_pp - 1.96 * 100.0 * se,
        "ci95_high_pp": delta_pp + 1.96 * 100.0 * se,
        "crb_only_correct": int(delta.eq(1).sum()),
        "rb_only_correct": int(delta.eq(-1).sum()),
        "p_value": p_value,
    }


def holm(values: pd.Series) -> pd.Series:
    """Holm-adjust a family of p-values."""
    order = values.sort_values().index
    adjusted, running, m = pd.Series(index=values.index, dtype=float), 0.0, len(values)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values.loc[idx]))
        adjusted.loc[idx] = running
    return adjusted


def ptext(value: float) -> str:
    return "<1e-99" if value < 1e-99 else f"{value:.2g}"


def main() -> None:
    global OUT, OUTCOMES, METRIC
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric",type=metric_name,choices=METRICS,default="v2")
    parser.add_argument("--outcomes",type=Path)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args();OUT=args.output_dir or results_root(args.metric)/"additional/statistical_tests"
    OUTCOMES=args.outcomes or results_root(args.metric)/"additional/missing_species_results/uid_model_outcomes.csv"
    if (OUT/"results.csv").exists():raise FileExistsError("Choose a new --output-dir")
    OUT.mkdir(parents=True, exist_ok=True)
    header=pd.read_csv(OUTCOMES,nrows=0).columns
    if "scoring_metric" in header:
        saved=pd.read_csv(OUTCOMES,usecols=["scoring_metric"])["scoring_metric"].map(metric_name)
        if not saved.eq(args.metric).all():raise ValueError("Outcome metric does not match --metric")
    elif args.metric!="v1":
        raise ValueError("Current outcomes must identify their scoring_metric")
    if "equivalence_correct" in header: METRIC="equivalence_correct"
    outcomes = pd.read_csv(
        OUTCOMES,
        usecols=["Unique_identifier", "model", "split_type", "rep", METRIC],
    )
    outcomes = outcomes[outcomes["model"].isin(["RB", "CRB"])].copy()
    outcomes[METRIC] = outcomes[METRIC].astype(str).str.lower().isin(["true", "1"])
    keys = ["Unique_identifier", "split_type", "rep"]
    if outcomes.duplicated(keys + ["model"]).any():
        raise ValueError("Duplicate model outcome for a UID/split/repetition")
    paired = outcomes.pivot(index=keys, columns="model", values=METRIC).dropna().reset_index()
    if set(["RB", "CRB"]) - set(paired.columns):
        raise ValueError("Could not find paired RB and CRB outcomes")

    benchmark = pd.read_csv(
        BENCHMARK, usecols=["Unique_identifier", "MissingNrCarbons"], low_memory=False
    )
    benchmark["Unique_identifier"] = benchmark["Unique_identifier"].astype(str)
    if benchmark["Unique_identifier"].duplicated().any():
        raise ValueError("Duplicate benchmark UID")
    paired = paired.merge(benchmark, on="Unique_identifier", how="left", validate="many_to_one")
    paired["MissingNrCarbons"] = pd.to_numeric(
        paired["MissingNrCarbons"], errors="raise"
    )

    rows = []
    for split, split_rows in paired.groupby("split_type", sort=False):
        rows.append({"split": split, "missing_carbons": "All", **paired_test(split_rows)})
        for label, low, high in BINS:
            selected = split_rows["MissingNrCarbons"].between(low, high)
            if selected.any():
                rows.append(
                    {
                        "split": split,
                        "missing_carbons": label,
                        **paired_test(split_rows.loc[selected]),
                    }
                )

    results = pd.DataFrame(rows)
    results["p_holm"] = holm(results["p_value"])
    results["significant_0.05"] = results["p_holm"].lt(0.05)
    results.to_csv(OUT / "results.csv", index=False)

    provenance(OUT, args.metric, [Path(__file__), OUTCOMES, BENCHMARK],
               population="filtered", aggregation="pooled observations",
               test="paired sign flip clustered by reaction UID; Holm correction",
               normal_approximation_above=20)
    print(f"Wrote {len(results)} paired comparisons to {OUT}")


if __name__ == "__main__":
    main()

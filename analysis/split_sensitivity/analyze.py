#!/usr/bin/env python3
"""Rebuild the tables that check how OOD split settings affect the test sets."""

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from utils.result_paths import METRICS, metric_name, results_root, verify_score_release
from utils.scoring import provenance
SWEEP = ROOT / "data/Template_curated/sensitivity_sweeps/extreme"
BENCH = ROOT / "data/Template_curated/benchmark"
PAPER_OOD = BENCH / "ood_folds_missing_stats"
SCORES = ROOT / "results/main/scores/per_reaction"
OUT = ROOT / "results/additional/split_sensitivity"


def repo_path(path):
    """Record paths inside the repository relative to it, so no local paths are published."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)



def split_statistics():
    rows = []
    for meta_path in sorted(SWEEP.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text())
        args = meta["args"]
        fold_metrics = pd.read_csv(
            Path(str(meta_path).replace("_meta.json", "_fold_metrics.csv"))
        )
        fold_sizes = list(meta["fold_sizes"].values())
        rows.append(
            {
                "threshold": args["leakage_tanimoto"],
                "k": args["knn_k"],
                "n": sum(fold_sizes),
                "median_similarity": fold_metrics[
                    "tanimoto_to_train_median"
                ].mean(),
                "random_median_similarity": fold_metrics[
                    "random_tanimoto_to_train_median"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def manuscript_burden_sensitivity():
    groups = pd.read_csv(
        PAPER_OOD / "benchmark_final_ood_group_summary_RI_t055.csv"
    )
    split = pd.read_csv(
        PAPER_OOD / "benchmark_final_ood_splits_RI_t055.csv"
    ).merge(
        groups[
            [
                "group_id",
                "group_missing_atoms_median",
                "group_missing_carbons_median",
            ]
        ],
        left_on="leakage_group_id_RI",
        right_on="group_id",
        validate="many_to_one",
    )
    by_uid = split.set_index("Unique_identifier")
    cutoffs = {
        q: (
            np.percentile(groups["group_missing_atoms_median"], q),
            np.percentile(groups["group_missing_carbons_median"], q),
        )
        for q in (80, 85, 90)
    }
    assigned = split["ood_fold_RI"] >= 0
    raw_n = {
        q: int(
            (
                assigned
                & (
                    (split["group_missing_atoms_median"] >= atom_cut)
                    | (split["group_missing_carbons_median"] >= carbon_cut)
                )
            ).sum()
        )
        for q, (atom_cut, carbon_cut) in cutoffs.items()
    }

    scores = {model: {q: [] for q in cutoffs} for model in ("RB", "CRB", "SynRBL")}
    evaluated_n = {q: 0 for q in cutoffs}

    for rep in range(5):
        model_frames = {}
        for model in scores:
            frame = pd.read_csv(
                SCORES / f"{model}_extreme_ood_rep{rep}.csv",
                usecols=["Unique_identifier", "retained", "equivalence"],
            )
            frame = frame.set_index("Unique_identifier", verify_integrity=True)
            model_frames[model] = frame
        population_ids = model_frames["CRB"].index
        assert all(frame.index.equals(population_ids) for frame in model_frames.values())
        assert all((frame["retained"] == 1).all() for frame in model_frames.values())

        # Use the manuscript's vocabulary-filtered evaluation population for
        # all models, matching the corrected manuscript summaries.
        population = model_frames["CRB"].join(
            by_uid[
                [
                    "group_missing_atoms_median",
                    "group_missing_carbons_median",
                ]
            ],
            how="left",
            validate="one_to_one",
        )
        assert population["group_missing_atoms_median"].notna().all()
        assert population["group_missing_carbons_median"].notna().all()

        for q, (atom_cut, carbon_cut) in cutoffs.items():
            keep = (
                (population["group_missing_atoms_median"] >= atom_cut)
                | (population["group_missing_carbons_median"] >= carbon_cut)
            )
            evaluated_n[q] += int(keep.sum())
            for model, frame in model_frames.items():
                scores[model][q].append(
                    100 * frame.loc[keep, "equivalence"].mean()
                )

    rows = []
    for q, (atom_cut, carbon_cut) in cutoffs.items():
        row = {
            "percentile": q,
            "atom_cut": atom_cut,
            "carbon_cut": carbon_cut,
            "raw_n": raw_n[q],
            "evaluated_n": evaluated_n[q],
        }
        for model in scores:
            row[f"{model}_mean"] = np.mean(scores[model][q])
            row[f"{model}_sd"] = np.std(scores[model][q], ddof=1)
        rows.append(row)
    return pd.DataFrame(rows), groups


def main():
    global SCORES, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", type=metric_name, choices=METRICS, default="v2")
    parser.add_argument("--scores-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    SCORES = args.scores_dir or results_root(args.metric)/"main/scores/per_reaction"
    OUT = args.output_dir or results_root(args.metric)/"additional/split_sensitivity"
    verify_score_release(SCORES, args.metric)
    if any((OUT/name).exists() for name in ("burden_sensitivity.csv", "split_sweep.csv")):
        parser.error("Choose a new --output-dir; results already exist")
    split = split_statistics()
    burden, groups = manuscript_burden_sensitivity()
    composition = pd.DataFrame([{
        "p75_carbon_cut": np.percentile(groups["group_missing_carbons_median"], 75),
        "p80_inclusive_group_fraction": (
            (groups["group_missing_atoms_median"] >= np.percentile(groups["group_missing_atoms_median"], 80))
            | (groups["group_missing_carbons_median"] >= np.percentile(groups["group_missing_carbons_median"], 80))
        ).mean(),
    }])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in (("split_sweep.csv", split), ("burden_sensitivity.csv", burden),
                        ("composition.csv", composition)):
        frame.to_csv(OUT/name, index=False)
    inputs = [Path(__file__), *sorted(SWEEP.glob("*_meta.json")),
              *sorted(SWEEP.glob("*_fold_metrics.csv")),
              PAPER_OOD/"benchmark_final_ood_group_summary_RI_t055.csv",
              PAPER_OOD/"benchmark_final_ood_splits_RI_t055.csv",
              *(SCORES/f"{model}_extreme_ood_rep{rep}.csv"
                for model in ("RB", "CRB", "SynRBL") for rep in range(5))]
    provenance(OUT, args.metric, inputs, population="filtered", repetitions=5,
               aggregation="mean and sample SD of fold percentages",
               score_provenance=repo_path(SCORES.parent/"provenance.json"))
    print("Wrote CSVs to", OUT)


if __name__ == "__main__":
    main()

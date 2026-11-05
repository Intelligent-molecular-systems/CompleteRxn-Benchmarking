#!/usr/bin/env python3
"""Summarize random-split accuracy by reaction template.

Check which benchmark reactions have template labels and match them to saved
predictions. Write template coverage, accuracy for each fold, and summaries
of how template size relates to accuracy.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0,_REPO_ROOT)

RANDOM_SPLIT_RE = re.compile(r"^benchmark_final_random_splits_rep(\d+)_seed(\d+)\.csv$")


@dataclass
class FoldMeta:
    rep: int
    seed: int
    split_csv: str
    pred_tsv: str


def _abs_path(path: str, repo_root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root, path)


def _safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0) != 0
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _normalize_template(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    invalid = s.str.lower().isin({"", "nan", "none", "null"})
    s[invalid] = pd.NA
    return s


def discover_random_folds(random_splits_dir: str, results_dir: str, experiment: str) -> List[FoldMeta]:
    folds: List[FoldMeta] = []
    for name in sorted(os.listdir(random_splits_dir)):
        m = RANDOM_SPLIT_RE.match(name)
        if m is None:
            continue
        rep = int(m.group(1))
        seed = int(m.group(2))
        split_csv = os.path.join(random_splits_dir, name)
        pred_tsv = os.path.join(
            results_dir,
            {"BS_PT": "rb", "CBS_PT": "crb"}[experiment],
            "random",
            f"random_seed{seed}_rep{rep}",
            "top1_predictions_with_stats.txt",
        )
        if not os.path.isfile(pred_tsv):
            print(f"Warning: prediction file not found for rep={rep}, seed={seed}: {pred_tsv}")
            continue
        folds.append(FoldMeta(rep=rep, seed=seed, split_csv=split_csv, pred_tsv=pred_tsv))
    return folds


def map_predictions_to_test_subset(
    bench_test: pd.DataFrame,
    pred_df: pd.DataFrame,
    metric_col: str,
) -> pd.DataFrame:
    if "original_index" not in pred_df.columns:
        raise ValueError("Column 'original_index' missing in prediction file.")
    if metric_col not in pred_df.columns:
        raise ValueError(f"Column '{metric_col}' missing in prediction file.")

    idx = pd.to_numeric(pred_df["original_index"], errors="coerce")
    valid = idx.notna() & (idx >= 0) & (idx < len(bench_test))
    idx_valid = idx.loc[valid].astype(int)

    mapped = bench_test.iloc[idx_valid.values].reset_index(drop=True).copy()
    pred_valid = pred_df.loc[valid].reset_index(drop=True)
    mapped["metric"] = _safe_bool(pred_valid[metric_col]).values
    return mapped


def compute_label_coverage(bench_df: pd.DataFrame, template_col: str) -> Dict[str, int]:
    templ = _normalize_template(bench_df[template_col])
    labeled = templ.notna()
    return {
        "n_benchmark_total": int(len(bench_df)),
        "n_benchmark_with_template_label": int(labeled.sum()),
        "n_benchmark_without_template_label": int((~labeled).sum()),
        "n_unique_template_labels": int(templ[labeled].nunique()),
    }


def compute_accuracy_distribution(template_summary: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"({bins[i]:.1f}, {bins[i + 1]:.1f}]" for i in range(len(bins) - 1)]

    out = template_summary.copy()
    out["accuracy_bin"] = pd.cut(
        out["accuracy_weighted"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    dist = (
        out.groupby("accuracy_bin", observed=False)
        .agg(
            n_templates=("template_label", "count"),
            n_samples=("n_samples_total", "sum"),
            mean_template_size=("n_samples_total", "mean"),
            median_template_size=("n_samples_total", "median"),
        )
        .reset_index()
    )
    return dist


def run_analysis(args: argparse.Namespace, repo_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark_csv = _abs_path(args.benchmark_csv, repo_root)
    random_splits_dir = _abs_path(args.random_splits_dir, repo_root)
    results_dir = _abs_path(args.results_dir, repo_root)
    output_dir = _abs_path(args.output_dir, repo_root)

    os.makedirs(output_dir, exist_ok=True)

    bench_df = pd.read_csv(benchmark_csv, low_memory=False)
    if args.template_col not in bench_df.columns:
        raise ValueError(f"Template column '{args.template_col}' not found in benchmark file.")

    label_cov = compute_label_coverage(bench_df, args.template_col)

    folds = discover_random_folds(random_splits_dir, results_dir, args.experiment)
    if not folds:
        raise ValueError("No valid random-split folds found with matching prediction files.")

    all_rows: List[pd.DataFrame] = []
    overlap_rows: List[Dict[str, object]] = []

    for fold in folds:
        split_df = pd.read_csv(fold.split_csv)
        if args.split_col not in split_df.columns:
            raise ValueError(f"Column '{args.split_col}' missing in split CSV: {fold.split_csv}")
        if "Unique_identifier" not in split_df.columns:
            raise ValueError(f"Column 'Unique_identifier' missing in split CSV: {fold.split_csv}")

        test_uids = split_df.loc[split_df[args.split_col] == args.test_label, "Unique_identifier"]
        bench_test = bench_df[bench_df["Unique_identifier"].isin(test_uids)].reset_index(drop=True)

        pred_df = pd.read_csv(fold.pred_tsv, sep="\t")
        from utils.scoring import overlay_scores
        pred_df=overlay_scores(pred_df,fold.pred_tsv)
        mapped = map_predictions_to_test_subset(bench_test, pred_df, args.metric_col)
        mapped["template_label"] = _normalize_template(mapped[args.template_col])

        labeled_test = _normalize_template(bench_test[args.template_col]).notna()
        labeled_pred = mapped["template_label"].notna()

        overlap_rows.append(
            {
                "rep": fold.rep,
                "seed": fold.seed,
                "split_csv": os.path.basename(fold.split_csv),
                "prediction_file": os.path.relpath(fold.pred_tsv, repo_root),
                "n_test_samples": int(len(bench_test)),
                "n_test_with_template_label": int(labeled_test.sum()),
                "n_predictions_rows": int(len(pred_df)),
                "n_predictions_mapped_to_test": int(len(mapped)),
                "n_predictions_mapped_with_template_label": int(labeled_pred.sum()),
                "frac_test_with_template_label": float(labeled_test.mean()) if len(bench_test) else np.nan,
                "frac_mapped_predictions_with_template_label": float(labeled_pred.mean()) if len(mapped) else np.nan,
            }
        )

        fold_labeled = mapped[mapped["template_label"].notna()].copy()
        if fold_labeled.empty:
            continue

        fold_labeled["rep"] = fold.rep
        fold_labeled["seed"] = fold.seed
        fold_labeled["split_csv"] = os.path.basename(fold.split_csv)
        all_rows.append(
            fold_labeled[["rep", "seed", "split_csv", "Unique_identifier", "template_label", "metric"]]
        )

    if not all_rows:
        raise ValueError("No mapped prediction rows with template labels found across random splits.")

    all_labeled_preds = pd.concat(all_rows, ignore_index=True)

    per_template_fold = (
        all_labeled_preds.groupby(["rep", "seed", "split_csv", "template_label"], as_index=False)
        .agg(
            n_samples=("metric", "size"),
            n_correct=("metric", "sum"),
        )
        .sort_values(["rep", "template_label"])
    )
    per_template_fold["accuracy"] = per_template_fold["n_correct"] / per_template_fold["n_samples"]

    template_summary = (
        per_template_fold.groupby("template_label", as_index=False)
        .agg(
            n_folds_present=("rep", "nunique"),
            n_samples_total=("n_samples", "sum"),
            n_correct_total=("n_correct", "sum"),
            mean_fold_accuracy=("accuracy", "mean"),
            std_fold_accuracy=("accuracy", "std"),
        )
        .sort_values("n_samples_total", ascending=False)
    )
    template_summary["std_fold_accuracy"] = template_summary["std_fold_accuracy"].fillna(0.0)
    template_summary["accuracy_weighted"] = (
        template_summary["n_correct_total"] / template_summary["n_samples_total"]
    )

    accuracy_distribution = compute_accuracy_distribution(template_summary)
    overlap_summary = pd.DataFrame(overlap_rows).sort_values(["rep", "seed"]).reset_index(drop=True)

    benchmark_label_coverage = pd.DataFrame([label_cov])

    # Templates with at least one error across all folds, sorted by total errors descending
    template_summary["n_errors_total"] = template_summary["n_samples_total"] - template_summary["n_correct_total"]
    failing_templates = (
        template_summary[template_summary["n_errors_total"] > 0]
        .sort_values("n_errors_total", ascending=False)
        .reset_index(drop=True)
    )
    # Attach all sample Unique_identifiers that were incorrect (metric==False) across folds
    failed_rows = all_labeled_preds[~all_labeled_preds["metric"]]
    failed_uids_per_template = (
        failed_rows.groupby("template_label")["Unique_identifier"]
        .apply(lambda x: sorted(x.unique().tolist()))
        .reset_index()
        .rename(columns={"Unique_identifier": "failed_unique_identifiers"})
    )
    failing_templates = failing_templates.merge(failed_uids_per_template, on="template_label", how="left")

    benchmark_label_coverage.to_csv(
        os.path.join(output_dir, "template_label_coverage_benchmark.csv"), index=False
    )
    overlap_summary.to_csv(
        os.path.join(output_dir, "random_split_prediction_template_overlap.csv"), index=False
    )
    per_template_fold.to_csv(
        os.path.join(output_dir, "random_split_per_template_accuracy_per_fold.csv"), index=False
    )
    template_summary.to_csv(
        os.path.join(output_dir, "random_split_per_template_accuracy_summary.csv"), index=False
    )
    accuracy_distribution.to_csv(
        os.path.join(output_dir, "random_split_template_accuracy_distribution.csv"), index=False
    )
    failing_templates.to_csv(
        os.path.join(output_dir, "random_split_failing_templates.csv"), index=False
    )

    return benchmark_label_coverage, overlap_summary, per_template_fold, template_summary, failing_templates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze random-split performance stratified by 1kTPLLabel."
    )
    parser.add_argument(
        "--benchmark-csv",
        default="data/Template_curated/benchmark/benchmark_final.csv",
        help="Path to benchmark_final.csv",
    )
    parser.add_argument(
        "--random-splits-dir",
        default="data/Template_curated/benchmark/random_splits",
        help="Directory containing benchmark_final_random_splits_rep*_seed*.csv",
    )
    parser.add_argument(
        "--results-dir",
        default="results/main/predictions",
        help="CRB results root directory",
    )
    parser.add_argument(
        "--experiment",
        default="CBS_PT",
        help="Experiment name under results_dir to analyze (for random split predictions)",
    )
    parser.add_argument(
        "--template-col",
        default="1kTPLLabel",
        help="Template-label column in benchmark_final.csv",
    )
    parser.add_argument(
        "--split-col",
        default="random_split",
        help="Split column in random split CSV files",
    )
    parser.add_argument(
        "--test-label",
        type=int,
        default=2,
        help="Label value in split column indicating test rows",
    )
    parser.add_argument(
        "--metric-col",
        default="matches_selected_metric",
        help="Accuracy metric column in top1_predictions_with_stats.txt",
    )
    parser.add_argument(
        "--output-dir",
        default="results/main/figures/template_label_random_split",
        help="Output directory for generated CSV summaries",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    release=os.environ.get("COMPLETERXN_RESULTS_ROOT")
    if release and args.output_dir.startswith("results/main/figures/"):
        args.output_dir=os.path.join(release,args.output_dir[len("results/"):])

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    coverage, overlap, per_fold, summary, failing = run_analysis(args, repo_root)

    cov = coverage.iloc[0].to_dict()
    print("\nBenchmark template-label coverage")
    print("--------------------------------")
    print(f"Total benchmark rows: {cov['n_benchmark_total']}")
    print(f"Rows with template label: {cov['n_benchmark_with_template_label']}")
    print(f"Rows without template label: {cov['n_benchmark_without_template_label']}")
    print(f"Unique template labels: {cov['n_unique_template_labels']}")

    print("\nRandom-split overlap summary (averaged across folds)")
    print("----------------------------------------------------")
    overlap_means = overlap[
        [
            "n_test_samples",
            "n_test_with_template_label",
            "n_predictions_rows",
            "n_predictions_mapped_to_test",
            "n_predictions_mapped_with_template_label",
            "frac_test_with_template_label",
            "frac_mapped_predictions_with_template_label",
        ]
    ].mean(numeric_only=True)
    for k, v in overlap_means.items():
        if k.startswith("frac_"):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v:.2f}")

    print("\nStratified template summary")
    print("---------------------------")
    print(f"Templates in summary: {len(summary)}")
    n_perfect   = int((summary["accuracy_weighted"] == 1.0).sum())
    n_near      = int(((summary["accuracy_weighted"] >= 0.99) & (summary["accuracy_weighted"] < 1.0)).sum())
    n_good      = int(((summary["accuracy_weighted"] >= 0.9)  & (summary["accuracy_weighted"] < 0.99)).sum())
    n_poor      = int(((summary["accuracy_weighted"] >  0.0)  & (summary["accuracy_weighted"] < 0.9)).sum())
    n_zero      = int((summary["accuracy_weighted"] == 0.0).sum())
    print(f"  acc == 1.0           (perfect):      {n_perfect}")
    print(f"  acc in [0.99, 1.0)   (near-perfect): {n_near}")
    print(f"  acc in [0.9,  0.99)  (good):         {n_good}")
    print(f"  acc in (0.0,  0.9)   (poor):         {n_poor}")
    print(f"  acc == 0.0           (zero):         {n_zero}")
    print(f"Overall weighted accuracy across templates: {summary['accuracy_weighted'].mean():.4f}")

    print(f"\nSaved outputs to: {args.output_dir}")
    print(f"Failing templates: {len(failing)} (saved to random_split_failing_templates.csv)")


if __name__ == "__main__":
    main()

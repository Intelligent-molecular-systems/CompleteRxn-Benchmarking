#!/usr/bin/env python3
"""
Summarize SynRBL scores for each split and repetition.

Input: CSV with all reactions and metric columns (from evaluate.py --add-all-metrics)
Output: summary CSV with per-split-replica statistics

Expected metric columns in CSV:
- exact_match, proton_shuffle_match, all_normalizations_match
- pragmatic_equivalence_match, deep_pragmatic_equivalence_match
- solved, solved_by, confidence (for mcs/rb distinction)
- similarity_RI or similarity_noRI (similarity column)
"""

import argparse
import logging
import os
import re
import sys
from typing import Dict, Any, Optional, Set

# Add workspace root to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

logger = logging.getLogger(__name__)


SUPPORTED_METRICS = [
    "exact_match",
    "proton_shuffle_match",
    "all_normalizations_match",
    "pragmatic_equivalence_match",
    "deep_pragmatic_equivalence_match",
]


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _format_ratio(r):
    return "-" if r is None else f"{r:.2%}"


def _resolve_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(_ROOT, path_value)


def _get_ml_top1_path(ml_results_dir: str, ml_experiment: str, split_type: str, rep: int, split_fname: str) -> Optional[str]:
    """Return top1_predictions_with_stats.txt path for the corresponding ML fold."""
    if split_type == "random":
        m_seed = re.search(r"seed(\d+)", split_fname)
        if not m_seed:
            return None
        seed = int(m_seed.group(1))
        fold_dir = f"random_seed{seed}_rep{rep}"
    elif split_type == "group":
        fold_dir = f"group_rep{rep}"
    elif split_type == "extreme_ood":
        fold_dir = f"extreme_ood_rep{rep}"
    else:
        return None

    model_dir = {"BS_PT": "rb", "CBS_PT": "crb"}[ml_experiment]
    return os.path.join(ml_results_dir, model_dir, split_type, fold_dir, "top1_predictions_with_stats.txt")


def _uids_from_ml_top1(
    benchmark_df: pd.DataFrame,
    test_uids: Set[Any],
    top1_path: str,
) -> Set[Any]:
    """Map ML top1 original_index rows back to benchmark test UIDs."""
    top1 = pd.read_csv(top1_path, sep="\t")
    if "original_index" not in top1.columns:
        raise KeyError(f"Column 'original_index' missing in ML top1 file: {top1_path}")

    bench_sub = benchmark_df[benchmark_df["Unique_identifier"].isin(test_uids)].reset_index(drop=True)
    if bench_sub.empty:
        return set()

    idx = pd.to_numeric(top1["original_index"], errors="coerce").dropna().astype(int)
    idx = idx[(idx >= 0) & (idx < len(bench_sub))]
    if idx.empty:
        return set()

    return set(bench_sub.iloc[idx.values]["Unique_identifier"].tolist())


def compute_stats_for_rows(
    df_subset: pd.DataFrame,
    metric_col: str,
    min_confidence: float = 0.5,
    confidence_col: str = "confidence",
    solved_by_col: str = "solved_by",
) -> Dict[str, Any]:
    """Compute accuracy stats for a subset of reactions on a given metric."""
    total = len(df_subset)
    solved_mask = (
        df_subset["solved"].map(_safe_bool)
        if "solved" in df_subset.columns
        else pd.Series(True, index=df_subset.index)
    )
    solved_cnt = int(solved_mask.sum())
    
    # Filter to solved rows only
    df_solved = df_subset[solved_mask]
    
    rb_correct = 0
    mcs_correct = 0
    confident_mcs_cnt = 0
    confident_mcs_correct = 0

    # GT metric correctness: solved-only, no confidence masking.
    metric_true_solved = df_solved[metric_col].map(_safe_bool)
    total_correct = int(metric_true_solved.sum())

    # Solver-specific breakdowns are diagnostic; self-reported stats remain
    # confidence-thresholded for mcs-based rows.
    if solved_by_col in df_solved.columns:
        solved_by = df_solved[solved_by_col].astype(str).str.lower()
        rb_mask = solved_by == "rule-based"
        mcs_mask = solved_by == "mcs-based"

        rb_correct = int((rb_mask & metric_true_solved).sum())
        mcs_correct = int((mcs_mask & metric_true_solved).sum())

        if confidence_col in df_solved.columns:
            conf_mask = pd.to_numeric(df_solved[confidence_col], errors="coerce").fillna(0.0) >= min_confidence
            confident_mcs_cnt = int((mcs_mask & conf_mask).sum())
            confident_mcs_correct = int((mcs_mask & conf_mask & metric_true_solved).sum())
        else:
            confident_mcs_cnt = int(mcs_mask.sum())
            confident_mcs_correct = int((mcs_mask & metric_true_solved).sum())
    else:
        # No solver info available; keep total correctness from solved-only GT match.
        rb_correct = total_correct

    success_rate = _ratio(solved_cnt, total)
    accuracy_solved = _ratio(total_correct, solved_cnt)
    accuracy = _ratio(total_correct, total)
    mcs_accuracy = _ratio(confident_mcs_correct, confident_mcs_cnt)
    synrbl_self_reported_acc = _ratio(confident_mcs_cnt, total)
    
    return {
        "reaction_cnt": total,
        "solved_cnt": solved_cnt,
        "solved_pct": success_rate,
        "confident_mcs_cnt": confident_mcs_cnt,
        "rb_correct": rb_correct,
        "mcs_correct": mcs_correct,
        "total_correct": total_correct,
        "success_rate": success_rate,
        "accuracy_solved": accuracy_solved,
        "accuracy": accuracy,
        "mcs_accuracy": mcs_accuracy,
        "SynRBL_self_reported_acc": synrbl_self_reported_acc,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics by split type/replica from full evaluation CSV."
    )
    parser.add_argument(
        "eval_csv",
        help="Full evaluation CSV (output from evaluate.py --add-all-metrics)"
    )
    parser.add_argument(
        "--benchmark-dir",
        default="data/Template_curated/benchmark",
        help="Root benchmark directory containing split files (optional)."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_METRICS,
        default=SUPPORTED_METRICS,
        help="Metric columns to aggregate (default: all supported metrics)."
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default=None,
        help="Backward-compatible alias for selecting a single metric. Overrides --metrics when set."
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output summary CSV"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Min confidence for mcs-based correctness"
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Aggregate only full-results rows (skip split-file based rows)."
    )
    parser.add_argument(
        "--filter-to-ml-top1",
        action="store_true",
        help="Restrict each split fold to the same UID universe used by ML top1 files (original_index mapping)."
    )
    parser.add_argument(
        "--ml-results-dir",
        default="results/main/predictions",
        help="Root directory containing ML top1 fold outputs."
    )
    parser.add_argument(
        "--ml-experiment",
        default="CBS_PT",
        help="ML experiment folder under --ml-results-dir used for row filtering."
    )
    parser.add_argument(
        "--benchmark-final",
        default="data/Template_curated/benchmark/benchmark_final.csv",
        help="benchmark_final.csv path used to map ML original_index to Unique_identifier."
    )
    args = parser.parse_args()
    
    # Resolve paths
    if not os.path.isabs(args.eval_csv):
        eval_csv = os.path.join(_ROOT, args.eval_csv)
    else:
        eval_csv = args.eval_csv
    
    benchdir = _resolve_path(args.benchmark_dir)
    ml_results_dir = _resolve_path(args.ml_results_dir)
    benchmark_final_path = _resolve_path(args.benchmark_final)
    
    if args.output_csv is None:
        args.output_csv = os.path.join(
            os.path.dirname(eval_csv),
            "split_aggregate_summary.csv"
        )
    
    # Load evaluation CSV
    logger.info(f"Loading evaluation CSV: {eval_csv}")
    df_full = pd.read_csv(eval_csv, low_memory=False)
    logger.info(f"Loaded {len(df_full)} reactions")

    benchmark_df = None
    if args.filter_to_ml_top1:
        logger.info("ML top1 filtering enabled: experiment=%s, results_dir=%s", args.ml_experiment, ml_results_dir)
        benchmark_df = pd.read_csv(benchmark_final_path, low_memory=False)
        if "Unique_identifier" not in benchmark_df.columns:
            raise KeyError("benchmark_final must contain 'Unique_identifier' for ML filtering.")
    
    selected_metrics = [args.metric] if args.metric else list(args.metrics)
    missing_metrics = [m for m in selected_metrics if m not in df_full.columns]
    if missing_metrics:
        raise KeyError(
            "Missing metric column(s) in CSV: " + ", ".join(missing_metrics)
        )
    
    summary_rows = []
    full_uid_union: Set[Any] = set()

    # Optionally compute split-file based rows.
    if not args.full_only:
        for split_type, split_dir, split_col in [
            ("random", os.path.join(benchdir, "random_splits"), "random_split"),
            ("group", os.path.join(benchdir, "group_splits"), "group_split"),
            ("extreme_ood", os.path.join(benchdir, "extreme_ood_splits"), "extreme_ood"),
        ]:
            if not os.path.isdir(split_dir):
                logger.warning(f"Split directory not found: {split_dir}")
                continue

            for fname in sorted(os.listdir(split_dir)):
                if not fname.endswith(".csv") or fname.endswith("_meta.csv"):
                    continue

                import re
                m = re.search(r"rep(\d+)", fname)
                if not m:
                    continue
                rep = int(m.group(1))

                split_file = os.path.join(split_dir, fname)
                logger.info(f"  Loading {split_type} rep {rep}: {fname}")
                split_df = pd.read_csv(split_file, low_memory=False)

                if split_col not in split_df.columns:
                    logger.warning(f"  Missing split column '{split_col}' in {fname}")
                    continue
                if "Unique_identifier" not in split_df.columns:
                    logger.warning(f"  Missing Unique_identifier in {fname}")
                    continue

                test_uids = set(split_df.loc[split_df[split_col] == 2, "Unique_identifier"])
                if not test_uids:
                    logger.warning(f"No test-set identifiers in {fname}")
                    continue

                selected_uids = test_uids
                if args.filter_to_ml_top1:
                    top1_path = _get_ml_top1_path(
                        ml_results_dir=ml_results_dir,
                        ml_experiment=args.ml_experiment,
                        split_type=split_type,
                        rep=rep,
                        split_fname=fname,
                    )
                    if top1_path is None or not os.path.isfile(top1_path):
                        logger.warning("  ML top1 file missing for %s rep %s (%s): %s", split_type, rep, fname, top1_path)
                        continue
                    selected_uids = _uids_from_ml_top1(
                        benchmark_df=benchmark_df,
                        test_uids=test_uids,
                        top1_path=top1_path,
                    )
                    if not selected_uids:
                        logger.warning("  No ML-mapped UIDs for %s rep %s (%s)", split_type, rep, fname)
                        continue

                full_uid_union.update(selected_uids)

                df_subset = df_full[df_full["Unique_identifier"].isin(selected_uids)].copy()
                if df_subset.empty:
                    logger.warning(f"No matched reactions in eval CSV for {split_type} rep {rep}")
                    continue

                for metric_col in selected_metrics:
                    stats = compute_stats_for_rows(
                        df_subset,
                        metric_col=metric_col,
                        min_confidence=args.min_confidence,
                    )
                    summary_rows.append({
                        "split_type": split_type,
                        "rep": rep,
                        "metric": metric_col,
                        **stats,
                    })

    # Always compute full-results stats first row, but on filtered UID union when
    # ML top1 filtering is active.
    if args.filter_to_ml_top1 and full_uid_union:
        df_full_for_summary = df_full[df_full["Unique_identifier"].isin(full_uid_union)].copy()
    else:
        df_full_for_summary = df_full

    full_rows = []
    for metric_col in selected_metrics:
        stats = compute_stats_for_rows(
            df_full_for_summary,
            metric_col=metric_col,
            min_confidence=args.min_confidence,
        )
        full_rows.append({
            "split_type": "full",
            "rep": -1,
            "metric": metric_col,
            **stats,
        })

    summary_rows = full_rows + summary_rows
    
    # Write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_csv, index=False)
    logger.info(f"Wrote summary CSV: {args.output_csv}")
    logger.info(f"  Rows: {len(summary_df)}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"Summary by split type (metrics: {', '.join(selected_metrics)})")
    print("="*60)
    for metric_col in selected_metrics:
        print(f"\nMetric: {metric_col}")
        for split_type in summary_df["split_type"].unique():
            df_split = summary_df[
                (summary_df["split_type"] == split_type)
                & (summary_df["metric"] == metric_col)
            ]
            if df_split.empty:
                continue
            print(f"  {split_type}:")
            for _, row in df_split.iterrows():
                print(
                    f"    rep{int(row['rep'])}: accuracy={_format_ratio(row['accuracy'])}, "
                    f"accuracy_solved={_format_ratio(row['accuracy_solved'])}, "
                    f"synrbl_balanced={_format_ratio(row['solved_pct'])}, "
                    f"solved={row['solved_cnt']}/{row['reaction_cnt']}"
                )


if __name__ == "__main__":
    main()

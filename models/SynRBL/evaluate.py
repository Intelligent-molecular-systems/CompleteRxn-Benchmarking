"""Compare saved SynRBL predictions with the benchmark references.

Count solved reactions and correct predictions under each selected metric.
Require atom and charge balance before a prediction can count as correct.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any

# Add workspace root to path so utils.metrics is importable from any working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress RDKit warnings (e.g. "not removing hydrogen atom without neighbors")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import rdkit.Chem.rdChemReactions as rdChemReactions

from utils.result_paths import metric_name, ensure_v1_seed
from utils.chemistry import reaction_similarity
from utils.reaction_balance import is_balanced_reaction
from utils.scoring import compare as selected_compare, METRICS


logger = logging.getLogger(__name__)


ACCURACY_METRIC_TO_COL = {
    "exact": "exact_match",
    "proton_shuffle": "proton_shuffle_match",
    "all_normalizations": "all_normalizations_match",
    "pragmatic_equivalence": "pragmatic_equivalence_match",
    "deep_pragmatic_equivalence": "deep_pragmatic_equivalence_match",
}


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


def check_columns(df: pd.DataFrame, reaction_col: str, target_col: str) -> None:
    if df.empty:
        raise ValueError("No reactions found in input.")

    if reaction_col not in df.columns:
        raise KeyError(f"No column '{reaction_col}' found in input.")

    if target_col not in df.columns:
        raise KeyError(f"No column '{target_col}' found in input.")

    sample = next((v for v in df[reaction_col].tolist() if isinstance(v, str)), None)
    if sample is None:
        raise TypeError(f"Reaction column '{reaction_col}' must contain strings.")

    reaction = None
    try:
        reaction = rdChemReactions.ReactionFromSmarts(sample, useSmiles=True)
    except Exception:
        pass
    if reaction is None:
        raise ValueError(
            f"Value '{sample[:30]}...' in reaction column '{reaction_col}' is not a valid reaction SMILES."
        )


def _infer_split_col(split_df: pd.DataFrame) -> str:
    for c in ("random_split", "group_split", "extreme_ood"):
        if c in split_df.columns:
            return c
    raise KeyError(
        "Could not infer split column from split file. "
        "Expected one of: random_split, group_split, extreme_ood"
    )


def filter_df_by_split(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.split_file is None:
        return df

    if args.id_col not in df.columns:
        raise KeyError(
            f"Input file must contain id column '{args.id_col}' when --split-file is used."
        )

    split_df = pd.read_csv(args.split_file, low_memory=False)
    if "Unique_identifier" not in split_df.columns:
        raise KeyError("Split file must contain column 'Unique_identifier'.")

    split_col = args.split_col or _infer_split_col(split_df)
    if split_col not in split_df.columns:
        raise KeyError(
            f"Requested split column '{split_col}' not found in split file '{args.split_file}'."
        )

    split_value = {"train": 0, "valid": 1, "test": 2}[args.split_label]
    split_uids = set(split_df.loc[split_df[split_col] == split_value, "Unique_identifier"])

    if not split_uids:
        raise ValueError(
            f"No rows found for split_label='{args.split_label}' in '{args.split_file}'."
        )

    before = len(df)
    out = df[df[args.id_col].isin(split_uids)].copy()
    logger.info(
        "Applied split filter: %s=%s (%s). Rows kept: %s / %s",
        split_col,
        split_value,
        args.split_label,
        len(out),
        before,
    )

    missing = len(split_uids) - out[args.id_col].nunique()
    if missing > 0:
        logger.warning(
            "Split contains %s identifiers not present in input predictions.",
            missing,
        )
        # Print some identifiers that are missing for debugging
        missing_uids = split_uids - set(out[args.id_col].unique())
        print("Missing identifiers: %s" % list(missing_uids)[:10])

    if out.empty:
        raise ValueError("Split filtering produced an empty dataframe.")

    return out


def _compute_all_metrics(predicted: str, expected: str, scoring_metric="v2") -> Dict[str, bool]:
    """Compute all accuracy metric matches for a reaction pair."""
    scoring_metric = metric_name(scoring_metric)
    columns = list(ACCURACY_METRIC_TO_COL.values()) if scoring_metric == "v1" else ["exact_match", "equivalent_match"]
    metrics: Dict[str, bool] = {}
    if not is_balanced_reaction(predicted):
        return {name: False for name in columns}
    try:
        comp = selected_compare(predicted, expected, metric=scoring_metric, stereo=scoring_metric == "v1")
        for col_name in columns:
            try:
                metrics[col_name] = bool(getattr(comp, col_name))
            except Exception:
                metrics[col_name] = False
    except Exception as exc:
        if scoring_metric != 'v1' and not isinstance(exc, ValueError):
            raise
        for col_name in columns:
            metrics[col_name] = False
    return metrics


def evaluate(df: pd.DataFrame, args) -> Dict[str, Any]:
    total = len(df)
    solved_cnt = 0
    confident_mcs_cnt = 0
    rb_correct = 0
    mcs_correct = 0

    similarities = []
    sim_and_match = []  # list of (similarity, correct) pairs for solved reactions

    if args.reagents_identified:
        similarity_col_name = "similarity_RI"
    else:
        # reagents are not identified in the ground truth
        similarity_col_name = "similarity_noRI"

    # Independent check; the solver's solved flag is not proof of balance.
    df["is_balanced"] = df[args.col].map(is_balanced_reaction)

    # Initialize similarity column with None/NaN
    df[similarity_col_name] = None

    scoring_metric = metric_name(getattr(args, "scoring_metric", "v2"))
    metric_columns = ACCURACY_METRIC_TO_COL if scoring_metric == "v1" else {"exact":"exact_match", "equivalence":"equivalent_match"}
    if args.accuracy_metric not in metric_columns: raise ValueError("Selected metric only provides exact and equivalence")
    if scoring_metric != "v1":
        df.drop(columns=[c for c in ACCURACY_METRIC_TO_COL.values() if c != "exact_match"], errors="ignore", inplace=True)
        df["scoring_metric"]=scoring_metric
    # If computing all metrics, initialize metric columns
    if args.add_all_metrics or scoring_metric != "v1":
        for col in metric_columns.values():
            df[col] = False
        _summary_metric_col = metric_columns[args.accuracy_metric]
    else:
        # Select the single match function based on --accuracy-metric
        _metric_attr = metric_columns[args.accuracy_metric]

        def _is_correct(predicted: str, expected: str) -> bool:
            if not is_balanced_reaction(predicted):
                return False
            try:
                return getattr(selected_compare(predicted, expected, metric=scoring_metric, stereo=scoring_metric == "v1"), _metric_attr)
            except Exception:
                return False

    for i, row in tqdm(df.iterrows(), total=total, desc="Evaluating", unit="rxn"):
        try:
            solved = _safe_bool(row.get(args.solved_col, False))
            if not solved:
                continue

            solved_cnt += 1
            solved_by = str(row.get(args.solved_by_col, "")).strip().lower()
            confidence = _safe_float(row.get(args.confidence_col, 0.0))

            if solved_by == "mcs-based" and confidence >= args.min_confidence:
                confident_mcs_cnt += 1

            expected = row.get(args.target_col)
            predicted = row.get(args.col)
            if pd.isna(expected) or pd.isna(predicted):
                continue

            sim = reaction_similarity(
                expected, predicted,
                method=args.similarity_method,
                ignore_reagents=True,
            )
            similarities.append(sim)

            # Store similarity in the dataframe at this row
            df.at[i, similarity_col_name] = sim

            # Compute metrics
            if args.add_all_metrics or scoring_metric != "v1":
                metrics = _compute_all_metrics(str(predicted), str(expected), scoring_metric)
                for col_name, val in metrics.items():
                    df.at[i, col_name] = val
                # For summary stats, use the mapped metric column
                correct = metrics.get(_summary_metric_col, False)
            else:
                correct = _is_correct(str(predicted), str(expected))

            sim_and_match.append((i, sim, correct))

            if solved_by == "rule-based":
                if correct:
                    rb_correct += 1
            elif solved_by == "mcs-based":
                if confidence >= args.min_confidence and correct:
                    mcs_correct += 1

        except Exception as exc:
            if scoring_metric != 'v1' and not isinstance(exc, ValueError):
                raise
            logger.warning("Failed to evaluate row %s (%s)", i, exc)

    total_correct = rb_correct + mcs_correct
    success_rate = _ratio(solved_cnt, total)
    accuracy_solved = _ratio(total_correct, solved_cnt)
    accuracy = _ratio(total_correct, total)
    mcs_accuracy = _ratio(mcs_correct, confident_mcs_cnt)

    return {
        "reaction_cnt": total,
        "solved_cnt": solved_cnt,
        "confident_mcs_cnt": confident_mcs_cnt,
        "rb_correct": rb_correct,
        "mcs_correct": mcs_correct,
        "total_correct": total_correct,
        "success_rate": success_rate,
        "accuracy_solved": accuracy_solved,
        "accuracy": accuracy,
        "mcs_accuracy": mcs_accuracy,
        "similarities_match": sim_and_match,
        "similarities": similarities,
    }


def print_summary(stats: Dict[str, Any], accuracy_metric: str) -> None:
    logger.info("%s Simple Benchmark Summary %s", "#" * 15, "#" * 15)
    logger.info("Reactions:      %s", stats["reaction_cnt"])
    logger.info("Solved:         %s (%s)", stats["solved_cnt"], _format_ratio(stats["success_rate"]))
    logger.info("Correct total:  %s  [accuracy metric: %s]", stats["total_correct"], accuracy_metric)
    logger.info("Accuracy (solved): %s", _format_ratio(stats["accuracy_solved"]))
    logger.info("Accuracy (all): %s", _format_ratio(stats["accuracy"]))
    logger.info("MCS confident:  %s", stats["confident_mcs_cnt"])
    logger.info("MCS accuracy:   %s", _format_ratio(stats["mcs_accuracy"]))
    logger.info("Average similarity:   %s", sum(stats["similarities"]) / len(stats["similarities"]) if stats["similarities"] else 0)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple benchmark evaluator for SynRBL output CSV.")
    parser.add_argument("--scoring-metric", type=metric_name, choices=METRICS, default="v2")
    parser.add_argument("inputfile", help="Path to SynRBL output CSV.")
    parser.add_argument("-o", default=None, help="Optional JSON output file for summary stats.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV output file with similarity and metric columns added.")
    parser.add_argument("--col", default="Reaction", help="Predicted reaction column name.")
    parser.add_argument("--reagents-identified", type=int, default=0, help="Whether the ground truth has reagents identified (1) or not (0). Should be 1 for fair benchmarking against SynRBL results.")
    parser.add_argument("--target-col", default="rxn_flower", help="Expected reaction column name.")
    parser.add_argument("--solved-col", default="solved", help="Solved flag column name.")
    parser.add_argument("--solved-by-col", default="solved_by", help="Column indicating solver type.")
    parser.add_argument("--confidence-col", default="confidence", help="Confidence score column name.")
    parser.add_argument("--id-col", default="Unique_identifier", help="Identifier column used to align with canonical split files.")
    parser.add_argument("--split-file", default=None, help="Optional canonical split CSV to evaluate only one partition (train/valid/test).")
    parser.add_argument("--split-col", default=None, choices=["random_split", "group_split", "extreme_ood"], help="Optional explicit split column in --split-file. If omitted, inferred automatically.")
    parser.add_argument("--split-label", default="test", choices=["train", "valid", "test"], help="Which partition to evaluate when --split-file is provided.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional debug cap on number of rows to evaluate (disabled by default).")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for counting mcs-based solutions as confident/correct.",
    )
    parser.add_argument(
        "--accuracy-metric",
        type=str,
        choices=list(ACCURACY_METRIC_TO_COL.keys()) + ["equivalence"],
        default="exact",
        help=(
            "Match level used to decide whether a prediction is correct (when --add-all-metrics is NOT used). "
            "'exact': pure canonical SMILES comparison — atom-map removal and RDKit canonicalization only, no EQUIV_MAP expansion. Strictest level. "
            "'proton_shuffle': proton-transfer-aware normalization checked both before and after EQUIV_MAP expansion; match passes if either pathway matches. "
            "'all_normalizations': aggregate mode that computes all normalization variants and compares predicted/ground-truth signatures all-against-all. "
            "'pragmatic_equivalence': catch-all metric combining all_normalizations, difference_normalized, and same-path spectator-cancelled core matching. "
            "'deep_pragmatic_equivalence': pragmatic metric plus targeted deep rewrites for common dataset artifacts (H+/H-/H2, acid/base neutralization, SOCl2-family, borohydride shorthand). "
            "Most permissive level. (default: exact)"
        ),
    )
    parser.add_argument(
        "--add-all-metrics",
        action="store_true",
        help="If set, compute all configured accuracy metrics (exact_match, proton_shuffle_match, all_normalizations_match, pragmatic_equivalence_match, deep_pragmatic_equivalence_match) "
             "and add them as boolean columns to output CSV. Ignores --accuracy-metric for column computation (but uses it for JSON summary).",
    )
    parser.add_argument(
        "--similarity-method",
        type=str,
        choices=["ecfp", "set_match"],
        default="ecfp",
        help="Method to compute the continuous reaction  score stored in the output CSV: "
             "'ecfp' (differential Morgan fingerprint Tanimoto) or 'set_match' (0 or 1).",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    ensure_v1_seed(args.scoring_metric)
    if not args.output_csv:
        raise ValueError("Evaluation requires --output-csv to preserve saved predictions")

    if args.output_csv:
        from pathlib import Path
        output=Path(args.output_csv)
        if output.resolve()==Path(args.inputfile).resolve() or output.exists():
            raise ValueError("Choose a new output CSV, separate from predictions")
        output.parent.mkdir(parents=True,exist_ok=True)
    df = pd.read_csv(args.inputfile, low_memory=False)
    logger.info("Loaded %s reactions from %s", len(df), args.inputfile)

    df = filter_df_by_split(df, args)

    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive when provided.")
        df = df.head(args.max_rows).copy()
        logger.info("Limiting evaluation to first %s rows (--max-rows).", args.max_rows)

    logger.info("Compute benchmark results for %s reactions (accuracy metric: %s)", len(df), args.accuracy_metric)

    check_columns(df, args.col, args.target_col)
    stats = evaluate(df, args)
    stats["scoring_metric"]=args.scoring_metric
    print_summary(stats, args.accuracy_metric)

    if args.o:
        with open(args.o, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Detailed stats written to %s", args.o)
    
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        logger.info("Updated CSV with similarity column written to %s", args.output_csv)
        if args.scoring_metric != "v1":
            from utils.scoring import provenance
            inputs=[args.inputfile,__file__] + ([args.split_file] if args.split_file else [])
            provenance(output.parent,args.scoring_metric,inputs,stereo=False,requires_solved=True,
                       confidence_threshold=args.min_confidence,max_rows=args.max_rows)
    else:
        if args.split_file:
            base, ext = os.path.splitext(args.inputfile)
            split_safe_out = f"{base}.split_eval{ext or '.csv'}"
            df.to_csv(split_safe_out, index=False)
            logger.info(
                "No --output-csv provided and --split-file is active; wrote split-filtered output to %s",
                split_safe_out,
            )
        else:
            # Overwrite original CSV with similarity column if no separate output CSV specified.
            df.to_csv(args.inputfile, index=False)
            logger.info("Original CSV updated with similarity column at %s", args.inputfile)


if __name__ == "__main__":
    main()

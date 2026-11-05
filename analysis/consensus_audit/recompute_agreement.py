#!/usr/bin/env python3
"""Measure CRB and SynRBL agreement on the filtered benchmark.

Both predictions must be balanced. The default comparison ignores
stereochemistry. Use --stereo-sensitive for the original comparison.
Reference-match labels come from the saved scores for each prediction.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

# The metric normalizes several halides by iterating a set. Fix Python's hash
# seed before importing it so that this ordering, and the released counts, are
# reproducible. Setting the variable after Python starts is insufficient.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.execvpe(sys.executable, [sys.executable, *sys.argv],
               dict(os.environ, PYTHONHASHSEED="0"))

import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.result_paths import metric_name
from utils.scoring import METRICS, labels, results_root, provenance, verify_score_release
SCORING_METRIC = "v2"
INPUTS = []

RDLogger.DisableLog("rdApp.*")
SPLITS = ("random", "group", "extreme_ood")
SCORES = ROOT / "results/main/scores/per_reaction"
OUTPUT = ROOT / "results/additional/consensus_audit"


def read_csv(path: Path) -> list[dict[str, str]]:
    INPUTS.append(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flag(value: object) -> bool:
    return str(value).strip().lower() in ("true", "1")


def completed_reaction(source: str, difference: str) -> str:
    source_sides = source.split(">")
    difference_sides = difference.split(">")
    if len(source_sides) != 3 or len(difference_sides) != 3:
        return ""
    return ">".join(".".join(molecule for part in (base, add)
                              for molecule in part.split(".") if molecule)
                    for base, add in zip(source_sides, difference_sides))


def load_crb(fold: str, n_best: int) -> pd.DataFrame:
    scored = pd.read_csv(SCORES / f"CRB_{fold}.csv", low_memory=False)
    prepared = ROOT / "models/constrained_reaction_balancer/Template_curated/benchmark/prepared" / fold
    split = 'random' if fold.startswith('random_') else 'group' if fold.startswith('group_') else 'extreme_ood'
    predictions = ROOT / "results/main/predictions/crb" / split / fold / "beamSize5_nBest5.txt"
    log_probs = predictions.with_name(predictions.name + "_log_probs")
    with (prepared / "test_src.txt").open() as handle:
        inputs = ["".join(line.split()) for line in handle]
    with predictions.open() as handle:
        differences = ["".join(line.split()) for line in handle][::n_best]
    with log_probs.open() as handle:
        confidences = [10 ** float(line.strip()) for line in handle][::n_best]
    if not len(inputs) == len(differences) == len(confidences):
        raise ValueError(f"Prediction/input/confidence count mismatch: {fold}")
    indices = scored["original_index"].astype(int).tolist()
    with (prepared / "test_tgt.txt").open() as handle:
        targets = ["".join(line.split()) for line in handle]
    INPUTS.extend([prepared/"test_src.txt", prepared/"test_tgt.txt", predictions, log_probs])
    if len(targets) != len(inputs): raise ValueError("Reference/input count mismatch")
    return pd.DataFrame({
        "Unique_identifier": scored["Unique_identifier"].astype(str),
        "crb_prediction": [completed_reaction(inputs[i], differences[i]) for i in indices],
        "crb_confidence": [confidences[i] for i in indices],
        "reference": [completed_reaction(inputs[i], targets[i]) for i in indices],
    })


def strip_stereo(reaction: str) -> str:
    sides = []
    for side in reaction.split(">"):
        molecules = []
        for smiles in side.split("."):
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            molecules.append(Chem.MolToSmiles(mol, isomericSmiles=False) if mol else smiles)
        sides.append(".".join(molecules))
    return ">".join(sides)


def direct_match(job):
    row, stereo_sensitive, metric = job
    syn, crb, balanced = row["syn_prediction"], row["crb_prediction"], row["both_balanced"]
    if metric != "v1":
        cs = labels(crb, row["reference"], metric=metric, stereo=stereo_sensitive)[1]
        ss = bool(row["syn_solved"]) and labels(syn, row["reference"], metric=metric, stereo=stereo_sensitive)[1]
        bucket = ("correct" if cs else "wrong") + "_" + ("correct" if ss else "wrong")
        agreement = bool(balanced and row["syn_solved"] and labels(syn, crb, metric=metric, stereo=stereo_sensitive)[1])
        if agreement and cs != ss: raise AssertionError("Common-reference inconsistency")
        return agreement, bucket
    if not balanced:
        return False, row["bucket"]
    if not stereo_sensitive:
        syn, crb = strip_stereo(syn), strip_stereo(crb)
    try:
        return bool(compare_reactions(syn, crb).deep_pragmatic_equivalence_match), row["bucket"]
    except Exception:
        return False, row["bucket"]


def fold_name(split: str, rep: int) -> str:
    return f"random_seed{42 + rep}_rep{rep}" if split == "random" else f"{split}_rep{rep}"


def load_records(splits: list[str], reps: list[int]) -> list[dict]:
    syn_source = {row["Unique_identifier"]: row for row in
                  read_csv(results_root("v1")/"main/scores/synrbl_evaluated.csv" if SCORING_METRIC == "v1" else ROOT/"results/main/predictions/synrbl/eval_all_splits_noRI_complete/full_results.csv")}
    records = []
    for rep in reps:
        for split in splits:
            fold = fold_name(split, rep)
            crb = load_crb(fold, 5)
            crb_scores = {r["Unique_identifier"]: r for r in read_csv(SCORES / f"CRB_{fold}.csv")}
            syn_scores = {r["Unique_identifier"]: r for r in read_csv(SCORES / f"SynRBL_{fold}.csv")}
            if len(crb) != len(crb_scores) or crb_scores.keys() != syn_scores.keys():
                raise ValueError(f"Filtered score rows do not align: {fold}")
            for row in crb.itertuples(index=False):
                uid = row.Unique_identifier
                cs, ss, sr = crb_scores[uid], syn_scores[uid], syn_source[uid]
                bucket = ("correct" if flag(cs["equivalence"]) else "wrong") + "_" + (
                    "correct" if flag(ss["equivalence"]) else "wrong")
                records.append({
                    "fold": f"{split}_rep{rep}",
                    "Unique_identifier": uid,
                    "bucket": bucket,
                    "main_accuracy_bucket": bucket,
                    "reference": row.reference,
                    "syn_prediction": sr["Reaction"],
                    "crb_prediction": row.crb_prediction,
                    "both_balanced": int(flag(cs["balanced"]) and flag(ss["balanced"])),
                    "syn_solved": int(flag(sr["solved"])),
                    "syn_confidence": sr["confidence"],
                    "crb_confidence": row.crb_confidence,
                })
            print(f"loaded {fold}: {len(crb)}", flush=True)
    return records


def summarize(records: list[dict], splits: list[str]) -> tuple[list[dict], list[dict]]:
    summary, eligible_summary = [], []
    for split in splits:
        selected = [r for r in records if r["fold"].startswith(split + "_rep")]
        agreed = [r for r in selected if r["direct_agreement"]]
        counts = Counter(r["bucket"] for r in agreed)
        high = [r for r in agreed if r["syn_solved"] and
                float(r["syn_confidence"] or 0) >= .90 and
                float(r["crb_confidence"] or 0) >= .95]
        high_counts = Counter(r["bucket"] for r in high)
        confident_pairs=[r for r in selected if r['syn_solved'] and r['both_balanced']
                         and float(r['syn_confidence'] or 0)>=.90 and float(r['crb_confidence'] or 0)>=.95]
        n = len(agreed)
        summary.append(dict(
            split=split, observations=len(selected), direct_agreement=n,
            direct_agreement_pct=100 * n / len(selected),
            both_correct=counts["correct_correct"],
            crb_correct_syn_wrong=counts["correct_wrong"],
            crb_wrong_syn_correct=counts["wrong_correct"],
            both_wrong=counts["wrong_wrong"],
            strict_reference_precision_pct=100 * counts["correct_correct"] / n if n else None,
            at_least_one_correct_pct=100 * (n - counts["wrong_wrong"]) / n if n else None,
            high_confidence_agreement=len(high),
            both_confident_observations=len(confident_pairs),
            high_confidence_agreement_pct_eligible=100*len(high)/len(confident_pairs) if confident_pairs else None,
            high_confidence_agreement_pct_all=100*len(high)/len(selected),
            high_confidence_both_correct=high_counts["correct_correct"],
            high_confidence_both_wrong=high_counts["wrong_wrong"],
        ))
        eligible = [r for r in selected if r["syn_solved"] and r["both_balanced"]]
        eligible_agreed = [r for r in eligible if r["direct_agreement"]]
        eligible_counts = Counter(r["bucket"] for r in eligible_agreed)
        m = len(eligible_agreed)
        eligible_summary.append(dict(
            split=split, eligible_solved_both_balanced=len(eligible),
            direct_agreement=m, agreement_pct_eligible=100 * m / len(eligible) if eligible else None,
            agreement_pct_all_filtered=100 * m / len(selected),
            both_correct=eligible_counts["correct_correct"],
            one_correct=eligible_counts["correct_wrong"] + eligible_counts["wrong_correct"],
            both_wrong=eligible_counts["wrong_wrong"],
            at_least_one_correct_pct=100 * (m - eligible_counts["wrong_wrong"]) / m if m else None,
        ))
    return summary, eligible_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-sensitive", action="store_true",
                        help="Use the historical direct comparison including stereochemistry")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split-types", default=",".join(SPLITS))
    parser.add_argument("--reps", default="0,1,2,3,4")
    parser.add_argument("--metric", type=metric_name, choices=METRICS, default="v2")
    parser.add_argument("--scores-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    global SCORING_METRIC, SCORES
    SCORING_METRIC = args.metric
    SCORES = args.scores_dir or results_root(args.metric)/"main/scores/per_reaction"
    args.output_dir = args.output_dir or results_root(args.metric)/"additional/consensus_audit"
    if (args.output_dir/"provenance.json").exists(): parser.error("Choose a new output directory")
    splits = [s.strip() for s in args.split_types.split(",") if s.strip()]
    reps = [int(s.strip()) for s in args.reps.split(",") if s.strip()]
    if not splits or any(s not in SPLITS for s in splits) or not reps or args.workers < 1:
        parser.error("Provide valid split types, repetitions and a positive worker count")

    verify_score_release(SCORES,args.metric)
    records = load_records(splits, reps)
    jobs = [(r, args.stereo_sensitive, args.metric) for r in records]
    print(f"comparing {len(records)} observations in fold order", flush=True)
    # Preserve the historical per-observation call sequence. Reusing a result
    # across folds changed some salt-containing matches in this metric version.
    with Pool(processes=args.workers) as pool:
        for i, match in enumerate(pool.imap(direct_match, jobs, chunksize=64)):
            records[i]["direct_agreement"] = int(match[0])
            records[i]["bucket"] = match[1]
            if (i + 1) % 10000 == 0:
                print(f"checked {i + 1}/{len(records)}", flush=True)

    summary, eligible_summary = summarize(records, splits)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    mode = "stereo_sensitive" if args.stereo_sensitive else "stereo_free"
    if args.metric != "v1":
        for row, eligible in zip(summary,eligible_summary):
            row['eligible_observations']=eligible['eligible_solved_both_balanced']
            row['agreement_pct_eligible']=eligible['agreement_pct_eligible']
            h=row['high_confidence_agreement']
            row['high_confidence_reference_precision_pct']=100*row['high_confidence_both_correct']/h if h else None
        write_csv(output/'agreement_summary.csv',summary,list(summary[0]))
        wrong=[r for r in records if r['direct_agreement'] and r['bucket']=='wrong_wrong']
        write_csv(output/'agreed_incorrect.csv',wrong,list(records[0]))
    else:
        write_csv(output / f"{mode}_agreement_summary.csv", summary, list(summary[0]))
        nonboth = [r for r in records if r["direct_agreement"] and r["bucket"] != "correct_correct"]
        write_csv(output / f"{mode}_agreement_not_both_correct.csv", nonboth, list(records[0]))
        both_wrong = [r for r in nonboth if r["bucket"] == "wrong_wrong"]
        write_csv(output / f"{mode}_both_wrong_cases.csv", both_wrong, list(records[0]))
        write_csv(output / f"{mode}_solved_balanced_agreement_summary.csv",
                  eligible_summary, list(eligible_summary[0]))
        wrong = [r for r in both_wrong if r["syn_solved"]]
        write_csv(output / f"{mode}_solved_balanced_both_wrong_cases.csv", wrong, list(records[0]))
    provenance(output, args.metric, INPUTS + [Path(__file__)],
               common_reference=args.metric != "v1", stereo=args.stereo_sensitive,
               requires_syn_solved=args.metric != "v1", requires_both_balanced=True,
               syn_confidence_threshold=.90,crb_confidence_threshold=.95)
    for row in summary:
        print(row, flush=True)
    print("TOTAL", Counter(r["bucket"] for r in records if r["direct_agreement"]), flush=True)


if __name__ == "__main__":
    main()

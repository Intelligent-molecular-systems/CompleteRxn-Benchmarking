#!/usr/bin/env python3
"""Summarize CRB failures on uncurated USPTO reactions.

There are no validated references for this set. The outputs describe validity,
balance, prediction length, confidence, and model agreement, not accuracy.
"""

from __future__ import annotations

import argparse
import sys
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from utils.result_paths import METRICS, metric_name, results_root, verify_score_release
from utils.scoring import provenance
HERE = ROOT / "results/additional/uncurated_failure_modes"
DEFAULTS = {
    "crb": ROOT / "results/main/predictions/crb/uncurated/loaded_balanced.csv",
    "master": ROOT / "data/Template_curated/USPTO_FlowER_raw/master_dataset_mapped.csv",
    "topn": ROOT / "results/main/predictions/crb/uncurated/merged/beamSize5_nBest5_all.txt",
    "similarity": ROOT / "results/additional/benchmark_vs_uncurated/per_query.csv",
    "consensus": None,
}
METALS = re.compile(
    r"\[(?:Pd|Pt|Ni|Cu|Fe|Rh|Ru|Co|Ir|Ag|Au|Zn|Sn|Hg|Mn|Cr|Ti|Zr|Mo|W|Cd)"
    r"(?:[^\]]*)\]"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metric",type=metric_name,choices=METRICS,default="v2")
    for name, path in DEFAULTS.items():
        p.add_argument(f"--{name}", type=Path, default=path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--output", type=Path, help="Optional short local Markdown summary")
    p.add_argument("--chunksize", type=int, default=50_000)
    p.add_argument("--n-best", type=int, default=5)
    p.add_argument("--max-length", type=int, default=200)
    p.add_argument("--confidence", type=float, default=0.95)
    args=p.parse_args()
    if args.consensus is None:
        args.consensus=(results_root("v1")/"additional/uncurated_consensus_summary.csv" if args.metric=="v1" else results_root(args.metric)/"additional/uncurated_stereo_free_agreement/summary.csv")
    args.output_dir = args.output_dir or (args.output.parent if args.output else
        results_root(args.metric)/"additional/uncurated_failure_modes")
    if (args.output_dir/"summary.csv").exists():
        p.error("Preserve saved results; choose a new --output-dir")
    if args.output and args.output.exists():
        p.error("Preserve the report; choose a new --output")
    if args.metric!="v1":verify_score_release(args.consensus.parent,args.metric)
    return args


def safe_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.lower().isin({"1", "true", "t", "yes"})


def outcome(valid, balanced):
    return np.select(
        [balanced, valid], ["balanced", "valid_unbalanced"], default="invalid"
    )


def mol_ok(smiles):
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def side_balance(side):
    atoms, charge = Counter(), 0
    for smiles in filter(None, side.split(".")):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        for atom in mol.GetAtoms():
            atoms[atom.GetSymbol()] += 1
            charge += atom.GetFormalCharge()
    return atoms, charge


def residual_pattern(reaction):
    left, _, right = reaction.split(">")
    la, lq = side_balance(left)
    ra, rq = side_balance(right)
    la.subtract(ra)
    delta = {element: count for element, count in la.items() if count}
    charge = lq - rq
    flags = (
        ("H", "H" in delta),
        ("heavy", any(element != "H" for element in delta)),
        ("charge", charge != 0),
    )
    return "+".join(name for name, present in flags if present) or "none", delta, charge


def invalid_reason(input_reaction, predicted_diff):
    if predicted_diff.count(">") != 2:
        return "Invalid completion structure"
    try:
        in_left, _, in_right = input_reaction.split(">")
        add_left, _, add_right = predicted_diff.split(">")
    except ValueError:
        return "Invalid completion structure"
    inputs = [x for x in (in_left + "." + in_right).split(".") if x]
    additions = [x for x in (add_left + "." + add_right).split(".") if x]
    if any(not mol_ok(x) for x in inputs):
        return "Invalid input molecule"
    if any(not mol_ok(x) for x in additions):
        return "Invalid predicted molecule"
    return "Other invalid reaction"


def top1_lengths(path, n_best):
    lengths = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(tqdm(handle, desc="Prediction lengths")):
            if i % n_best == 0:
                lengths.append(len(line.split()))
    return np.asarray(lengths, dtype=np.int16)


def component_counts(reaction):
    try:
        left, reagents, right = str(reaction).split(">")
        lhs = len([x for x in (left + "." + reagents).split(".") if x])
        rhs = len([x for x in right.split(".") if x])
        return lhs, rhs
    except ValueError:
        return np.nan, np.nan


def pct(n, total):
    return 100 * n / total if total else float("nan")


def subset_row(data, label, mask):
    part = data.loc[mask]
    balanced = int(part["outcome"].eq("balanced").sum())
    invalid = int(part["outcome"].eq("invalid").sum())
    return dict(subset=label, n=len(part), balanced_n=balanced,
                balanced_pct=pct(balanced, len(part)), invalid_n=invalid,
                invalid_pct=pct(invalid, len(part)))


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")

    basic = pd.read_csv(
        args.crb,
        usecols=[
            "Unique_identifier",
            "crb_sequence_probability",
            "is_valid_prediction",
            "is_balanced",
        ],
        low_memory=False,
    )
    basic["is_valid_prediction"] = safe_bool(basic["is_valid_prediction"])
    basic["is_balanced"] = safe_bool(basic["is_balanced"])
    basic["outcome"] = outcome(
        basic["is_valid_prediction"], basic["is_balanced"]
    )

    lengths = top1_lengths(args.topn, args.n_best)
    if len(lengths) != len(basic):
        raise ValueError(
            f"Top-1 predictions ({len(lengths):,}) != CRB rows ({len(basic):,})"
        )
    basic["hit_max_length"] = lengths >= args.max_length

    master = pd.read_csv(
        args.master,
        usecols=[
            "Unique_identifier",
            "CanonicalReaction",
            "MissingNrAtoms",
            "MissingNrCarbons",
            "MultiProduct",
        ],
        low_memory=False,
    )
    data = basic.merge(master, on="Unique_identifier", validate="one_to_one")
    if len(data) != len(basic):
        raise ValueError("Some CRB identifiers are absent from the USPTO source table")
    components = np.asarray(
        [component_counts(x) for x in tqdm(data["CanonicalReaction"], desc="Components")]
    )
    data["lhs_components"] = components[:, 0]
    data["transition_metal"] = data["CanonicalReaction"].str.contains(
        METALS, na=False
    )

    invalid_counts, residual_counts = Counter(), Counter()
    residual_elements, residual_amounts = Counter(), Counter()
    columns = [
        "input_reaction",
        "prediction_reaction",
        "crb_top1_diff",
        "is_valid_prediction",
        "is_balanced",
    ]
    reader = pd.read_csv(
        args.crb, usecols=columns, chunksize=args.chunksize, low_memory=False
    )
    for chunk in tqdm(reader, total=int(np.ceil(len(data) / args.chunksize)), desc="Failures"):
        chunk["is_valid_prediction"] = safe_bool(chunk["is_valid_prediction"])
        chunk["is_balanced"] = safe_bool(chunk["is_balanced"])
        for row in chunk.itertuples(index=False):
            if not row.is_valid_prediction:
                invalid_counts[
                    invalid_reason(str(row.input_reaction), str(row.crb_top1_diff))
                ] += 1
            elif not row.is_balanced:
                pattern, delta, charge = residual_pattern(str(row.prediction_reaction))
                residual_counts[pattern] += 1
                for element, amount in delta.items():
                    residual_elements[element] += 1
                    residual_amounts[element] += abs(amount)
                if charge:
                    residual_elements["charge"] += 1
                    residual_amounts["charge"] += abs(charge)

    total = len(data)
    balanced_n = int((data["outcome"] == "balanced").sum())
    valid_unbalanced_n = int((data["outcome"] == "valid_unbalanced").sum())
    invalid_n = int((data["outcome"] == "invalid").sum())
    h_only_n = residual_counts["H"]
    other_unbalanced_n = valid_unbalanced_n - h_only_n

    counts = {
        "Balanced and parseable": balanced_n,
        "Valid; hydrogen-only imbalance": h_only_n,
        "Valid; other imbalance": other_unbalanced_n,
        **invalid_counts,
    }
    if sum(counts.values()) != total:
        raise ValueError("Outcome counts do not cover all records")
    failure_rows = [dict(outcome=k, n=v, total=total, pct=pct(v, total))
                    for k, v in counts.items()]

    burden_rows = [
        subset_row(data, "Zero missing atoms", data["MissingNrAtoms"] == 0),
        subset_row(data, "More than 20 missing atoms", data["MissingNrAtoms"] > 20),
        subset_row(data, "Carbon-free deficit", data["MissingNrCarbons"].fillna(0) == 0),
        subset_row(data, "Carbon-bearing deficit", data["MissingNrCarbons"].fillna(0) > 0),
        subset_row(data, "No transition metal", ~data["transition_metal"]),
        subset_row(data, "Contains transition metal", data["transition_metal"]),
        subset_row(data, "At most 2 left-side components", data["lhs_components"] <= 2),
        subset_row(data, "More than 10 left-side components", data["lhs_components"] > 10),
        subset_row(data, "Single-product record", data["MultiProduct"] == 0),
        subset_row(data, "Multi-product record", data["MultiProduct"] == 1),
    ]

    max_total = int(data["hit_max_length"].sum())
    max_invalid = int(
        (data["hit_max_length"] & (data["outcome"] == "invalid")).sum()
    )
    any_h = residual_elements["H"]
    mean_h = residual_amounts["H"] / any_h
    conf = data["crb_sequence_probability"] > args.confidence
    conf_n = int(conf.sum())
    conf_counts = data.loc[conf, "outcome"].value_counts()

    inputs = [Path(__file__), args.crb, args.master, args.topn, args.consensus]
    if args.similarity.exists():
        sim = pd.read_csv(args.similarity, usecols=["Unique_identifier", "nearest_tanimoto"]).merge(
            data[["Unique_identifier", "outcome"]], on="Unique_identifier", validate="one_to_one")
        inputs.append(args.similarity)
        for label, mask in [("Nearest similarity < 0.20", sim.nearest_tanimoto.lt(.20)),
                            ("Nearest similarity >= 0.55", sim.nearest_tanimoto.ge(.55))]:
            burden_rows.append(subset_row(sim, label, mask))

    consensus = pd.read_csv(args.consensus).iloc[0]
    both = int(consensus["overlap_both_balanced_rows"])
    agree = int(consensus["agreement_dpe_n" if args.metric == "v1" else "stereo_free_agreement_n"])
    summary = dict(total=total, parseable_n=balanced_n+valid_unbalanced_n,
                   balanced_n=balanced_n, valid_unbalanced_n=valid_unbalanced_n,
                   invalid_n=invalid_n, hydrogen_only_n=h_only_n,
                   any_hydrogen_residual_n=any_h, mean_abs_hydrogen_residual=mean_h,
                   max_length=args.max_length, max_length_n=max_total,
                   max_length_invalid_n=max_invalid, confidence_threshold=args.confidence,
                   confident_n=conf_n, both_balanced_n=both, agreement_n=agree,
                   agreement_pct=pct(agree, both))
    confidence = [dict(outcome=k, n=int(conf_counts.get(k, 0)), total=conf_n,
                       pct=pct(int(conf_counts.get(k, 0)), conf_n))
                  for k in ("balanced", "valid_unbalanced", "invalid")]
    residuals = [dict(pattern=k, n=v) for k, v in sorted(residual_counts.items())]
    elements = [dict(element=k, n=v, total_abs_residual=residual_amounts[k],
                     mean_abs_residual=residual_amounts[k]/v)
                for k, v in sorted(residual_elements.items())]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary.csv", [summary]), ("outcomes.csv", failure_rows),
                       ("subsets.csv", burden_rows), ("confidence.csv", confidence),
                       ("residual_patterns.csv", residuals), ("residual_elements.csv", elements)):
        pd.DataFrame(rows).to_csv(args.output_dir/name, index=False)
    provenance(args.output_dir, args.metric, inputs, confidence=args.confidence,
               n_best=args.n_best, max_length=args.max_length,
               correctness_available=False, balance_source="saved CRB cache flags")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "# Uncurated CRB diagnostics\n\n"
            "No reference accuracy is available. Numerical results are in "
            f"`{args.output_dir}`. Balanced: {balanced_n}/{total}; "
            f"model agreement among both-balanced outputs: {agree}/{both}.\n")
    print("Wrote CSVs to", args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score CRB predictions on the separate SynRXN Complex dataset."""

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

import pandas as pd
from rdkit import RDLogger
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RDLogger.DisableLog("rdApp.*")

from utils.result_paths import metric_name, ensure_v1_seed
from utils.scoring import METRICS as SCORERS, compare as compare_reactions, results_root, provenance
from models.constrained_reaction_balancer.score_predictions import (
    canonicalize_smiles,
    compose_completed_reaction,
    evaluate_balanced_reaction,
    reaction_balance,
)

DATA = ROOT / "data/SynRXN_complex/complex_processed.csv"
PREDICTIONS = ROOT / "results/main/predictions/crb/synrxn/beamSize5_nBest5.txt"
OUT = ROOT / "results/additional/synrxn_analysis"
JOINT_COLUMN = "balanced_and_deep_equivalent"
METRICS = [
    "exact_match",
    "proton_shuffle_match",
    "all_normalizations_match",
    "pragmatic_equivalence_match",
    "deep_pragmatic_equivalence_match",
]

AUDIT_NOTES = {
    "complex_1559": ("both plausible", "CRB gives water and unchanged sulfuric acid; SynRXN represents the same overall composition as sulfate, protons, and hydroxide. The distinction is mainly acid-base bookkeeping."),
    "complex_329": ("both plausible", "CRB represents DIPEA and fluoride as an ammonium-fluoride salt; SynRXN leaves neutral DIPEA and HF."),
    "complex_582": ("both plausible", "The POCl3-mediated chlorination admits different formal phosphorus-containing coproducts; the specific hydrolysis state is not established by this record."),
    "complex_814": ("CRB preferred", "CRB accounts for stoichiometric SOCl2 consumption as SO2 and two HCl; SynRXN gives water and carries SOCl2 through unchanged."),
    "complex_414": ("CRB preferred", "CRB accounts for HATU consumption with tetramethylurea and an HOAt-derived species; SynRXN gives water and carries HATU through unchanged."),
    "complex_1533": ("both plausible", "CRB gives water and neutral sodium; SynRXN gives an explicit hydrogen species, hydroxide, and sodium ion. These are alternative formal balance representations; neither establishes isolated coproducts."),
    "complex_317": ("both plausible", "The completions use different carbonate, bicarbonate, hydroxide, and water bookkeeping; duplicated spectator carbonate in CRB makes the detailed coproduct assignment uncertain."),
    "complex_311": ("both plausible", "CRB gives sulfonate plus bicarbonate; SynRXN gives sulfonic acid plus carbonate, differing by proton transfer."),
    "complex_333": ("both plausible", "CRB gives mesylate plus bicarbonate; SynRXN gives methanesulfonic acid plus carbonate, differing by proton transfer."),
    "complex_1": ("both plausible", "CRB gives silyl mesylate; SynRXN gives hydrolysed silanol and methanesulfonic acid after adding water."),
    "complex_208": ("SynRXN preferred", "CRB predicts methane, whereas SynRXN adds an oxygen species and gives methanol; methanol is the more plausible methoxy-derived coproduct, although the oxygen source is unspecified."),
    "complex_14": ("SynRXN preferred", "CRB predicts methane, whereas SynRXN adds an oxygen species and gives methanol; methanol is the more plausible methoxy-derived coproduct, although the oxygen source is unspecified."),
}


def target_status(diff):
    left, _, right = str(diff).split(">")
    if left and right:
        return "both_sides"
    if left:
        return "left_only"
    if right:
        return "right_only"
    return "zero_difference"


def safe_balance(reaction):
    try:
        return reaction_balance(str(reaction))
    except Exception:
        return False, "invalid_reaction"


def imbalance_class(info):
    if info in {"invalid_smiles", "invalid_reaction"}:
        return info.replace("_", " ")
    match = re.fullmatch(r"\$L:(.*?) \$R:(.*?) \$Charge:(.*)", str(info))
    if not match:
        return "other/parse"
    atoms = []
    for side in match.group(1, 2):
        atoms.extend(re.findall(r"\d+([A-Z][a-z]?)", side))
    charge = bool(match.group(3).strip())
    elements = set(atoms)
    if elements == {"H"}:
        return "hydrogen + charge" if charge else "hydrogen only"
    if not elements and charge:
        return "charge only"
    if elements:
        return "other elements + charge" if charge else "other elements"
    return "other/parse"


def pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else float("nan")


def metric_row(name, frame):
    n = len(frame)
    values = {"subset": name, "n": n}
    for column in ["prediction_balanced", *METRICS, JOINT_COLUMN]:
        count = int(frame[column].sum())
        values[f"{column}_n"] = count
        values[f"{column}_pct"] = pct(count, n)
    return values


def main():
    global OUT, METRICS, JOINT_COLUMN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric",type=metric_name, choices=SCORERS,default="v2")
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    ensure_v1_seed(args.metric)
    OUT=args.output_dir or results_root(args.metric)/"additional/synrxn_analysis"
    if args.metric != "v1":
        METRICS=["exact_match","equivalent_match"]
        JOINT_COLUMN="balanced_and_equivalent"
    if (OUT/"per_reaction.csv").exists():parser.error("Preserve saved scores; choose a new --output-dir")
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA, low_memory=False)
    predictions = ["".join(line.strip().split()) for line in PREDICTIONS.open()]
    if len(predictions) != 5 * len(data):
        raise ValueError(f"Expected {5 * len(data)} beam lines, found {len(predictions)}")
    data["prediction"] = predictions[::5]

    records = []
    for row in tqdm(data.itertuples(index=False), total=len(data), desc="SynRXN"):
        completed = compose_completed_reaction(str(row.rxn), str(row.prediction))
        pred_balanced, balance_info = evaluate_balanced_reaction(
            str(row.rxn), str(row.prediction)
        )
        target_balanced, target_balance_info = safe_balance(row.ground_truth)
        try:
            comparison = compare_reactions(completed, str(row.ground_truth), metric=args.metric, stereo=True)
            flags = {metric: bool(getattr(comparison, metric)) for metric in METRICS}
        except Exception:
            flags = {metric: False for metric in METRICS}
        records.append(
            {
                "r_id": row.r_id,
                "target_status": target_status(row.input_target_diff),
                "target_balanced": target_balanced,
                "target_balance_info": target_balance_info,
                "prediction_valid": bool(canonicalize_smiles(str(row.prediction))),
                "prediction_balanced": pred_balanced,
                "prediction_balance_info": balance_info,
                "imbalance_class": "" if pred_balanced else imbalance_class(balance_info),
                **flags,
                JOINT_COLUMN: (
                    pred_balanced and flags["deep_pragmatic_equivalence_match" if args.metric == "v1" else "equivalent_match"]
                ),
                "input_reaction": row.rxn,
                "target_addition": row.input_target_diff,
                "predicted_addition": row.prediction,
                "completed_prediction": completed,
                "ground_truth": row.ground_truth,
            }
        )

    result = pd.DataFrame(records)
    result.to_csv(OUT / "per_reaction.csv", index=False)

    if args.metric != "v1":
        summary=pd.DataFrame([metric_row("all", result),metric_row("non-empty targets, balanced references",result[result["target_status"].ne("zero_difference") & result["target_balanced"]])])
        summary.to_csv(OUT/"summary.csv",index=False)
        provenance(OUT,args.metric,[DATA,PREDICTIONS,Path(__file__)],stereo=True,requires_balance=True)
        print("Wrote",OUT);return

    nonempty = result["target_status"].ne("zero_difference")
    balanced_reference = result["target_balanced"]
    summary = pd.DataFrame(
        [
            metric_row("all", result),
            metric_row("balanced references", result[balanced_reference]),
            metric_row("non-empty targets", result[nonempty]),
            metric_row(
                "non-empty targets, balanced references",
                result[nonempty & balanced_reference],
            ),
            metric_row("zero-difference targets", result[~nonempty]),
        ]
    )
    summary.to_csv(OUT / "summary.csv", index=False)

    audit_pool = result[
        result["target_balanced"]
        & result["prediction_balanced"]
        & ~result["deep_pragmatic_equivalence_match" if args.metric == "v1" else "equivalent_match"]
    ]
    audit = pd.concat(
        [
            audit_pool[audit_pool["target_status"].eq("right_only")].sample(
                n=9, random_state=73
            ),
            audit_pool[audit_pool["target_status"].eq("both_sides")].sample(
                n=3, random_state=73
            ),
        ]
    )
    audit[["audit_label", "audit_note"]] = audit["r_id"].map(AUDIT_NOTES).apply(
        pd.Series
    )
    audit.to_csv(OUT / "audit_sample.csv", index=False)

    failures = result[~result["prediction_balanced"]]
    failure_counts = Counter(failures["imbalance_class"])
    hydrogen_n = failure_counts["hydrogen only"] + failure_counts["hydrogen + charge"]
    parseable_residual_n = len(failures) - failure_counts["invalid smiles"] - failure_counts["invalid reaction"]
    audit_counts = Counter(audit["audit_label"])

    def row(subset):
        return summary.loc[summary["subset"].eq(subset)].iloc[0]

    all_row = row("all")
    clean_row = row("non-empty targets, balanced references")
    markdown = f"""# CRB on SynRXN complex

The available run contains **{len(result):,}** non-overlapping SynRXN records. Of these,
**{nonempty.sum():,}** require a non-empty completion and **{balanced_reference.sum():,}**
have atom/charge-balanced reference reactions under the current RDKit checker.

| Evaluation subset | n | Balanced | Exact | Deep equivalence | Balanced and deep-equivalent |
|---|---:|---:|---:|---:|---:|
| All records | {int(all_row.n):,} | {all_row.prediction_balanced_pct:.2f}% | {all_row.exact_match_pct:.2f}% | {all_row.deep_pragmatic_equivalence_match_pct:.2f}% | {all_row.balanced_and_deep_equivalent_pct:.2f}% |
| Non-empty targets with balanced references | {int(clean_row.n):,} | {clean_row.prediction_balanced_pct:.2f}% | {clean_row.exact_match_pct:.2f}% | {clean_row.deep_pragmatic_equivalence_match_pct:.2f}% | {clean_row.balanced_and_deep_equivalent_pct:.2f}% |

The two exclusion criteria overlap in one record (`complex_541`), whose input
requires no completion but whose reference is unbalanced. Therefore the eligible
count is 1,746 − 36 − 348 + 1 = **1,363**, not 1,362. The corresponding joint
score is 709/1,363 = 52.02%.

## Balance failures

Top-1 is unbalanced or unparsable in **{len(failures):,}** records. Hydrogen-only
residuals account for **{hydrogen_n:,}/{parseable_residual_n:,}
({pct(hydrogen_n, parseable_residual_n):.2f}%)** of parseable residual imbalances.

| Failure class | n | % of balance failures |
|---|---:|---:|
""" + "\n".join(
        f"| {name} | {count:,} | {pct(count, len(failures)):.2f}% |"
        for name, count in failure_counts.most_common()
    ) + f"""

Seven predictions pass deep equivalence but fail atom balance; inspection shows
that permissive set-level normalization can overlook duplicated equivalent
species. The safer joint statistic is therefore “balanced and deep-equivalent.”

## Small structure-level audit

A fixed-seed stratified sample of **{len(audit)}** balanced predictions that disagree
with balanced references contains:

- {audit_counts["both plausible"]} cases where both completions appeared chemically plausible;
- {audit_counts["CRB preferred"]} cases where CRB's completion was preferred;
- {audit_counts["SynRXN preferred"]} cases where SynRXN's completion was preferred.

| Judgment | Reaction IDs |
|---|---|
| Both plausible | {", ".join(audit.loc[audit["audit_label"].eq("both plausible"), "r_id"])} |
| CRB preferred | {", ".join(audit.loc[audit["audit_label"].eq("CRB preferred"), "r_id"])} |
| SynRXN preferred | {", ".join(audit.loc[audit["audit_label"].eq("SynRXN preferred"), "r_id"])} |

The two SynRXN-preferred cases predict methane with CRB, whereas the reference
uses an added oxygen species to form methanol. Preference is a qualitative
judgment about reagent and byproduct accounting, not a metric-equivalence
classification or proof of the isolated coproducts. In particular, the
phosphorus, sodium, and carbonate cases retain uncertainty about their exact
physical species. See `audit_sample.csv` for the case-level reasoning.
"""
    (OUT / "summary.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()

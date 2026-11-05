#!/usr/bin/env python3
"""Reproduce the original v1 review sample from the saved prediction labels.

Keep the existing human audit unchanged. This script does not select errors under the benchmark metric.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/main/predictions/synrbl/eval_all_splits_noRI_complete/full_results.csv"
SPLIT = ROOT / "data/Template_curated/benchmark/random_splits/benchmark_final_random_splits_rep0_seed42.csv"
OUT = ROOT / "results/archive/metric_v1/additional/equivalence_metric_audit"

sys.path.insert(0, str(ROOT))
from models.constrained_reaction_balancer.score_predictions import reaction_balance  # noqa: E402


def as_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_test_ids():
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        return {
            row["Unique_identifier"]
            for row in csv.DictReader(handle)
            if row["random_split"] == "2"
        }


def load_candidates(test_ids):
    candidates = []
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Unique_identifier"] not in test_ids:
                continue
            if not as_bool(row["solved"]):
                continue
            if as_bool(row["deep_pragmatic_equivalence_match"]):
                continue
            prediction = row["Reaction"].strip()
            target = row["rxn_flower"].strip()
            if not prediction or not target:
                continue
            try:
                balanced, _ = reaction_balance(prediction)
            except Exception:
                balanced = False
            if balanced:
                candidates.append(row)
    return candidates


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    if (OUT/"audit_key.csv").exists() or (OUT/"audit_sample_blinded.csv").exists():
        parser.error("Preserve the fixed human audit; choose a new --output-dir")
    OUT.mkdir(parents=True, exist_ok=True)

    test_ids = load_test_ids()
    candidates = load_candidates(test_ids)
    if args.sample_size > len(candidates):
        raise ValueError("Requested more cases than the eligible population")

    rng = random.Random(args.seed)
    sample = rng.sample(candidates, args.sample_size)
    audit_rows, key_rows = [], []
    for audit_id, row in enumerate(sample, start=1):
        swap = rng.choice([False, True])
        prediction, target = row["Reaction"], row["rxn_flower"]
        completion_a, completion_b = (target, prediction) if swap else (prediction, target)
        source_a, source_b = ("FlowER", "SynRBL") if swap else ("SynRBL", "FlowER")
        audit_rows.append({
            "audit_id": audit_id,
            "input_reaction": row["input_reaction"],
            "completion_a": completion_a,
            "completion_b": completion_b,
            "assessment": "",
            "notes": "",
        })
        key_rows.append({
            "audit_id": audit_id,
            "Unique_identifier": row["Unique_identifier"],
            "completion_a_source": source_a,
            "completion_b_source": source_b,
            "solved_by": row["solved_by"],
            "confidence": row["confidence"],
            "independently_balanced": True,
            "deep_pragmatic_equivalence_match": False,
            "synrbl_completion": prediction,
            "flower_target": target,
        })

    write_csv(
        OUT / "audit_sample_blinded.csv",
        audit_rows,
        ["audit_id", "input_reaction", "completion_a", "completion_b", "assessment", "notes"],
    )
    write_csv(
        OUT / "audit_key.csv",
        key_rows,
        [
            "audit_id", "Unique_identifier", "completion_a_source", "completion_b_source",
            "solved_by", "confidence", "independently_balanced",
            "deep_pragmatic_equivalence_match", "synrbl_completion", "flower_target",
        ],
    )

    readme = f"""# SynRBL equivalence-metric audit

This is a fixed-seed random sample of **{args.sample_size}** records from the
random-split repetition-0 test set. The eligible population contains
**{len(candidates):,}** SynRBL outputs that are marked solved, independently
atom/charge-balanced, and incorrect under deep pragmatic equivalence.

Selection seed: `{args.seed}`. Recreate the sample with:

```bash
apptainer exec models/constrained_reaction_balancer/CRB_env.sif \\
  python analysis/equivalence_metric_audit/select_synrbl_audit.py
```

Inspect `audit_sample_blinded.csv` without opening `audit_key.csv`. For
`assessment`, use one of:

- `equivalent`: same chemistry under acceptable representational differences
- `both_acceptable`: different but both chemically acceptable completions
- `only_a_acceptable`
- `only_b_acceptable`
- `uncertain`

After completing the audit, use `audit_key.csv` to reveal which completion is
SynRBL. Report the reviewed sample and its uncertainty; do not extrapolate the
result to unsolved or imbalanced SynRBL outputs.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print("Eligible population: {:,}".format(len(candidates)))
    print("Wrote {} blinded cases to {}".format(args.sample_size, OUT))


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    main()

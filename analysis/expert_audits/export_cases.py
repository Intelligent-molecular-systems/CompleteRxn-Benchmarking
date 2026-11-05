#!/usr/bin/env python3
"""Export the two expert audits as compact, anonymized case tables.

Target audit (Appendix A.2): FlowER controls embedded in the curation app with
exactly three expert completions. Inputs are the outputs of
data/Expert_curation/analyze_flower_controls.py, which stay local because the
raw curation export contains usernames. Experts appear only as "Expert 1-7".

Metric audit (Appendix B.4): the 20 blinded SynRBL-FlowER pairs and the
expert's judgement, from results/archive/metric_v1/additional/
equivalence_metric_audit. The LLM comparison there is deliberately not exported.

Both tables are relabelled with the current equivalence metric. build_appendix.py
renders figures and LaTeX from these files only.

    python analysis/expert_audits/export_cases.py
"""

from pathlib import Path
import sys

import pandas as pd
from rdkit import RDLogger


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.equivalence_v2 import METRIC_VERSION, compare_reactions  # noqa: E402
from utils.scoring import provenance  # noqa: E402

CONTROLS = ROOT / "data/Expert_curation/control_reactions.csv"
VOTES = ROOT / "data/Expert_curation/expert_votes.csv"
METRIC_AUDIT = ROOT / "results/archive/metric_v1/additional/equivalence_metric_audit"
KEY = METRIC_AUDIT / "audit_key.csv"
BLINDED = METRIC_AUDIT / "audit_sample_blinded.csv"
DETAILS = METRIC_AUDIT / "expert_audit_details.csv"
OUT = ROOT / "results/additional/expert_audits"


def equivalent(prediction, reference, stereo):
    try:
        return compare_reactions(prediction, reference, stereo=stereo).equivalent_match
    except ValueError:
        return False


def export_target_audit():
    controls = pd.read_csv(CONTROLS)
    controls = controls[controls["n_experts"].eq(3)]
    votes = pd.read_csv(VOTES)
    votes = votes[votes["reaction_id"].isin(controls["reaction_id"])]

    completions = []
    for vote in votes.itertuples():
        target = controls.loc[controls["reaction_id"].eq(vote.reaction_id), "flower_target"].item()
        solutions = [s.strip() for s in str(vote.curated_solutions).split(" | ")]
        for index, solution in enumerate(solutions, 1):
            if not solution or solution.upper() == "SKIPPED":
                continue
            completions.append({
                "control_id": vote.reaction_id,
                "expert_id": vote.expert_id,
                "solution_rank": index,
                "completion": solution,
                "equivalent_match": equivalent(solution, target, stereo=True),
            })
    completions = pd.DataFrame(completions)

    per_expert = completions.groupby(["control_id", "expert_id"])["equivalent_match"].any()
    matching = per_expert.groupby("control_id").sum().rename("experts_matching")
    table = (
        controls.rename(columns={
            "reaction_id": "control_id",
            "Unique_identifier": "unique_identifier",
            "original_rxn_smiles": "input_reaction",
        })[["control_id", "unique_identifier", "input_reaction", "flower_target", "n_experts"]]
        .merge(matching, on="control_id")
        .sort_values("control_id")
    )
    table.to_csv(OUT / "target_audit_controls.csv", index=False)
    completions.sort_values(["control_id", "expert_id", "solution_rank"]).to_csv(
        OUT / "target_audit_completions.csv", index=False)
    print("Target audit: {} controls, {} with >=1 matching expert, {} with a majority".format(
        len(table), int(table["experts_matching"].gt(0).sum()),
        int(table["experts_matching"].ge(2).sum())))


def export_metric_audit():
    key = pd.read_csv(KEY)
    blinded = pd.read_csv(BLINDED)[["audit_id", "input_reaction"]]
    details = pd.read_csv(DETAILS)
    details = details[details["assessor"].eq("expert")].rename(
        columns={"confidence": "expert_confidence"})
    table = key.drop(columns="confidence").merge(blinded, on="audit_id").merge(
        details[["audit_id", "equivalence", "outcome", "expert_confidence", "notes"]],
        on="audit_id")
    # SynRBL drops stereochemistry, so it is compared without it, as in the benchmark.
    table["equivalent_match"] = [
        equivalent(row.synrbl_completion, row.flower_target, stereo=False)
        for row in table.itertuples()
    ]
    table = table.rename(columns={
        "Unique_identifier": "unique_identifier",
        "equivalence": "expert_equivalence",
        "outcome": "expert_outcome",
        "notes": "expert_notes",
        "deep_pragmatic_equivalence_match": "initial_metric_match",
    })[[
        "audit_id", "unique_identifier", "input_reaction", "flower_target",
        "synrbl_completion", "expert_equivalence", "expert_outcome",
        "expert_confidence", "expert_notes", "initial_metric_match", "equivalent_match",
    ]].sort_values("audit_id")
    table.to_csv(OUT / "metric_audit_cases.csv", index=False)
    accepted = table.loc[table["equivalent_match"], "audit_id"].tolist()
    expert = table.loc[table["expert_equivalence"].eq("equivalent"), "audit_id"].tolist()
    print("Metric audit: revised metric accepts {}; expert judged equivalent {}".format(
        accepted, expert))


def main():
    RDLogger.DisableLog("rdApp.*")
    OUT.mkdir(parents=True, exist_ok=True)
    export_target_audit()
    export_metric_audit()
    provenance(OUT, "v2", [CONTROLS, VOTES, KEY, BLINDED, DETAILS],
               target_audit_stereo=True, metric_audit_stereo=False)
    print("Wrote {} (metric v{})".format(OUT, METRIC_VERSION))


if __name__ == "__main__":
    main()

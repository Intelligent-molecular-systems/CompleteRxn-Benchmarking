#!/usr/bin/env python3
"""Classify the molecules missing from each reaction.

Compare CanonicalReaction with flower_target_proton_reagents_identified from
analysis/bias_analysis/analyze.py. Match input reactants and reagents against
target reactants, then compare products separately. Target reagents are excluded.
Molecule matching allows the proton transfers used by the category analysis.
The original unmatched target molecules determine their categories.

Each missing molecule is a helper from the fixed chem_balancer list, a
non-helper without carbon, or a non-helper containing carbon. A reaction's
category records which of those types are present, such as "Helper + carbon".
These categories do not depend on the selected accuracy metric.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm


RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from analysis.bias_analysis.analyze import proton_key

# Exact SMILES list from chem_balancer/helpCompound.py at commit
# c4e67a605a094b1140171f0324b76f490a377e10.
HELPER_SMILES = [
    "CC(=O)O", "O", "[H][H]", "O=O", "N", "N#N", "[Na+]", "[K+]",
    "O=[N+]=O", "O=S(O)[O-]", "C", "Cl", "Br", "I", "F", "P", "S",
    "O=P(O)O", "O=S(=O)(O)O", "O=N(=O)O", "O=Cl(=O)O", "CO", "CCO",
    "CCCO", "CCCCO", "CCC(=O)O", "CCCC(=O)O", "COCC(=O)O",
    "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Clc1ccccc1",
]

CATEGORY_ORDER = [
    "Helper only",
    "Non-carbon only",
    "Carbon only",
    "Helper + non-carbon",
    "Helper + carbon",
    "Non-carbon + carbon",
    "Helper + non-carbon + carbon",
    "No missing molecules",
    "Unclassified",
]


def repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


@lru_cache(maxsize=None)
def molecule(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol.GetMol()


@lru_cache(maxsize=None)
def identity(smiles: str) -> str:
    """Return canonical SMILES without stereochemistry or other chemical changes."""
    return Chem.MolToSmiles(
        molecule(smiles), canonical=True, isomericSmiles=False
    )


@lru_cache(maxsize=None)
def contains_carbon(smiles: str) -> bool:
    return any(atom.GetAtomicNum() == 6 for atom in molecule(smiles).GetAtoms())


HELPER_IDENTITIES = frozenset(identity(smiles) for smiles in HELPER_SMILES)


def split_reaction(reaction: str) -> tuple[list[str], list[str], list[str]]:
    parts = str(reaction).split(">")
    if len(parts) != 3:
        raise ValueError(f"Expected three-part reaction: {reaction!r}")
    return tuple(
        [component for component in side.split(".") if component]
        for side in parts
    )  # type: ignore[return-value]


def missing_on_side(source: list[str], target: list[str]) -> list[str]:
    """List target molecules missing from the source, keeping counts and target order."""
    available = Counter(proton_key(component) for component in source)
    missing = []
    for component in target:
        key = proton_key(component)
        if available[key] > 0:
            available[key] -= 1
        else:
            missing.append(identity(component))
    return missing


def component_type(smiles: str) -> str:
    if smiles in HELPER_IDENTITIES:
        return "helper"
    return "carbon" if contains_carbon(smiles) else "non-carbon"


def combination_label(types: set[str]) -> str:
    labels = {
        frozenset({"helper"}): "Helper only",
        frozenset({"non-carbon"}): "Non-carbon only",
        frozenset({"carbon"}): "Carbon only",
        frozenset({"helper", "non-carbon"}): "Helper + non-carbon",
        frozenset({"helper", "carbon"}): "Helper + carbon",
        frozenset({"non-carbon", "carbon"}): "Non-carbon + carbon",
        frozenset({"helper", "non-carbon", "carbon"}):
            "Helper + non-carbon + carbon",
        frozenset(): "No missing molecules",
    }
    return labels[frozenset(types)]


def classify(row: pd.Series) -> dict:
    uid = str(row["Unique_identifier"])
    try:
        src_r, src_g, src_p = split_reaction(row["CanonicalReaction"])
        tgt_r, tgt_g, tgt_p = split_reaction(
            row["flower_target_proton_reagents_identified"]
        )
        missing_preproduct = missing_on_side(src_r + src_g, tgt_r)
        missing_products = missing_on_side(src_p, tgt_p)
        missing = missing_preproduct + missing_products
        typed = [(smiles, component_type(smiles)) for smiles in missing]
        types = {kind for _, kind in typed}
        return {
            "Unique_identifier": uid,
            "missing_category": combination_label(types),
            "n_missing": len(missing),
            "n_helper": sum(kind == "helper" for _, kind in typed),
            "n_noncarbon": sum(kind == "non-carbon" for _, kind in typed),
            "n_carbon": sum(kind == "carbon" for _, kind in typed),
            "missing_preproduct": ".".join(missing_preproduct),
            "missing_products": ".".join(missing_products),
            "helper_molecules": ".".join(
                smiles for smiles, kind in typed if kind == "helper"
            ),
            "noncarbon_molecules": ".".join(
                smiles for smiles, kind in typed if kind == "non-carbon"
            ),
            "carbon_molecules": ".".join(
                smiles for smiles, kind in typed if kind == "carbon"
            ),
            "classification_error": "",
        }
    except Exception as exc:
        return {
            "Unique_identifier": uid,
            "missing_category": "Unclassified",
            "n_missing": 0,
            "n_helper": 0,
            "n_noncarbon": 0,
            "n_carbon": 0,
            "missing_preproduct": "",
            "missing_products": "",
            "helper_molecules": "",
            "noncarbon_molecules": "",
            "carbon_molecules": "",
            "classification_error": f"{type(exc).__name__}: {exc}",
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', default='data/Template_curated/benchmark/benchmark_final.csv')
    parser.add_argument('--derived-targets', default='results/additional/bias_analysis/derived_targets.csv')
    parser.add_argument('--output-dir', default='results/additional/missing_species_results')
    parser.add_argument('--chunk-size', type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_path = repo_path(args.benchmark)
    output_dir = repo_path(args.output_dir)
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if (output_dir/"reaction_categories.csv").exists():
        raise FileExistsError("Categories already exist; choose a new --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "Unique_identifier",
        "CanonicalReaction",
    }
    columns = pd.read_csv(benchmark_path, nrows=0).columns
    missing_columns = required.difference(columns)
    if missing_columns:
        raise KeyError(f"Missing benchmark columns: {sorted(missing_columns)}")

    with benchmark_path.open(encoding="utf-8") as handle:
        total = sum(1 for _ in handle) - 1

    categories_path = output_dir / "reaction_categories.csv"
    category_parts = []
    category_counts: Counter = Counter()
    classification_errors = 0
    target_path = repo_path(args.derived_targets)
    targets = pd.read_csv(
        target_path,
        usecols=["Unique_identifier", "flower_target_proton_reagents_identified"],
        dtype={"Unique_identifier": str},
    ).set_index("Unique_identifier")["flower_target_proton_reagents_identified"]
    chunks = pd.read_csv(
        benchmark_path,
        usecols=[
            "Unique_identifier",
            "CanonicalReaction",
        ],
        dtype={"Unique_identifier": str},
        chunksize=args.chunk_size,
        low_memory=False,
    )
    with tqdm(total=total, unit="reaction", desc="Classifying") as overall:
        for chunk_number, chunk in enumerate(chunks):
            chunk["flower_target_proton_reagents_identified"] = (
                chunk["Unique_identifier"].map(targets)
            )
            if chunk["flower_target_proton_reagents_identified"].isna().any():
                raise KeyError("Derived FlowER target missing for one or more UIDs")
            records = [
                classify(row)
                for _, row in tqdm(
                    chunk.iterrows(),
                    total=len(chunk),
                    desc=f"Chunk {chunk_number + 1}",
                    leave=False,
                )
            ]
            chunk_categories = pd.DataFrame(records)
            chunk_categories.to_csv(
                categories_path,
                mode="w" if chunk_number == 0 else "a",
                header=chunk_number == 0,
                index=False,
            )
            category_parts.append(
                chunk_categories[["Unique_identifier", "missing_category"]]
            )
            category_counts.update(chunk_categories["missing_category"])
            classification_errors += int(
                chunk_categories["classification_error"]
                .fillna("")
                .ne("")
                .sum()
            )
            overall.update(len(chunk))

            # Keep RDKit's parsed-molecule memory bounded to one chunk.
            contains_carbon.cache_clear()
            identity.cache_clear()
            molecule.cache_clear()

    categories = pd.concat(category_parts, ignore_index=True)
    distribution = pd.DataFrame(
        category_counts.items(), columns=["missing_category", "n"]
    )
    distribution["percent"] = 100.0 * distribution["n"] / total
    order = {category: i for i, category in enumerate(CATEGORY_ORDER)}
    distribution = distribution.assign(
        _order=distribution["missing_category"].map(order)
    ).sort_values("_order").drop(columns="_order")

    distribution.to_csv(output_dir / "category_distribution.csv", index=False)

    print(distribution.to_string(index=False))
    print(f"Classification errors: {classification_errors}")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()

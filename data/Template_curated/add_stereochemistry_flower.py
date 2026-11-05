from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.warning")


# ============================================================
# Basic helpers
# ============================================================

def split_reaction_smiles(rxn_smiles: str):
    """Split reaction SMILES into reactant and product component SMILES lists."""
    try:
        left, right = rxn_smiles.split(">>")
    except ValueError as exc:
        raise ValueError(f"Invalid reaction SMILES (missing '>>'): {rxn_smiles}") from exc

    reactants = [s for s in left.split(".") if s]
    products = [s for s in right.split(".") if s]
    return reactants, products


def mol_from_smiles(smi: str) -> Chem.Mol:
    """
    Parse one molecule SMILES and force stereochemistry perception.
    Important for E/Z input encoded via / and \\.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smi}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def canonicalize_mol(mol: Chem.Mol) -> Chem.Mol:
    """Convert a molecule to canonical isomeric SMILES and parse it again."""
    mol = Chem.Mol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    out = Chem.MolFromSmiles(smi)
    if out is None:
        raise ValueError(f"Could not reparse canonical SMILES: {smi}")
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    return out


def canonical_smiles_from_mol(mol: Chem.Mol, isomeric: bool = True) -> str:
    """Return canonical SMILES from a Mol."""
    mol = canonicalize_mol(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def canonicalize_molecule_smiles(smi: str, isomeric: bool = True) -> str:
    """Canonicalize one molecule SMILES."""
    mol = mol_from_smiles(smi)
    return canonical_smiles_from_mol(mol, isomeric=isomeric)


def canonicalize_reaction_smiles(rxn_smiles: str) -> str:
    """
    Canonicalize each component molecule and sort molecules on each side.
    This ensures same molecules => same reaction representation.
    """
    reactants, products = split_reaction_smiles(rxn_smiles)
    reactants_can = sorted(canonicalize_molecule_smiles(s, isomeric=True) for s in reactants)
    products_can = sorted(canonicalize_molecule_smiles(s, isomeric=True) for s in products)
    return ".".join(reactants_can) + ">>" + ".".join(products_can)


def join_reaction_smiles(reactants, products):
    """
    Join reactant/product molecule lists back into a canonicalized reaction SMILES.
    """
    left = ".".join(canonical_smiles_from_mol(m, isomeric=True) for m in reactants)
    right = ".".join(canonical_smiles_from_mol(m, isomeric=True) for m in products)
    rxn = f"{left}>>{right}"
    return canonicalize_reaction_smiles(rxn)


def remove_stereo(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of mol with all stereochemistry removed."""
    out = Chem.Mol(mol)
    Chem.RemoveStereochemistry(out)
    return out


def achiral_key(mol: Chem.Mol) -> str:
    """Canonical non-isomeric SMILES used for achiral matching."""
    return Chem.MolToSmiles(remove_stereo(mol), isomericSmiles=False, canonical=True)


# ============================================================
# Stereo detection
# ============================================================

def has_defined_atom_stereo(mol: Chem.Mol) -> bool:
    """True if molecule contains defined tetrahedral atom stereo."""
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            return True
    return False


def has_defined_bond_stereo(mol: Chem.Mol) -> bool:
    """
    True if molecule contains defined bond stereo.
    Includes:
    - explicit double-bond stereo assignments
    - directional single bonds (/ or \\ in SMILES)
    """
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            return True
        if bond.GetBondDir() != Chem.rdchem.BondDir.NONE:
            return True
    return False


def has_defined_stereo(mol: Chem.Mol) -> bool:
    """True if molecule already contains any defined atom or bond stereo."""
    return has_defined_atom_stereo(mol) or has_defined_bond_stereo(mol)


def has_unresolved_tetrahedral_stereo(mol: Chem.Mol) -> bool:
    """
    True if molecule contains at least one tetrahedral stereocenter
    that is possible but not assigned.
    """
    centers = Chem.FindMolChiralCenters(
        mol,
        includeUnassigned=True,
        includeCIP=False,
        useLegacyImplementation=False,
    )
    return any(label == "?" for _, label in centers)


def has_unresolved_bond_stereo(mol: Chem.Mol) -> bool:
    """True if RDKit sees an actually unresolved bond stereo element."""
    try:
        stereo_info = Chem.FindPotentialStereo(mol)
    except Exception:
        return False

    for si in stereo_info:
        if (
            si.type in (
                Chem.rdchem.StereoType.Bond_Double,
                Chem.rdchem.StereoType.Bond_Cumulene_Even,
            )
            and si.specified == Chem.rdchem.StereoSpecified.Unspecified
        ):
            return True

    return False


def has_unresolved_stereo(mol: Chem.Mol) -> bool:
    """
    Only flag molecules that genuinely have unresolved stereochemistry:
    - unassigned tetrahedral center(s)
    - unassigned E/Z-type bond stereo
    """
    return has_unresolved_tetrahedral_stereo(mol) or has_unresolved_bond_stereo(mol)


# ============================================================
# Safe stereo restoration
# ============================================================

def same_achiral_graph(template: Chem.Mol, target: Chem.Mol) -> bool:
    """Check that two molecules have the same graph after removing stereochemistry.

    Require equal canonical keys, atom and bond counts, and substructure matches
    in both directions. This is stricter than comparing keys alone.
    """
    if achiral_key(template) != achiral_key(target):
        return False

    if template.GetNumAtoms() != target.GetNumAtoms():
        return False

    if template.GetNumBonds() != target.GetNumBonds():
        return False

    template_ns = remove_stereo(template)
    target_ns = remove_stereo(target)

    match_fwd = target_ns.GetSubstructMatch(template_ns)
    match_rev = template_ns.GetSubstructMatch(target_ns)
    return bool(match_fwd) and bool(match_rev)


def transfer_stereo_from_template(template: Chem.Mol, target: Chem.Mol) -> Chem.Mol:
    """Restore stereochemistry from a template with the same non-stereo graph.

    Reuse that template molecule rather than copying individual stereo labels.
    Its final canonical SMILES then retains the template's stereochemistry.
    """
    if not same_achiral_graph(template, target):
        raise ValueError("Template and target do not have the same achiral graph.")

    return canonicalize_mol(Chem.Mol(template))


# ============================================================
# Template pool logic
# ============================================================

def build_stereo_template_pool(stereo_rxn_smiles: str):
    """
    Build a pool of stereo-defined template molecules from BOTH sides
    of the stereo-containing reaction, grouped by achiral graph.
    """
    reactant_smis, product_smis = split_reaction_smiles(stereo_rxn_smiles)
    all_mols = [mol_from_smiles(s) for s in (reactant_smis + product_smis)]

    pool = defaultdict(list)
    for mol in all_mols:
        if has_defined_stereo(mol):
            pool[achiral_key(mol)].append(canonicalize_mol(mol))

    return pool


def choose_template(templates):
    """
    Choose a unique stereo template for one achiral key.

    Returns:
        (template_mol_or_none, status)

    status:
        - 'unique'
        - 'no_template'
        - 'ambiguous'
    """
    if not templates:
        return None, "no_template"

    uniq = {}
    for t in templates:
        t_can = canonicalize_mol(t)
        smi = Chem.MolToSmiles(t_can, isomericSmiles=True, canonical=True)
        uniq[smi] = t_can

    uniq_templates = list(uniq.values())

    if len(uniq_templates) == 1:
        return uniq_templates[0], "unique"

    return None, "ambiguous"


# ============================================================
# Main API
# ============================================================

def restore_stereo_to_balanced_reaction(
    stereo_rxn_smiles: str,
    balanced_rxn_smiles: str,
):
    """
    Restore stereochemistry from stereo_rxn_smiles to balanced_rxn_smiles.

    Behavior:
    - stereo templates are collected from BOTH sides of the stereo reaction
    - every molecule on either side of the balanced reaction can inherit stereo
    - reagent copies on the product side can receive stereo
    - unmatched balanced-only molecules are left unchanged
    - only molecules with an ambiguous template or failed transfer are flagged
    - final reaction output is canonicalized molecule-by-molecule

    Important implementation detail:
    - when a balanced molecule matches a stereo template achirally, the output uses
      the source template molecule directly, rather than reconstructing stereo on the
      target graph atom-by-atom.
    """
    bal_reactant_smis, bal_product_smis = split_reaction_smiles(balanced_rxn_smiles)
    bal_reactants = [mol_from_smiles(s) for s in bal_reactant_smis]
    bal_products = [mol_from_smiles(s) for s in bal_product_smis]

    template_pool = build_stereo_template_pool(stereo_rxn_smiles)

    report = {
        "restored_reactants": 0,
        "restored_products": 0,
        "flagged_unresolved_stereo": [],
        "has_unresolved_stereo": False,
        "n_unresolved_stereo": 0,
    }

    def maybe_flag(side_name, idx, mol, reason):
        if has_unresolved_stereo(mol):
            report["flagged_unresolved_stereo"].append(
                {
                    "side": side_name,
                    "index": idx,
                    "smiles_achiral": Chem.MolToSmiles(
                        remove_stereo(mol), isomericSmiles=False, canonical=True
                    ),
                    "smiles_input": canonical_smiles_from_mol(mol, isomeric=True),
                    "reason": reason,
                }
            )

    def process_side(mols, side_name):
        out = []
        restored_count = 0

        for idx, mol in enumerate(mols):
            key = achiral_key(mol)
            template, status = choose_template(template_pool.get(key, []))

            if status == "unique":
                try:
                    new_mol = transfer_stereo_from_template(template, mol)
                    out.append(new_mol)
                    restored_count += 1
                except Exception:
                    out.append(canonicalize_mol(mol))
                    maybe_flag(side_name, idx, mol, "transfer_failed")
            elif status == "ambiguous":
                out.append(canonicalize_mol(mol))
                maybe_flag(side_name, idx, mol, "ambiguous_template")
            else:
                out.append(canonicalize_mol(mol))

        return out, restored_count

    new_reactants, n_r = process_side(bal_reactants, "reactant")
    new_products, n_p = process_side(bal_products, "product")

    report["restored_reactants"] = n_r
    report["restored_products"] = n_p
    report["n_unresolved_stereo"] = len(report["flagged_unresolved_stereo"])
    report["has_unresolved_stereo"] = report["n_unresolved_stereo"] > 0

    restored_rxn = join_reaction_smiles(new_reactants, new_products)
    return restored_rxn, report


# ============================================================
# Optional dataframe helpers
# ============================================================

def restore_stereo_row(
    row,
    stereo_col="rxn_with_stereo",
    balanced_col="rxn_balanced_no_stereo",
):
    restored_rxn, report = restore_stereo_to_balanced_reaction(
        row[stereo_col],
        row[balanced_col],
    )
    return restored_rxn, report["has_unresolved_stereo"], report["n_unresolved_stereo"]


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Test for a single reaction
    src = "C=CCBr.CC(C)(C)OC(=O)N1[C@H](C(=O)O)C[C@H]2C[C@H]21.CN(C)C=O.O=C([O-])[O-].[Cs+].[Cs+]>>C=CCOC(=O)[C@@H]1C[C@H]2C[C@H]2N1C(=O)OC(C)(C)C"
    tgt = "C=CCBr.CC(C)(C)OC(=O)N1C(C(=O)O)CC2CC21.CN(C)C=O.O=C([O-])[O-].[Cs+].[Cs+]>>C=CCOC(=O)C1CC2CC2N1C(=O)OC(C)(C)C.C[NH+](C)C=O.O=C([O-])[O-].[Br-].[Cs+].[Cs+]"

    src = canonicalize_reaction_smiles(src)
    tgt = canonicalize_reaction_smiles(tgt)

    print("Canonical source:", src)
    print("Canonical target:", tgt)

    restored, report = restore_stereo_to_balanced_reaction(src, tgt)
    print("Restored reaction:", restored)
    print("Report:", report)


    # ============================================================
    # Whole dataset
    # ============================================================

    df = pd.read_csv("data/Template_curated/benchmark/benchmark.csv")

    tqdm.pandas(desc="Restoring stereochemistry")

    # Canonicalize the columns to make sure that also the InputReaction column is in the same format as the TargetReaction column in terms of stereochemistry. 
    # With progress apply
    df["InputReaction_canonical"] = df["InputReaction"].progress_apply(canonicalize_reaction_smiles)
    df["TargetReaction_canonical"] = df["TargetReaction"].progress_apply(canonicalize_reaction_smiles)

    df[[
        "TargetReaction_with_stereo",
        "has_unresolved_stereo",
        "n_unresolved_stereo",
    ]] = df.progress_apply(
        lambda row: restore_stereo_row(
            row,
            stereo_col="InputReaction_canonical",
            balanced_col="TargetReaction_canonical",
        ),
        axis=1,
        result_type="expand",
    )

    n_total = len(df)
    n_unresolved = df["has_unresolved_stereo"].sum()
    print(f"Total rows: {n_total}")
    print(f"Rows with unresolved stereo after transfer: {n_unresolved} ({100*n_unresolved/n_total:.1f}%)")

    df_resolved = df[~df["has_unresolved_stereo"]].copy()
    df_unresolved = df[df["has_unresolved_stereo"]].copy()

    df_resolved.to_csv("data/Template_curated/benchmark/benchmark_stereo_resolved.csv", index=False)
    df_unresolved.to_csv("data/Template_curated/benchmark/benchmark_stereo_unresolved.csv", index=False)

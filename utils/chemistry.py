"""Match molecule identities for missing-species categories and similarity analysis.

These helpers keep the original category definitions. Accuracy scoring uses
the equivalence metric in completerxn.equivalence.
"""
from functools import lru_cache
from collections import Counter
from typing import List
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


def _sanitize_problematic_smiles(smiles: str) -> str:
    """Normalize a few known non-canonical notations before RDKit parsing."""
    s = smiles.strip()

    # Dative-bond AlCl3 forms frequently appear as [Cl][Al]<-[Cl]Cl.
    # Convert directional dative bonds to plain single bonds so RDKit can parse.
    s = s.replace("<-", "-").replace("->", "-")

    return s

def _mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(_sanitize_problematic_smiles(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol

def _remove_atom_map(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol.GetMol()

@lru_cache(maxsize=None)
def canonicalize_smiles(smiles: str) -> str:
    """Canonical SMILES with atom-map numbers stripped."""
    mol = _mol_from_smiles(smiles)
    mol = _remove_atom_map(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

@lru_cache(maxsize=None)
def _charge_parent_smiles(smiles: str) -> str:
    """ChargeParent only (no fragment removal). Cached per unique SMILES."""
    mol = _mol_from_smiles(smiles)
    mol = rdMolStandardize.ChargeParent(mol)
    return Chem.MolToSmiles(mol)

def _parse_formula(formula: str):
    """Return (heavy_atom_str, h_count, charge_sign) from a molecular formula string.

    Charge sign is +1 for cations (formula ends with '+'), -1 for anions ('-'), 0 otherwise.
    """
    charge_sign = 0
    if formula.endswith("+"):
        charge_sign = +1
    elif formula.endswith("-"):
        charge_sign = -1
    formula_no_charge = re.sub(r"[+-]+$", "", formula)
    # Extract H count explicitly (e.g. "H4B" -> 4, "CH4O" -> 4, "C2H3O2" -> 3)
    m = re.search(r"H(\d*)", formula_no_charge)
    if m:
        h_count = int(m.group(1)) if m.group(1) else 1
        heavy = formula_no_charge[:m.start()] + formula_no_charge[m.end():]
    else:
        h_count = 0
        heavy = formula_no_charge
    return heavy, h_count, charge_sign

@lru_cache(maxsize=None)
def proton_transfer_identity(smiles: str) -> str:
    """Apply ChargeParent only when the transform corresponds to a simple H+ transfer.

    Allowed: Δcharge and ΔH have the *same* sign (gain/lose a proton together),
             i.e. protonation (charge+1, H+1) or deprotonation (charge−1, H−1),
             and heavy-atom skeleton is unchanged.
    Blocked: Δcharge and ΔH have *opposite* signs — this indicates hydride (H−)
             transfer or other non-proton-transfer chemistry (e.g. [BH4−] → B).
    """
    parent = _charge_parent_smiles(smiles)
    if parent == smiles:
        return smiles
    try:
        src_formula = rdMolDescriptors.CalcMolFormula(_mol_from_smiles(smiles))
        dst_formula = rdMolDescriptors.CalcMolFormula(_mol_from_smiles(parent))
    except Exception:
        return smiles
    src_heavy, src_h, src_q = _parse_formula(src_formula)
    dst_heavy, dst_h, dst_q = _parse_formula(dst_formula)
    # Heavy-atom skeleton must be identical.
    if src_heavy != dst_heavy:
        return smiles
    delta_q = dst_q - src_q
    delta_h = dst_h - src_h
    # Allow only when ΔH and Δcharge agree in sign (pure proton transfer).
    if delta_q == 0 and delta_h == 0:
        return parent  # no change — always fine
    if delta_q * delta_h > 0 or (delta_h == 0 and delta_q != 0):
        # same sign or pure charge redistribution without H change — allow
        return parent
    # Opposite signs → hydride or other non-proton chemistry — block.
    return smiles

def _parse_sides(rxn_smiles: str):
    """Return (reactant_smiles_list, product_smiles_list)."""
    parts = rxn_smiles.strip().split(">")
    if len(parts) == 3:
        r_str, _, p_str = parts
    elif len(parts) == 2:
        r_str, p_str = parts
    else:
        raise ValueError(f"Cannot parse reaction SMILES: {rxn_smiles!r}")
    reactants = [s for s in r_str.split(".") if s.strip()]
    products  = [s for s in p_str.split(".") if s.strip()]
    return reactants, products

@lru_cache(maxsize=None)
def _smi_ecfp(smi: str, radius: int, n_bits: int) -> np.ndarray:
    """ECFP bit-vector for one SMILES. Cached per (smi, radius, n_bits)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def _side_fp(smiles_list: List[str], radius: int, n_bits: int) -> np.ndarray:
    """OR-combine per-molecule ECFP bit vectors for a reaction side."""
    combined = np.zeros(n_bits, dtype=np.uint8)
    for smi in smiles_list:
        combined |= _smi_ecfp(smi, radius, n_bits)
    return combined

def reaction_similarity(
    rxn1: str,
    rxn2: str,
    method: str = "ecfp",
    radius: int = 3,
    n_bits: int = 2048,
    ignore_reagents: bool = True,
) -> float:
    """
    Continuous similarity in [0, 1] between two reaction SMILES.

    method='ecfp'
        Differential Morgan fingerprint Tanimoto (DrFP-style): per-molecule
        ECFP bit vectors are OR-combined per side, XOR-ed to get the
        differential fingerprint, and Tanimoto is computed between the two.
    method='set_match'
        Binary exact match via :func:`reaction_set_match` (returns 0.0 or 1.0).
    """
    if method == "set_match":
        return 1.0 if reaction_set_match(rxn1, rxn2) else 0.0

    if method == "ecfp":
        try:
            r1, p1 = _parse_sides(rxn1)
            r2, p2 = _parse_sides(rxn2)
        except ValueError:
            return 0.0
        diff1 = _side_fp(r1, radius, n_bits) ^ _side_fp(p1, radius, n_bits)
        diff2 = _side_fp(r2, radius, n_bits) ^ _side_fp(p2, radius, n_bits)
        inter = int(np.sum(diff1 & diff2))
        union = int(np.sum(diff1 | diff2))
        return inter / union if union else 0.0

    raise ValueError(f"Unknown method: {method!r}. Choose 'ecfp' or 'set_match'.")

def reaction_set_match(prediction, target):
    """Canonical side match used only by the optional similarity indicator."""
    try:
        sides = [_parse_sides(r) for r in (prediction, target)]
        return all(Counter(canonicalize_smiles(s) for s in left) ==
                   Counter(canonicalize_smiles(s) for s in right)
                   for left, right in zip(*sides))
    except Exception:
        return False

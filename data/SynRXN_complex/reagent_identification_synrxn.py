"""
Identify reagents in a reaction SMILES: move unchanged species from reactants to the
reagent field.

Usage in a pandas DataFrame::

    from reagent_identification_synrxn import identify_reagents

    df[["rxn_reagents_identified", "moved_metadata"]] = df["rxn"].apply(
        lambda smi: pd.Series(identify_reagents(smi))
    )
"""

from functools import lru_cache
from collections import Counter
from rdkit import Chem


# ===============================================================
#  Cached RDKit helpers
# ===============================================================

@lru_cache(maxsize=200_000)
def mol_from_smiles_cached(smi: str):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


@lru_cache(maxsize=200_000)
def nomap_smiles_cached(smi: str):
    """Return SMILES with atom-map numbers removed (cached)."""
    m = mol_from_smiles_cached(smi)
    if not m:
        return None
    tmp = Chem.Mol(m)

    # Remove atom mapping completely
    for atom in tmp.GetAtoms():
        atom.SetAtomMapNum(0)

    # Invalidate cached properties that affect canonicalization
    Chem.rdmolops.AssignStereochemistry(tmp, cleanIt=True, force=True)
    Chem.rdmolops.FastFindRings(tmp)
    Chem.rdmolops.Cleanup(tmp)

    return Chem.MolToSmiles(tmp, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=200_000)
def atommap_set_cached(smi: str):
    """Return frozenset of atom-map numbers appearing in the molecule."""
    m = mol_from_smiles_cached(smi)
    if not m:
        return None
    out = []
    for a in m.GetAtoms():
        if a.HasProp("molAtomMapNumber"):
            out.append(a.GetProp("molAtomMapNumber"))
    return frozenset(out)


# ===============================================================
#  Reagent identification logic
# ===============================================================

def identify_reagents(rxn):
    """Reassign reagents. Remove unchanged species with correct multiplicity.
    Return:
      - rewritten_rxn (original order preserved)
      - moved_metadata (list of dicts with smiles and reason)
    """

    try:
        react_s, reag_s, prod_s = rxn.split(">")
    except ValueError:
        return rxn, []

    react_list = [x for x in react_s.split(".") if x]
    reag_list  = [x for x in reag_s.split(".") if x]
    prod_list  = [x for x in prod_s.split(".") if x]

    moved = []
    used_reactants = []

    # Cached nomap strings
    react_nomap = [nomap_smiles_cached(s) for s in react_list]
    prod_nomap  = [nomap_smiles_cached(s) for s in prod_list]

    # ---- 1) symmetric unchanged removal with multiplicity ----
    rcount = Counter([s for s in react_nomap if s])
    pcount = Counter([s for s in prod_nomap if s])
    needed = rcount & pcount

    # Remove from products (keep original order)
    need_left = needed.copy()
    prod_keep = []
    for smi, nm in zip(prod_list, prod_nomap):
        if nm and need_left.get(nm, 0) > 0:
            need_left[nm] -= 1
        else:
            prod_keep.append(smi)

    # ---- 2) Reactant handling (order preserved) ----
    need_left = needed.copy()
    for smi, nm in zip(react_list, react_nomap):
        # unchanged → moved to reagents
        if nm and need_left.get(nm, 0) > 0:
            moved.append({"smiles": nm, "reason": "mol_on_both_sides"})
            need_left[nm] -= 1
        elif nm:
            # Molecule only on reactant side → keep as reactant
            used_reactants.append(smi)
        else:
            # Failed to parse
            moved.append({"smiles": smi, "reason": "bad_parse"})

    # ---- 3) Build rewritten (order-preserving) reaction ----
    reag_final = reag_list + [m["smiles"] for m in moved]
    react_final = used_reactants
    prod_final  = prod_keep

    rewritten = (
        f"{'.'.join(react_final)}>"
        f"{'.'.join(reag_final)}>"
        f"{'.'.join(prod_final)}"
    )

    return rewritten, moved

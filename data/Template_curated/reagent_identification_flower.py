#!/usr/bin/env python3
"""
Identify reagents in FlowER reactions.

Process FlowER dataset to identify and reassign reagents, adding:
- moved_metadata: molecules moved to reagents with reasons
- rxn_reagents_identified: rewritten reaction with reagents identified
"""

import csv
import json
from tqdm import tqdm
from functools import lru_cache
from collections import Counter
from rdkit import Chem

LOG_FILE = "reagent_identification_failures.txt"


# ===============================================================
#  Cached RDKit Molecule operations
# ===============================================================

@lru_cache(maxsize=200_000)
def mol_from_smiles_cached(smi: str):
    """Shared Mol parser with logging."""
    try:
        m = Chem.MolFromSmiles(smi)
    except Exception as e:
        with open(LOG_FILE, "a") as f:
            f.write(f"MolFromSmiles crash: {smi}\n{str(e)}\n\n")
        return None
    if m is None:
        with open(LOG_FILE, "a") as f:
            f.write(f"MolFromSmiles failed: {smi}\n")
    return m


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


# ===============================================================
#  Main processing
# ===============================================================

def main(input_file, output_file):
    """Process FlowER dataset and identify reagents."""
    
    # Reset log
    open(LOG_FILE, "w").close()
    
    rows_out = []
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        
        for row in tqdm(reader, desc="Processing reactions"):
            reactants = row.get('reactants', '').strip()
            products = row.get('products', '').strip()
            
            if not reactants or not products:
                # Keep original row, add empty new columns
                row['rxn'] = ''
                row['moved_metadata'] = ''
                row['rxn_reagents_identified'] = ''
                rows_out.append(row)
                continue
            
            # Construct reaction SMILES: reactants>>products (no reagents initially)
            rxn = f"{reactants}>>{products}"
            
            # Identify reagents
            rewritten_rxn, moved_metadata = identify_reagents(rxn)
            
            # Add new columns
            row['rxn'] = rxn
            row['moved_metadata'] = json.dumps(moved_metadata)
            row['rxn_reagents_identified'] = rewritten_rxn
            
            rows_out.append(row)
    
    # Summary statistics
    total_reactions = len(rows_out)
    with_moved = sum(1 for r in rows_out if r['moved_metadata'] and r['moved_metadata'] != '[]')
    
    print(f"\nSummary:")
    print(f"  Total reactions: {total_reactions}")
    print(f"  Reactions with reassigned reagents: {with_moved} ({100*with_moved/total_reactions:.1f}%)")
    
    # Save to file
    output_fieldnames = list(fieldnames) + ['rxn', 'moved_metadata', 'rxn_reagents_identified']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows_out)
    
    print(f"\nResults saved to {output_file}")
    
    return rows_out


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Identify reagents in FlowER reactions')
    parser.add_argument('--input', required=True, 
                       help='Path to FlowER TSV file with reactants and products columns')
    parser.add_argument('--output', required=True,
                       help='Output TSV file path')
    
    args = parser.parse_args()
    
    main(args.input, args.output)

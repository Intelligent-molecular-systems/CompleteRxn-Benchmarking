"""Check atom and charge balance before scoring a completed reaction.

Keep the counting rules and balance reports from the April 17 CRB scorer.
RDKit adds explicit hydrogens, and repeated molecules keep their counts.
The middle reagent field is not included in the reaction equation.
"""
from rdkit import Chem


def mol_with_hs_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.AddHs(mol)


def reaction_balance(reaction):
    """Check whether a reaction is atom and charge balanced."""
    reactants, _, products = reaction.split('>')
    reactants = [s for s in reactants.split('.') if s]
    products = products.split('|')[0]
    products = [s for s in products.split('.') if s]

    reactant_mols = [mol_with_hs_from_smiles(smi) for smi in reactants]
    product_mols = [mol_with_hs_from_smiles(smi) for smi in products]

    if None in reactant_mols or None in product_mols:
        return False, "invalid_smiles"

    reactant_atoms = [mol.GetAtoms() for mol in reactant_mols]
    product_atoms = [mol.GetAtoms() for mol in product_mols]

    atom_types = []
    for molecule in reactant_atoms:
        for atom in molecule:
            atom_types.append(atom.GetSymbol())
    for molecule in product_atoms:
        for atom in molecule:
            atom_types.append(atom.GetSymbol())

    atom_types = set(atom_types)
    reactants_dict = dict([(atom, 0) for atom in atom_types])
    products_dict = dict([(atom, 0) for atom in atom_types])
    reactants_dict['charge'] = 0
    products_dict['charge'] = 0

    for mol in reactant_atoms:
        for atom in mol:
            atom_symbol = atom.GetSymbol()
            if atom_symbol in reactants_dict:
                reactants_dict[atom_symbol] += 1
            else:
                reactants_dict[atom_symbol] = 1
            reactants_dict['charge'] += atom.GetFormalCharge()

    for mol in product_atoms:
        for atom in mol:
            atom_symbol = atom.GetSymbol()
            if atom_symbol in products_dict:
                products_dict[atom_symbol] += 1
            else:
                products_dict[atom_symbol] = 1
            products_dict['charge'] += atom.GetFormalCharge()

    if reactants_dict != products_dict:
        diff = {k: v - reactants_dict[k] for k, v in products_dict.items()}
        for atom in list(diff):
            if diff[atom] == 0:
                del diff[atom]

        diff = {k: v for k, v in sorted(diff.items(), key=lambda x: x[0])}
        lhs, rhs = '$L:', '$R:'
        charge = '$Charge:'
        for atom in diff:
            if atom == "charge":
                charge += str(diff[atom])
            else:
                if diff[atom] < 0:
                    rhs += str(abs(diff[atom])) + atom
                else:
                    lhs += str(diff[atom]) + atom
        combination = f'{lhs} {rhs} {charge}'
        return False, combination

    return True, '$L $R $Charge'


def is_balanced_reaction(reaction):
    """Return false for empty, malformed, or unparseable reactions, then check balance."""
    if not isinstance(reaction, str) or reaction.count('>') != 2:
        return False
    left, _, right = reaction.split('>')
    if not left.strip() or not right.split('|')[0].strip():
        return False
    try:
        return bool(reaction_balance(reaction)[0])
    except Exception:
        return False

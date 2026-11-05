"""Compare completed reaction SMILES for exact or conservative equivalent matches.

Convert each reaction to a fixed normalized representation. Two reactions are
equivalent when their representations are equal. Predictions that agree then
have the same reference-match label when using the same stereochemistry setting.
Rules are in analysis/equivalence_metric_audit/equivalence_rules.tex.
See analysis/README.md for the paper analyses.

Normalize only restricted N/O/S proton placement and named ionic salts.
Keep hydrides, carbon-anion protonation, radical recombination, carbonate
decomposition, hydrolysis, and activator consumption distinct. The only
decomposition exception is HO-S(=O)-Cl -> SO2 + HCl and its conjugate base.
These are scoring conventions. Balance and equivalence do not establish
chemical plausibility. Use the same reference and stereo setting for agreement.
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import re

import rdkit
from rdkit import Chem


METRIC_VERSION = "2.3.0"


@dataclass(frozen=True)
class ReactionMatch:
    exact_match: bool
    equivalent_match: bool


def _read_molecule(smiles, stereo=False):
    if not smiles or any(c.isspace() for c in smiles) or "|" in smiles:
        raise ValueError("Expected a molecule SMILES without names or CXSMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError("Invalid or unspecified molecule: {!r}".format(smiles))
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if not stereo:
        Chem.RemoveStereochemistry(mol)
    return mol


def _smiles(mol):
    # Removing stereo above, rather than isomericSmiles=False, retains isotopes.
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _inventory(mol):
    atoms = Counter((a.GetAtomicNum(), a.GetIsotope())
                    for a in Chem.AddHs(mol).GetAtoms())
    return atoms, sum(a.GetFormalCharge() for a in mol.GetAtoms())


def _build_expansions():
    # Exact identities only: never infer a salt from its elemental formula.
    rules = [(acid, ("[H+]", "[{}-]".format(acid)))
             for acid in ("F", "Cl", "Br", "I")]
    # A named labile byproduct, not a general reagent-decomposition rule.
    rules.append(("O=S(O)Cl", ("O=S=O", "[H+]", "[Cl-]")))
    for metal in ("Li", "Na", "K", "Rb", "Cs"):
        cation = "[{}+]".format(metal)
        rules.extend(("[{}]{}".format(metal, halide),
                      (cation, "[{}-]".format(halide)))
                     for halide in ("F", "Cl", "Br", "I"))
        rules.extend([
            ("[OH][{}]".format(metal), (cation, "[OH-]")),
            ("[{}H]".format(metal), (cation, "[H-]")),
            ("C[O][{}]".format(metal), (cation, "C[O-]")),
            ("CC[O][{}]".format(metal), (cation, "CC[O-]")),
            ("CC(=O)O[{}]".format(metal), (cation, "CC(=O)[O-]")),
            ("O=S(=O)(O)O[{}]".format(metal),
             (cation, "O=S(=O)(O)[O-]")),
        ])
    expansions = {}
    for source, targets in rules:
        src = _read_molecule(source)
        dst = [_read_molecule(s) for s in targets]
        atoms, charge = Counter(), 0
        for mol in dst:
            inventory, q = _inventory(mol)
            atoms.update(inventory)
            charge += q
        if _inventory(src) != (atoms, charge):
            raise RuntimeError("Nonconserving equivalence rule: " + source)
        key = _smiles(src)
        value = tuple(_smiles(mol) for mol in dst)
        if key in expansions and expansions[key] != value:
            raise RuntimeError("Conflicting equivalence rule: " + source)
        expansions[key] = value
    return expansions


_EXPANSIONS = _build_expansions()


def _proton_form(mol):
    """Normalize restricted N/O/S proton states. Retain signed free H+.

    Adjacent opposite charges (e.g. nitro/azide), radicals, and sites bonded
    to explicit hydrogens are retained. Carbon/boron charges, hydrides,
    and permanent ammonium ions are never neutralized. Protonating carbon
    anions is excluded to preserve carbon reagent and intermediate identities.
    """
    parent = Chem.Mol(mol)
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return parent, 0
    protons = 0
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        permitted = atom.GetAtomicNum() in (7, 8, 16) and charge in (-1, 1)
        if not permitted:
            continue
        neighbors = list(atom.GetNeighbors())
        if (any(a.GetAtomicNum() == 1 for a in neighbors)
                or any(a.GetFormalCharge() * charge < 0 for a in neighbors)):
            continue
        hydrogens = atom.GetTotalNumHs()
        if charge == 1 and hydrogens == 0:
            continue
        edited = parent.GetAtomWithIdx(atom.GetIdx())
        edited.SetFormalCharge(0)
        edited.SetNumExplicitHs(hydrogens - charge)
        edited.SetNoImplicit(True)
        protons += charge
    Chem.SanitizeMol(parent)
    atoms, q = _inventory(parent)
    atoms[(1, 0)] += protons
    atoms = Counter({k: v for k, v in atoms.items() if v})
    if _inventory(mol) != (atoms, q + protons):
        raise RuntimeError("Proton normalization changed atoms or charge")
    return parent, protons


@lru_cache(maxsize=50000)
def _normal_form(smiles, stereo):
    """Apply proton and named-species rules until reaching a fixed representation.

    Normalize protonation before expanding species so chlorosulfite and its
    conjugate acid reach the same result. Apply the same rules to expansion
    products. The allowed expansions do not contain cycles.
    """
    if smiles == "[H+]":
        return (("[H+]", 1),)
    parent, protons = _proton_form(_read_molecule(smiles, stereo))
    canonical = _smiles(parent)
    normalized = Counter({"[H+]": protons})
    if canonical in _EXPANSIONS:
        for component in _EXPANSIONS[canonical]:
            normalized.update(dict(_normal_form(component, stereo)))
    else:
        normalized[canonical] += 1
    return tuple(sorted((s, n) for s, n in normalized.items() if n))


@lru_cache(maxsize=50000)
def _species(smiles, stereo):
    mol = _read_molecule(smiles, stereo)
    canonical = _smiles(mol)
    atoms, charge = _inventory(mol)
    return canonical, tuple(sorted(atoms.items())), charge, _normal_form(canonical, stereo)


def _side(text, stereo):
    exact, normalized, atoms = Counter(), Counter(), Counter()
    charge = 0
    for smiles in text.strip().split("."):
        canonical, inventory, q, form = _species(smiles, stereo)
        exact[canonical] += 1
        atoms.update(dict(inventory))
        charge += q
        # Counter += would discard negative proton counts; update retains them.
        normalized.update(dict(form))
    return exact, normalized, atoms, charge


def _reaction(reaction, stereo):
    if not isinstance(reaction, str):
        raise ValueError("Reaction must be a SMILES string")
    if not isinstance(stereo, bool):
        raise ValueError("stereo must be True or False")
    return _cached_reaction(reaction, stereo)


@lru_cache(maxsize=50000)
def _cached_reaction(reaction, stereo):
    # A dative bond '->' is a bond, not a reaction-field delimiter.
    sides = re.split(r"(?<!-)>", reaction.strip())
    if len(sides) != 3 or not sides[0].strip() or not sides[2].strip():
        raise ValueError("Expected a completed reactants>agents>products equation")
    left, right = _side(sides[0], stereo), _side(sides[2], stereo)
    exact = tuple(sorted(left[0].items())), tuple(sorted(right[0].items()))
    net = right[1].copy()
    net.subtract(left[1])
    equivalent = tuple(sorted((s, n) for s, n in net.items() if n))
    balanced = left[2:] == right[2:]
    return exact, equivalent, balanced


def _build_lenient_rules():
    """Check signed rewrite inventories, including consumed water."""
    rules = {
        "O=C(O)O": {"O=C=O": 2, "O": 2},
        "[H-]": {"[H][H]": 2, "[H+]": -2},
        "[H]": {"[H][H]": 1},
        "O=S(Cl)Cl": {"O=S=O": 2, "[H+]": 4, "[Cl-]": 4, "O": -2},
    }
    for source, targets in rules.items():
        source_atoms, source_charge = _inventory(_read_molecule(source))
        atoms, charge = Counter(), 0
        for target, coefficient in targets.items():
            inventory, q = _inventory(_read_molecule(target))
            for atom, count in inventory.items():
                atoms[atom] += coefficient * count
            charge += coefficient * q
        atoms = {a: n for a, n in atoms.items() if n}
        if atoms != {a: 2 * n for a, n in source_atoms.items()} or charge != 2 * source_charge:
            raise RuntimeError("Nonconserving lenient rule: " + source)
    return rules


_LENIENT_RULES = _build_lenient_rules()


@lru_cache(maxsize=50000)
def _lenient_signature(signature):
    """Apply the optional, atom-conserving versions of the older byproduct rules.

    Double the counts to represent half an H2 without rounding or losing atoms.
    SOCl2 hydrolysis consumes water. These rules can merge different reagent fates.
    They do not reproduce older shortcuts that changed the atom inventory.
    """
    result = Counter()
    for species, count in signature:
        for terminal, coefficient in _LENIENT_RULES.get(species, {species: 2}).items():
            result[terminal] += count * coefficient
    return tuple(sorted((s, n) for s, n in result.items() if n))


def _comparison_form(reaction, stereo, equivalence_lenient):
    if not isinstance(equivalence_lenient, bool):
        raise ValueError("equivalence_lenient must be True or False")
    exact, signature, balanced = _reaction(reaction, stereo)
    if equivalence_lenient:
        signature = _lenient_signature(signature)
    return exact, signature, balanced


def reaction_signature(reaction: str, *, stereo: bool = False,
                       equivalence_lenient: bool = False) -> tuple:
    """Return the normalized representation and metric version for a balanced reaction.

    Ignore stereochemistry by default, but keep isotopes and molecule counts.
    Invalid or unbalanced reactions raise ValueError. The reagent field is excluded.
    Lenient mode uses a separate version suffix and doubled counts.
    """
    _, signature, balanced = _comparison_form(reaction, stereo, equivalence_lenient)
    if not balanced:
        raise ValueError("Reaction is not atom/isotope/charge balanced")
    version = METRIC_VERSION + (":equivalence_lenient" if equivalence_lenient else "")
    return version, rdkit.__version__, stereo, signature


def compare_reactions(prediction: str, reference: str, *, stereo: bool = False,
                      equivalence_lenient: bool = False) -> ReactionMatch:
    """Return exact and equivalent matches using the same stereochemistry setting.

    Invalid SMILES raise ValueError. Unbalanced reactions receive two false labels.
    The caller separately checks whether SynRBL reported the prediction as solved.
    The default uses strict rules. equivalence_lenient=True adds atom-conserving
    versions of older byproduct rules without changing exact matching or balance.
    """
    pred = _comparison_form(prediction, stereo, equivalence_lenient)
    target = _comparison_form(reference, stereo, equivalence_lenient)
    balanced = pred[2] and target[2]
    return ReactionMatch(balanced and pred[0] == target[0],
                         balanced and pred[1] == target[1])

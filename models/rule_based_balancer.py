"""A small rule-based model that completes reactions by balancing atoms and charge.

The model balances each input reaction in three steps:

1. Count the atoms (by element, including hydrogens) and the charge on both sides.
   Everything on the reactant side that is not in the products must be added
   to the product side.
2. Copy unchanged reactant-side molecules (solvents, bases, unused reagents) to
   the product side, keeping as many as the balance allows.
3. Cover the remaining atoms with as few small byproducts ("help compounds")
   as possible, such as water or HCl. If the products contain atoms missing
   from the reactant side, first try adding a small reactant such as water or
   hydrogen.

Molecules are dropped from step 2 only when the balance requires it, as few as
possible. All rules and compound lists are fixed in advance and were not tuned
on benchmark reactions. The model needs no training and no atom mapping.
Reactions it cannot balance are returned unchanged and marked as unsolved.

This is a lightweight reimplementation of the help-compound idea in
chem_balancer (Zhang et al., ACS Omega 2024,
https://github.com/chonghuanzhang/balancing_rxn). The help-compound list below
is taken from that project, which is released under the MIT License,
Copyright (c) 2022 chonghuanzhang.

Usage, with RDKit and the completerxn package installed:

    from rule_based_balancer import RuleBasedBalancer
    predictions = test.predict(RuleBasedBalancer(workers=8))

See examples/run_ruleBased_ood.py for a complete benchmark run.
"""
from collections import Counter
from functools import lru_cache
from itertools import combinations, combinations_with_replacement
from multiprocessing import Pool

from rdkit import Chem, RDLogger

from completerxn.predictors import Completion

RDLogger.DisableLog('rdApp.*')

# Help compounds used to cover the remaining atoms, in order of preference.
# This is chem_balancer's list (helpCompound.py), unchanged.
HELP_PRODUCTS = [
    'CC(=O)O', 'O', '[H][H]', 'O=O', 'N', 'N#N',
    '[Na+]', '[K+]', 'O=[N+]=O', 'O=S(O)[O-]',
    'C', 'Cl', 'Br', 'I', 'F', 'P', 'S',
    'O=P(O)O', 'O=S(=O)(O)O', 'O=N(=O)O', 'O=Cl(=O)O',
    'CO', 'CCO', 'CCCO', 'CCCCO', 'CCC(=O)O', 'CCCC(=O)O', 'COCC(=O)O',
    'c1ccccc1', 'Cc1ccccc1', 'Oc1ccccc1', 'Clc1ccccc1',
]

# Small reactants tried when the products contain atoms the inputs lack: common
# missing reactants (hydrolysis, hydrogenation, oxidation, ...). Not part of
# chem_balancer's default settings; chosen in advance, not tuned.
HELP_REACTANTS = ['O', '[H][H]', 'O=O', 'Cl', 'N']


@lru_cache(maxsize=None)
def composition(smiles):
    """Return element counts (including H) and the net charge of a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    counts = Counter()
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
        counts['H'] += atom.GetTotalNumHs()
        counts['charge'] += atom.GetFormalCharge()
    return counts


def total(molecules):
    counts = Counter()
    for smiles in molecules:
        counts.update(composition(smiles))
    return counts


def difference(a, b):
    """a - b, keeping zero and negative entries (unlike Counter subtraction)."""
    return {key: a.get(key, 0) - b.get(key, 0) for key in set(a) | set(b)}


def feasible(residual):
    return all(value >= 0 for key, value in residual.items() if key != 'charge')


def decompose(residual, max_molecules=6):
    """Return the fewest help compounds whose atoms and charge add up exactly to the residual.

    Ties are broken by the order of HELP_PRODUCTS. Returns (count, compounds),
    or None if no combination of at most max_molecules compounds fits.
    """
    # Work with fixed-order tuples: one entry per element present, charge last.
    elements = sorted(key for key, value in residual.items() if key != 'charge' and value > 0)
    keys = elements + ['charge']
    candidates = []
    for smiles in HELP_PRODUCTS:
        counts = composition(smiles)
        if {k for k in counts if k != 'charge'} <= set(elements):
            candidates.append((smiles, tuple(counts.get(k, 0) for k in keys)))
    if not candidates:
        return None if any(residual.values()) else (0, [])
    # A residual larger than max_molecules of the largest byproduct cannot fit.
    largest = [max(vector[i] for _, vector in candidates) for i in range(len(elements))]
    memo = {}

    def search(rest, limit, start):
        if not any(rest):
            return 0, []
        if any(rest[i] > limit * largest[i] for i in range(len(elements))):
            return None
        key = (rest, limit, start)
        if key not in memo:
            best = None
            for i in range(start, len(candidates)):
                smiles, vector = candidates[i]
                remaining = tuple(a - b for a, b in zip(rest, vector))
                if min(remaining[:-1], default=0) >= 0:
                    found = search(remaining, limit - 1, i)
                    if found is not None:
                        if best is None or found[0] + 1 < best[0]:
                            best = (found[0] + 1, [smiles] + found[1])
            memo[key] = best
        return memo[key]

    return search(tuple(residual.get(k, 0) for k in keys), max_molecules, 0)


class RuleBasedBalancer:
    """Complete reactions by copying spectators and adding small byproducts."""

    def __init__(self, max_dropped=4, max_added_reactants=2, workers=1):
        self.max_dropped = max_dropped
        self.max_added_reactants = max_added_reactants
        self.workers = workers

    def predict(self, inputs):
        if self.workers == 1:
            return [self.complete(reaction) for reaction in inputs]
        with Pool(self.workers) as pool:
            return pool.map(self.complete, inputs, chunksize=64)

    def complete(self, reaction):
        try:
            reactants, agents, products = reaction.split('>')
            left = [m for m in (reactants.split('.') + agents.split('.')) if m]
            right = [m for m in products.split('.') if m]
            solution = self.balance(left, right)
        except ValueError:
            solution = None
        if solution is None:
            return Completion(reaction, solved=False)
        added_reactants, added_products = solution
        return Completion('.'.join(left + added_reactants) + '>>' + '.'.join(right + added_products))

    def balance(self, left, right):
        for size in range(self.max_added_reactants + 1):
            for added in combinations_with_replacement(HELP_REACTANTS, size):
                deficit = difference(total(left + list(added)), total(right))
                products = self.fill_products(left, deficit)
                if products is not None:
                    return list(added), products
        return None

    def fill_products(self, left, deficit):
        """Copy reactant-side molecules as spectators and cover the rest with help compounds.

        Tries dropping 0, 1, 2, ... molecules (the ones consumed by the reaction)
        and stops at the first level with a solution. Within that level, the
        solution with the fewest help compounds wins.
        """
        for dropped_count in range(min(self.max_dropped, len(left)) + 1):
            best, seen = None, set()
            for dropped in combinations(range(len(left)), dropped_count):
                spectators = [m for i, m in enumerate(left) if i not in dropped]
                signature = tuple(sorted(spectators))
                if signature in seen:
                    continue
                seen.add(signature)
                residual = difference(deficit, total(spectators))
                if not feasible(residual):
                    continue
                found = decompose(residual)
                if found is not None and (best is None or found[0] < best[0]):
                    best = (found[0], spectators + found[1])
            if best is not None:
                return best[1]
        return None

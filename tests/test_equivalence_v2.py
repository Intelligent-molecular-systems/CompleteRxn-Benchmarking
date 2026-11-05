"""Chemical examples and consistency checks for the benchmark equivalence metric."""

import itertools
import os
from pathlib import Path
import subprocess
import sys
import unittest

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import equivalence_v2 as metric


SUBSTITUTIONS = (
    "CNCCCO.OCCCCl.[OH][Na]>>CN(CCCO)CCCO.O.[Na][Cl]",
    "CNCCCO.OCCCCl.[Na+].[OH-]>>CN(CCCO)CCCO.O.[Na+].[Cl-]",
    "CNCCCO.OCCCCl>>CN(CCCO)CCCO.Cl",
)
HYDROLYSIS = "COC(C)=O.O>>CC(=O)O.CO"
MIXED_HALIDES = (
    "CC#N.ClCc1ccc(Cl)cc1.O=Cc1ccc(O)cc1O.[F][K]>>"
    "O=Cc1ccc(OCc2ccc(Cl)cc2)cc1O.CC#N.Cl.[F][K]",
    "CC#N.ClCc1ccc(Cl)cc1.O=Cc1ccc(O)cc1O.[F-].[K+]>>"
    "CC#N.F.O=Cc1ccc(OCc2ccc(Cl)cc2)cc1O.[Cl-].[K+]",
)


REFERENCE_CONFLICTS = (
    (
        "USPTO_US05066810_133249",
        "COc1c(C)cnc(CO)c1C.O=S(Cl)Cl>>COc1c(C)cnc(CCl)c1C.O=S(O)Cl",
        "COc1c(C)cnc(CO)c1C.O=S(Cl)Cl>>COc1c(C)cnc(CCl)c1C.Cl.O=S=O",
        "CCOCC.COc1c(C)cnc(CO)c1C.ClCCl.O=S(Cl)Cl>>CC[OH+]CC.COc1c(C)cnc(CCl)c1C.ClCCl.O=S=O.[Cl-]",
        (True, True, True),
    ),
    (
        "USPTO_US05905149_232676",
        "C1CCOC1.CCOC(=O)C(C)Br.ClCCl.O=C1Nc2ccccc2C1=Cc1ccc(O)c2ncccc12.[F-].CCCC[N+](CCCC)(CCCC)CCCC>>CCOC(=O)C(C)Oc1ccc(C=C2C(=O)Nc3ccccc32)c2cccnc12.Br.C1CCOC1.ClCCl.CCCC[N+](CCCC)(CCCC)CCCC.[F-]",
        "C1CCOC1.CCCC[N+](CCCC)(CCCC)CCCC.CCOC(=O)C(C)Br.ClCCl.O=C1Nc2ccccc2C1=Cc1ccc(O)c2ncccc12.[F-]>>CCOC(=O)C(C)Oc1ccc(C=C2C(=O)Nc3ccccc32)c2cccnc12.C1CCOC1.CCCC[N+](CCCC)(CCCC)CCCC.ClCCl.F.[Br-]",
        "C1CCOC1.CCCC[N+](CCCC)(CCCC)CCCC.CCOC(=O)C(C)Br.ClCCl.O=C1Nc2ccccc2C1=Cc1ccc(O)c2ncccc12.[F-]>>C1CCOC1.CCCC[N+](CCCC)(CCCC)CCCC.CCOC(=O)C(C)Oc1ccc(C=C2C(=O)Nc3ccccc32)c2cccnc12.ClCCl.F.[Br-]",
        (True, True, True),
    ),
)


class EquivalenceV2Tests(unittest.TestCase):
    def test_exact_canonicalization_and_agents(self):
        result = metric.compare_reactions(
            "[CH3:1][OH:2]>O>CO", "OC>>[H]OC([H])([H])[H]")
        self.assertTrue(result.exact_match)
        self.assertTrue(result.equivalent_match)
        self.assertTrue(metric.compare_reactions(
            "N->[Cu]>>N->[Cu]", "N->[Cu]>>N->[Cu]").exact_match)
        reaction = SUBSTITUTIONS[1]
        reordered = []
        for side in reaction.split(">>"):
            components = []
            for smiles in reversed(side.split(".")):
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
                components.append(Chem.MolToSmiles(mol, canonical=False))
            reordered.append(".".join(components))
        self.assertEqual(metric.reaction_signature(">>".join(reordered)),
                         metric.reaction_signature(reaction))

    def test_equivalence_relation_and_common_reference_labels(self):
        # All permutations exercise reflexivity, symmetry, and transitivity.
        for a, b, c in itertools.product(SUBSTITUTIONS, repeat=3):
            self.assertTrue(metric.compare_reactions(a, b).equivalent_match)
            self.assertTrue(metric.compare_reactions(b, c).equivalent_match)
            self.assertTrue(metric.compare_reactions(a, c).equivalent_match)
            for reference, expected in ((c, True), (HYDROLYSIS, False)):
                self.assertEqual(metric.compare_reactions(a, reference).equivalent_match,
                                 expected)
                self.assertEqual(metric.compare_reactions(b, reference).equivalent_match,
                                 expected)
        self.assertFalse(metric.compare_reactions(
            SUBSTITUTIONS[0], SUBSTITUTIONS[1]).exact_match)

    def test_saved_reference_conflicts(self):
        # Saved audit examples previously agreed while only CRB matched.
        for uid, syn, crb, reference, expected in REFERENCE_CONFLICTS:
            with self.subTest(uid=uid):
                labels = tuple(metric.compare_reactions(a, b).equivalent_match
                               for a, b in ((syn, crb), (syn, reference),
                                            (crb, reference)))
                self.assertEqual(labels, expected)
                if labels[0]:
                    self.assertEqual(labels[1], labels[2])

    def test_mixed_halide_regression_and_hash_independence(self):
        self.assertTrue(metric.compare_reactions(*MIXED_HALIDES).equivalent_match)
        code = (
            "from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*'); "
            "from utils.equivalence_v2 import compare_reactions, reaction_signature; "
            "print(compare_reactions({0!r}, {1!r})); "
            "print(reaction_signature({0!r})); print(reaction_signature({0!r}, equivalence_lenient=True))"
        ).format(*MIXED_HALIDES)
        outputs = []
        for seed in (0, 7, 42):
            env = dict(os.environ, PYTHONHASHSEED=str(seed))
            run = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True, check=True)
            outputs.append(run.stdout)
        self.assertEqual(len(set(outputs)), 1)

    def test_proton_transfers_and_charge_separation(self):
        for reaction in (
            "C[NH3+].CC(=O)[O-]>>CN.CC(=O)O",
            "[OH3+].[OH-]>>O.O",
            "c1cc[nH+]cc1.C[O-]>>c1ccncc1.CO",
            "C[N+](C)(C)CC(=O)[O-].Cl>>C[N+](C)(C)CC(=O)O.[Cl-]",
            "[O-]S(=O)(=O)[O-].Cl.Cl>>OS(=O)(=O)O.[Cl-].[Cl-]",
        ):
            with self.subTest(reaction=reaction):
                self.assertTrue(metric.compare_reactions(
                    reaction, "CCO>>CCO").equivalent_match)
        for smiles in ("C[N+](=O)[O-]", "[N-]=[N+]=N", "[BH4-]", "[CH3+]",
                       "[H-]", "[H]", "[O]", "C[N+](C)(C)C",
                       "[cH-]1cccc1", "C[N+]#[C-]", "[CH2-]CCC"):
            mol = metric._read_molecule(smiles)
            parent, protons = metric._proton_form(mol)
            self.assertEqual(metric._smiles(parent), metric._smiles(mol))
            self.assertEqual(protons, 0)

    def test_normalizations_conserve_inventory_and_are_idempotent(self):
        # Includes a +2 molecular charge, zwitterion, and isotope-bound proton.
        for smiles in ("[NH3+]CC[NH3+]", "[NH3+]CC(=O)[O-]", "C[O-]",
                       "[OH3+]", "[2H][O-]", "[n-]1cccc1"):
            mol = metric._read_molecule(smiles)
            parent, protons = metric._proton_form(mol)
            atoms, charge = metric._inventory(parent)
            atoms[(1, 0)] += protons
            atoms = {k: v for k, v in atoms.items() if v}
            self.assertEqual(metric._inventory(mol), (atoms, charge + protons))
            again, extra = metric._proton_form(parent)
            self.assertEqual(metric._smiles(again), metric._smiles(parent))
            self.assertEqual(extra, 0)
        for source, products in metric._EXPANSIONS.items():
            self.assertTrue(metric.compare_reactions(
                source + ">>" + ".".join(products), "O>>O").equivalent_match)

    def test_stereo_policy_and_isotopes(self):
        a = "N[C@@H](C)C(=O)OC.O>>N[C@@H](C)C(=O)O.CO"
        b = "N[C@H](C)C(=O)OC.O>>N[C@H](C)C(=O)O.CO"
        reference = "NC(C)C(=O)OC.O>>NC(C)C(=O)O.CO"
        self.assertTrue(metric.compare_reactions(a, b).equivalent_match)
        self.assertTrue(metric.compare_reactions(a, reference).equivalent_match)
        self.assertTrue(metric.compare_reactions(b, reference).equivalent_match)
        self.assertFalse(metric.compare_reactions(a, b, stereo=True).exact_match)
        self.assertFalse(metric.compare_reactions(a, b, stereo=True).equivalent_match)
        labelled = "[13CH3]OC(C)=O.O>>CC(=O)O.[13CH3]O"
        self.assertFalse(metric.compare_reactions(labelled, HYDROLYSIS).exact_match)
        self.assertFalse(metric.compare_reactions(labelled, HYDROLYSIS).equivalent_match)
        self.assertEqual(metric.reaction_signature(a)[:3],
                         (metric.METRIC_VERSION, metric.rdkit.__version__, False))

    def test_spectators_stoichiometry_and_reaction_direction(self):
        left, right = HYDROLYSIS.split(">>")
        spectator = left + ".[Na+].[Cl-]>>" + right + ".[Na+].[Cl-]"
        result = metric.compare_reactions(HYDROLYSIS, spectator)
        self.assertFalse(result.exact_match)
        self.assertTrue(result.equivalent_match)
        doubled = left + "." + left + ">>" + right + "." + right
        for other in (doubled, right + ">>" + left, "CCO>>COC"):
            self.assertFalse(metric.compare_reactions(HYDROLYSIS, other).equivalent_match)
        # The legacy asymmetric guardrail is unnecessary with a fixed net form.
        self.assertEqual(metric.compare_reactions("[O].C>>[O].C", "C>>C"),
                         metric.compare_reactions("C>>C", "[O].C>>[O].C"))

    def test_redox_and_hydrolysis_are_not_representation_rewrites(self):
        for reaction in ("[H].[H]>>[H][H]",
                         "O=S(Cl)Cl.O>>O=S=O.Cl.Cl"):
            self.assertFalse(metric.compare_reactions(
                reaction, "O>>O").equivalent_match)

    def test_extended_acid_base_salt_and_byproduct_rules(self):
        for reaction in (
            "O=S(O)Cl>>O=S=O.Cl",
            "O=S([O-])Cl>>O=S=O.[Cl-]",
            "O=S(=O)(O)O[Na]>>[Na+].O=S(=O)(O)[O-]",
            "O=S(=O)(O)O[K]>>[K+].O=S(=O)(O)[O-]",
            "Cl[Cs]>>[Cs+].[Cl-]",
            "I[Cs]>>[Cs+].[I-]",
            "CC(=O)O[Na]>>CC(=O)[O-].[Na+]",
        ):
            with self.subTest(reaction=reaction):
                self.assertTrue(metric.compare_reactions(
                    reaction, "O>>O").equivalent_match)
        # Proton placement can vary while unused hydride remains unchanged.
        family = (
            "CO.CCl.[NaH]>>COC.Cl.[NaH]",
            "CO.CCl.[NaH].N>>COC.[NH4+].[Cl-].[NaH]",
        )
        for a, b, c in itertools.product(family, repeat=3):
            self.assertTrue(metric.compare_reactions(a, b).equivalent_match)
            self.assertEqual(metric.compare_reactions(a, c).equivalent_match,
                             metric.compare_reactions(b, c).equivalent_match)
        for reaction in (
            "[2H-].[H+]>>[2H][H]",
            "O=C(O)C[Na]>>CC(=O)[O-].[Na+]",
            "[BH4-].[H+]>>B.[H][H]",
        ):
            with self.subTest(reaction=reaction):
                self.assertFalse(metric.compare_reactions(
                    reaction, "O>>O").equivalent_match)
        # Every named expansion also composes with acid/base normalization.
        for source in metric._EXPANSIONS:
            form = dict(metric._normal_form(source, False))
            repeated = {}
            for species, count in form.items():
                for terminal, coefficient in metric._normal_form(species, False):
                    repeated[terminal] = repeated.get(terminal, 0) + count * coefficient
            self.assertEqual(form, {s: n for s, n in repeated.items() if n})

    def test_excluded_byproduct_conversions_are_balanced_but_distinct(self):
        # A negative must reflect the chemical boundary, not invalid input.
        reactions = (
            "O=C(O)O>>O=C=O.O",              # carbonic acid dehydration
            "O=S(Cl)Cl.O>>O=S=O.Cl.Cl",      # activator hydrolysis
            "[H+].[H-]>>[H][H]",            # hydride quenching
            "[NaH].Cl>>[Na+].[Cl-].[H][H]",
            "[CH2-]CCC.Cl>>CCCC.[Cl-]",     # carbon-anion quenching
            "[H].[H]>>[H][H]",             # radical recombination
            "C[Si](C)(C)Cl.O>>C[Si](C)(C)O.Cl",
            "O=P(Cl)(Cl)Cl.O>>O=P(O)(Cl)Cl.Cl",
            "[BH4-].[H+]>>B.[H][H]", # distinct reducing-agent fate
        )
        for reaction in reactions:
            with self.subTest(reaction=reaction):
                metric.reaction_signature(reaction)
                self.assertFalse(metric.compare_reactions(
                    reaction, "O>>O").equivalent_match)

    def test_hydride_and_hydrogen_reductions_remain_distinct(self):
        a = "CC=O.[H-].[Na+].O>>CCO.[Na+].[OH-]"
        b = "CC=O.[H][H]>>CCO"
        c = "CC=O.[NaH].O>>CCO.[OH][Na]"
        for reaction in (a, b, c):
            metric.reaction_signature(reaction)
        self.assertTrue(metric.compare_reactions(a, c).equivalent_match)
        for x in (a, c):
            self.assertFalse(metric.compare_reactions(x, b).equivalent_match)
            self.assertFalse(metric.compare_reactions(b, x).equivalent_match)
        for reference in (a, b, c, HYDROLYSIS):
            self.assertEqual(metric.compare_reactions(a, reference).equivalent_match,
                             metric.compare_reactions(c, reference).equivalent_match)

    def test_coupling_reagent_consumption_remains_distinct(self):
        for reagent, urea in (
            ("C(=NC1CCCCC1)=NC1CCCCC1", "O=C(NC1CCCCC1)NC1CCCCC1"),
            ("CCN=C=NCCCN(C)C", "CCNC(=O)NCCCN(C)C"),
        ):
            with self.subTest(reagent=reagent):
                left = "CC(=O)O.CN." + reagent
                retained = left + ">>CC(=O)NC.O." + reagent
                consumed = left + ">>CC(=O)NC." + urea
                for reaction in (retained, consumed):
                    metric.reaction_signature(reaction)
                self.assertFalse(metric.compare_reactions(
                    retained, consumed).equivalent_match)

    def test_saved_activator_and_workup_differences(self):
        # Actual SynRXN audit pairs; these plausible alternatives stay distinct.
        cases = (
            ('complex_582',
             'Cc1nnc(O)c(-c2cccc(F)c2F)c1-c1ccc(Cl)cc1.O=P(Cl)(Cl)Cl>>Cc1nnc(Cl)c(-c2cccc(F)c2F)c1-c1ccc(Cl)cc1.Cl.O=P(=O)Cl',
             'Cc1nnc(O)c(-c2cccc(F)c2F)c1-c1ccc(Cl)cc1.O=P(Cl)(Cl)Cl>>Cc1nnc(Cl)c(-c2cccc(F)c2F)c1-c1ccc(Cl)cc1.O=P(O)(Cl)Cl'),
            ('complex_814',
             'CN.O=C(O)Cc1ccc(Cl)cc1.O=S(Cl)Cl>>CNC(=O)Cc1ccc(Cl)cc1.Cl.Cl.O=S=O',
             'CN.O=C(O)Cc1ccc(Cl)cc1.O=S(Cl)Cl>>CNC(=O)Cc1ccc(Cl)cc1.O.O=S(Cl)Cl'),
            ('complex_414',
             'C1CCOC1.CCn1c(C(=O)N(C2CC2)C2CC2)cc2c3c(ncn3C)c(Nc3cc(C(=O)O)n(C)n3)nc21.CN(C)C(On1nnc2cccnc21)=[N+](C)C.CNC.F[P-](F)(F)(F)(F)F>>CCn1c(C(=O)N(C2CC2)C2CC2)cc2c3c(ncn3C)c(Nc3cc(C(=O)N(C)C)n(C)n3)nc21.C1CCOC1.CN(C)C(=O)N(C)C.F[P-](F)(F)(F)(F)F.On1nnc2ccc[nH+]c21',
             'C1CCOC1.CCn1c(C(=O)N(C2CC2)C2CC2)cc2c3c(ncn3C)c(Nc3cc(C(=O)O)n(C)n3)nc21.CN(C)C(On1nnc2cccnc21)=[N+](C)C.CNC.F[P-](F)(F)(F)(F)F>>C1CCOC1.CCn1c(C(=O)N(C2CC2)C2CC2)cc2c3c(ncn3C)c(Nc3cc(C(=O)N(C)C)n(C)n3)nc21.CN(C)C(On1nnc2cccnc21)=[N+](C)C.F[P-](F)(F)(F)(F)F.O'),
            ('complex_1',
             'C=CCC(CCC(COCc1ccccc1)O[Si](C)(C)C(C)(C)C)OS(C)(=O)=O>>C=CCC1CCC(COCc2ccccc2)O1.CC(C)(C)[Si](C)(C)OS(C)(=O)=O',
             'C=CCC(CCC(COCc1ccccc1)O[Si](C)(C)C(C)(C)C)OS(C)(=O)=O.O>>C=CCC1CCC(COCc2ccccc2)O1.CC(C)(C)[Si](C)(C)O.CS(=O)(=O)O'),
        )
        for uid, prediction, reference in cases:
            with self.subTest(uid=uid):
                metric.reaction_signature(prediction)
                metric.reaction_signature(reference)
                self.assertFalse(metric.compare_reactions(
                    prediction, reference).equivalent_match)
                self.assertFalse(metric.compare_reactions(
                    reference, prediction).equivalent_match)

    def test_imbalance_and_invalid_input(self):
        for reaction in ("CCO>>CC=O", "[Na+]>>[Na]", "O>>[2H]O",
                         "[BH2][Na]>>[BH4-].[Na+]", "O=S(Cl)Cl>>O=S=O.Cl.Cl"):
            self.assertEqual(metric.compare_reactions(reaction, reaction),
                             metric.ReactionMatch(False, False))
            with self.assertRaises(ValueError):
                metric.reaction_signature(reaction)
        for reaction in (None, [], "", ">>", "CCO", "CCO>CCO", "CCO>>invalid", "*>>*",
                         "CCO.>>CCO", "CCO>>CCO name", "O>>O |metadata|"):
            with self.assertRaises(ValueError):
                metric.compare_reactions(reaction, "O>>O")
        with self.assertRaises(ValueError):
            metric.compare_reactions("O>>O", "O>>O", stereo="false")

    def test_lenient_conventions_are_opt_in_and_conserving(self):
        for reaction in (
            "O=C(O)O>>O=C=O.O",
            "[H+].[H-]>>[H][H]",
            "[H].[H]>>[H][H]",
            "O=S(Cl)Cl.O>>O=S=O.Cl.Cl",
        ):
            with self.subTest(reaction=reaction):
                metric.reaction_signature(reaction)
                self.assertFalse(metric.compare_reactions(reaction, "O>>O").equivalent_match)
                match = metric.compare_reactions(reaction, "O>>O", equivalence_lenient=True)
                self.assertFalse(match.exact_match)
                self.assertTrue(match.equivalent_match)
                self.assertEqual(metric.reaction_signature(
                    reaction, equivalence_lenient=True)[3], ())
        for reaction in ("O=S(Cl)Cl>>O=S=O.Cl.Cl", "[BH2][Na]>>[BH4-].[Na+]"):
            self.assertEqual(metric.compare_reactions(
                reaction, reaction, equivalence_lenient=True), metric.ReactionMatch(False, False))
        for reaction in ("[2H].[2H]>>[2H][2H]", "[BH4-].[H+]>>B.[H][H]"):
            metric.reaction_signature(reaction)
            self.assertFalse(metric.compare_reactions(
                reaction, "O>>O", equivalence_lenient=True).equivalent_match)
        self.assertNotEqual(metric.reaction_signature(HYDROLYSIS),
                            metric.reaction_signature(HYDROLYSIS, equivalence_lenient=True))
        with self.assertRaises(ValueError):
            metric.compare_reactions("O>>O", "O>>O", equivalence_lenient="true")

    def test_lenient_transitivity_and_multiplicity(self):
        family = (
            "CC=O.[H-].[Na+].O>>CCO.[Na+].[OH-]",
            "CC=O.[H][H]>>CCO",
            "CC=O.[H].[H]>>CCO",
        )
        for a, b, reference in itertools.product(family, repeat=3):
            self.assertTrue(metric.compare_reactions(a, b, equivalence_lenient=True).equivalent_match)
            self.assertEqual(metric.compare_reactions(a, reference, equivalence_lenient=True).equivalent_match,
                             metric.compare_reactions(b, reference, equivalence_lenient=True).equivalent_match)
        left, right = family[1].split(">>")
        doubled = left + "." + left + ">>" + right + "." + right
        self.assertFalse(metric.compare_reactions(
            family[1], doubled, equivalence_lenient=True).equivalent_match)
        for a, b in itertools.product(SUBSTITUTIONS, repeat=2):
            self.assertTrue(metric.compare_reactions(a, b, equivalence_lenient=True).equivalent_match)


if __name__ == "__main__":
    print("Metric {} / RDKit {}".format(metric.METRIC_VERSION,
                                       metric.rdkit.__version__), flush=True)
    unittest.main()

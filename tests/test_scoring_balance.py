"""Regression tests: equivalence must never award accuracy to imbalance."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.reaction_balance import is_balanced_reaction
from utils.metrics import compare_reactions


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ml = load('ml_scorer', 'models/constrained_reaction_balancer/score_predictions.py')
syn = load('syn_scorer', 'models/SynRBL/evaluate.py')
benchmark = load('benchmark_scorer', 'analysis/score_benchmark.py')


class BalanceGateTests(unittest.TestCase):
    def test_chemistry_and_malformed_inputs(self):
        for rxn in ['CCO>>CCO', '[Na+].[Cl-]>>[Na+].[Cl-]', 'c1ccccc1>>c1ccccc1']:
            self.assertTrue(is_balanced_reaction(rxn), rxn)
        for rxn in ['', '>>', 'CCO', 'CCO>>invalid', 'CCO>>CC=O', '[Na+]>>[Na]', 'Cl.Cl>>Cl.Cl.Cl']:
            self.assertFalse(is_balanced_reaction(rxn), rxn)

    def test_permissive_match_is_not_accuracy(self):
        target, pred = 'Cl.Cl>>Cl.Cl', 'Cl.Cl>>Cl.Cl.Cl'
        self.assertTrue(compare_reactions(pred, target).deep_pragmatic_equivalence_match)
        self.assertFalse(any(syn._compute_all_metrics(pred, target).values()))

    def test_unbalanced_exact_match_rejected(self):
        rxn = 'CCO>>CC=O'
        self.assertTrue(compare_reactions(rxn, rxn).exact_match)
        self.assertFalse(any(syn._compute_all_metrics(rxn, rxn).values()))

    def test_beam_skips_unbalanced_candidate(self):
        row = dict(input='Cl.Cl>>', target='>>Cl.Cl', prediction_1='>>Cl.Cl.Cl', prediction_2='>>Cl.Cl')
        self.assertEqual(ml.get_ranks_rich(row, 'prediction_', 2, ['exact_match', 'deep_pragmatic_equivalence_match'], scoring_metric='v1'), {'exact_match': 2, 'deep_pragmatic_equivalence_match': 2})

    def test_balanced_positive_and_input_validation(self):
        self.assertTrue(syn._compute_all_metrics('CCO>>CCO', 'CCO>>CCO')['exact_match'])
        with self.assertRaises(ValueError):
            ml.get_ranks_rich(dict(input='bad', target='>>'), 'prediction_', 1, ['exact_match'])

    def test_v2_metric_selection_in_model_scorers(self):
        row = dict(input='CO.O=S(Cl)Cl>>CCl', target='>>Cl.O=S=O', prediction_1='>>O=S(O)Cl')
        ranks=ml.get_ranks_rich(row,'prediction_',1,['exact_match','equivalent_match'],scoring_metric='v2')
        self.assertEqual(ranks,{'exact_match':0,'equivalent_match':1})
        self.assertEqual(syn._compute_all_metrics('CCO>>CCO','CCO>>CCO',scoring_metric='v2'),
                         {'exact_match':True,'equivalent_match':True})
        self.assertFalse(any(syn._compute_all_metrics('CCO>>CC=O','CCO>>CC=O',scoring_metric='v2').values()))
        from utils.scoring import labels
        reaction='O=C(O)O>>O=C=O.O'
        self.assertFalse(labels(reaction,'O>>O',metric='v2')[1])
        self.assertTrue(labels(reaction,'O>>O',metric='equivalence_lenient')[1])

    def test_single_repetition_summary_has_no_invented_sd(self):
        row=dict(population='retained',split='random',model='CRB',rep=0,n=10,
                 exact_pct=90,equivalence_pct=100,balanced_pct=100)
        single=benchmark.aggregate_repetitions([row])[0]
        self.assertIsNone(single['exact_pct_sd'])
        second=dict(row,rep=1,exact_pct=80)
        full=benchmark.aggregate_repetitions([row,second])[0]
        self.assertEqual(full['n_observations'],20)
        self.assertEqual(full['exact_pct_mean'],85)
        self.assertAlmostEqual(full['exact_pct_sd'],50**.5)

    def test_syn_solved_flag_cannot_bypass_gate(self):
        import pandas as pd
        args = SimpleNamespace(reagents_identified=False, add_all_metrics=True, accuracy_metric='deep_pragmatic_equivalence', solved_col='solved', solved_by_col='solved_by', confidence_col='confidence', min_confidence=.5, scoring_metric='v1', target_col='target', col='Reaction', similarity_method='tanimoto')
        # Similarity is unrelated to this test; stub it to exercise both scoring branches.
        original = syn.reaction_similarity
        syn.reaction_similarity = lambda *a, **kw: 1.0
        try:
            for all_metrics in [True, False]:
                args.add_all_metrics = all_metrics
                df = pd.DataFrame([dict(Reaction='Cl.Cl>>Cl.Cl.Cl', target='Cl.Cl>>Cl.Cl', solved=True, solved_by='rule-based', confidence=1.)])
                result = syn.evaluate(df, args)
                self.assertEqual(result['total_correct'], 0)
                self.assertFalse(df.iloc[0]['is_balanced'])
        finally:
            syn.reaction_similarity = original


if __name__ == '__main__':
    unittest.main()

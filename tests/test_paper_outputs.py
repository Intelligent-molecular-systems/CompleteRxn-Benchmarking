"""Check compact diagnostics and the historical audit's current match labels."""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location('paper_output', ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PaperOutputTests(unittest.TestCase):
    def test_failure_csvs_keep_invalid_and_unbalanced_records(self):
        module = load('analysis/uncurated_failure_modes/analyze.py')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            crb, master, topn, consensus = (base/n for n in ('crb.csv', 'master.csv', 'topn.txt', 'agreement.csv'))
            pd.DataFrame([
                dict(Unique_identifier='balanced', crb_sequence_probability=.99,
                     is_valid_prediction=True, is_balanced=True, input_reaction='CCO>>',
                     prediction_reaction='CCO>>CCO', crb_top1_diff='>>CCO'),
                dict(Unique_identifier='hydrogen', crb_sequence_probability=.96,
                     is_valid_prediction=True, is_balanced=False, input_reaction='CCO>>',
                     prediction_reaction='CCO>>CC=O', crb_top1_diff='>>CC=O'),
                dict(Unique_identifier='invalid', crb_sequence_probability=.1,
                     is_valid_prediction=False, is_balanced=False, input_reaction='CCO>>',
                     prediction_reaction='CCO>>C1', crb_top1_diff='>>C1'),
            ]).to_csv(crb, index=False)
            pd.DataFrame([dict(Unique_identifier=uid, CanonicalReaction='CCO>>CCO',
                               MissingNrAtoms=1, MissingNrCarbons=0, MultiProduct=0)
                          for uid in ('balanced', 'hydrogen', 'invalid')]).to_csv(master, index=False)
            topn.write_text('fixture\n')
            pd.DataFrame([dict(overlap_both_balanced_rows=2, stereo_free_agreement_n=1)]).to_csv(consensus, index=False)
            args = SimpleNamespace(crb=crb, master=master, topn=topn, consensus=consensus,
                                   similarity=base/'absent.csv', metric='v2', output=None,
                                   output_dir=base/'out', chunksize=2, n_best=1,
                                   max_length=3, confidence=.95)
            with patch.object(module, 'parse_args', return_value=args), \
                 patch.object(module, 'top1_lengths', return_value=np.array([3, 3, 1])), \
                 contextlib.redirect_stdout(io.StringIO()):
                module.main()
            result = pd.read_csv(args.output_dir/'summary.csv').iloc[0]
            self.assertEqual((result.total, result.balanced_n, result.valid_unbalanced_n,
                              result.invalid_n, result.hydrogen_only_n), (3, 1, 1, 1, 1))
            self.assertEqual(result.mean_abs_hydrogen_residual, 2)
            self.assertEqual(result.agreement_pct, 50)
            self.assertEqual(pd.read_csv(args.output_dir/'outcomes.csv').n.sum(), 3)
            self.assertEqual(pd.read_csv(args.output_dir/'confidence.csv').n.sum(), 2)
            self.assertTrue((args.output_dir/'provenance.json').is_file())
            self.assertFalse((args.output_dir/'results.md').exists())

    def test_synrxn_plots_select_remaining_negatives_without_changing_judgments(self):
        module = load('analysis/synrxn_analysis/plot_ambiguous_GT.py')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            rows = [dict(r_id=uid, input_reaction='CCO>>', ground_truth='CCO>>CCO',
                         completed_prediction='CCO>>CCO', prediction_balanced=True,
                         target_balanced=True, equivalent_match=matched)
                    for uid, matched in [('accepted', True), ('remaining', False)]]
            pd.DataFrame(rows).to_csv(base/'current.csv', index=False)
            pd.DataFrame([dict(r_id=r['r_id'], audit_label='both plausible',
                               audit_note='Fixed historical judgment', target_status='right_only')
                          for r in rows]).to_csv(base/'audit.csv', index=False)
            with patch('sys.argv', ['plot', '--data', str(base/'current.csv'),
                                   '--audit', str(base/'audit.csv'), '--output-dir', str(base/'out')]), \
                 contextlib.redirect_stdout(io.StringIO()):
                module.main()
            self.assertEqual([p.name for p in (base/'out').glob('*.png')], ['remaining.png'])
            status = pd.read_csv(base/'out/audit_status.csv')
            self.assertEqual(len(status), 2)
            self.assertTrue(status.audit_label.eq('both plausible').all())


if __name__ == '__main__':
    unittest.main()

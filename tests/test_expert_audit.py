"""Retabulate released judgments without chemistry, model calls or private inputs."""
import copy
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'expert_audit', ROOT/'analysis/equivalence_metric_audit/analyze_assessments.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ExpertAuditTests(unittest.TestCase):
    def setUp(self):
        self.expert = audit.read_rows(audit.HERE/'reaction_completion_assessment_expert_first20.csv')
        self.blinded = audit.read_rows(audit.HERE/'audit_sample_blinded.csv')
        self.key = audit.read_rows(audit.HERE/'audit_key.csv')

    def test_released_judgments_and_source_counts(self):
        details = audit.assess(self.expert, self.blinded, self.key, 'expert')
        summary = audit.summarize(details)
        counts = {(row['measure'], row['category']): row['count'] for row in summary}
        self.assertEqual(len(details), 20)
        self.assertEqual({r['audit_id'] for r in details if r['equivalence']=='equivalent'},
                         {4, 10, 11, 19})
        self.assertEqual(counts[('equivalence', 'not_equivalent')], 14)
        self.assertEqual(counts[('outcome', 'FlowER')], 8)
        self.assertEqual(counts[('outcome', 'SynRBL')], 3)
        self.assertEqual(counts[('outcome', 'both')], 3)
        self.assertEqual(counts[('outcome', 'uncertain')], 2)
        for measure in ('equivalence', 'outcome'):
            self.assertEqual(sum(r['count'] for r in summary if r['measure']==measure),20)

    def test_changed_completion_or_mislabeled_key_is_rejected(self):
        expert = copy.deepcopy(self.expert)
        expert[1]['completion_a'] = 'O>>O'
        with self.assertRaisesRegex(ValueError, 'differs from blinded'):
            audit.assess(expert, self.blinded, self.key, 'expert')
        key = copy.deepcopy(self.key)
        key[1]['completion_a_source'], key[1]['completion_b_source'] = (
            key[1]['completion_b_source'], key[1]['completion_a_source'])
        with self.assertRaisesRegex(ValueError, 'does not identify completion'):
            audit.assess(self.expert, self.blinded, key, 'expert')

    def test_default_needs_only_released_inputs_and_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            audit.main(['--output-dir',directory])
            files = sorted(p.name for p in Path(directory).iterdir())
            self.assertEqual(files, ['expert_audit_details.csv','expert_audit_summary.csv'])
            with (Path(directory)/files[0]).open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),20)
            with self.assertRaises(FileExistsError):
                audit.main(['--output-dir',directory])

    def test_score_join_rejects_a_filtered_file_missing_an_audit_case(self):
        details = audit.assess(self.expert, self.blinded, self.key, 'expert')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scores.csv'
            path.write_text('Unique_identifier,equivalence,retained\n')
            with self.assertRaisesRegex(ValueError, 'use the full-test fold'):
                audit.add_scores(details, path)


if __name__ == '__main__':
    unittest.main()

"""Keep frozen v1 reproducible and prevent current analysis from using old flags."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from utils import chemistry
from utils.result_paths import (ROOT, metric_name, results_root, resolve_recorded_path,
                                verify_score_release, verify_cached_agreement)
from utils.scoring import labels, overlay_scores, provenance
from src.completerxn.legacy import equivalence_v1


class MetricReleaseTests(unittest.TestCase):
    def test_v1_is_frozen_and_v2_is_the_default(self):
        source=ROOT/'src/completerxn/legacy/equivalence_v1.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         '84e2219ecaf6c9208ff5a7fec561b96bfc4f5588d256b4f7eec2851224562640')
        prediction, reference='O=C(O)O>>O=C=O.O','O>>O'
        self.assertFalse(labels(prediction,reference)[1])
        self.assertTrue(labels(prediction,reference,metric='v1')[1])
        self.assertEqual(labels(prediction,reference,metric='legacy'),
                         labels(prediction,reference,metric='v1'))
        self.assertEqual(metric_name('legacy'),'v1')
        self.assertEqual(results_root(),ROOT/'results')
        self.assertEqual(results_root('v1'),ROOT/'results/archive/metric_v1')

    def test_classification_and_similarity_helpers_preserve_their_conventions(self):
        for smiles in ('CO','[CH3:1][OH:2]','[Na+]','Cl','C[NH3+]','[BH4-]'):
            self.assertEqual(chemistry.canonicalize_smiles(smiles),
                             equivalence_v1.canonicalize_smiles(smiles))
            canonical=chemistry.canonicalize_smiles(smiles)
            self.assertEqual(chemistry.proton_transfer_identity(canonical),
                             equivalence_v1._charge_parent_if_proton_transfer(canonical))
        for method in ('ecfp','set_match'):
            for left,right in [('CO>>C=O','OC>>C=O'),('CCO>>CC=O','CO>>C=O'),('bad','CO>>CO')]:
                self.assertEqual(chemistry.reaction_similarity(left,right,method=method),
                                 equivalence_v1.reaction_similarity(left,right,method=method))

    def test_moved_score_paths_and_cached_agreement_remain_verifiable(self):
        self.assertEqual(resolve_recorded_path('results/benchmark/summary.csv'),
                         results_root('v1')/'main/scores/summary.csv')
        self.assertEqual(resolve_recorded_path('results/v2_3_0/benchmark/summary.csv'),
                         results_root()/'main/scores/summary.csv')
        # This check uses small published score files, not raw model predictions.
        for metric in ('v1','v2'):
            base=results_root(metric)
            verify_score_release(base/'main/scores',metric)
        path=verify_cached_agreement(results_root()/'additional/consensus_audit',
                                     results_root()/'main/scores')
        self.assertEqual(path.name,'agreement_summary.csv')

    def test_plot_overlay_uses_selected_score_rows_even_without_environment_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            release=Path(directory)
            path=release/'main/scores/per_reaction/CRB_fold.csv'
            path.parent.mkdir(parents=True)
            path.write_text('original_index,exact,equivalence,balanced\n0,0,1,1\n1,0,0,1\n')
            frame=pd.DataFrame({'original_index':[1,0],'matches_selected_metric':[True,False]})
            with patch('utils.scoring.selected_root',return_value=release):
                output=overlay_scores(frame,'predictions/crb/fold/top1_predictions_with_stats.txt')
            self.assertEqual(output['matches_selected_metric'].tolist(),[False,True])
            self.assertEqual(frame['matches_selected_metric'].tolist(),[True,False])

    def test_provenance_cannot_cross_metric_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            provenance(output,'legacy',[],population='test')
            saved=json.loads((output/'provenance.json').read_text())
            self.assertEqual((saved['metric'],saved['version']),('v1','v1'))
            self.assertIn('src/completerxn/legacy/equivalence_v1.py',saved['sha256'])
            verify_score_release(output,'v1')
            with self.assertRaises(ValueError):verify_score_release(output,'v2')
            saved['metric']='legacy';saved['version']='legacy'
            (output/'provenance.json').write_text(json.dumps(saved))
            verify_score_release(output,'v1')

    def test_category_analysis_needs_scores_not_old_prediction_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scores=root/'scores';rows=scores/'per_reaction'
            rows.mkdir(parents=True)
            (scores/'provenance.json').write_text(json.dumps({'metric':'v2','version':'2.3.0'}))
            categories=root/'categories.csv'
            categories.write_text('Unique_identifier,missing_category,n_missing\na,Helper only,1\nb,Helper only,1\n')
            for model in ('RB','CRB','SynRBL'):
                (rows/f'{model}_random_seed42_rep0.csv').write_text(
                    'Unique_identifier,retained,exact,equivalence,balanced\na,1,1,1,1\nb,1,0,1,1\n')
            output=root/'out'
            command=[sys.executable,str(ROOT/'analysis/missing_species/build_category_accuracy_table.py'),
                     '--scores-dir',str(rows),'--categories',str(categories),'--output-dir',str(output),
                     '--split-types','random','--reps','0']
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            per_rep=pd.read_csv(output/'uid_accuracy_per_rep.csv')
            self.assertTrue(per_rep['equivalence_accuracy_pct'].eq(100).all())
            self.assertTrue(per_rep['exact_accuracy_pct'].eq(50).all())
            self.assertEqual(set(pd.read_csv(output/'uid_model_outcomes.csv')['scoring_metric']),{'v2'})
            run=subprocess.run(command+['--metric','v1','--output-dir',str(root/'wrong')],capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertFalse((root/'wrong').exists())

    def test_compressed_release_can_verify_cached_agreement_without_expanded_data(self):
        import io,tarfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scores=root/'main/scores';output=root/'additional/consensus_audit'
            scores.mkdir(parents=True);output.mkdir(parents=True)
            name='CRB_random_seed42_rep0.csv';content=b'fixture score rows\n'
            with tarfile.open(scores/'filtered_scores.tar.gz','w:gz') as bundle:
                info=tarfile.TarInfo(name);info.size=len(content);bundle.addfile(info,io.BytesIO(content))
            saved={'metric':'v2','version':'2.3.0','policies':{'stereo':False},
                   'sha256':{str(scores/'per_reaction'/name):hashlib.sha256(content).hexdigest()}}
            (output/'provenance.json').write_text(json.dumps(saved))
            (output/'agreement_summary.csv').write_text('summary\n')
            verify_cached_agreement(output,scores)
            (scores/'filtered_scores.tar.gz').unlink()
            with self.assertRaisesRegex(ValueError,'No agreement score inputs'):
                verify_cached_agreement(output,scores)


if __name__=='__main__':unittest.main()

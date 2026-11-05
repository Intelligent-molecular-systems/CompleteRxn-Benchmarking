"""Release paths, repeatable analysis commands and examples without large data."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from completerxn.benchmark import Example, Partition

ROOT = Path(__file__).resolve().parents[1]


def load_refresh():
    spec = importlib.util.spec_from_file_location('refresh', ROOT/'analysis/refresh_main_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseWorkflowTests(unittest.TestCase):
    def test_label_column_uses_provenance_and_resolves_legacy_path(self):
        refresh = load_refresh()
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            self.assertEqual(refresh.equivalence_column('results/archive/metric_v1/main/scores'),
                             'deep_pragmatic_equivalence_match')
        finally:
            os.chdir(previous)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for metric in ('v1', 'v2', 'equivalence_lenient'):
                (path/'provenance.json').write_text(json.dumps({'metric': metric}))
                expected = 'deep_pragmatic_equivalence_match' if metric == 'v1' else 'equivalent_match'
                self.assertEqual(refresh.equivalence_column(path), expected)
            (path/'provenance.json').unlink()
            with self.assertRaises(ValueError):
                refresh.equivalence_column(path)

    def test_outcome_figure_uses_selected_release(self):
        from analysis.figures.plot_synrbl_vs_crb_outcome_fractions import _abs_path
        with patch.dict(os.environ, {'COMPLETERXN_RESULTS_ROOT': '/tmp/selected-release'}):
            self.assertEqual(_abs_path('results/main/figures/example.png'),
                             '/tmp/selected-release/main/figures/example.png')
            self.assertEqual(_abs_path('results/main/predictions/example.csv'),
                             str(ROOT/'results/main/predictions/example.csv'))
            self.assertEqual(_abs_path('/tmp/explicit.png'), '/tmp/explicit.png')

    def test_synrbl_example_saves_failed_reaction(self):
        test = Partition([Example('case', 'CCO>>', 'CCO>>CCO', {})], {})
        class Loader:
            def __init__(self, **kwargs): pass
            def load(self, **kwargs): return test, test, test
        class Model:
            def __init__(self, **kwargs): pass
            def predict(self, inputs): return inputs
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'output'
            with patch('completerxn.DataLoader', Loader), \
                 patch('completerxn.predictors.SynRBLPredictor', Model), \
                 patch.object(sys, 'argv', ['run_synrbl.py', '--output-dir', str(output)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT/'examples/run_synrbl.py'), run_name='__main__')
            self.assertTrue((output/'failed_reaction.svg').is_file())
            self.assertTrue((output/'summary.csv').is_file())

    def test_reproduction_reuses_agreement_and_rejects_overwrite_or_changed_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            release = work/'release'
            scores = release/'main/scores'
            scores.mkdir(parents=True)
            (scores/'per_reaction').mkdir()
            (scores/'per_repetition.csv').write_text('fixture\n')
            (scores/'main_table_auxiliary_per_repetition.csv').write_text('fixture\n')
            metadata = {'metric': 'v2', 'version': '2.3.0'}
            (scores/'provenance.json').write_text(json.dumps(metadata))
            score = scores/'per_reaction/CRB_random_seed42_rep0.csv'
            score.write_text('fixture scores\n')
            agreement = release/'additional/consensus_audit'
            agreement.mkdir(parents=True)
            metadata.update(policies={'stereo': False}, sha256={
                str(score): hashlib.sha256(score.read_bytes()).hexdigest()})
            (agreement/'provenance.json').write_text(json.dumps(metadata))
            (agreement/'agreement_summary.csv').write_text('fixture agreement\n')
            # Run the real shell control flow while replacing expensive workers.
            binaries = work/'bin'
            binaries.mkdir()
            python = binaries/'python3'
            python.write_text('#!'+sys.executable+'\n'
                              'import os, sys\n'
                              'if len(sys.argv)>1 and sys.argv[1].startswith("analysis/"):\n'
                              '    raise SystemExit(0)\n'
                              'os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n')
            python.chmod(0o755)
            apptainer = binaries/'apptainer'
            apptainer.write_text('#!/bin/sh\nexit 0\n')
            apptainer.chmod(0o755)
            env = dict(os.environ, PATH=str(binaries)+os.pathsep+os.environ['PATH'])
            command = ['bash', str(ROOT/'analysis/reproduce_main_results.sh'),
                       '--output-root', str(release)]
            for _ in range(2):
                run = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn('Using completed agreement', run.stdout)
            before = score.read_bytes()
            run = subprocess.run(command+['--rescore'], env=env, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('not empty', run.stderr)
            self.assertEqual(score.read_bytes(), before)
            score.write_text('changed scores\n')
            run = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('score inputs changed', run.stderr)

    def test_slurm_spool_copy_resolves_repository_submission_directory(self):
        base = ROOT/'models/constrained_reaction_balancer'
        scripts = ['translate_batch_PT_BS_parametrized.sh',
                   'translate_batch_PT_CBS_parametrized.sh',
                   'translate_batch_PT_BS_synrxn.sh',
                   'translate_batch_PT_CBS_synrxn.sh',
                   'translate_USPTO_uncurated_chunk.sh']
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            repo = work/'repo'
            model = repo/'models/constrained_reaction_balancer'
            model.mkdir(parents=True)
            for name in ['translate.py', 'CRB_env.sif',
                         'Template_curated/benchmark/prepared/random_seed42_rep0/test_src.txt',
                         'SynRXN_complex/prepared/test_src.txt',
                         'USPTO_uncurated/chunks/test_src_chunk_0000',
                         'models/checkpoints_pretrained/Stereo_separated/random_seed42_rep0/model_step_50000.pt',
                         'models/checkpoints_pretrained/Stereo_separated/random_seed42_rep1/model_step_50000.pt']:
                path = model/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            binaries = work/'bin'
            binaries.mkdir()
            for name in ('module', 'apptainer'):
                path = binaries/name
                path.write_text('#!/bin/sh\nexit 0\n')
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(binaries)+os.pathsep+os.environ['PATH'],
                       SLURM_SUBMIT_DIR=str(repo), SLURM_ARRAY_TASK_ID='0')
            env.pop('SCRIPT_DIR', None)
            for name in scripts:
                spool = work/'slurm_script'
                shutil.copyfile(base/name, spool)
                run = subprocess.run(['bash', str(spool), 'random', '42', '0'],
                                     cwd=work, env=env, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, (name, run.stdout, run.stderr))
                self.assertIn(str(model), run.stdout)


if __name__ == '__main__':
    unittest.main()

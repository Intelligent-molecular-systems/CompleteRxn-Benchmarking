"""Public data/scoring contracts, tested without network or manuscript artifacts."""
import csv
import hashlib
import io
import json
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from completerxn.cli import main
from completerxn.data import download, eligibility_file, export_split, load_split, split_file
from completerxn.io import sha256, write_csv
from completerxn.scoring import score


class BenchmarkCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.references = self.root/'references.csv'
        self.predictions = self.root/'predictions.csv'

    def fixture(self):
        rows = [dict(Unique_identifier=uid, InputReaction='CCO>>',
                     TargetReaction='CCO>>CCO', TargetReaction_with_stereo='CCO>>CCO',
                     InputReaction_harmonized='O>>')
                for uid in ('train', 'val', 'test1', 'test2')]
        write_csv(self.root/'benchmark_final.csv', rows, list(rows[0]))
        for split, column in [('random','random_split'), ('group','group_split'), ('extreme_ood','extreme_ood')]:
            path=self.root/split_file(split, 0)
            path.parent.mkdir(parents=True)
            assignments=[dict(Unique_identifier=r['Unique_identifier'], **{column:str(min(i,2))})
                         for i,r in enumerate(rows)]
            write_csv(path, list(reversed(assignments)), ['Unique_identifier',column])

    def inputs(self, references, predictions):
        write_csv(self.references, [dict(reaction_id=k,reference=v) for k,v in references],
                  ['reaction_id','reference'])
        write_csv(self.predictions, [dict(reaction_id=k,prediction=v) for k,v in predictions],
                  ['reaction_id','prediction'])

    def test_published_partitions_join_by_id_and_preserve_order(self):
        self.fixture()
        for split in ('random','group','extreme_ood'):
            for partition, expected in [('train',['train']),('val',['val']),('test',['test1','test2'])]:
                rows=load_split(self.root,split=split,partition=partition)
                self.assertEqual([r['reaction_id'] for r in rows],expected)

    def test_default_input_is_harmonized_and_original_requires_explicit_selection(self):
        self.fixture()
        self.assertEqual(load_split(self.root)[0]['input'],'O>>')
        self.assertEqual(load_split(self.root,input_column='InputReaction')[0]['input'],'CCO>>')
        output=self.root/'default.csv'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(['data','export','--data-dir',str(self.root),
                                   '--output',str(output)]),0)
        with output.open() as handle:
            self.assertEqual(next(csv.DictReader(handle))['input'],'O>>')
        # Older local tables must be selected explicitly, not silently substituted.
        path=self.root/'benchmark_final.csv'
        with path.open() as handle:
            rows=list(csv.DictReader(handle))
        for row in rows:row.pop('InputReaction_harmonized')
        path.unlink()
        write_csv(path,rows,list(rows[0]))
        with self.assertRaises(ValueError):load_split(self.root)
        self.assertEqual(len(load_split(self.root,input_column='InputReaction')),2)

    def test_filter_requires_explicit_valid_mask_and_metadata(self):
        self.fixture()
        with self.assertRaises(ValueError):load_split(self.root,population='filtered')
        mask=self.root/'mask.csv'
        write_csv(mask,[{'Unique_identifier':'test2'}],['Unique_identifier'])
        output=self.root/'test.csv'
        self.assertEqual(export_split(output,self.root,population='filtered',eligibility=mask),1)
        metadata=json.loads(output.with_suffix('.csv.metadata.json').read_text())
        self.assertFalse(metadata['matches_published_files'])
        self.assertIsNone(metadata['revision'])
        self.assertEqual(metadata['export_sha256'],sha256(output))
        with self.assertRaises(FileExistsError):export_split(output,self.root)
        mask.write_text('Unique_identifier\ntrain\n')
        with self.assertRaises(ValueError):load_split(self.root,population='filtered',eligibility=mask)

    def test_duplicate_split_ids_rejected(self):
        self.fixture()
        with (self.root/split_file('random',0)).open('a') as f:f.write('test1,2\n')
        with self.assertRaises(ValueError):load_split(self.root)

    def test_missing_reference_never_silently_falls_back(self):
        self.fixture()
        path=self.root/'benchmark_final.csv'
        path.write_text(path.read_text().replace('test1,CCO>>,CCO>>CCO,CCO>>CCO','test1,CCO>>,CCO>>CCO,'))
        with self.assertRaises(ValueError):load_split(self.root)
        self.assertEqual(len(load_split(self.root,reference_column='TargetReaction')),2)

    def test_missing_invalid_unbalanced_predictions_stay_in_denominator(self):
        self.inputs([(k,'CCO>>CCO') for k in ('a','b','c','d')],
                    [('c','CCO>>CC=O'),('a','CCO>>CCO'),('b','invalid')])
        result=score(self.references,self.predictions,self.root/'scores')
        self.assertEqual((result['observations'],result['missing'],result['invalid'],result['unbalanced']),
                         (4,1,1,1))
        self.assertEqual(result['equivalence_pct'],25)
        with self.assertRaises(FileExistsError):score(self.references,self.predictions,self.root/'scores')

    def test_extra_duplicate_and_bad_reference_fail_before_writing(self):
        for i,(references,predictions) in enumerate([
            ([('a','CCO>>CCO')],[('extra','CCO>>CCO')]),
            ([('a','CCO>>CCO')],[('a','CCO>>CCO'),('a','CCO>>CCO')]),
            ([('a','CCO>>CC=O')],[('a','CCO>>CCO')]),
        ]):
            self.references=self.root/f'r{i}.csv';self.predictions=self.root/f'p{i}.csv'
            self.inputs(references,predictions)
            output=self.root/f'bad{i}'
            with self.assertRaises(ValueError):score(self.references,self.predictions,output)
            self.assertFalse(output.exists())

    def test_stereo_is_explicit_and_lenient_is_opt_in(self):
        self.inputs([('a','F[C@H](Cl)Br>>F[C@H](Cl)Br')], [('a','FC(Cl)Br>>FC(Cl)Br')])
        # Exact matching compares each side, so stereo-sensitive exact must fail.
        free=score(self.references,self.predictions,self.root/'free')
        stereo=score(self.references,self.predictions,self.root/'stereo',stereo=True)
        self.assertEqual(free['exact_pct'],100)
        self.assertEqual(stereo['exact_pct'],0)
        self.references=self.root/'lenient-r.csv';self.predictions=self.root/'lenient-p.csv'
        self.inputs([('a','O>>O')],[('a','O=C(O)O>>O=C=O.O')])
        self.assertEqual(score(self.references,self.predictions,self.root/'strict')['equivalence_pct'],0)
        self.assertEqual(score(self.references,self.predictions,self.root/'lenient',equivalence_lenient=True)['equivalence_pct'],100)

    def test_unsolved_prediction_cannot_receive_accuracy(self):
        self.inputs([('a','CCO>>CCO')],[('a','CCO>>CCO')])
        self.predictions.write_text('reaction_id,prediction,solved\na,CCO>>CCO,false\n')
        self.assertEqual(score(self.references,self.predictions,self.root/'scores')['equivalence_pct'],0)

    def test_invalid_confidence_is_rejected(self):
        self.inputs([('a','CCO>>CCO')],[('a','CCO>>CCO')])
        for value in ('nan','inf','1.1','not-a-number'):
            self.predictions.write_text(f'reaction_id,prediction,confidence\na,CCO>>CCO,{value}\n')
            with self.assertRaises(ValueError):
                score(self.references,self.predictions,self.root/'scores')

    def test_manuscript_independent_balance_gate_is_preserved(self):
        self.inputs([('a','O=C([O-])O.[Na+]>>O=C([O-])O.[Na+]')],
                    [('a','OC(O)=[O]->[Na]>>OC(O)=[O]->[Na]')])
        result=score(self.references,self.predictions,self.root/'scores')
        self.assertEqual(result['balanced_pct'],0)
        self.assertEqual(result['equivalence_pct'],0)

    def test_modified_export_rejected(self):
        self.fixture();export_split(self.references,self.root)
        self.references.write_text(self.references.read_text()+'extra,CCO>>,CCO>>CCO\n')
        write_csv(self.predictions,[],['reaction_id','prediction'])
        with self.assertRaisesRegex(ValueError,'metadata'):
            score(self.references,self.predictions,self.root/'scores')

    def fake_download(self, content, **kwargs):
        path = Path(kwargs['local_dir'])/kwargs['filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    def test_download_verifies_cache_and_cleans_up_failed_download(self):
        content=b'small test artifact\n';digest=hashlib.sha256(content).hexdigest()
        manifest=dict(repo_id='test/repo',revision='a'*40,files={
            name:dict(sha256=digest,size=len(content))
            for name in ['benchmark_final.csv',split_file('random',0),eligibility_file('random',0)]})
        with patch('completerxn.data.MANIFEST',manifest),patch(
                'completerxn.data.hf_hub_download',
                side_effect=lambda **kwargs:self.fake_download(content,**kwargs)) as request:
            download(self.root);self.assertEqual(request.call_count,3)
            for call in request.call_args_list:
                self.assertEqual(call.kwargs['repo_id'],'test/repo')
                self.assertEqual(call.kwargs['revision'],'a'*40)
                self.assertEqual(call.kwargs['repo_type'],'dataset')
            download(self.root);self.assertEqual(request.call_count,3)
            (self.root/'benchmark_final.csv').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):download(self.root)
        target=self.root/'failed'
        with patch('completerxn.data.MANIFEST',manifest),patch(
                'completerxn.data.hf_hub_download',
                side_effect=lambda **kwargs:self.fake_download(b'wrong',**kwargs)):
            with self.assertRaises(ValueError):download(target)
        self.assertEqual(list(target.iterdir()),[])

    def test_private_download_error_explains_login_and_removes_temporary_files(self):
        from huggingface_hub.errors import HfHubHTTPError
        error=HfHubHTTPError('Access denied',response=SimpleNamespace(
            status_code=401,headers={},request=None))
        with patch('completerxn.data.hf_hub_download',side_effect=error):
            with self.assertRaisesRegex(PermissionError,'hf auth login'):
                download(self.root)
        self.assertEqual(list(self.root.iterdir()),[])

    def test_hf_loading_downloads_mask_and_tracks_filtered_population(self):
        from completerxn import DataLoader
        self.fixture()
        mask=self.root/eligibility_file('random',0)
        mask.parent.mkdir()
        write_csv(mask,[{'Unique_identifier':'test2'}],['Unique_identifier'])
        names=['benchmark_final.csv',split_file('random',0),eligibility_file('random',0)]
        contents={name:(self.root/name).read_bytes() for name in names}
        manifest=dict(repo_id='test/repo',revision='a'*40,files={name:dict(
            sha256=hashlib.sha256(content).hexdigest(),size=len(content))
            for name,content in contents.items()})
        def fetch(**kwargs):
            return self.fake_download(contents[kwargs['filename']],**kwargs)
        with patch('completerxn.data.MANIFEST',manifest),patch(
                'completerxn.benchmark.MANIFEST',manifest),patch(
                'completerxn.data.hf_hub_download',side_effect=fetch) as request:
            cache=self.root/'downloaded'
            loader=DataLoader(cache_dir=cache,revision=manifest['revision'])
            train,val,test=loader.load(population='filtered')
            self.assertEqual((len(train),len(val),test.reaction_ids),(1,1,('test2',)))
            self.assertEqual(request.call_count,3)
            self.assertEqual(test.provenance['revision'],'a'*40)
            self.assertEqual(test.provenance['repo_id'],'test/repo')
            self.assertEqual(len(loader.test()),2)
            self.assertEqual(request.call_count,3)
            # A different valid local selection is not the published population.
            custom=self.root/'custom';custom.mkdir()
            write_csv(custom/'random_rep0.csv',[{'Unique_identifier':'test1'}],['Unique_identifier'])
            changed=DataLoader(data_dir=cache,eligibility_dir=custom,revision=manifest['revision']).test(population='filtered')
            self.assertEqual(changed.reaction_ids,('test1',))
            self.assertIsNone(changed.provenance['revision'])
            output=self.root/'custom_export.csv'
            export_split(output,cache,population='filtered',eligibility=custom/'random_rep0.csv')
            self.assertFalse(json.loads(output.with_suffix('.csv.metadata.json').read_text())['matches_published_files'])
            # Cached masks receive the same integrity checks as the main data.
            (cache/eligibility_file('random',0)).write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'Checksum mismatch'):
                loader.test(population='filtered')

    def test_cli_reports_errors_and_exports(self):
        self.fixture()
        with redirect_stdout(io.StringIO()):
            code=main(['data','export','--data-dir',str(self.root),'--partition','val',
                       '--output',str(self.root/'val.csv')])
        self.assertEqual(code,0)
        with redirect_stderr(io.StringIO()):
            code=main(['score','--references','missing.csv','--predictions','missing.csv',
                       '--output-dir',str(self.root/'scores')])
        self.assertEqual(code,2)


if __name__ == '__main__':
    unittest.main()

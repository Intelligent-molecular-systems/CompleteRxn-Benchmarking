"""End-to-end Python workflow, predictor isolation, and failure inspection."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from completerxn import DataLoader, load_benchmark, plot_reaction
from completerxn.benchmark import Example, Partition
from completerxn.io import write_csv
from completerxn.predictors import Completion, CRBPredictor, SynRBLPredictor
from completerxn.scoring import score


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        examples=[Example(str(i),'CCO>>CCO','CCO>>CCO',{'MissingNrCarbons':str(i%2)}) for i in range(6)]
        self.test=Partition(examples,{'split':'random','rep':0,'partition':'test','population':'full'})

    def test_batch_prediction_preserves_ids_and_hides_references(self):
        batches=[]
        class Model:
            def predict(self,inputs):
                batches.append(inputs)
                return inputs
        predictions=self.test.predict(Model(),batch_size=4)
        self.assertEqual([len(b) for b in batches],[4,2])
        self.assertTrue(all(isinstance(s,str) for b in batches for s in b))
        self.assertEqual(tuple(predictions),self.test.reaction_ids)
        result=self.test.evaluate(predictions)
        self.assertEqual(result.summary['equivalence_pct'],100)
        self.assertEqual(result.failures,[])
        ET.fromstring(result.plot_failures().svg)

    def test_wrong_count_and_foreign_partition_are_errors(self):
        class Wrong:
            def predict(self,inputs):return ['CCO>>CCO']
        with self.assertRaises(ValueError):self.test.predict(Wrong())
        other=Partition([Example('different','CCO>>CCO','CCO>>CCO',{})],{})
        class Model:
            def predict(self,inputs):return inputs
        with self.assertRaises(ValueError):other.evaluate(self.test.predict(Model()))
        with self.assertRaises(ValueError):self.test.evaluate({'unexpected':'CCO>>CCO'})
        with self.assertRaises(ValueError):self.test.evaluate(['CCO>>CCO'])

    def test_external_predictor_can_batch_internally(self):
        calls=[]
        class Model:
            preferred_batch_size=None
            def predict(self,inputs):
                calls.append(inputs)
                return inputs
        self.test.predict(Model())
        self.assertEqual(len(calls),1)
        self.assertEqual(len(calls[0]),len(self.test))

    def test_failure_categories_csv_parity_and_saved_provenance(self):
        values={'0':'CCO>>CCO','1':'invalid','2':'CCO>>CC=O',
                '3':Completion('CCO>>CCO',solved=False),
                '4':'CCO>>COC'}
        result=self.test.evaluate(values)
        self.assertEqual(result.failure_summary(),dict(invalid=1,unbalanced=1,unsolved=1,
                                                       reference_mismatch=1,missing=1))
        self.assertEqual(len(result.failures),5)
        groups=result.failure_summary(group_by='MissingNrCarbons')
        self.assertEqual(sum(g['observations'] for g in groups),6)
        with self.assertRaises(ValueError):result.failure_summary(group_by='absent')
        result.save(self.root/'api')
        references=[dict(reaction_id=r.reaction_id,reference=r.reference) for r in self.test]
        write_csv(self.root/'ref.csv',references,['reaction_id','reference'])
        csv_summary=score(self.root/'ref.csv',self.root/'api/predictions.csv',self.root/'csv')
        self.assertEqual(csv_summary,result.summary)
        metadata=json.loads((self.root/'api/provenance.json').read_text())
        self.assertEqual(metadata['partition_fingerprint'],self.test.fingerprint)
        with self.assertRaises(FileExistsError):result.save(self.root/'api')
        with (self.root/'api/failures.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),5)

    def test_reaction_and_failure_plots_are_valid_svg_and_escape_text(self):
        result=self.test.evaluate({'0':'invalid <script>'})
        view=result.plot('0')
        ET.fromstring(view.svg)
        self.assertIn('Input',view.svg);self.assertIn('Reference',view.svg)
        self.assertNotIn('<script>',view.svg)
        ET.fromstring(result.plot('1').svg) # Missing prediction remains inspectable.
        with self.assertRaises(KeyError):result.plot('unknown')
        simple=plot_reaction('CCO>>CC=O')
        simple.save(self.root/'reaction.svg')
        with self.assertRaises(FileExistsError):simple.save(self.root/'reaction.svg')
        with self.assertRaises(ValueError):simple.save(self.root/'reaction.png')
        with self.assertRaises(ValueError):plot_reaction('invalid')

    def test_local_loading_train_val_test_and_mask_discovery(self):
        rows=[dict(Unique_identifier=str(i),InputReaction='CCO>>CCO',
                   InputReaction_harmonized='O>>O',TargetReaction_with_stereo='CCO>>CCO',
                   MissingNrCarbons=str(i%2)) for i in range(6)]
        write_csv(self.root/'benchmark_final.csv',rows,list(rows[0]))
        splits=self.root/'random_splits';splits.mkdir()
        write_csv(splits/'benchmark_final_random_splits_rep0_seed42.csv',
                  [dict(Unique_identifier=str(i),random_split=str(min(i,2))) for i in range(6)],
                  ['Unique_identifier','random_split'])
        benchmark=load_benchmark(self.root)
        self.assertEqual(len(benchmark.train()),1)
        self.assertEqual(len(benchmark.val()),1)
        self.assertEqual(len(benchmark.test()),4)
        train,val,test=DataLoader(data_dir=self.root).load(split='random',rep=0)
        self.assertEqual((len(train),len(val),len(test)),(1,1,4))
        from completerxn.data import DataLoader as DataLoaderFromData
        self.assertIs(DataLoaderFromData,DataLoader)
        harmonized=DataLoader(data_dir=self.root).test()
        self.assertEqual(harmonized.inputs,('O>>O',)*4)
        self.assertEqual(harmonized.provenance['input_column'],'InputReaction_harmonized')
        original=DataLoader(data_dir=self.root,input_column='InputReaction').test()
        self.assertEqual(original.inputs,('CCO>>CCO',)*4)
        with self.assertRaises(FileNotFoundError):benchmark.test(population='filtered')
        masks=self.root/'eligibility';masks.mkdir()
        write_csv(masks/'random_rep0.csv',[dict(Unique_identifier='3')],['Unique_identifier'])
        filtered=benchmark.test(population='filtered')
        self.assertEqual(filtered.reaction_ids,('3',))
        self.assertEqual(filtered[0].metadata['MissingNrCarbons'],'1')
        ET.fromstring(filtered.plot('3').svg)
        train,val,test=DataLoader(data_dir=self.root).load(population='filtered')
        self.assertEqual((len(train),len(val),len(test)),(1,1,1))

    def test_hf_loader_uses_requested_revision_cache_and_published_splits(self):
        from completerxn.data import MANIFEST
        loader=DataLoader(cache_dir=self.root/'hf-cache')
        self.assertTrue(loader.download_missing)
        self.assertEqual(loader.data_dir,self.root/'hf-cache')
        with self.assertRaises(ValueError):DataLoader(revision='main')
        with self.assertRaises(ValueError):DataLoader(data_dir=self.root,cache_dir=self.root)

    def test_synrbl_adapter_restores_reordered_results(self):
        class Config:
            def __init__(self,**kwargs):pass
        class Rebalancer:
            def __init__(self,**kwargs):pass
            def rebalance(self,rows,**kwargs):
                assert kwargs['keep_extra'] is True # False drops solved/confidence in SynRBL.
                return [dict(row,solved='false',confidence=.5) for row in reversed(rows)]
        module=types.SimpleNamespace(ReactionRebalancer=Rebalancer,RebalanceConfig=Config)
        with patch.dict(sys.modules,{'synrbl':module}),patch('importlib.metadata.version',return_value='test'):
            predictor=SynRBLPredictor()
        values=predictor.predict(['CCO>>CCO','O>>O'])
        self.assertEqual([p.reaction for p in values],['CCO>>CCO','O>>O'])
        self.assertFalse(values[0].solved)

    def test_crb_adapter_tokenizes_and_combines_missing_fragments(self):
        checkpoint=self.root/'model.pt';checkpoint.write_bytes(b'test checkpoint')
        script=self.root/'translate.py'
        script.write_text("""import sys
from pathlib import Path
args=sys.argv
assert '-constrained_atom_balance' in args and '-constrained_charge_balance' in args
source=Path(args[args.index('-src')+1]).read_text()
assert 'C C O > > C C = O' in source
Path(args[args.index('-output')+1]).write_text('> > [ H ] [ H ]\\n')
""")
        predictor=CRBPredictor(checkpoint,script)
        self.assertEqual(predictor.predict(['CCO>>CC=O'])[0].reaction,'CCO>>CC=O.[H][H]')
        self.assertIn('checkpoint_sha256',predictor.metadata)

    def test_crb_hf_loader_pins_model_revision_and_records_source(self):
        checkpoint=self.root/'finetuned-random-rep0.pt';checkpoint.write_bytes(b'fixture')
        script=self.root/'translate.py';script.write_text('')
        calls=[]
        def download(**kwargs):calls.append(kwargs);return str(checkpoint)
        module=types.SimpleNamespace(hf_hub_download=download)
        with patch.dict(sys.modules,{'huggingface_hub':module}):
            model=CRBPredictor.from_pretrained('owner/crb',filename='random_rep0.pt',revision='a'*40,
                                              translate_script=script)
            with self.assertRaises(ValueError):
                CRBPredictor.from_pretrained('owner/crb',filename='random_rep0.pt',revision='main',
                                            translate_script=script)
        self.assertEqual(calls[0]['repo_type'],'model')
        self.assertEqual(calls[0]['revision'],'a'*40)
        self.assertEqual(model.metadata['huggingface']['filename'],'random_rep0.pt')


if __name__=='__main__':unittest.main()

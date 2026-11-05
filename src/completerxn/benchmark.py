"""The benchmark workflow: load a partition, predict, evaluate, inspect."""
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

from .data import MANIFEST, cache_dir, download, eligibility_file, load_split, split_file
from .io import sha256, write_csv, write_json
from .predictors import Completion


def _completion(value):
    if isinstance(value,Completion):
        return value
    if isinstance(value,str):
        return Completion(value)
    raise TypeError('Predictions must be completed reaction strings or Completion objects')


@dataclass(frozen=True)
class Example:
    reaction_id: str
    input: str
    reference: str
    metadata: dict


class Predictions(Mapping):
    """Completed reactions stored by ID, optionally linked to a selected data partition."""
    def __init__(self, values, *, partition_fingerprint=None, model=None):
        self._values={uid:_completion(value) for uid,value in values.items()}
        self.partition_fingerprint=partition_fingerprint
        self.model=model

    def __getitem__(self,key):return self._values[key]
    def __iter__(self):return iter(self._values)
    def __len__(self):return len(self._values)

    def save(self,path):
        path=Path(path)
        sidecar=path.with_suffix(path.suffix+'.metadata.json')
        if path.exists() or sidecar.exists():
            raise FileExistsError(path)
        rows=[dict(reaction_id=uid,prediction=value.reaction,solved=value.solved,
                   confidence=value.confidence) for uid,value in self.items()]
        path.parent.mkdir(parents=True,exist_ok=True)
        write_csv(path,rows,['reaction_id','prediction','solved','confidence'])
        write_json(sidecar,dict(partition_fingerprint=self.partition_fingerprint,
                               model=self.model,sha256=sha256(path)))


class Partition(Sequence):
    def __init__(self, examples, provenance):
        self._examples=tuple(examples)
        self.provenance=dict(provenance)
        ids=self.reaction_ids
        if not ids or any(not uid for uid in ids) or len(ids)!=len(set(ids)):
            raise ValueError('A partition must contain unique, nonempty reaction IDs')
        policy={k:v for k,v in self.provenance.items() if k!='sha256'}
        payload=[policy,[(r.reaction_id,r.input,r.reference) for r in self]]
        self.fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

    def __len__(self):return len(self._examples)
    def __getitem__(self,key):
        if isinstance(key,slice):
            return Partition(self._examples[key],dict(self.provenance,subset=True))
        return self._examples[key]

    @property
    def inputs(self):return tuple(r.input for r in self)
    @property
    def reaction_ids(self):return tuple(r.reaction_id for r in self)

    def predict(self, model, *, batch_size=None):
        """Run model.predict(inputs) in batches, without giving it references."""
        if batch_size is None:
            batch_size=getattr(model,'preferred_batch_size',64)
            if batch_size is None:
                batch_size=len(self) # External runners batch internally; start them once.
        if not isinstance(batch_size,int) or batch_size<1:
            raise ValueError('batch_size must be positive')
        values={}
        for start in range(0,len(self),batch_size):
            batch=self._examples[start:start+batch_size]
            outputs=model.predict([r.input for r in batch])
            if isinstance(outputs,(str,bytes,Mapping)):
                raise TypeError('predict must return a sequence, in the same order as its inputs')
            outputs=list(outputs)
            if len(outputs)!=len(batch):
                raise ValueError('Predictor returned a different number of outputs than inputs')
            values.update((r.reaction_id,_completion(output)) for r,output in zip(batch,outputs))
        metadata=getattr(model,'metadata',f'{type(model).__module__}.{type(model).__qualname__}')
        return Predictions(values,partition_fingerprint=self.fingerprint,model=metadata)

    def evaluate(self,predictions,*,stereo=False,equivalence_lenient=False):
        """Score predictions supplied by ID, as Predictions, or as a full ordered sequence."""
        from .scoring import evaluate_rows
        if isinstance(predictions,Predictions):
            if predictions.partition_fingerprint not in (None,self.fingerprint):
                raise ValueError('Predictions belong to a different partition or subset')
        elif isinstance(predictions,Mapping):
            predictions=Predictions(predictions)
        else:
            if isinstance(predictions,(str,bytes)):
                raise TypeError('Provide predictions by ID or as an ordered sequence')
            predictions=list(predictions)
            if len(predictions)!=len(self):
                raise ValueError('Ordered predictions must cover the full partition; use IDs for partial results')
            predictions=Predictions(dict(zip(self.reaction_ids,predictions)))
        reference_rows=[dict(reaction_id=r.reaction_id,reference=r.reference) for r in self]
        predicted_rows=[dict(reaction_id=uid,prediction=p.reaction,solved=p.solved)
                        for uid,p in predictions.items()]
        records,summary=evaluate_rows(reference_rows,predicted_rows,stereo=stereo,
                                       equivalence_lenient=equivalence_lenient)
        return Evaluation(self,predictions,records,summary)

    def plot(self,reaction_id):
        from .plotting import compare_reactions
        example=next((r for r in self if r.reaction_id==reaction_id),None)
        if example is None:raise KeyError(reaction_id)
        return compare_reactions([('Input',example.input),('Reference',example.reference)],title=reaction_id)


class Evaluation:
    def __init__(self,partition,predictions,records,summary):
        self.partition,self.predictions=partition,predictions
        self.records,self.summary=records,summary

    @property
    def failures(self):
        """Return failure records for inspection. A reference mismatch is not a chemical judgment."""
        examples={r.reaction_id:r for r in self.partition}
        rows=[]
        for record in self.records:
            if record['reason']=='correct':continue
            example=examples[record['reaction_id']]
            completion=self.predictions.get(example.reaction_id)
            rows.append(dict(record,input=example.input,reference=example.reference,
                             prediction=completion.reaction if completion else '',
                             confidence=completion.confidence if completion else None,
                             **example.metadata))
        return rows

    def failure_summary(self,*,group_by=None):
        if group_by is None:
            return dict(Counter(r['reason'] for r in self.records if r['reason']!='correct'))
        examples={r.reaction_id:r for r in self.partition}
        if not any(group_by in r.metadata for r in self.partition):
            raise ValueError(f'Metadata field is unavailable: {group_by}')
        groups={}
        for record in self.records:
            group=examples[record['reaction_id']].metadata.get(group_by,'unknown')
            groups.setdefault(group,[]).append(record)
        return [dict(group=key,observations=len(rows),failures=sum(not r['equivalence'] for r in rows),
                     equivalence_pct=100*sum(r['equivalence'] for r in rows)/len(rows))
                for key,rows in sorted(groups.items(),key=lambda item:str(item[0]))]

    def plot(self,reaction_id):
        from .plotting import compare_reactions
        example=next((r for r in self.partition if r.reaction_id==reaction_id),None)
        if example is None:raise KeyError(reaction_id)
        completion=self.predictions.get(reaction_id)
        reason=next(r['reason'] for r in self.records if r['reaction_id']==reaction_id)
        return compare_reactions([('Input',example.input),
                                  ('Prediction',completion.reaction if completion else ''),
                                  ('Reference',example.reference)],title=f'{reaction_id}: {reason}')

    def plot_failures(self):
        from .plotting import failure_bars
        return failure_bars(self.failure_summary())

    def save(self,output_dir):
        """Save numerical results, failure records, scoring settings, and input file hashes."""
        from . import __version__
        output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
        write_csv(output/'per_reaction.csv',self.records,list(self.records[0]))
        write_csv(output/'summary.csv',[self.summary],list(self.summary))
        predicted=[dict(reaction_id=uid,prediction=p.reaction,solved=p.solved,confidence=p.confidence)
                   for uid,p in self.predictions.items()]
        write_csv(output/'predictions.csv',predicted,['reaction_id','prediction','solved','confidence'])
        failures=self.failures
        fields=list(dict.fromkeys(k for row in failures for k in row)) if failures else ['reaction_id','reason']
        write_csv(output/'failures.csv',failures,fields)
        sources=[Path(__file__),Path(__file__).with_name('scoring.py'),
                 Path(__file__).with_name('equivalence.py'),Path(__file__).with_name('balance.py')]
        write_json(output/'provenance.json',dict(package_version=__version__,
                   dataset=self.partition.provenance,partition_fingerprint=self.partition.fingerprint,
                   model=self.predictions.model,scoring=self.summary,
                   predictions_sha256=sha256(output/'predictions.csv'),
                   sha256={p.name:sha256(p) for p in sources}))


class Benchmark:
    def __init__(self,data_dir=None,*,eligibility_dir=None,reference_column='TargetReaction_with_stereo',
                 input_column='InputReaction_harmonized'):
        self.data_dir=Path(data_dir) if data_dir is not None else cache_dir()
        self.download_missing=data_dir is None
        self.eligibility_dir=Path(eligibility_dir) if eligibility_dir is not None else self.data_dir/'eligibility'
        self.reference_column=reference_column
        self.input_column=input_column

    def partition(self,partition,*,split='random',rep=0,population='full'):
        name=split_file(split,rep)
        if self.download_missing:
            download(self.data_dir,split=split,rep=rep)
        eligibility=None
        if population=='filtered':
            eligibility=self.eligibility_dir/f'{split}_rep{rep}.csv'
            if not eligibility.is_file():
                fold=f'random_seed{42+rep}_rep{rep}' if split=='random' else f'{split}_rep{rep}'
                eligibility=self.eligibility_dir/f'CRB_{fold}.csv'
            if not eligibility.is_file():
                raise FileNotFoundError('Published eligibility mask unavailable for this fold; '
                                        'provide eligibility_dir or choose population="full"')
        rows=load_split(self.data_dir,split=split,rep=rep,partition=partition,population=population,
                        eligibility=eligibility,reference_column=self.reference_column,include_metadata=True,
                        input_column=self.input_column)
        files=[self.data_dir/'benchmark_final.csv',self.data_dir/name]
        if eligibility is not None:files.append(eligibility)
        hashes={str(p.resolve()):sha256(p) for p in files}
        published=all(hashes[str((self.data_dir/n).resolve())]==MANIFEST['files'][n]['sha256']
                      for n in ('benchmark_final.csv',name))
        if eligibility is not None:
            expected = MANIFEST['files'].get(eligibility_file(split, rep), {})
            published = published and hashes[str(eligibility.resolve())] == expected.get('sha256')
        provenance=dict(split=split,rep=rep,partition=partition,population=population,
                        reference_column=self.reference_column,input_column=self.input_column,sha256=hashes,
                        revision=MANIFEST['revision'] if published else None,
                        repo_id=MANIFEST['repo_id'])
        return Partition([Example(r['reaction_id'],r['input'],r['reference'],r['metadata']) for r in rows],provenance)

    def train(self,**options):return self.partition('train',**options)
    def val(self,**options):return self.partition('val',**options)
    def test(self,**options):return self.partition('test',**options)
    def evaluate(self,predictions,partition,**options):return partition.evaluate(predictions,**options)


def load_benchmark(data_dir=None,**options):
    """Open local benchmark files, or download the specified Hugging Face version when needed."""
    return Benchmark(data_dir,**options)


class DatasetSplits(NamedTuple):
    train: Partition
    val: Partition
    test: Partition


class DataLoader(Benchmark):
    """Load train, validation, and test sets from Hugging Face or local files.

    load(split='random', rep=0) returns train, val, test. Downloads use the dataset
    version specified in dataset.json and check file hashes. Set data_dir for
    local files or cache_dir to change the download directory.
    """
    def __init__(self, *, data_dir=None, cache_dir=None, eligibility_dir=None,
                 reference_column='TargetReaction_with_stereo',input_column='InputReaction_harmonized',
                 revision=MANIFEST['revision']):
        if data_dir is not None and cache_dir is not None:
            raise ValueError('Use data_dir for local files or cache_dir for HF downloads, not both')
        if revision!=MANIFEST['revision']:
            raise ValueError('This package supports the dataset revision in dataset.json only')
        super().__init__(data_dir if data_dir is not None else cache_dir,
                         eligibility_dir=eligibility_dir,reference_column=reference_column,input_column=input_column)
        self.download_missing=data_dir is None

    def load(self, *, split='random', rep=0, population='full'):
        """Return train, validation, and test sets for one split and repetition.

        Training and validation assignments are unchanged. population='filtered'
        selects the paper's test reactions using the released retained-ID list.
        The loader downloads the retained-ID list for the requested repetition.
        """
        test=self.test(split=split,rep=rep,population=population)
        return DatasetSplits(self.train(split=split,rep=rep),self.val(split=split,rep=rep),test)

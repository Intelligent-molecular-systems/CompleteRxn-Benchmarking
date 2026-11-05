"""Connect models to the benchmark's prediction interface.

Models receive incomplete reaction strings and return one completed reaction
per input in the same order. Optional adapters support SynRBL and CRB.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Completion:
    reaction: str
    solved: bool = True
    confidence: float | None = None

    def __post_init__(self):
        if not isinstance(self.reaction,str) or not isinstance(self.solved,bool):
            raise ValueError('Completion requires a reaction string and boolean solved flag')
        if self.confidence is not None and (not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError('Confidence must be finite and between 0 and 1')


class Predictor(Protocol):
    def predict(self, inputs: Sequence[str]) -> Sequence[str | Completion]:
        """Return one completed reaction for each input. Do not return fragments."""
        ...


class SynRBLPredictor:
    """Run the optional SynRBL package installed in the current environment."""
    def __init__(self, *, n_jobs=1, use_default_reduction=True):
        try:
            from synrbl import ReactionRebalancer, RebalanceConfig
        except ImportError as error:
            raise ImportError('Install SynRBL in its model environment before using this adapter') from error
        config=RebalanceConfig(reaction_col='reaction',id_col='id',n_jobs=n_jobs,
                               enable_logging=False,use_default_reduction=use_default_reduction)
        self.model=ReactionRebalancer(config=config,user_logger=None)
        from importlib.metadata import version
        self.metadata=dict(model='SynRBL',version=version('synrbl'),n_jobs=n_jobs,
                           use_default_reduction=use_default_reduction)

    def predict(self, inputs):
        rows=self.model.rebalance([{'id':i,'reaction':rxn} for i,rxn in enumerate(inputs)],keep_extra=True)
        indexed={int(r['id']):r for r in rows}
        if len(indexed)!=len(rows) or set(indexed)!=set(range(len(inputs))):
            raise ValueError('SynRBL returned missing, duplicate, or unexpected IDs')
        # SynRBL may reorder its results. Restore input order by ID.
        result=[]
        for i in range(len(inputs)):
            row=indexed[i];solved=str(row.get('solved',False)).lower()
            if solved not in ('true','false','1','0'):
                raise ValueError('Invalid SynRBL solved flag')
            result.append(Completion(row['reaction'],solved in ('true','1'),row.get('confidence')))
        return result



class CRBPredictor:
    """Run the repository's CRB translator with a local model checkpoint.

    Use command_prefix to select the model environment, for example
    ['apptainer', 'exec', '/path/to/CRB_env.sif', 'python'].
    Use from_pretrained to download a checkpoint from Hugging Face.
    """
    preferred_batch_size=None # Submit once; OpenNMT batches internally.

    def __init__(self, checkpoint, translate_script, *, command_prefix=None, beam_size=5, gpu=-1):
        self.checkpoint=Path(checkpoint).resolve()
        self.script=Path(translate_script).resolve()
        if not self.checkpoint.is_file() or not self.script.is_file():
            raise FileNotFoundError('Provide an existing checkpoint and the CRB translate.py script')
        if not isinstance(beam_size,int) or beam_size<1:
            raise ValueError('beam_size must be positive')
        self.prefix=list(command_prefix) if command_prefix is not None else [sys.executable]
        self.beam_size,self.gpu=beam_size,gpu
        self.hub_source=None

    @classmethod
    def from_pretrained(cls,repo_id,*,filename,revision,translate_script,cache_dir=None,**options):
        """Download a checkpoint from a specified Hugging Face model version.

        Choose the fine-tuned model for the evaluation fold. No default model
        repository is configured. Prediction still needs the existing CRB environment.
        """
        if not re.fullmatch(r'[0-9a-fA-F]{40}',revision):
            raise ValueError('Provide a full 40-character HF model commit revision')
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise ImportError('huggingface_hub is missing; reinstall the package with pip install -e .') from error
        checkpoint=hf_hub_download(repo_id=repo_id,repo_type='model',filename=filename,
                                   revision=revision,cache_dir=cache_dir)
        model=cls(checkpoint,translate_script,**options)
        model.hub_source=dict(repo_id=repo_id,filename=filename,revision=revision)
        return model

    @property
    def metadata(self):
        from .io import sha256
        return dict(model='CRB',checkpoint=str(self.checkpoint),checkpoint_sha256=sha256(self.checkpoint),
                    translate_sha256=sha256(self.script),command_prefix=self.prefix,
                    beam_size=self.beam_size,gpu=self.gpu,huggingface=self.hub_source)

    def predict(self, inputs):
        pattern=re.compile(r'(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])')
        tokenized=[]
        for reaction in inputs:
            tokens=pattern.findall(reaction)
            if ''.join(tokens)!=reaction:
                raise ValueError('CRB tokenizer cannot represent an input reaction')
            tokenized.append(' '.join(tokens))
        with tempfile.TemporaryDirectory(prefix='completerxn-crb-') as temporary:
            source=Path(temporary)/'inputs.txt';output=Path(temporary)/'predictions.txt'
            source.write_text('\n'.join(tokenized)+'\n')
            command=self.prefix+[str(self.script),'-model',str(self.checkpoint),'-src',str(source),
                '-output',str(output),'-beam_size',str(self.beam_size),'-n_best','1','-batch_size','64',
                '-gpu',str(self.gpu),'-replace_unk','-max_length','200','-fast',
                '-constrained_atom_balance','-constrained_charge_balance']
            subprocess.run(command,check=True)
            differences=[''.join(line.split()) for line in output.read_text().splitlines()]
        if len(differences)!=len(inputs):
            raise ValueError('CRB returned a different number of predictions than inputs')
        completed=[]
        for original,difference in zip(inputs,differences):
            left=re.split(r'(?<!-)>',original);right=re.split(r'(?<!-)>',difference)
            if len(left)!=3 or len(right)!=3:
                completed.append(Completion('',solved=False))
            else:
                completed.append(Completion('>'.join('.'.join(x for x in (a,b) if x)
                                                      for a,b in zip(left,right))))
        return completed

"""Check saved scores and rebuild the main summary tables and plotting inputs.

Run after analysis/score_benchmark.py. No chemistry packages are needed.
"""
import argparse
import csv
import gzip
import io
import json
import sys
import tarfile
import statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.result_paths import metric_name, results_root
OUT=ROOT/'results/main/scores'

def equivalence_column(scores_dir):
    """Read the metric from the run record and choose its saved accuracy column."""
    scores_dir=Path(scores_dir).resolve()
    metadata=scores_dir/'provenance.json'
    if metadata.exists():
        metric=metric_name(json.loads(metadata.read_text())['metric'])
        return 'deep_pragmatic_equivalence_match' if metric=='v1' else 'equivalent_match'
    if scores_dir==results_root('v1')/'main/scores':
        return 'deep_pragmatic_equivalence_match'
    raise ValueError('Missing score provenance: '+str(metadata))

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir",type=Path,default=OUT)
    args=parser.parse_args();OUT=args.scores_dir.resolve()
    if OUT==results_root("v1")/"main/scores":
        parser.error("The v1 archive is read-only; rescore into a new --output-root")
    column=equivalence_column(OUT)
    bench={r['Unique_identifier']:r for r in read(ROOT/'data/Template_curated/benchmark/benchmark_final.csv')}
    syn_source={r['Unique_identifier']:r for r in read(ROOT/'results/main/predictions/synrbl/eval_all_splits_noRI_complete/full_results.csv')}
    syn_scores={};folds={};per_rep=[];template_counts={m:defaultdict(lambda:[0,0]) for m in ['RB','CRB','SynRBL']}
    archive=OUT/'filtered_scores.tar.gz'
    if not (OUT/'per_reaction').exists() and archive.exists():
        (OUT/'per_reaction').mkdir()
        with tarfile.open(archive,'r:gz') as bundle:
            for member in bundle.getmembers():
                if not member.isfile() or Path(member.name).name!=member.name or not member.name.endswith('.csv'):
                    raise ValueError('Unexpected score archive member: '+member.name)
                (OUT/'per_reaction'/member.name).write_bytes(bundle.extractfile(member).read())
    files=sorted((OUT/'per_reaction').glob('*.csv'))
    assert len(files)==45
    for path in files:
        model,fold=path.stem.split('_',1);rep=int(fold.rsplit('rep',1)[1]);split='random' if fold.startswith('random') else 'group' if fold.startswith('group') else 'extreme_ood'
        rows=read(path);assert rows and all(r['retained']=='1' for r in rows)
        support=[(r['Unique_identifier'],r['original_index']) for r in rows]
        assert len(set(support))==len(rows)
        if fold in folds:assert folds[fold]==support
        else:folds[fold]=support
        assert all(not (int(r['exact']) or int(r['equivalence'])) or int(r['balanced']) for r in rows)
        result=dict(model=model,split=split,rep=rep,population='retained',n=len(rows))
        for metric in ['exact','equivalence','balanced']:result[metric+'_pct']=100*sum(int(r[metric]) for r in rows)/len(rows)
        per_rep.append(result)
        for r in rows:
            uid=r['Unique_identifier']
            if model=='SynRBL':
                flags=tuple(int(r[k]) for k in ['exact','equivalence','balanced'])
                if uid in syn_scores:assert syn_scores[uid]==flags
                syn_scores[uid]=flags
            if split=='random':
                label=bench[uid]['1kTPLLabel'].strip()
                if label.lower() not in {'','nan','none','null'}:
                    template_counts[model][label][0]+=1
                    template_counts[model][label][1]+=int(r['equivalence'])
    aggregate=[]
    for split in ['random','group','extreme_ood']:
        for model in ['RB','CRB','SynRBL']:
            group=[r for r in per_rep if r['model']==model and r['split']==split];assert len(group)==5
            row=dict(population='retained',split=split,model=model,n_observations=sum(r['n'] for r in group))
            for metric in ['exact_pct','equivalence_pct','balanced_pct']:
                row[metric+'_mean']=statistics.mean(r[metric] for r in group)
                row[metric+'_sd']=statistics.stdev(r[metric] for r in group)
            aggregate.append(row)
    write(OUT/'per_repetition.csv',per_rep);write(OUT/'summary.csv',aggregate)
    syn=[]
    for uid,flags in sorted(syn_scores.items()):
        r=syn_source[uid]
        # Do not carry forward outdated normalization-score columns.
        row={k:r[k] for k in ['Unique_identifier','Reaction','solved','solved_by','confidence','rxn_flower']}
        row.update(exact_match=bool(flags[0]),is_balanced=bool(flags[2]))
        row[column]=bool(flags[1])
        syn.append(row)
    write(OUT/'synrbl_evaluated.csv',syn)
    templates=[]
    for model,counts in template_counts.items():
        assert counts.keys()==template_counts['CRB'].keys()
        assert all(n==template_counts['CRB'][label][0] for label,(n,c) in counts.items())
        errors=sorted([n-c for n,c in counts.values()],reverse=True);total=sum(errors);cumulative=0;n50=n90=None
        for i,e in enumerate(errors,1):
            cumulative+=e
            if n50 is None and cumulative>=.5*total:n50=i
            if n90 is None and cumulative>=.9*total:n90=i
        n=len(counts);perfect=sum(size==correct for size,correct in counts.values())
        templates.append(dict(model=model,n_templates=n,n_perfect_templates=perfect,frac_perfect_templates=perfect/n,n_failing_templates=n-perfect,total_errors=total,n_templates_for_50pct_errors=n50,n_templates_for_90pct_errors=n90,frac_templates_for_50pct_errors=n50/n,frac_templates_for_90pct_errors=n90/n))
    write(OUT/'template_error_concentration.csv',templates)
    # Deterministic compact release of score flags (no raw structures/predictions).
    with archive.open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as zipped:
            with tarfile.open(fileobj=zipped,mode='w') as bundle:
                for path in files:
                    data=path.read_bytes();info=tarfile.TarInfo(path.name)
                    info.size=len(data);info.mtime=0;info.mode=0o644
                    bundle.addfile(info,io.BytesIO(data))
    print('Validated 45 files; published filtered summaries, {} SynRBL UIDs, and matched template statistics.'.format(len(syn)))

if __name__=='__main__':main()

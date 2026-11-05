"""Build Table 2 from the saved filtered scores, rounding only for display.

By default, read the saved counts of invalid SMILES and confident predictions.
This needs only Python's standard library. Use --recompute-auxiliary in the
CRB container to recalculate those counts from the original predictions.
"""
import argparse
import csv
import importlib.util
import statistics
import sys
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/main/scores'

def read(path,delimiter=','):
    with path.open() as f:return list(csv.DictReader(f,delimiter=delimiter))

def recompute_auxiliary():
    sys.path.insert(0,str(ROOT))
    spec=importlib.util.spec_from_file_location('scorer',str(ROOT/'models/constrained_reaction_balancer/score_predictions.py'))
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    @lru_cache(maxsize=None)
    def invalid(pred):return scorer.canonicalize_smiles(pred)==''
    syn={r['Unique_identifier']:r for r in read(ROOT/'results/main/predictions/synrbl/eval_all_splits_noRI_complete/full_results.csv')}
    rows=[]
    for split in ['random','group','extreme_ood']:
        for model,directory in [('RB','BS_PT'),('CRB','CBS_PT'),('SynRBL',None)]:
            for rep in range(5):
                fold=f'{split}_rep{rep}' if split!='random' else f'random_seed{42+rep}_rep{rep}'
                scores=read(OUT/'per_reaction'/f'{model}_{fold}.csv')
                assert all(r['retained']=='1' for r in scores)
                if directory:
                    old=read(ROOT/'results/main/predictions'/('rb' if directory=='BS_PT' else 'crb')/split/fold/'top1_predictions_with_stats.txt','\t')
                    assert [r['original_index'] for r in old]==[r['original_index'] for r in scores]
                    n=sum(invalid(r['top1_prediction']) for r in old)
                else:
                    n=sum(syn[r['Unique_identifier']]['solved'].lower()=='true' and syn[r['Unique_identifier']]['solved_by'].lower()=='mcs-based' and float(syn[r['Unique_identifier']]['confidence'] or 0)>=.5 for r in scores)
                rows.append(dict(model=model,split=split,rep=rep,n=len(scores),count=n,metric='invalid_smiles' if directory else 'confident_mcs',pct=100*n/len(scores)))
    with (OUT/'main_table_auxiliary_per_repetition.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute-auxiliary',action='store_true')
    parser.add_argument('--scores-dir',type=Path,default=OUT)
    parser.add_argument('--output-dir',type=Path,help='Write display tables separately from the score release')
    args=parser.parse_args()
    OUT=args.scores_dir
    frozen=ROOT/'results/archive/metric_v1/main/scores'
    if OUT.resolve()==frozen.resolve() and args.output_dir is None:
        parser.error('The v1 archive is read-only; provide --output-dir')
    if args.output_dir is not None and args.recompute_auxiliary:
        parser.error('Recompute auxiliary counts in a new score release instead')
    destination=args.output_dir or OUT
    destination.mkdir(parents=True,exist_ok=True)
    if args.recompute_auxiliary:recompute_auxiliary()
    summary=read(OUT/'summary.csv');aux=read(OUT/'main_table_auxiliary_per_repetition.csv');per_rep=read(OUT/'per_repetition.csv')
    tex=[];table=[]
    def fmt(mean,sd):return f'{mean:.2f} ± {sd:.2f}'
    for split,label in [('random','R'),('group','G'),('extreme_ood','OOD')]:
        tex+=['\\hline','',label]
        for model in ['RB','CRB','SynRBL']:
            selected=[r for r in aux if r['model']==model and r['split']==split];assert len(selected)==5
            for r in selected:
                original=next(x for x in per_rep if x['model']==model and x['split']==split and x['rep']==r['rep'])
                assert int(r['n'])==int(original['n'])
            values=[100*int(r['count'])/int(r['n']) for r in selected]
            auxiliary=fmt(statistics.mean(values),statistics.stdev(values))
            r=next(r for r in summary if (r['population'],r['split'],r['model'])==('retained',split,model))
            exact=fmt(float(r['exact_pct_mean']),float(r['exact_pct_sd']))
            equiv=fmt(float(r['equivalence_pct_mean']),float(r['equivalence_pct_sd']))
            balance=fmt(float(r['balanced_pct_mean']),float(r['balanced_pct_sd']))
            confidence,invalid=(auxiliary,'-') if model=='SynRBL' else ('-',auxiliary)
            table.append(dict(Split=label,Model=model,Exact=exact,Equivalence=equiv,Confidence=confidence,Invalid_SMILES=invalid,Balanced=balance))
            cells=[exact,equiv,confidence,invalid,balance]
            cells=[c.replace('±','$\\pm$') for c in cells]
            if model=='CRB':cells[1]='\\textbf{'+cells[1]+'}'
            tex.append('& '+model+' & '+' & '.join(cells)+' \\\\')
        tex.append('')
    tex.append('\\hline')
    (destination/'main_table_filtered.tex').write_text('\n'.join(tex)+'\n')
    with (destination/'main_table_filtered.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    md=['# Main manuscript: filtered results','', 'All entries are percentages (mean ± sample SD over five repetitions). All models use identical retained test IDs. Accuracy requires independent atom and charge balance. Confidence is the solved MCS-based fraction at confidence ≥0.5, not accuracy. Invalid SMILES uses the historical top-1 predicted-completion parsing definition.','', '| Split | Model | Exact | Equivalence | Confidence | Invalid SMILES | Balanced |','|---|---|---:|---:|---:|---:|---:|']
    for r in table:md.append('| '+' | '.join(r.values())+' |')
    (destination/'main_table_filtered.md').write_text('\n'.join(md)+'\n')
    print(f'Wrote {destination}/main_table_filtered.{{tex,csv,md}}')

if __name__=='__main__':main()

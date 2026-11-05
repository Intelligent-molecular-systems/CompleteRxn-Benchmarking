"""Build the main filtered table, preserving the historical invalid-SMILES definition."""
import csv
import importlib.util
import statistics
import sys
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT / "results/additional/unfiltered_evaluation"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('scorer',str(ROOT/'models/constrained_reaction_balancer/score_predictions.py'))
scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
def read(p,delimiter=','):
 with p.open() as f:return list(csv.DictReader(f,delimiter=delimiter))
@lru_cache(maxsize=None)
def invalid(pred):return scorer.canonicalize_smiles(pred)==''
summary=read(OUT/'summary.csv')
syn={r['Unique_identifier']:r for r in read(ROOT/'results/main/predictions/synrbl/eval_all_splits_noRI_complete/full_results.csv')}
rows=[];perfold=[]
for split,label in [('random','R'),('group','G'),('extreme_ood','OOD')]:
 rows+=['\\hline','',label]
 for model,directory in [('RB','BS_PT'),('CRB','CBS_PT'),('SynRBL',None)]:
  fractions=[]
  for rep in range(5):
   fold=f'{split}_rep{rep}' if split!='random' else f'random_seed{42+rep}_rep{rep}'
   scores=[r for r in read(OUT/'per_reaction'/f'{model}_{fold}.csv') if r['retained']=='1']
   if directory:
    old=read(ROOT/'results/main/predictions'/('rb' if directory=='BS_PT' else 'crb')/split/fold/'top1_predictions_with_stats.txt','\t')
    assert len(old)==len(scores)
    assert [r['original_index'] for r in old]==[r['original_index'] for r in scores]
    n=sum(invalid(r['top1_prediction']) for r in old)
   else:
    n=sum(syn[r['Unique_identifier']]['solved'].lower()=='true' and syn[r['Unique_identifier']]['solved_by'].lower()=='mcs-based' and float(syn[r['Unique_identifier']]['confidence'] or 0)>=.5 for r in scores)
   fractions.append(100*n/len(scores))
   perfold.append(dict(model=model,split=split,rep=rep,n=len(scores),count=n,metric='invalid_smiles' if directory else 'confident_mcs',pct=fractions[-1]))
  def fmt(m,s):return f'{m:.2f} $\\pm$ {s:.2f}'
  auxiliary=fmt(statistics.mean(fractions),statistics.stdev(fractions))
  r=next(r for r in summary if (r['population'],r['split'],r['model'])==('retained',split,model))
  exact=fmt(float(r['exact_pct_mean']),float(r['exact_pct_sd']))
  equiv=fmt(float(r['equivalence_pct_mean']),float(r['equivalence_pct_sd']))
  if model=='CRB':equiv='\\textbf{'+equiv+'}'
  balance=fmt(float(r['balanced_pct_mean']),float(r['balanced_pct_sd']))
  confidence,inv=('-',auxiliary) if directory else (auxiliary,'-')
  rows.append(f'& {model:6} & {exact} & {equiv} & {confidence} & {inv} & {balance} \\\\')
 rows.append('')
rows.append('\\hline')
(OUT/'main_table_filtered.tex').write_text('\n'.join(rows)+'\n')
with (OUT/'main_table_auxiliary_per_repetition.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(perfold[0]));w.writeheader();w.writerows(perfold)
print((OUT/'main_table_filtered.tex').read_text())

"""Match scores to missing-species categories using reaction IDs.

Read benchmark scores by default, or use --metric v1 for historical scores.
The category definitions are the same for both metrics. Accuracy labels come
from the selected score files, not older columns in the prediction files.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from utils.result_paths import METRICS, metric_name, results_root, verify_score_release
from utils.scoring import provenance
CATEGORIES=["Helper only","Non-carbon only","Carbon only","Helper + non-carbon",
            "Helper + carbon","Non-carbon + carbon","Helper + non-carbon + carbon",
            "No missing molecules","Unclassified"]


def require_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    duplicated = frame[column].astype(str).duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, column].astype(str).head(5).tolist()
        raise ValueError(f"Duplicate {column} values in {label}: {examples}")

def aggregate(
    rows: pd.DataFrame, stratum: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model", "split_type", "rep", stratum]
    per_rep = (
        rows.groupby(keys, sort=False)
        .agg(
            n=("Unique_identifier", "nunique"),
            n_rows=("Unique_identifier", "size"),
            exact_correct=("exact_correct", "sum"),
            deep_pragmatic_correct=("deep_pragmatic_correct", "sum"),
        )
        .reset_index()
    )
    if not per_rep["n"].equals(per_rep["n_rows"]):
        raise ValueError("A UID occurs more than once within a model/split/rep/category")
    per_rep["exact_accuracy_pct"] = 100 * per_rep.exact_correct / per_rep.n
    per_rep["deep_pragmatic_accuracy_pct"] = (
        100 * per_rep.deep_pragmatic_correct / per_rep.n
    )
    summary = (
        per_rep.groupby(["model", "split_type", stratum], sort=False)
        .agg(
            n_reps=("rep", "nunique"),
            mean_n=("n", "mean"),
            min_n=("n", "min"),
            max_n=("n", "max"),
            exact_mean_pct=("exact_accuracy_pct", "mean"),
            exact_std_pct=("exact_accuracy_pct", "std"),
            deep_mean_pct=("deep_pragmatic_accuracy_pct", "mean"),
            deep_std_pct=("deep_pragmatic_accuracy_pct", "std"),
        )
        .reset_index()
    )
    return per_rep, summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric',type=metric_name,choices=METRICS,default='v2')
    parser.add_argument('--scores-dir',type=Path)
    parser.add_argument('--categories',type=Path,default=ROOT/'results/additional/missing_species_results/reaction_categories.csv')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--split-types',nargs='+',choices=['random','group','extreme_ood'],default=['random','group','extreme_ood'])
    parser.add_argument('--reps',nargs='+',type=int,choices=range(5),default=list(range(5)))
    args=parser.parse_args()
    scores=args.scores_dir or results_root(args.metric)/'main/scores/per_reaction'
    output=args.output_dir or results_root(args.metric)/'additional/missing_species_results'
    verify_score_release(scores,args.metric)
    if (output/'uid_model_outcomes.csv').exists():
        raise FileExistsError('Preserve existing outcomes; choose a new --output-dir')
    categories=pd.read_csv(args.categories,usecols=['Unique_identifier','missing_category','n_missing'])
    categories['Unique_identifier']=categories['Unique_identifier'].astype(str)
    require_unique(categories,'Unique_identifier','categories')
    frames=[];inputs=[Path(__file__),args.categories]
    for split in args.split_types:
        for rep in args.reps:
            fold=f'random_seed{42+rep}_rep{rep}' if split=='random' else f'{split}_rep{rep}'
            support=None
            for model in ('RB','CRB','SynRBL'):
                path=scores/f'{model}_{fold}.csv';inputs.append(path)
                frame=pd.read_csv(path)
                if not frame['retained'].eq(1).all():raise ValueError('Expected filtered scores')
                for column in ('exact','equivalence','balanced'):
                    if not frame[column].isin([0,1]).all():raise ValueError('Invalid binary score')
                if ((frame['exact'].eq(1)|frame['equivalence'].eq(1))&frame['balanced'].ne(1)).any():
                    raise ValueError('Accuracy requires balance')
                frame['Unique_identifier']=frame['Unique_identifier'].astype(str)
                require_unique(frame,'Unique_identifier',fold+model)
                current=frame['Unique_identifier'].tolist()
                if support is not None and current!=support:raise ValueError('Model populations differ')
                support=current
                frame=frame.merge(categories,on='Unique_identifier',how='left',validate='one_to_one')
                if frame['missing_category'].isna().any():raise ValueError('Missing category for score UID')
                frame['model'],frame['split_type'],frame['rep']=model,split,rep
                frame['exact_correct']=frame['exact'].astype(bool)
                frame['deep_pragmatic_correct']=frame['equivalence'].astype(bool)
                frames.append(frame)
    rows=pd.concat(frames,ignore_index=True)
    rows['n_missing']=pd.to_numeric(rows['n_missing'],errors='raise').astype(int)
    rows['missing_count_bin']=rows['n_missing'].astype(str)
    rows.loc[rows.n_missing.ge(7),'missing_count_bin']='7+'
    per_rep,summary=aggregate(rows,'missing_category')
    count_per_rep,count_summary=aggregate(rows,'missing_count_bin')
    outputs={'uid_model_outcomes.csv':rows,'uid_accuracy_per_rep.csv':per_rep,
             'uid_accuracy_summary.csv':summary,'uid_accuracy_by_missing_count_per_rep.csv':count_per_rep,
             'uid_accuracy_by_missing_count_summary.csv':count_summary}
    output.mkdir(parents=True,exist_ok=True)
    for filename,frame in outputs.items():
        frame=frame.rename(columns={c:c.replace('deep_pragmatic','equivalence').replace('deep_','equivalence_') for c in frame.columns})
        frame['scoring_metric']=args.metric
        frame.to_csv(output/filename,index=False)
    provenance(output,args.metric,inputs,population='filtered',join='Unique_identifier')
    print(f'Validated and joined {len(rows):,} model outcomes; wrote {output}')

if __name__=='__main__':main()

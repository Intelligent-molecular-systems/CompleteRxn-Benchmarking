"""Summarize random-split template errors from the selected filtered score release."""
import argparse
from pathlib import Path
import sys
from typing import Dict
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from utils.result_paths import METRICS, metric_name, results_root, verify_score_release


def _normalize_template(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    invalid = s.str.lower().isin({"", "nan", "none", "null"})
    s[invalid] = pd.NA
    return s

def summarize_model(model_name: str, model_rows: pd.DataFrame) -> Dict[str, float]:
    if model_rows.empty:
        return {
            "model": model_name,
            "n_templates": 0,
            "n_perfect_templates": 0,
            "frac_perfect_templates": np.nan,
            "n_failing_templates": 0,
            "total_errors": 0,
            "n_templates_for_50pct_errors": 0,
            "n_templates_for_90pct_errors": 0,
            "frac_templates_for_50pct_errors": np.nan,
            "frac_templates_for_90pct_errors": np.nan,
        }

    per_template = (
        model_rows.groupby("template_label", as_index=False)
        .agg(
            n_samples_total=("metric", "size"),
            n_correct_total=("metric", "sum"),
        )
        .sort_values("n_samples_total", ascending=False)
    )
    per_template["accuracy_weighted"] = per_template["n_correct_total"] / per_template["n_samples_total"]
    per_template["n_errors_total"] = per_template["n_samples_total"] - per_template["n_correct_total"]

    n_templates = int(len(per_template))
    n_perfect = int((per_template["accuracy_weighted"] == 1.0).sum())

    fail = per_template[per_template["n_errors_total"] > 0].copy().sort_values("n_errors_total", ascending=False)
    total_errors = int(fail["n_errors_total"].sum())

    n_50 = 0
    n_90 = 0
    if total_errors > 0 and not fail.empty:
        cum = fail["n_errors_total"].cumsum().to_numpy(dtype=float)
        frac = cum / float(total_errors)
        n_50 = int(np.searchsorted(frac, 0.5, side="left") + 1)
        n_90 = int(np.searchsorted(frac, 0.9, side="left") + 1)

    return {
        "model": model_name,
        "n_templates": n_templates,
        "n_perfect_templates": n_perfect,
        "frac_perfect_templates": float(n_perfect / n_templates) if n_templates else np.nan,
        "n_failing_templates": int(len(fail)),
        "total_errors": total_errors,
        "n_templates_for_50pct_errors": n_50,
        "n_templates_for_90pct_errors": n_90,
        "frac_templates_for_50pct_errors": float(n_50 / n_templates) if n_templates else np.nan,
        "frac_templates_for_90pct_errors": float(n_90 / n_templates) if n_templates else np.nan,
    }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric',type=metric_name,choices=METRICS,default='v2')
    parser.add_argument('--scores-dir',type=Path)
    parser.add_argument('--benchmark-csv',type=Path,default=ROOT/'data/Template_curated/benchmark/benchmark_final.csv')
    parser.add_argument('--template-col',default='1kTPLLabel')
    parser.add_argument('--output-csv',type=Path)
    args=parser.parse_args()
    scores=args.scores_dir or results_root(args.metric)/'main/scores/per_reaction'
    output=args.output_csv or results_root(args.metric)/'additional/templates/random_template_error_concentration.csv'
    verify_score_release(scores,args.metric)
    if output.exists():raise FileExistsError('Choose a new --output-csv')
    benchmark=pd.read_csv(args.benchmark_csv,usecols=['Unique_identifier',args.template_col])
    frames={model:[] for model in ('CRB','SynRBL')}
    for rep in range(5):
        support=None
        for model in frames:
            frame=pd.read_csv(scores/f'{model}_random_seed{42+rep}_rep{rep}.csv')
            if not frame['retained'].eq(1).all():raise ValueError('Expected filtered scores')
            uids=frame['Unique_identifier'].tolist()
            if support is not None and uids!=support:raise ValueError('Model populations differ')
            support=uids
            frame=frame.merge(benchmark,on='Unique_identifier',validate='one_to_one')
            frame['template_label']=_normalize_template(frame[args.template_col])
            frame['metric']=frame['equivalence'].astype(bool)
            frames[model].append(frame[frame['template_label'].notna()])
    summary=pd.DataFrame([summarize_model(model,pd.concat(rows,ignore_index=True))
                          for model,rows in frames.items()])
    output.parent.mkdir(parents=True,exist_ok=True)
    summary.to_csv(output,index=False)
    print('Wrote',output)

if __name__=='__main__':main()

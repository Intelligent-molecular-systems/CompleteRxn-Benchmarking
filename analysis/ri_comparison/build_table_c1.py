"""Recalculate SynRBL scores with and without reagent identification on the same test reactions."""
import argparse
import csv
import importlib.util
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from utils.result_paths import metric_name, ensure_v1_seed
from utils.scoring import METRICS, labels, results_root, provenance, verify_score_release
_METRIC = "v2"
INPUTS = []
OUT = ROOT / "results/additional/ri_comparison"
RI = ROOT / 'results/SynRBL/eval_all_splits_CanonicalReaction/full_results.csv'
NO_RI = ROOT / 'results/main/predictions/synrbl/eval_all_splits_noRI_complete/full_results.csv'
SCORES = ROOT / 'results/main/scores/per_reaction'
_RI = None
_SCORER = None


def read(path):
    INPUTS.append(path)
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flag(value):
    return str(value).strip().lower() in ('1', 'true', 't', 'yes')


def score_ri(uid):
    source = _RI[uid]
    balanced = bool(_SCORER.is_balanced_reaction(source['Reaction']))
    exact = equivalent = False
    if flag(source['solved']) and balanced:
        try:
            exact, equivalent = labels(source['Reaction'], source['rxn_flower'], metric=_METRIC, stereo=False)
        except Exception:
            pass
    return uid, (int(exact), int(equivalent), int(balanced))


def main():
    global OUT, SCORES, _METRIC
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric",type=metric_name, choices=METRICS,default="v2")
    parser.add_argument("--scores-dir",type=Path)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    ensure_v1_seed(args.metric);_METRIC=args.metric
    SCORES=args.scores_dir or results_root(args.metric)/"main/scores/per_reaction"
    OUT=args.output_dir or results_root(args.metric)/"additional/ri_comparison"
    if (OUT/"table_c1_paired.csv").exists(): parser.error("Choose a new output directory")
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('scorer', ROOT / 'models/constrained_reaction_balancer/score_predictions.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    verify_score_release(SCORES,args.metric)
    ri_rows = read(RI)
    no_ri_rows = read(NO_RI)
    ri = {r['Unique_identifier']: r for r in ri_rows if r['Unique_identifier']}
    no_ri = {r['Unique_identifier']: r for r in no_ri_rows if r['Unique_identifier']}
    assert len(ri) == sum(bool(r['Unique_identifier']) for r in ri_rows)
    assert len(no_ri) == len(no_ri_rows) == 206423
    assert all(uid in no_ri for uid in ri)
    global _RI, _SCORER
    _RI, _SCORER = ri, scorer
    rescored = {}
    observations = []
    per_rep = []
    with Pool(processes=8) as pool:
      for split in ('random', 'group', 'extreme_ood'):
       for rep in range(5):
            fold = f'random_seed{42 + rep}_rep{rep}' if split == 'random' else f'{split}_rep{rep}'
            saved = read(SCORES / f'SynRBL_{fold}.csv')
            assert all(row['retained'] == '1' for row in saved)
            common = [row for row in saved if row['Unique_identifier'] in ri]
            assert len(common) == len({row['Unique_identifier'] for row in common})
            print(f'{fold}: paired {len(common)}/{len(saved)}', flush=True)
            pending = [row['Unique_identifier'] for row in common if row['Unique_identifier'] not in rescored]
            for count, (uid, values) in enumerate(pool.imap_unordered(score_ri, pending, chunksize=64), 1):
                rescored[uid] = values
                if count % 5000 == 0:
                    print(f'{fold}: rescored {count}/{len(pending)} new RI reactions', flush=True)
            for method in ('RI', 'noRI'):
                for row in common:
                    uid = row['Unique_identifier']
                    source = ri[uid] if method == 'RI' else no_ri[uid]
                    solved = flag(source['solved'])
                    if method == 'RI':
                        exact, equivalent, balanced = rescored[uid]
                    else:
                        exact, equivalent, balanced = (int(row[k]) for k in ('exact', 'equivalence', 'balanced'))
                    assert not (exact or equivalent) or balanced
                    confident = int(solved and source['solved_by'].strip().lower() == 'mcs-based' and float(source['confidence'] or 0) >= .5)
                    observations.append(dict(split=split, rep=rep, method=method, Unique_identifier=uid,
                                             exact=exact, equivalence=equivalent, balanced=balanced,
                                             confident_mcs=confident))
                subset = observations[-len(common):]
                n = len(common)
                per_rep.append(dict(split=split, rep=rep, method=method, n=n,
                                    **{key + '_pct': 100 * sum(int(x[key]) for x in subset) / n
                                       for key in ('exact', 'equivalence', 'balanced', 'confident_mcs')}))
    write(OUT / 'table_c1_paired_per_repetition.csv', per_rep)
    write(OUT / 'table_c1_paired_per_observation.csv', observations)
    table = []
    tex = [r'\hline']
    for method in ('noRI', 'RI'):
      for split, label in (('random', 'R'), ('group', 'G'), ('extreme_ood', 'OOD')):
            rows = [r for r in per_rep if r['split'] == split and r['method'] == method]
            assert len(rows) == 5
            n = sum(int(r['n']) for r in rows)
            values = {}
            for key in ('exact', 'equivalence', 'confident_mcs', 'balanced'):
                percentages = [float(r[key + '_pct']) for r in rows]
                values[key] = f'{statistics.mean(percentages):.2f} ± {statistics.stdev(percentages):.2f}'
            table.append(dict(split=label, method=method, n=n, **values))
            cells = [values[k].replace('±', r'$\pm$') for k in ('exact', 'equivalence', 'confident_mcs', 'balanced')]
            tex.append(f'SynRBL ({method}) & {label} & ' + ' & '.join(cells) + r' \\')
    tex.append(r'\hline')
    write(OUT / 'table_c1_paired.csv', table)
    (OUT / 'table_c1_paired.tex').write_text('\n'.join(tex) + '\n')
    provenance(OUT,args.metric,INPUTS+[Path(__file__)],stereo=False,requires_solved=True,paired=True)
    print('Wrote', OUT, flush=True)


if __name__ == '__main__':
    main()

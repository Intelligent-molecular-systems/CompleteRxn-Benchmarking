"""Recalculate the main benchmark scores from saved predictions. Run in the CRB container."""
import csv
import io
import tarfile
import importlib.util
import argparse
import hashlib
from functools import lru_cache
import re
import statistics
import sys
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/main/scores"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('existing_scorer', str(ROOT / 'models/constrained_reaction_balancer/score_predictions.py'))
from utils.result_paths import metric_name, ensure_v1_seed
from utils.scoring import METRICS, labels, results_root, provenance
SCORING_METRIC = 'v2'
FULL = False
INPUTS = []
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def read(path, sep=','):
    INPUTS.append(path)
    with path.open(newline='') as f:
        return list(csv.DictReader(f, delimiter=sep))


def write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def flag(value):
    assert str(value).lower() in {'true', 'false', '1', '0', '', 'nan'}, value
    return str(value).lower() in {'true', '1'}


def lines(path):
    INPUTS.append(path)
    return [''.join(s.split()) for s in path.read_text().splitlines()]


@lru_cache(maxsize=250000)
def score_ml(inp, target, pred, metric="v2"):
    metric = metric_name(metric)
    # Same independent balance requirement as the restored repository scorer.
    balanced = scorer.is_balanced_reaction(scorer.compose_completed_reaction(inp, pred))
    if not balanced:
        return False, False, False
    attr = 'deep_pragmatic_equivalence_match' if metric == 'v1' else 'equivalent_match'
    ranks = scorer.get_ranks_rich({'input': inp, 'target': target, 'prediction_1': pred}, 'prediction_', 1, ['exact_match', attr], metric)
    return ranks['exact_match'] == 1, ranks[attr] == 1, balanced



def released_scores(model, fold):
    folder=results_root('v1')/('additional/unfiltered_evaluation' if FULL else 'main/scores')
    path=folder/'per_reaction'/f'{model}_{fold}.csv'
    if path.exists():
        rows=read(path)
    elif not FULL and (folder/'filtered_scores.tar.gz').exists():
        archive=folder/'filtered_scores.tar.gz';INPUTS.append(archive)
        with tarfile.open(archive) as bundle:
            rows=list(csv.DictReader(io.TextIOWrapper(bundle.extractfile(path.name))))
    elif FULL:
        return {}  # Optional old full-test labels are not required for a fresh rescore.
    else:
        raise ValueError('Missing released baseline scores: '+str(path))
    indexed={r['Unique_identifier']:r for r in rows}
    assert len(indexed)==len(rows), ('Duplicate released score UID',path)
    return indexed


def main():
    bench = ROOT / 'data/Template_curated/benchmark'
    order = [r['Unique_identifier'] for r in read(bench / 'benchmark_final.csv')]
    syn_rows = read(ROOT / 'results/main/predictions/synrbl/eval_all_splits_noRI_complete/full_results.csv')
    syn = {r['Unique_identifier']: r for r in syn_rows}
    assert len(syn) == len(syn_rows) == len(order) == 206423
    del syn_rows
    summaries = []
    audit = []
    drift = []
    syn_cache = {}
    mlroot = ROOT / 'models/constrained_reaction_balancer'
    if (OUT / 'per_repetition.csv').exists():
        raise ValueError('Choose a new --output-dir; this score release already exists')
    (OUT / 'per_reaction').mkdir(parents=True, exist_ok=True)
    for split, col in [('random', 'random_split'), ('group', 'group_split'), ('extreme_ood', 'extreme_ood')]:
        if split not in args.split_types.split(','): continue
        paths = [p for p in sorted((bench / (split + '_splits')).glob('*.csv')) if re.search(r'rep\d+', p.name) and not p.name.endswith('_meta.csv')]
        assert len(paths) == 5
        for path in paths:
            rep = int(re.search(r'rep(\d+)', path.name)[1])
            if rep not in [int(x) for x in args.reps.split(',')]: continue
            ids = {r['Unique_identifier'] for r in read(path) if r[col] == '2'}
            uids = [u for u in order if u in ids]
            fold = '{}_rep{}'.format(split, rep)
            if split == 'random':
                fold = 'random_seed{}_rep{}'.format(re.search(r'seed(\d+)', path.name)[1], rep)
            prepared = mlroot / 'Template_curated/benchmark/prepared' / fold
            inputs, targets = lines(prepared / 'test_src.txt'), lines(prepared / 'test_tgt.txt')
            assert len(inputs) == len(targets) == len(uids)
            common_indices = None
            for model, directory in [('RB', 'BS_PT'), ('CRB', 'CBS_PT')]:
                base = ROOT / 'results/main/predictions' / ('rb' if model == 'RB' else 'crb') / split / fold
                predictions = lines(base / 'beamSize5_nBest5.txt')
                assert len(predictions) == 5 * len(uids), (model, fold, len(predictions), len(uids))
                top1 = predictions[::5]
                old_rows = read(base / 'top1_predictions_with_stats.txt', '\t')
                old = {int(r['original_index']): r for r in old_rows}
                assert len(old) == len(old_rows)
                if common_indices is None:
                    common_indices = set(old)
                else:
                    assert common_indices == set(old), ('different RB/CRB populations', fold)
                for idx, r in old.items():
                    assert top1[idx] == ''.join(r['top1_prediction'].split()), (model, fold, idx)
                    assert r['selected_metric'] == 'deep_pragmatic_equivalence'
                released=released_scores(model,fold)
                rows = []
                for idx, (uid, inp, target, pred) in enumerate(zip(uids, inputs, targets, top1)):
                    retained = idx in old
                    if not retained and not FULL:
                        continue
                    exact, equiv, balanced = score_ml(inp, target, pred, SCORING_METRIC)
                    if (idx+1)%5000==0: print(model,fold,idx+1,'/',len(uids),flush=True)
                    if retained and uid in released:
                        for metric, value, old_col in [('exact',exact,'matches_exact'),('equivalence',equiv,'matches_selected_metric'),('balanced',balanced,'is_balanced')]:
                            if value != flag(released[uid][metric]):
                                drift.append(dict(model=model, split=split, rep=rep, Unique_identifier=uid, metric=metric, historical=int(flag(released[uid][metric])), rescored=int(value)))
                    rows.append(dict(Unique_identifier=uid, original_index=idx, retained=int(retained), exact=int(exact), equivalence=int(equiv), balanced=int(balanced)))
                write(OUT / 'per_reaction' / '{}_{}.csv'.format(model, fold), rows)
                summarize(summaries, model, split, rep, rows)
                print('{} {}: scored {} retained reactions'.format(model, fold, len(rows)), flush=True)
            released=released_scores('SynRBL',fold)
            rows = []
            for idx, uid in enumerate(uids):
                if idx not in common_indices and not FULL:
                    continue
                r = syn[uid]
                solved = flag(r['solved'])
                balanced = scorer.is_balanced_reaction(r['Reaction'])
                # Recompute with the current metric and preserve the original reference convention.
                if uid not in syn_cache:
                    exact = equiv = False
                    if solved and balanced:
                        try:
                            exact, equiv = labels(r['Reaction'], r['rxn_flower'], metric=SCORING_METRIC, stereo=False)
                        except ValueError:
                            pass  # Invalid predictions remain incorrect and in the denominator.
                    syn_cache[uid] = (exact, equiv)
                exact, equiv = syn_cache[uid]
                for metric, value, old_col in [('exact',exact,'exact_match'),('equivalence',equiv,'deep_pragmatic_equivalence_match')]:
                    if uid not in released: continue
                    historical = flag(released[uid][metric])
                    if value != historical:
                        drift.append(dict(model='SynRBL', split=split, rep=rep, Unique_identifier=uid, metric=metric, historical=int(historical), rescored=int(value)))
                rows.append(dict(Unique_identifier=uid, original_index=idx, retained=int(idx in common_indices), exact=int(exact), equivalence=int(equiv), balanced=int(balanced)))
            write(OUT / 'per_reaction' / 'SynRBL_{}.csv'.format(fold), rows)
            summarize(summaries, 'SynRBL', split, rep, rows)
            audit.append(dict(split=split, rep=rep, full_n=len(uids), retained_n=len(common_indices), excluded_n=len(uids)-len(common_indices)))
    if drift:
        write(OUT / 'historical_score_changes.csv', drift)
    else:
        (OUT / 'historical_score_changes.csv').write_text('model,split,rep,Unique_identifier,metric,historical,rescored\n')
    if SCORING_METRIC == "v1":
        (OUT / 'metric_sha256.txt').write_text(''.join(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() + '  ' + path + '\n' for path in [('src/completerxn/legacy/equivalence_v1.py' if SCORING_METRIC == 'v1' else 'src/completerxn/equivalence.py'), 'utils/scoring.py', 'utils/reaction_balance.py', 'models/constrained_reaction_balancer/score_predictions.py', 'analysis/score_benchmark.py']))
    write(OUT / 'per_repetition.csv', summaries)
    write(OUT / 'coverage.csv', audit)
    aggregate = aggregate_repetitions(summaries)
    write(OUT / 'summary.csv', aggregate)
    provenance(OUT, SCORING_METRIC, INPUTS + [Path(__file__)],
               population='full' if FULL else 'filtered', splits=args.split_types, reps=args.reps,
               historical_comparison='available released labels, not raw prediction flags',
               crb_reference_stereo=True, synrbl_reference_stereo=False,
               synrbl_requires_solved=True, accuracy_requires_balance=True)
    print('All checks passed; wrote versioned CSVs.', flush=True)



def aggregate_repetitions(summaries):
    output=[]
    groups=sorted({(r['population'],r['split'],r['model']) for r in summaries})
    for population,split,model in groups:
        group=[r for r in summaries if (r['population'],r['split'],r['model'])==(population,split,model)]
        row=dict(population=population,split=split,model=model,
                 n_observations=sum(int(r['n']) for r in group))
        for metric in ('exact_pct','equivalence_pct','balanced_pct'):
            values=[float(r[metric]) for r in group]
            row[metric+'_mean']=statistics.mean(values)
            row[metric+'_sd']=statistics.stdev(values) if len(values)>1 else None
        output.append(row)
    return output


def summarize(output, model, split, rep, rows):
    assert all(not (r['exact'] or r['equivalence']) or r['balanced'] for r in rows), (model, split, rep, 'accuracy awarded to an unbalanced prediction')
    for population in (['full','retained','excluded'] if FULL else ['retained']):
        selected=[r for r in rows if population=='full' or r['retained']==int(population=='retained')]
        if not selected: continue
        row=dict(model=model, split=split, rep=rep, population=population,n=len(selected))
        for metric in ['exact','equivalence','balanced']:
            row[metric+'_pct']=100*sum(r[metric] for r in selected)/len(selected)
        output.append(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true', help='Check scoring and the known historical discrepancy; do not generate result tables.')
    parser.add_argument('--metric', type=metric_name, choices=METRICS, default='v2')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--population', choices=['filtered','full'], default='filtered')
    parser.add_argument('--split-types', default='random,group,extreme_ood')
    parser.add_argument('--reps', default='0,1,2,3,4')
    args = parser.parse_args()
    ensure_v1_seed(args.metric)
    if not set(args.split_types.split(',')) <= {'random','group','extreme_ood'} or not set(args.reps.split(',')) <= set('01234'):
        parser.error('Invalid split types or repetitions')
    SCORING_METRIC, FULL = args.metric, args.population == 'full'
    OUT = args.output_dir or (results_root(args.metric) / ('additional/unfiltered_evaluation' if FULL else 'main/scores'))
    if args.smoke_test:
        base = ROOT / 'models/constrained_reaction_balancer'
        prep = base / 'Template_curated/benchmark/prepared/group_rep0'
        inputs, targets = lines(prep / 'test_src.txt'), lines(prep / 'test_tgt.txt')
        pred = lines(ROOT / 'results/main/predictions/rb/group/group_rep0/beamSize5_nBest5.txt')[::5]
        assert len(inputs) == len(targets) == len(pred)
        for idx in [0, 1, len(inputs)-1]:
            print('group_rep0 RB index', idx, 'exact/equivalence/balanced:', score_ml(inputs[idx], targets[idx], pred[idx], SCORING_METRIC))
        assert score_ml(inputs[1], targets[1], pred[1], SCORING_METRIC) == (False, False, False)
        print('Smoke test passed. Full run requires omitting --smoke-test.')
    else:
        main()

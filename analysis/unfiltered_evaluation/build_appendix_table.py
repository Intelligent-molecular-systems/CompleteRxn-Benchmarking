"""Build the appendix table from scores on the full test sets."""
import argparse
import csv
import importlib.util
import statistics
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/additional/unfiltered_evaluation"


def read(path, delimiter=','):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir",type=Path,default=OUT)
    args=parser.parse_args();OUT=args.scores_dir
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        'scorer', ROOT / 'models/constrained_reaction_balancer/score_predictions.py'
    )
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)

    @lru_cache(maxsize=None)
    def invalid(prediction):
        return scorer.canonicalize_smiles(prediction) == ''

    syn = {
        row['Unique_identifier']: row
        for row in read(ROOT / 'results/main/predictions/synrbl/eval_all_splits_noRI_complete/full_results.csv')
    }
    summary = read(OUT / 'summary.csv')
    per_rep = read(OUT / 'per_repetition.csv')
    auxiliary = []
    for split in ('random', 'group', 'extreme_ood'):
        for model, directory in (('RB', 'BS_PT'), ('CRB', 'CBS_PT'), ('SynRBL', None)):
            for rep in range(5):
                fold = f'random_seed{42 + rep}_rep{rep}' if split == 'random' else f'{split}_rep{rep}'
                scores = read(OUT / 'per_reaction' / f'{model}_{fold}.csv')
                n = len(scores)
                expected = next(row for row in per_rep if (row['population'], row['split'], row['model'], row['rep']) == ('full', split, model, str(rep)))
                assert n == int(expected['n'])
                assert len({row['Unique_identifier'] for row in scores}) == n
                if directory:
                    raw = (ROOT / 'results/main/predictions' / ('rb' if directory == 'BS_PT' else 'crb') / split / fold / 'beamSize5_nBest5.txt').read_text().splitlines()
                    assert len(raw) == 5 * n
                    top1 = raw[::5]
                    assert {int(row['original_index']) for row in scores} == set(range(n))
                    count = sum(invalid(''.join(top1[int(row['original_index'])].split())) for row in scores)
                    metric = 'invalid_smiles'
                else:
                    assert all(row['Unique_identifier'] in syn for row in scores)
                    count = sum(
                        syn[row['Unique_identifier']]['solved'].lower() == 'true'
                        and syn[row['Unique_identifier']]['solved_by'].lower() == 'mcs-based'
                        and float(syn[row['Unique_identifier']]['confidence'] or 0) >= .5
                        for row in scores
                    )
                    metric = 'confident_mcs'
                auxiliary.append(dict(model=model, split=split, rep=rep, n=n, count=count, metric=metric, pct=100 * count / n))
    write_csv(OUT / 'appendix_table_unfiltered_auxiliary_per_repetition.csv', auxiliary)

    def fmt(mean, sd):
        return f'{mean:.2f} ± {sd:.2f}'

    table = []
    tex = []
    for split, label in (('random', 'R'), ('group', 'G'), ('extreme_ood', 'OOD')):
        tex.extend(('\\hline', '', label))
        for model in ('RB', 'CRB', 'SynRBL'):
            selected = [row for row in auxiliary if row['split'] == split and row['model'] == model]
            assert len(selected) == 5
            percentages = [100 * int(row['count']) / int(row['n']) for row in selected]
            aux = fmt(statistics.mean(percentages), statistics.stdev(percentages))
            source = next(row for row in summary if (row['population'], row['split'], row['model']) == ('full', split, model))
            assert int(source['n_observations']) == sum(int(row['n']) for row in selected)
            exact = fmt(float(source['exact_pct_mean']), float(source['exact_pct_sd']))
            equivalence = fmt(float(source['equivalence_pct_mean']), float(source['equivalence_pct_sd']))
            balanced = fmt(float(source['balanced_pct_mean']), float(source['balanced_pct_sd']))
            confidence, invalid_smiles = (aux, '-') if model == 'SynRBL' else ('-', aux)
            row = dict(Split=label, Model=model, Exact=exact, Equivalence=equivalence,
                       Confidence=confidence, Invalid_SMILES=invalid_smiles, Balanced=balanced)
            table.append(row)
            cells = [exact, equivalence, confidence, invalid_smiles, balanced]
            cells = [cell.replace('±', '$\\pm$') for cell in cells]
            if model == 'CRB':
                cells[1] = '\\textbf{' + cells[1] + '}'
            tex.append('& ' + model + ' & ' + ' & '.join(cells) + ' \\\\')
        tex.append('')
    tex.append('\\hline')
    (OUT / 'appendix_table_unfiltered.tex').write_text('\n'.join(tex) + '\n')
    write_csv(OUT / 'appendix_table_unfiltered.csv', table)
    md = [
        '# Appendix: unfiltered benchmark results', '',
        'All entries are percentages (mean ± sample SD over five repetitions). All models use identical full test sets. Accuracy requires independent atom and charge balance. Confidence is the solved MCS-based fraction at confidence ≥0.5, not accuracy. Invalid SMILES uses the historical top-1 predicted-completion parsing definition. Repetition-weighted observation counts are R: 103,215; G: 105,078; OOD: 7,457.', '',
        '| Split | Model | Exact | Equivalence | Confidence | Invalid SMILES | Balanced |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    md.extend('| ' + ' | '.join(row.values()) + ' |' for row in table)
    (OUT / 'appendix_table_unfiltered.md').write_text('\n'.join(md) + '\n')
    print('Wrote results/additional/unfiltered_evaluation/appendix_table_unfiltered.{tex,csv,md}')


if __name__ == '__main__':
    main()

"""Count CRB errors that are caused by stereochemistry alone.

RB and CRB are scored against the reference with stereochemistry, SynRBL against
the reference without it. This script checks how much the stricter evaluation
affects CRB: it rescores every balanced but incorrect top-1 prediction with the
same metric while ignoring stereochemistry, and counts the ones that then match.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/constrained_reaction_balancer'))
import score_predictions as scorer
from utils.scoring import compare, provenance

RDLogger.DisableLog('rdApp.*')
PREPARED = ROOT / 'models/constrained_reaction_balancer/Template_curated/benchmark/prepared'
SPLITS = ['random', 'group', 'extreme_ood']


def lines(path):
    return [''.join(s.split()) for s in Path(path).read_text().splitlines()]


def stereo_marks(side):
    """Count specified stereocenters and stereo double bonds in one reaction side."""
    count = 0
    for smiles in side.split('.'):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        count += sum(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
        count += sum(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/additional/stereo_errors')
    args = parser.parse_args()
    if (args.output_dir / 'summary.csv').exists():
        raise FileExistsError('Choose a new --output-dir; these results already exist')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, cases, inputs = [], [], [Path(__file__)]
    for split in SPLITS:
        totals = Counter()
        for rep in range(5):
            fold = f'random_seed{42 + rep}_rep{rep}' if split == 'random' else f'{split}_rep{rep}'
            paths = [PREPARED / fold / 'test_src.txt', PREPARED / fold / 'test_tgt.txt',
                     ROOT / f'results/main/predictions/crb/{split}/{fold}/beamSize5_nBest5.txt',
                     ROOT / f'results/main/scores/per_reaction/CRB_{fold}.csv']
            inputs += paths
            sources, targets = lines(paths[0]), lines(paths[1])
            top1 = lines(paths[2])[::5]
            with open(paths[3], newline='') as handle:
                for row in csv.DictReader(handle):
                    totals['reactions'] += 1
                    if row['equivalence'] == '1':
                        continue
                    totals['incorrect'] += 1
                    if row['balanced'] != '1':
                        continue
                    totals['balanced_incorrect'] += 1
                    index = int(row['original_index'])
                    prediction = scorer.compose_completed_reaction(sources[index], top1[index])
                    reference = scorer.compose_completed_reaction(sources[index], targets[index])
                    try:
                        match = compare(prediction, reference, metric='v2', stereo=False).equivalent_match
                    except ValueError:
                        match = False
                    if not match:
                        continue
                    totals['stereo_only'] += 1
                    predicted = stereo_marks(prediction.split('>>')[-1])
                    expected = stereo_marks(reference.split('>>')[-1])
                    direction = ('prediction has more stereo' if predicted > expected else
                                 'prediction has less stereo' if predicted < expected else
                                 'same amount, different assignment')
                    totals[direction] += 1
                    cases.append(dict(split=split, rep=rep, Unique_identifier=row['Unique_identifier'],
                                      direction=direction, prediction=prediction, reference=reference))
        summary.append(dict(
            split=split, reactions=totals['reactions'], incorrect=totals['incorrect'],
            balanced_incorrect=totals['balanced_incorrect'], stereo_only=totals['stereo_only'],
            stereo_only_pct_of_incorrect=100 * totals['stereo_only'] / totals['incorrect'],
            stereo_only_pct_of_balanced_incorrect=100 * totals['stereo_only'] / totals['balanced_incorrect'],
            stereo_only_pct_of_reactions=100 * totals['stereo_only'] / totals['reactions'],
            more_stereo=totals['prediction has more stereo'], less_stereo=totals['prediction has less stereo'],
            different_assignment=totals['same amount, different assignment']))
        print(f"{split}: {totals['stereo_only']} of {totals['incorrect']} CRB errors are stereo-only")

    for name, rows in [('summary.csv', summary), ('cases.csv', cases)]:
        with open(args.output_dir / name, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    provenance(args.output_dir, 'v2', inputs, population='filtered', reference='TargetReaction_with_stereo',
               rescored_with_stereo=False)
    print(f'Wrote {args.output_dir}')


if __name__ == '__main__':
    main()

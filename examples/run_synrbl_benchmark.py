"""Benchmark SynRBL on the filtered test sets, as in "Benchmark on all test sets" in the README.

SynRBL needs no training, so only the test sets are loaded. Requires `pip install synrbl==1.0.5`.
Run from the repository root:

    python examples/run_synrbl_benchmark.py --output-dir results/local/synrbl_benchmark --n-jobs 8
    python examples/run_synrbl_benchmark.py --output-dir results/local/synrbl_ood --splits extreme_ood --reps 0 1

The full run (3 splits x 5 repetitions) takes a while; use --splits and --reps for a subset.
Compare the printed means with results/main/scores/main_table_filtered.csv.
"""
import argparse
import statistics
from pathlib import Path

from completerxn import DataLoader
from completerxn.predictors import SynRBLPredictor

SPLITS = ['random', 'group', 'extreme_ood']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=SPLITS)
    parser.add_argument('--reps', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--data-dir', type=Path, help='Use existing local files instead of HF')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a new output directory')

    model = SynRBLPredictor(n_jobs=args.n_jobs)
    loader = DataLoader(data_dir=args.data_dir)
    scores = {}
    for split in args.splits:
        for rep in args.reps:
            test = loader.test(split=split, rep=rep, population='filtered')
            result = test.evaluate(test.predict(model))
            result.save(args.output_dir / f'{split}_rep{rep}')
            scores.setdefault(split, []).append(result.summary['equivalence_pct'])
            print(f'{split} rep{rep}: {scores[split][-1]:.1f} % ({len(test)} reactions)')

    for split, values in scores.items():
        spread = f' ± {statistics.stdev(values):.1f}' if len(values) > 1 else ''
        print(f'{split}: {statistics.mean(values):.1f}{spread} % over {len(values)} repetition(s)')


if __name__ == '__main__':
    main()

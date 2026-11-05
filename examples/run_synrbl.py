"""Complete ten HF random-rep0 test reactions with installed SynRBL, then score."""
import argparse
from pathlib import Path

from completerxn import DataLoader
from completerxn.predictors import SynRBLPredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,help='Use existing local files instead of HF')
    parser.add_argument('--limit',type=int,default=10,help='Test examples; 0 means the full test partition')
    parser.add_argument('--n-jobs',type=int,default=1)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.limit<0:parser.error('--limit must be nonnegative')
    if args.output_dir.exists():parser.error('Choose a new output directory')
    train,val,test=DataLoader(data_dir=args.data_dir).load(split='random',rep=0)
    print(f'Random rep0: train={len(train)}, val={len(val)}, test={len(test)}')
    if args.limit:test=test[:args.limit]
    predictions=test.predict(SynRBLPredictor(n_jobs=args.n_jobs))
    result=test.evaluate(predictions)
    result.save(args.output_dir)
    # Extract one failed reaction and save its plot
    if result.failures:
        result.plot(result.failures[0]['reaction_id']).save(args.output_dir/'failed_reaction.svg')
    print(result.summary)
    print(result.failure_summary())


if __name__=='__main__':main()

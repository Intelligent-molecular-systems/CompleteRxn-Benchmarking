"""Load the HF random split, return inputs unchanged, score, and inspect failures."""
import argparse
from pathlib import Path

from completerxn import DataLoader


class UnchangedPredictor:
    def predict(self, inputs):
        return list(inputs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,help='Use existing local files instead of HF')
    parser.add_argument('--limit',type=int,default=10,help='Test examples; 0 means the full test partition')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.limit<0:parser.error('--limit must be nonnegative')
    if args.output_dir.exists():parser.error('Choose a new output directory')
    train,val,test=DataLoader(data_dir=args.data_dir).load(split='random',rep=0)
    print(f'Random rep0: train={len(train)}, val={len(val)}, test={len(test)}')
    if args.limit:test=test[:args.limit]
    predictions=test.predict(UnchangedPredictor())
    result=test.evaluate(predictions)
    result.save(args.output_dir)
    result.plot_failures().save(args.output_dir/'failure_counts.svg')
    result.plot(test[0].reaction_id).save(args.output_dir/'reaction.svg')
    print(result.summary)
    print(result.failure_summary())


if __name__=='__main__':main()

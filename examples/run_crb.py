"""Use the HF checkpoint fine-tuned on random rep0, predict ten test records, score."""
import argparse
from pathlib import Path
import sys

from completerxn import DataLoader
from completerxn.predictors import CRBPredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-repo',required=True,help='HF model repository containing the random-rep0 checkpoint')
    parser.add_argument('--model-file',required=True,help='Path to the fine-tuned .pt file within that repository')
    parser.add_argument('--model-revision',required=True,help='Full HF model commit SHA')
    parser.add_argument('--translate-script',type=Path,default=Path(__file__).resolve().parents[1]/
                        'models/constrained_reaction_balancer/translate.py')
    parser.add_argument('--container',type=Path,help='Existing CRB Apptainer image; otherwise use this Python environment')
    parser.add_argument('--gpu',type=int,default=-1,help='-1 for CPU, 0 for first GPU')
    parser.add_argument('--data-dir',type=Path,help='Use existing local files instead of HF')
    parser.add_argument('--input-column',default='InputReaction_harmonized',help='Input field (default: InputReaction_harmonized)')
    parser.add_argument('--limit',type=int,default=10,help='Test examples; 0 means the full test partition')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.limit<0:parser.error('--limit must be nonnegative')
    if args.output_dir.exists():parser.error('Choose a new output directory')
    if args.container:
        if not args.container.is_file():parser.error('CRB container does not exist')
        prefix=['apptainer','exec']+(['--nv'] if args.gpu>=0 else [])+[str(args.container.resolve()),'python']
    else:
        prefix=[sys.executable]
    model=CRBPredictor.from_pretrained(args.model_repo,filename=args.model_file,
        revision=args.model_revision,translate_script=args.translate_script,
        command_prefix=prefix,gpu=args.gpu)
    train,val,test=DataLoader(data_dir=args.data_dir,input_column=args.input_column).load(split='random',rep=0)
    print(f'Random rep0: train={len(train)}, val={len(val)}, test={len(test)}')
    if args.limit:test=test[:args.limit]
    predictions=test.predict(model)
    result=test.evaluate(predictions,stereo=True)
    result.save(args.output_dir)
    result.plot_failures().save(args.output_dir/'failure_counts.svg')
    result.plot(test[0].reaction_id).save(args.output_dir/'reaction.svg')
    print(result.summary)
    print(result.failure_summary())


if __name__=='__main__':main()

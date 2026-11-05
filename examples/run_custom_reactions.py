"""Complete your own reactions with one of the models and check that they balance.

The input CSV needs the columns reaction_id and input (incomplete reaction SMILES, with reagents and solvents on the reactant side).
The output predictions.csv uses the benchmark's prediction format, so if you also have references you can score it with `completerxn score`.

Run from the repository root:

    python examples/run_custom_reactions.py --model synrbl --input path/to/my_reactions.csv --output path/to/predictions.csv
    python examples/run_custom_reactions.py --model crb --input path/to/my_reactions.csv --output path/to/predictions.csv

`--model rule-based` runs the simple example model in models/rule_based_balancer.py, which is not part of the paper.
"""
import argparse
import csv
import sys
from pathlib import Path

from completerxn.balance import is_balanced_reaction
from completerxn.predictors import Completion

ROOT = Path(__file__).resolve().parents[1]
CRB_REPO = 'Intelligent-molecular-systems/completeRXN-CRB'


def load_model(name, container, gpu):
    if name == 'rule-based':
        sys.path.insert(0, str(ROOT / 'models'))
        from rule_based_balancer import RuleBasedBalancer
        return RuleBasedBalancer()
    if name == 'synrbl':
        from completerxn.predictors import SynRBLPredictor
        return SynRBLPredictor()
    from huggingface_hub import HfApi
    from completerxn.predictors import CRBPredictor
    if not container.is_file():
        raise SystemExit(f'CRB container not found: {container}. See "Run the small examples" in the README.')
    prefix = ['apptainer', 'exec'] + (['--nv'] if gpu >= 0 else []) + [str(container.resolve()), 'python']
    return CRBPredictor.from_pretrained(
        CRB_REPO, filename='random_seed42_rep0/model_step_50000.pt',
        revision=HfApi().model_info(CRB_REPO).sha,
        translate_script=ROOT / 'models/constrained_reaction_balancer/translate.py',
        command_prefix=prefix, gpu=gpu)


def balanced(reaction):
    try:
        return is_balanced_reaction(reaction)
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=['synrbl', 'crb', 'rule-based'], required=True)
    parser.add_argument('--input', type=Path, default=ROOT / 'examples/custom_reactions.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--container', type=Path, default=ROOT / 'models/constrained_reaction_balancer/CRB_env.sif')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index for CRB; -1 uses the CPU')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file')

    with args.input.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    model = load_model(args.model, args.container, args.gpu)
    completions = [c if isinstance(c, Completion) else Completion(c)
                   for c in model.predict([row['input'] for row in rows])]

    with args.output.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['reaction_id', 'prediction', 'solved', 'confidence'])
        for row, completion in zip(rows, completions):
            writer.writerow([row['reaction_id'], completion.reaction, completion.solved,
                             '' if completion.confidence is None else completion.confidence])
            status = 'balanced' if completion.solved and balanced(completion.reaction) else 'not balanced'
            print(f"{row['reaction_id']}: {status}: {completion.reaction}")
    print(f'Saved {len(rows)} predictions to {args.output}')


if __name__ == '__main__':
    main()

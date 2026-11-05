"""Select a scoring metric and keep new releases separate from legacy results."""
import hashlib
import json
from pathlib import Path

from rdkit import Chem, rdBase
from utils import equivalence_v2
from utils.result_paths import METRICS, metric_name, results_root, verify_score_release, selected_root
from utils.reaction_balance import is_balanced_reaction

ROOT = Path(__file__).resolve().parents[1]



def compare(prediction, reference, *, metric='v2', stereo=False):
    metric = metric_name(metric)
    if metric != 'v1':
        return equivalence_v2.compare_reactions(
            prediction, reference, stereo=stereo,
            equivalence_lenient=metric == 'equivalence_lenient')
    if not stereo:
        def strip(reaction):
            sides = []
            for side in reaction.split('>'):
                molecules = []
                for smiles in filter(None, side.split('.')):
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        raise ValueError('Invalid molecule: ' + smiles)
                    Chem.RemoveStereochemistry(mol)
                    molecules.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                sides.append('.'.join(molecules))
            return '>'.join(sides)
        prediction, reference = strip(prediction), strip(reference)
    from src.completerxn.legacy.equivalence_v1 import compare_reactions
    return compare_reactions(prediction, reference)


def labels(prediction, reference, *, metric='v2', stereo=False):
    metric = metric_name(metric)
    if not isinstance(stereo,bool):
        raise ValueError('Invalid metric or stereo policy')
    try:
        match = compare(prediction, reference, metric=metric, stereo=stereo)
    except ValueError:
        # Invalid model output remains in the denominator as incorrect.
        return False, False
    return bool(match.exact_match), bool(getattr(match,
        'deep_pragmatic_equivalence_match' if metric == 'v1' else 'equivalent_match'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def provenance(output, metric, inputs, **policies):
    metric = metric_name(metric)
    def relative(path):
        try: return str(path.relative_to(ROOT))
        except ValueError: return str(path)
    files = list(dict.fromkeys(Path(p) for p in inputs))
    files += [ROOT/'utils/scoring.py', ROOT/'utils/reaction_balance.py',
              ROOT/('src/completerxn/legacy/equivalence_v1.py' if metric == 'v1' else 'src/completerxn/equivalence.py')]
    if metric != 'v1':
        files.append(Path(equivalence_v2.compare_reactions.__code__.co_filename).resolve())
    files.append(Path(is_balanced_reaction.__code__.co_filename).resolve())
    value = dict(metric=metric, version='v1' if metric == 'v1' else equivalence_v2.METRIC_VERSION,
                 rdkit=rdBase.rdkitVersion, policies=policies,
                 sha256={relative(p): digest(p)
                         for p in files})
    (Path(output)/'provenance.json').write_text(json.dumps(value, indent=2)+'\n')


def overlay_scores(frame, stats_path):
    """Replace older prediction labels with the selected scores before plotting."""
    import os
    release = selected_root()
    import pandas as pd
    path = Path(stats_path)
    model = 'CRB' if 'crb' in path.parts or 'CBS_PT' in path.parts else 'RB'
    scores = pd.read_csv(Path(release)/'main/scores/per_reaction'/f'{model}_{path.parent.name}.csv')
    if set(frame['original_index']) != set(scores['original_index']):
        raise ValueError('Prediction metadata and score populations differ: ' + str(path))
    scores = scores.set_index('original_index')
    frame = frame.copy()
    for target, source in [('matches_exact','exact'),('matches_selected_metric','equivalence'),
                           ('matches_ground_truth','equivalence'),('is_balanced','balanced')]:
        frame[target] = frame['original_index'].map(scores[source]).astype(bool)
    return frame

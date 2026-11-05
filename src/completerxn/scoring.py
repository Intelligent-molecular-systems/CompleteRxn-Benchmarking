"""Score one prediction per reaction ID with the benchmark equivalence metric."""
import json
import math
from pathlib import Path

import rdkit

from . import __version__
from .balance import is_balanced_reaction
from .equivalence import METRIC_VERSION, compare_reactions, reaction_signature
from .io import read_csv, sha256, write_csv, write_json


def evaluate_rows(reference_rows, prediction_rows, *, stereo=False, equivalence_lenient=False):
    """Score predictions in memory. Missing predictions count as incorrect."""
    reference_rows = list(reference_rows)
    ids = [r['reaction_id'] for r in reference_rows]
    if not ids or any(not uid for uid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Reference IDs must be nonempty and unique')
    predicted = {}
    for row in prediction_rows:
        uid = row['reaction_id']
        if not uid or uid in predicted:
            raise ValueError(f'Empty or duplicate prediction ID: {uid!r}')
        value = str(row.get('solved', 'true')).strip().lower()
        if value not in ('true', 'false', '1', '0'):
            raise ValueError(f'Invalid solved flag for {uid}: {value!r}')
        confidence=row.get('confidence')
        if confidence not in (None,''):
            number=float(confidence)
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError(f'Invalid confidence for {uid}')
        predicted[uid] = (row['prediction'], value in ('true', '1'))
    extras = predicted.keys() - set(ids)
    if extras:
        raise ValueError(f'{len(extras)} prediction IDs are outside the reference population')
    policy = dict(stereo=stereo, equivalence_lenient=equivalence_lenient)
    records = []
    for row in reference_rows:
        uid, reference = row['reaction_id'], row['reference']
        try:
            reaction_signature(reference, **policy)
        except ValueError as error:
            raise ValueError(f'Invalid or unbalanced reference {uid}: {error}') from error
        present = uid in predicted
        prediction, solved = predicted.get(uid, ('', False))
        valid = balanced = exact = equivalent = False
        if prediction:
            try:
                match = compare_reactions(prediction, reference, **policy)
                valid = True
                try:
                    reaction_signature(prediction, **policy)
                    balanced = is_balanced_reaction(prediction)
                except ValueError:
                    pass
                exact = solved and balanced and match.exact_match
                equivalent = solved and balanced and match.equivalent_match
            except ValueError:
                pass
        reason = ('missing' if not present else 'unsolved' if not solved else
                  'invalid' if not valid else 'unbalanced' if not balanced else
                  'reference_mismatch' if not equivalent else 'correct')
        records.append(dict(reaction_id=uid, reason=reason, present=int(present), solved=int(solved),
                            valid=int(valid), balanced=int(balanced), exact=int(exact),
                            equivalence=int(equivalent)))
    n = len(records)
    metric = 'equivalence_lenient' if equivalence_lenient else 'v2'
    summary = dict(observations=n, predictions=len(predicted), missing=n-len(predicted),
                   invalid=sum(r['present'] and not r['valid'] for r in records),
                   unbalanced=sum(r['valid'] and not r['balanced'] for r in records),
                   exact_pct=100*sum(r['exact'] for r in records)/n,
                   equivalence_pct=100*sum(r['equivalence'] for r in records)/n,
                   balanced_pct=100*sum(r['balanced'] for r in records)/n,
                   metric=metric, metric_version=METRIC_VERSION, rdkit=rdkit.__version__, stereo=stereo)
    return records, summary


def score(references, predictions, output_dir, *, stereo=False, equivalence_lenient=False):
    """Score CSVs with the same implementation as the Python API. Never overwrite."""
    records, summary = evaluate_rows(
        read_csv(references, ['reaction_id', 'reference']),
        read_csv(predictions, ['reaction_id', 'prediction']),
        stereo=stereo, equivalence_lenient=equivalence_lenient)
    metric = summary['metric']
    paths = [Path(references), Path(predictions), Path(__file__),
             Path(__file__).with_name('equivalence.py'), Path(__file__).with_name('balance.py')]
    provenance = dict(package_version=__version__, metric=metric, metric_version=METRIC_VERSION,
                      rdkit=rdkit.__version__, stereo=stereo,
                      missing_predictions='incorrect', requires_solved=True,
                      balance_gate='manuscript independent atom/charge gate plus metric isotope balance',
                      sha256={str(p.resolve()):sha256(p) for p in paths})
    metadata_path = Path(references).with_suffix(Path(references).suffix + '.metadata.json')
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['export_sha256'] != sha256(references):
            raise ValueError('Reference CSV no longer matches its export metadata')
        provenance['dataset'] = metadata
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'per_reaction.csv', records, list(records[0]))
    write_csv(output/'summary.csv', [summary], list(summary))
    write_json(output/'provenance.json', provenance)
    return summary

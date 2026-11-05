"""Choose metric versions and result folders without importing chemistry packages."""
import ast
import csv
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('v2', 'v1', 'legacy', 'equivalence_lenient')


def metric_name(value):
    """Accept the previous 'legacy' spelling, but publish the name v1."""
    if value not in METRICS:
        raise ValueError('Unknown metric: ' + str(value))
    return 'v1' if value == 'legacy' else value


def ensure_v1_seed(metric):
    """Restart v1 scoring with Python hash seed 0 before writing any files."""
    if metric_name(metric)=='v1' and os.environ.get('PYTHONHASHSEED')!='0':
        os.execvpe(sys.executable,[sys.executable,*sys.argv],dict(os.environ,PYTHONHASHSEED='0'))


def metric_version(metric='v2'):
    if metric_name(metric) == 'v1':
        return 'v1'
    tree = ast.parse((ROOT/'src/completerxn/equivalence.py').read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'METRIC_VERSION' for t in n.targets))


def results_root(metric='v2'):
    name = metric_name(metric)
    if name == 'v1':
        return ROOT/'results/archive/metric_v1'
    if name == 'equivalence_lenient':
        return ROOT/'results/local/equivalence_lenient'
    return ROOT/'results'


def selected_root():
    return Path(os.environ.get('COMPLETERXN_RESULTS_ROOT', results_root())).resolve()


def resolve_recorded_path(path):
    """Find a moved file using its recorded old path, leaving the run record unchanged."""
    path = Path(path)
    if path.is_absolute():
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError:
            return path
    else:
        relative = str(path)
    ledger = ROOT/'results/archive/path_relocations.csv'
    if ledger.exists():
        with ledger.open(newline='') as handle:
            moves = list(csv.DictReader(handle))
        for row in sorted(moves, key=lambda r: len(r['original_path']), reverse=True):
            old = row['original_path']
            if relative == old or relative.startswith(old + '/'):
                return ROOT/(row['current_path'] + relative[len(old):])
    return ROOT/relative


def verify_score_release(scores, metric='v2'):
    scores = Path(scores).resolve()
    metric = metric_name(metric)
    path = scores.parent/'provenance.json' if scores.name == 'per_reaction' else scores/'provenance.json'
    if not path.exists():
        # The original v1 release predates the common provenance schema.
        base = results_root('v1')/'main/scores'
        if metric == 'v1' and path.parent == base and (base/'score_provenance.json').is_file():
            return
        raise ValueError('Missing score provenance: ' + str(path))
    saved = json.loads(path.read_text())
    if metric_name(saved['metric']) != metric:
        raise ValueError('Score release and selected metric do not match')
    allowed = ('legacy', 'v1') if metric == 'v1' else (metric_version(metric),)
    if saved['version'] not in allowed:
        raise ValueError('Score release and selected metric version do not match')


def verify_cached_agreement(output, scores, metric='v2'):
    output, scores, metric = Path(output), Path(scores), metric_name(metric)
    saved = json.loads((output/'provenance.json').read_text())
    version = saved['version']
    if (metric_name(saved['metric']) != metric
            or version not in (('legacy', 'v1') if metric == 'v1' else (metric_version(metric),))
            or saved['policies']['stereo']):
        raise ValueError('Existing agreement uses a different metric or stereo policy')
    summary = 'stereo_free_agreement_summary.csv' if metric == 'v1' else 'agreement_summary.csv'
    if not (output/summary).is_file():
        raise ValueError('Agreement output is incomplete')
    recorded = {resolve_recorded_path(k).resolve(): v for k, v in saved['sha256'].items()}
    paths = list(scores.joinpath('per_reaction').glob('*.csv'))
    checked = set()
    def check(path, content):
        if path.resolve() in checked:
            raise ValueError("Duplicate agreement score input")
        if recorded.get(path.resolve()) != hashlib.sha256(content).hexdigest():
            raise ValueError('Agreement score inputs changed; choose a new --output-root')
        checked.add(path.resolve())
    if paths:
        for path in paths:
            if path.name.startswith(('CRB_', 'SynRBL_')):
                check(path, path.read_bytes())
    elif (scores/'filtered_scores.tar.gz').is_file():
        # Fresh clones contain the compressed scores, not expanded CSV copies.
        with tarfile.open(scores/'filtered_scores.tar.gz') as bundle:
            for member in bundle.getmembers():
                if not member.isfile() or Path(member.name).name != member.name:
                    raise ValueError('Unexpected score archive member')
                if member.name.startswith(('CRB_', 'SynRBL_')) and member.name.endswith('.csv'):
                    check(scores/'per_reaction'/member.name, bundle.extractfile(member).read())
    if not checked:
        raise ValueError('No agreement score inputs available')
    expected={p for p in recorded if p.parent==(scores/'per_reaction').resolve() and p.name.startswith(('CRB_','SynRBL_'))}
    if checked!=expected:
        raise ValueError('Cached agreement score inputs are incomplete')
    return output/summary

"""Download a fixed dataset version and match split assignments by reaction ID."""
import json
import os
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from . import __version__
from .io import read_csv, sha256, write_csv, write_json

MANIFEST = json.loads(Path(__file__).with_name('dataset.json').read_text())
SPLITS = {'random': 'random_split', 'group': 'group_split', 'extreme_ood': 'extreme_ood'}
PARTITIONS = {'train': '0', 'val': '1', 'test': '2'}


def __getattr__(name):
    if name == 'DataLoader':
        from .benchmark import DataLoader
        return DataLoader
    raise AttributeError(name)


def split_file(split, rep):
    if split not in SPLITS or rep not in range(5):
        raise ValueError('Choose random, group, or extreme_ood and repetition 0-4')
    suffix = f'_seed{42 + rep}' if split == 'random' else ''
    return f'{split}_splits/benchmark_final_{split}_splits_rep{rep}{suffix}.csv'


def cache_dir():
    base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return base / 'completerxn' / MANIFEST['revision']


def eligibility_file(split, rep):
    """Return the published list of retained test IDs for this split and repetition."""
    split_file(split, rep)  # Validate both arguments.
    return f'eligibility/{split}_rep{rep}.csv'


def download(data_dir=None, *, split='random', rep=0):
    """Download the benchmark, split, and retained test IDs into a verified local cache.

    Hugging Face uses your saved login or HF_TOKEN for private datasets.
    Downloaded files are checked before they become available to the loader.
    """
    destination = Path(data_dir) if data_dir is not None else cache_dir()
    paths = ['benchmark_final.csv', split_file(split, rep), eligibility_file(split, rep)]
    for name in paths:
        path = destination / name
        expected = MANIFEST['files'][name]
        if path.exists():
            if path.stat().st_size != expected['size'] or sha256(path) != expected['sha256']:
                raise ValueError(f'Checksum mismatch: {path}. Use a separate download directory.')
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # Download beside the destination so the final hard link needs no copy.
        with tempfile.TemporaryDirectory(prefix='.download-', dir=path.parent) as temporary:
            try:
                downloaded = Path(hf_hub_download(
                    repo_id=MANIFEST['repo_id'], repo_type='dataset',
                    revision=MANIFEST['revision'], filename=name, local_dir=temporary))
            except HfHubHTTPError as error:
                status = getattr(error.response, 'status_code', None)
                if status in (401, 403):
                    raise PermissionError(
                        f"Cannot access {MANIFEST['repo_id']}. Run hf auth login with an "
                        "account that has access, or set HF_TOKEN.") from error
                raise OSError(f'Could not download {name}: {error}') from error
            if downloaded.stat().st_size != expected['size'] or sha256(downloaded) != expected['sha256']:
                raise ValueError(f'Download checksum mismatch: {name}')
            # A concurrent download may have finished. Never replace its data.
            try:
                os.link(downloaded, path)
            except FileExistsError:
                if path.stat().st_size != expected['size'] or sha256(path) != expected['sha256']:
                    raise ValueError(f'Concurrent download conflict: {path}')
    return destination


def load_split(data_dir=None, *, split='random', rep=0, partition='test',
               population='full', eligibility=None, reference_column='TargetReaction_with_stereo',
               include_metadata=False, input_column='InputReaction_harmonized'):
    """Read reaction_id, input, and reference rows in the original benchmark order.

    Read local files only. The filtered test set requires a list of retained IDs
    in a Unique_identifier column. Do not infer exclusions from the predictions.
    Local files may have extra columns. export_split records their actual hashes.
    Missing stereo references raise an error rather than using another column.
    """
    if partition not in PARTITIONS or population not in ('full', 'filtered'):
        raise ValueError('Invalid partition or population')
    if (population == 'filtered') != (eligibility is not None):
        raise ValueError('Filtered export requires --eligibility; full export must omit it')
    if population == 'filtered' and partition != 'test':
        raise ValueError('The manuscript eligibility mask applies only to the test partition')
    directory = Path(data_dir) if data_dir is not None else cache_dir()
    assignments = {}
    split_path = directory / split_file(split, rep)
    for row in read_csv(split_path, ['Unique_identifier', SPLITS[split]]):
        uid, value = row['Unique_identifier'], row[SPLITS[split]]
        if not uid or uid in assignments or value not in PARTITIONS.values():
            raise ValueError(f'Invalid/duplicate split assignment: {uid!r}')
        assignments[uid] = value
    selected = {uid for uid, value in assignments.items() if value == PARTITIONS[partition]}
    if eligibility is not None:
        ids = [r['Unique_identifier'] for r in read_csv(eligibility, ['Unique_identifier'])]
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= selected:
            raise ValueError('Eligibility IDs must be unique and belong to this test fold')
        selected = set(ids)
    rows, seen = [], set()
    for row in read_csv(directory / 'benchmark_final.csv',
                        ['Unique_identifier', input_column, reference_column]):
        uid = row['Unique_identifier']
        if not uid or uid in seen:
            raise ValueError(f'Empty or duplicate benchmark ID: {uid!r}')
        seen.add(uid)
        if uid in selected:
            if not row[input_column] or not row[reference_column]:
                raise ValueError(f'Empty input/reference for {uid}; choose the reference column explicitly')
            item = dict(reaction_id=uid, input=row[input_column], reference=row[reference_column])
            if include_metadata:
                item['metadata'] = {k:row[k] for k in ('MissingNrAtoms','MissingNrCarbons','rxn_class') if k in row}
            rows.append(item)
    if set(assignments) - seen:
        raise ValueError('Split contains IDs absent from the benchmark')
    if not rows:
        raise ValueError('Selected partition is empty')
    return rows


def export_split(output, data_dir=None, **options):
    """Save one partition and its input details in new files without overwriting existing ones."""
    output = Path(output)
    metadata = output.with_suffix(output.suffix + '.metadata.json')
    if output.exists() or metadata.exists():
        raise FileExistsError(f'Output already exists: {output} or {metadata}')
    options = dict(split='random', rep=0, partition='test', population='full',
                   eligibility=None, reference_column='TargetReaction_with_stereo',input_column='InputReaction_harmonized') | options
    directory = Path(data_dir) if data_dir is not None else cache_dir()
    rows = load_split(directory, **options)
    split, rep = options.get('split', 'random'), options.get('rep', 0)
    sources = [directory/'benchmark_final.csv', directory/split_file(split, rep)]
    if options.get('eligibility') is not None:
        sources.append(Path(options['eligibility']))
    hashes = {str(p.resolve()): sha256(p) for p in sources}
    published = all(hashes[str((directory/name).resolve())] == MANIFEST['files'][name]['sha256']
                    for name in ['benchmark_final.csv', split_file(split, rep)])
    if options.get('eligibility') is not None:
        expected = MANIFEST['files'].get(eligibility_file(split, rep), {})
        published = published and sha256(Path(options['eligibility'])) == expected.get('sha256')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows, ['reaction_id', 'input', 'reference'])
    write_json(metadata, dict(package_version=__version__, rows=len(rows), options={k:str(v) if isinstance(v, Path) else v
                                                    for k,v in options.items()},
                              repo_id=MANIFEST['repo_id'],
                              revision=MANIFEST['revision'] if published else None,
                              matches_published_files=published, sha256=hashes,
                              export_sha256=sha256(output)))
    return len(rows)

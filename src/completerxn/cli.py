"""Command-line data export and reaction scoring."""
import argparse
import json
import sys
from pathlib import Path
from urllib.error import URLError

from . import __version__
from .data import SPLITS, PARTITIONS, cache_dir, download, eligibility_file, export_split


def main(argv=None):
    parser = argparse.ArgumentParser(prog='completerxn', description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    data = commands.add_parser('data', help='Download a split, or export it to a CSV for an external model')
    actions = data.add_subparsers(dest='action', required=True)
    for action in ('download', 'export'):
        sub = actions.add_parser(action)
        sub.add_argument('--data-dir', type=Path, help='Dataset directory; defaults to a versioned user cache')
        sub.add_argument('--split', choices=list(SPLITS), default='random')
        sub.add_argument('--rep', type=int, choices=range(5), default=0)
        if action == 'export':
            sub.add_argument('--partition', choices=list(PARTITIONS), default='test')
            sub.add_argument('--population', choices=['full', 'filtered'], default='full')
            sub.add_argument('--eligibility', type=Path,
                             help='CSV of Unique_identifier values in the filtered test set '
                                  '(default: the published list for this split and repetition)')
            sub.add_argument('--reference-column', default='TargetReaction_with_stereo')
            sub.add_argument('--input-column', default='InputReaction_harmonized')
            sub.add_argument('--output', type=Path, required=True)
    scoring = commands.add_parser('score', help='Score completed reaction SMILES, joined by reaction ID')
    scoring.add_argument('--references', type=Path, required=True)
    scoring.add_argument('--predictions', type=Path, required=True)
    scoring.add_argument('--output-dir', type=Path, required=True)
    scoring.add_argument('--stereo-sensitive', action='store_true', help='Preserve stereochemistry (default: stereo-free)')
    scoring.add_argument('--metric', choices=['v2', 'equivalence_lenient'], default='v2')
    args = parser.parse_args(argv)
    try:
        if args.command == 'data' and args.action == 'download':
            print(download(args.data_dir, split=args.split, rep=args.rep))
        elif args.command == 'data':
            # Without --data-dir, fetch the published files first, as the Python loader does.
            if args.data_dir is None:
                download(split=args.split, rep=args.rep)
            if args.population == 'filtered' and args.eligibility is None:
                args.eligibility = (args.data_dir or cache_dir()) / eligibility_file(args.split, args.rep)
            options = vars(args).copy()
            for name in ('command', 'action', 'output', 'data_dir'):
                options.pop(name)
            count = export_split(args.output, args.data_dir, **options)
            print(f'Exported {count} observations to {args.output}')
        else:
            from rdkit import rdBase
            # Invalid molecules are reported in the numerical outputs; avoid
            # repeating RDKit's parser messages for each failed prediction.
            with rdBase.BlockLogs():
                from .scoring import score
                result = score(args.references, args.predictions, args.output_dir,
                               stereo=args.stereo_sensitive, equivalence_lenient=args.metric=='equivalence_lenient')
            print(json.dumps(result))
    except (ValueError, OSError, URLError) as error:
        print(f'completerxn: {error}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
import os
import sys

# Add repository root so utils.metrics is importable regardless of launch directory.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir):
    cur = os.path.abspath(start_dir)
    while True:
        candidate = os.path.join(cur, 'utils', 'metrics.py')
        if os.path.isfile(candidate):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


# Try both script location and process CWD; apptainer launch context can vary.
_ROOT_CANDIDATES = [_find_repo_root(_HERE), _find_repo_root(os.getcwd())]
for _root in _ROOT_CANDIDATES:
    if _root and _root not in sys.path:
        sys.path.insert(0, _root)

from utils.result_paths import metric_name, ensure_v1_seed
from utils.reaction_balance import reaction_balance, is_balanced_reaction

import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False
try:
    import onmt.opts as _onmt_opts
    _HAS_ONMT = True
except ImportError:
    _onmt_opts = None
    _HAS_ONMT = False
import re


# NOTE: Legacy helpers kept for compatibility with older outputs.
def _canonical_mol_set(smiles_str):
    result = set()
    for smi in smiles_str.split('.'):
        smi = smi.strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        result.add(Chem.MolToSmiles(mol))
    return frozenset(result)


def reaction_set_match(pred_rxn, target_rxn):
    """True if pred and target describe the same reaction as unordered molecule sets."""
    try:
        pred_parts = pred_rxn.strip().split('>')
        tgt_parts = target_rxn.strip().split('>')
        if len(pred_parts) not in (2, 3) or len(tgt_parts) not in (2, 3):
            return False
        return (
            _canonical_mol_set(pred_parts[0]) == _canonical_mol_set(tgt_parts[0])
            and _canonical_mol_set(pred_parts[-1]) == _canonical_mol_set(tgt_parts[-1])
        )
    except Exception:
        return False


def canonicalize_smiles(smiles):
    res = ''
    components = re.split(r"([>.])", smiles)

    # Filter out empty strings
    components = list(filter(None, components))

    for component in components:
        # If the component is a punctuation mark, just add it to the result
        if component == '>' or component == '.':
            res += component
            continue

        mol_comp = Chem.MolFromSmiles(component)

        # invalid smiles
        if mol_comp is None:
            return ''

        # canonicalize
        smiles_comp = Chem.MolToSmiles(mol_comp, isomericSmiles=True)
        res += smiles_comp

    return res


def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        targets = row['target'].split('\t')
        for target in targets:
            if reaction_set_match(row['{}{}'.format(base, i)], target):
                return i
    return 0


def get_ranks_rich(row, base, max_rank, metric_attrs, scoring_metric="v2"):
    """Return the first matching beam rank for each selected accuracy metric.

    Ranks start at 1. A rank of 0 means no candidate matched. Compare each
    candidate once and read all selected metric labels from that comparison.
    """
    scoring_metric = metric_name(scoring_metric)
    input_rxn = row.get('input', '')
    if not isinstance(input_rxn, str) or input_rxn.count('>') != 2:
        raise ValueError("Rich scoring requires full input reactions in row['input'] (A>R>B format).")

    targets = row['target'].split('\t')  # target diffs
    # Compose all full target reactions once; comparisons must be full-vs-full.
    full_targets = []
    for target_diff in targets:
        full_target = compose_completed_reaction(input_rxn, target_diff)
        if full_target:
            full_targets.append(full_target)

    ranks = {attr: 0 for attr in metric_attrs}
    pending = set(metric_attrs)
    for i in range(1, max_rank + 1):
        if not pending:
            break
        pred_diff = row.get(f'{base}{i}', '')
        full_pred = compose_completed_reaction(input_rxn, pred_diff)
        if not full_pred:
            continue
        # Do not award metric matches to chemically unbalanced full predictions.
        try:
            pred_balanced = is_balanced_reaction(full_pred)
        except Exception:
            pred_balanced = False
        if not pred_balanced:
            continue
        for full_target in full_targets:
            try:
                from utils.scoring import compare
                result = compare(full_pred, full_target, metric=scoring_metric, stereo=True)
            except Exception as exc:
                if scoring_metric != 'v1' and not isinstance(exc, ValueError):
                    raise
                continue
            for attr in list(pending):
                if getattr(result, attr):
                    ranks[attr] = i
                    pending.discard(attr)
    return ranks


def _merge_side(base_side, add_side):
    base = [s for s in base_side.split('.') if s] if base_side else []
    add = [s for s in add_side.split('.') if s] if add_side else []
    return '.'.join(base + add)


def compose_completed_reaction(input_reaction, predicted_diff):
    """Add predicted missing molecules to the corresponding sides of the input reaction."""
    if not isinstance(input_reaction, str) or not isinstance(predicted_diff, str):
        return ''
    try:
        in_r, in_rg, in_p = input_reaction.split('>')
        d_r, d_rg, d_p = predicted_diff.split('>')
    except ValueError:
        return ''

    out_r = _merge_side(in_r, d_r)
    out_rg = _merge_side(in_rg, d_rg)
    out_p = _merge_side(in_p, d_p)
    return f"{out_r}>{out_rg}>{out_p}"


def evaluate_balanced_reaction(input_reaction, predicted_diff):
    """Return (is_balanced, balance_info) for input+predicted_diff completed reaction."""
    completed = compose_completed_reaction(input_reaction, predicted_diff)
    if not completed:
        return False, "invalid_reaction"
    try:
        balanced, balance_info = reaction_balance(completed)
        return balanced, balance_info
    except Exception:
        return False, "invalid_reaction"


def main(opt):
    ensure_v1_seed(opt.scoring_metric)
    if not opt.output_dir: raise ValueError("Scoring requires --output-dir to preserve saved predictions")
    from pathlib import Path
    destination=Path(opt.output_dir)
    if destination.exists() and any(destination.iterdir()): raise ValueError("Choose an empty output directory")
    destination.mkdir(parents=True,exist_ok=True)
    opt.write_top1=str(destination/"top1_predictions_with_stats.txt")
    if opt.write_wrong: opt.write_wrong=str(destination/"wrong_predictions.txt")
    synrbl_metric_attrs = [
        'exact_match',
        'proton_shuffle_match',
        'all_normalizations_match',
        'pragmatic_equivalence_match',
        'deep_pragmatic_equivalence_match',
    ]

    def load_vocab(vocab_path):
        if not _HAS_TORCH:
            raise RuntimeError('--exclude-unknown-tokens requires torch, which is not installed in this environment.')
        vocab_obj = torch.load(vocab_path, map_location='cpu')
        if isinstance(vocab_obj, dict) and 'vocab' in vocab_obj:
            vocab = vocab_obj['vocab']
        else:
            vocab = vocab_obj

        tokens = set()
        if isinstance(vocab, list):
            for _, v in vocab:
                if hasattr(v, 'itos'):
                    tokens.update(v.itos)
        elif isinstance(vocab, dict):
            for v in vocab.values():
                if hasattr(v, 'itos'):
                    tokens.update(v.itos)
        else:
            raise ValueError('Unknown vocab structure in --vocab-path file')
        return tokens

    def has_oov_token(tokenized_string, vocab_tokens):
        if not isinstance(tokenized_string, str):
            return False
        return any(tok not in vocab_tokens for tok in tokenized_string.strip().split())

    # Check if write-file already exists
    if opt.write_wrong and os.path.exists(opt.write_wrong):
        print(f"File {opt.write_wrong} already exists. Delete it or rename it to continue.")
        exit()
    if opt.write_top1 and os.path.exists(opt.write_top1):
        print(f"File {opt.write_top1} already exists. Delete it or rename it to continue.")
        exit()

    # Inputs are required: scoring compares completed reactions, not diffs.
    raw_inputs = None
    inputs = None
    if opt.inputs:
        with open(opt.inputs, 'r') as f:
            raw_input_lines = f.readlines()
        raw_inputs = [line.strip() for line in raw_input_lines]
        inputs = [''.join(line.split()) for line in raw_input_lines]
    else:
        raise ValueError("-inputs is required for rich-metric scoring.")

    # Determine scoring metric.
    _metric_attr_map = {
        'exact':               'exact_match',
        'equivalence':         'equivalence_match',
        'ionic_equivalence':   'ionic_equivalence_match',
        'proton_shuffle':      'proton_shuffle_match',
        'proton_shuffle_set':  'proton_shuffle_set_match',
        'all_normalizations':  'all_normalizations_match',
        'pragmatic_equivalence': 'pragmatic_equivalence_match',
        'deep_pragmatic_equivalence': 'deep_pragmatic_equivalence_match',
    }
    metric_attr = _metric_attr_map.get(
        getattr(opt, 'accuracy_metric', 'deep_pragmatic_equivalence'),
        'deep_pragmatic_equivalence_match',
    )

    if opt.scoring_metric != 'v1':
        if opt.accuracy_metric not in ('exact', 'equivalence', 'deep_pragmatic_equivalence'):
            raise ValueError('The benchmark metric provides only exact and equivalent matching')
        synrbl_metric_attrs = ['exact_match', 'equivalent_match']
        metric_attr = 'exact_match' if opt.accuracy_metric == 'exact' else 'equivalent_match'
        opt.accuracy_metric = 'exact' if metric_attr == 'exact_match' else 'equivalence'

    with open(opt.targets, 'r') as f:
        raw_target_lines = f.readlines()
    raw_targets = [line.strip() for line in raw_target_lines]
    targets = [''.join(line.split()) for line in raw_target_lines]

    predictions = [[] for i in range(opt.beam_size)]

    test_df = pd.DataFrame({'raw_target': raw_targets, 'target': targets})
    if inputs is not None:
        if len(inputs) != len(targets):
            raise ValueError(f"inputs/targets length mismatch: {len(inputs)} vs {len(targets)}")
        test_df['raw_input'] = raw_inputs
        test_df['input'] = inputs
    total = len(test_df)
    test_df['original_index'] = range(total)
    test_df['scoring_metric'] = opt.scoring_metric

    with open(opt.predictions, 'r') as f:
        for i, line in enumerate(f.readlines()):
            predictions[i % opt.beam_size].append(''.join(line.strip().split()))

    for i, preds in enumerate(predictions):
        if len(preds) != total:
            raise ValueError(
                f"Prediction count mismatch for beam {i+1}: {len(preds)} vs expected {total}. "
                f"Check -beam_size and predictions file length."
            )

    excluded_summary_line = ''
    if opt.exclude_unknown_tokens:
        if not opt.vocab_path:
            raise ValueError('--vocab-path is required when using --exclude-unknown-tokens')
        vocab_tokens = load_vocab(opt.vocab_path)
        keep_mask = ~test_df['raw_target'].apply(lambda s: has_oov_token(s, vocab_tokens))
        if 'raw_input' in test_df.columns:
            keep_mask &= ~test_df['raw_input'].apply(lambda s: has_oov_token(s, vocab_tokens))

        keep_indices = keep_mask[keep_mask].index.tolist()
        excluded_count = len(test_df) - len(keep_indices)
        excluded_pct = (excluded_count / len(test_df) * 100) if len(test_df) else 0.0

        test_df = test_df.iloc[keep_indices].reset_index(drop=True)
        predictions = [[preds[idx] for idx in keep_indices] for preds in predictions]
        total = len(test_df)

        if total == 0:
            raise ValueError('No reactions remain after filtering unknown tokens.')

        excluded_summary_line = (
            f"Excluded {excluded_count} reactions with unknown tokens "
            f"({excluded_pct:.2f}%). {total} reactions remain for evaluation."
        )
        print(excluded_summary_line)

    for i, preds in enumerate(predictions):
        test_df['prediction_{}'.format(i + 1)] = preds
        test_df['canonical_prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(
            canonicalize_smiles)

    if inputs is not None:
        for i in range(1, opt.beam_size + 1):
            tqdm.pandas(desc=f'Checking balance beam {i}')
            test_df[f'is_balanced_{i}'] = test_df.progress_apply(
                lambda row, _i=i: evaluate_balanced_reaction(row['input'], row[f'prediction_{_i}'])[0],
                axis=1,
            )

    # Always report SynRBL metrics + requested metric (deduplicated).
    _all_metrics_to_report = synrbl_metric_attrs + [metric_attr]
    _report_attrs = list(dict.fromkeys(_all_metrics_to_report))  # preserve order, deduplicate

    print(f"Scoring metrics: {', '.join(_report_attrs)}")
    tqdm.pandas(desc='Scoring')
    ranks_dicts = test_df.progress_apply(
        lambda row: get_ranks_rich(row, 'prediction_', opt.beam_size, _report_attrs, opt.scoring_metric), axis=1,
    )
    for attr in _report_attrs:
        test_df[f'rank_{attr}'] = ranks_dicts.apply(lambda d: d[attr])
    # Primary rank column (used for write_top1 / write_wrong) = requested metric.
    test_df['rank'] = test_df[f'rank_{metric_attr}']
    exact_rank_col = 'rank_exact_match'
    selected_rank_col = f'rank_{metric_attr}'

    for attr in synrbl_metric_attrs:
        rank_col = f'rank_{attr}'
        if rank_col in test_df.columns:
            test_df[attr] = test_df[rank_col] == 1

    if opt.write_top1:
        with open(opt.write_top1, 'w') as write_f:
            write_f.write(
                'original_index\ttop1_prediction\tis_balanced\tbalance_info\t'
                'matches_exact\tmatches_selected_metric\tselected_metric\tmatches_ground_truth\n'
            )
            for _, row in test_df.iterrows():
                pred = row['prediction_1']
                is_balanced, balance_info = evaluate_balanced_reaction(row['input'], row['prediction_1'])
                matches_exact = row[exact_rank_col] == 1
                matches_selected_metric = row[selected_rank_col] == 1
                matches_ground_truth = matches_selected_metric
                write_f.write(
                    f'{int(row["original_index"])}\t{pred}\t{is_balanced}\t{balance_info}\t'
                    f'{matches_exact}\t{matches_selected_metric}\t{opt.accuracy_metric}\t{matches_ground_truth}\n'
                )

    # Write down ranks for primary metric.
    rank_path = os.path.join(opt.output_dir,'ranks.txt')
    with open(rank_path, 'w') as write_f:
        for rank in test_df['rank']:
            write_f.write(f'{rank}\n')

    # Print top-k table.
    metric_labels = [a.replace('_match', '') for a in _report_attrs]
    rank_cols = [f'rank_{a}' for a in _report_attrs]

    header = 'k  | ' + ' | '.join(f'{lbl:>20}' for lbl in metric_labels)
    if opt.invalid_smiles or inputs is not None:
        header += ' | invalid_smiles | balanced_beam1'
    separator = '-' * len(header)
    print(header)
    print(separator)

    summary_lines = []
    if excluded_summary_line:
        summary_lines.append(excluded_summary_line)
        summary_lines.append('')
    summary_lines.append(header)
    summary_lines.append(separator)

    corrects = {col: 0 for col in rank_cols}
    for i in range(1, opt.beam_size + 1):
        for col in rank_cols:
            corrects[col] += (test_df[col] == i).sum()
        invalid_smiles = (test_df[f'canonical_prediction_{i}'] == '').sum()
        balanced_pct = test_df[f'is_balanced_{i}'].mean() * 100 if inputs is not None else None
        row_str = f'{i:<2} | ' + ' | '.join(
            f'{corrects[col] / total * 100:>19.1f}%' for col in rank_cols
        )
        if opt.invalid_smiles:
            row_str += f' | {invalid_smiles / total * 100:>13.2f}%'
        if balanced_pct is not None:
            row_str += f' | {balanced_pct:>13.2f}%'
        print(row_str)
        summary_lines.append(row_str)

    summary_path = os.path.join(opt.output_dir,'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f'Summary written to: {summary_path}')

    full_results_path = os.path.join(opt.output_dir, 'full_results.csv')
    full_results_cols = [
        'original_index',
        'scoring_metric',
        'raw_input',
        'input',
        'raw_target',
        'target',
        'prediction_1',
        'canonical_prediction_1',
    ]
    full_results_cols.extend([a for a in synrbl_metric_attrs if a in test_df.columns])
    full_results_cols.extend([
        'rank',
        exact_rank_col,
        selected_rank_col,
        'selected_metric',
        'matches_ground_truth',
        'is_balanced_1',
    ])
    full_results_cols = [c for c in full_results_cols if c in test_df.columns]

    test_df['selected_metric'] = opt.accuracy_metric
    test_df['matches_ground_truth'] = test_df[selected_rank_col] == 1
    test_df[full_results_cols].to_csv(full_results_path, index=False)
    print(f'Full results written to: {full_results_path}')

    if opt.write_wrong:
        wrong_df = test_df[test_df['rank'] == 0]
        valid_smiles_df = wrong_df[wrong_df['canonical_prediction_1'] != '']
        invalid_smiles_df = wrong_df[wrong_df['canonical_prediction_1'] == '']
        valid_smiles_df.to_csv(opt.write_wrong, sep='\t', index=False)
        invalid_smiles_df.to_csv(opt.write_wrong + '.invalid', sep='\t', index=False)

    if opt.scoring_metric != 'v1':
        from utils.scoring import provenance
        provenance(opt.output_dir,opt.scoring_metric,[opt.predictions,opt.inputs,opt.targets,__file__],
                   stereo=True,accuracy_requires_balance=True,exclude_unknown_tokens=opt.exclude_unknown_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score_predictions.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if _HAS_ONMT:
        _onmt_opts.add_md_help_argument(parser)

    parser.add_argument('-beam_size', type=int, default=5,
                       help='Beam size')
    parser.add_argument('-invalid_smiles', action="store_true",
                       help='Show % of invalid SMILES')
    parser.add_argument('-predictions', type=str, default="",
                       help="Path to file containing the predictions")
    parser.add_argument('-targets', type=str, default="",
                       help="Path to file containing targets")
    parser.add_argument('-inputs', type=str, default="",
                       help="Path to source/input reactions file (required for -write_top1)")
    parser.add_argument('-write_wrong', type=str, default="",
                        help="Path to file to write wrong predictions to")
    parser.add_argument('-write_top1', type=str, default="",
                        help="Path to TSV file with top1 prediction, balanced flag, and match flag")
    parser.add_argument('--exclude-unknown-tokens', action='store_true',
                        help='Exclude reactions with input/target tokens not present in model vocab. Requires --vocab-path.')
    parser.add_argument('--vocab-path', type=str, default='',
                        help='Path to model checkpoint or vocab file used for unknown-token filtering')
    parser.add_argument('--accuracy-metric', type=str, default='deep_pragmatic_equivalence',
                        choices=['exact', 'equivalence', 'ionic_equivalence',
                                 'proton_shuffle', 'proton_shuffle_set', 'all_normalizations',
                                 'pragmatic_equivalence', 'deep_pragmatic_equivalence'],
                        help=(
                            'Use exact or equivalence for benchmark scoring. '
                            'The default name deep_pragmatic_equivalence is kept for older commands '
                            'and selects equivalence with the benchmark metric. '
                            'The other comparison levels apply only with --scoring-metric v1.'

                        ))

    parser.add_argument('--output-dir', help='Required separate output directory to preserve saved predictions')
    parser.add_argument('--scoring-metric', type=metric_name, choices=['v1','v2','equivalence_lenient'], default='v2')
    opt = parser.parse_args()
    main(opt)
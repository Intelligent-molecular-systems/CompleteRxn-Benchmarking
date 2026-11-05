import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse

# Canonical split labels (matches benchmark_final_random_splits_*.csv)
SPLIT_TRAIN = 0
SPLIT_VALID = 1
SPLIT_TEST  = 2


def split_data_random(df, train_size=0.8, valid_size=0.1, test_size=0.1, random_state=42):
    """
    Perform random train/valid/test split on a DataFrame.

    Args:
        df: DataFrame to split (must contain 'Unique_identifier' column)
        train_size: Proportion for training set (default 0.8)
        valid_size: Proportion for validation set (default 0.1)
        test_size: Proportion for test set (default 0.1)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    # First split: train and temp (valid + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state
    )

    # Second split: valid and test from temp
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_size / (valid_size + test_size),
        random_state=random_state
    )

    return train_df, valid_df, test_df


def build_canonical_split_df(train_df, valid_df, test_df, uid_to_row):
    """Create split assignments as reaction IDs, row positions, and split labels.

    random_split uses 0 for training, 1 for validation, and 2 for testing.
    Sort rows by their position in benchmark_final.
    """
    frames = [
        train_df[['Unique_identifier']].assign(random_split=SPLIT_TRAIN),
        valid_df[['Unique_identifier']].assign(random_split=SPLIT_VALID),
        test_df[['Unique_identifier']].assign(random_split=SPLIT_TEST),
    ]
    merged = pd.concat(frames, ignore_index=True)
    merged['row_index'] = merged['Unique_identifier'].map(uid_to_row).astype('Int64')
    return (merged[['Unique_identifier', 'row_index', 'random_split']]
            .sort_values('row_index')
            .reset_index(drop=True))


def split_data_by_group(group_splits_csv, benchmark_df, test_fold, valid_fold=None):
    """
    Produce a (train, valid, test) split from a pre-computed rxn_groups CSV.

    Assignment:
        test  = reactions in fold `test_fold`
        valid = reactions in fold `valid_fold`  (default: (test_fold + 1) % n_folds)
        train = all remaining folds

    Args:
        group_splits_csv: Path to e.g. rxn_groups/benchmark_final_group_splits_RI_t055.csv
        benchmark_df:     Full benchmark DataFrame (for joining columns).
        test_fold:        Integer fold index to use as test set.
        valid_fold:       Integer fold index to use as validation set.
                          If None, defaults to (test_fold + 1) % n_folds.

    Returns:
        Tuple of (train_df, valid_df, test_df), fold_col, n_folds
    """
    splits_df = pd.read_csv(group_splits_csv, usecols=['Unique_identifier', 'row_index'] +
                            [c for c in pd.read_csv(group_splits_csv, nrows=0).columns
                             if c.startswith('ood_fold')])
    fold_col = [c for c in splits_df.columns if c.startswith('ood_fold')][0]
    folds = sorted(f for f in splits_df[fold_col].unique() if f != -1)
    n_folds = len(folds)

    if test_fold not in folds:
        raise ValueError(f"test_fold={test_fold} not in available folds {folds}")
    if valid_fold is None:
        valid_fold = folds[(folds.index(test_fold) + 1) % n_folds]
    if valid_fold not in folds:
        raise ValueError(f"valid_fold={valid_fold} not in available folds {folds}")
    if valid_fold == test_fold:
        raise ValueError("valid_fold and test_fold must be different")

    train_folds = [f for f in folds if f not in (test_fold, valid_fold)]

    uid_test  = set(splits_df.loc[splits_df[fold_col] == test_fold,  'Unique_identifier'])
    uid_valid = set(splits_df.loc[splits_df[fold_col] == valid_fold, 'Unique_identifier'])
    uid_train = set(splits_df.loc[splits_df[fold_col].isin(train_folds), 'Unique_identifier'])

    test_df  = benchmark_df[benchmark_df['Unique_identifier'].isin(uid_test)].reset_index(drop=True)
    valid_df = benchmark_df[benchmark_df['Unique_identifier'].isin(uid_valid)].reset_index(drop=True)
    train_df = benchmark_df[benchmark_df['Unique_identifier'].isin(uid_train)].reset_index(drop=True)

    return train_df, valid_df, test_df, fold_col, n_folds, valid_fold



def main():
    """
    Produce canonical split index files.

    Random split (--split-type random):
        5 reps with different seeds.
        Output: <random_splits>/benchmark_final_random_splits_rep{i}_seed{seed}.csv
        Columns: Unique_identifier, row_index, random_split  (0=train, 1=valid, 2=test)

    Group/OOD split (--split-type group):
        5 reps from a 10-fold group-splits file (each fold ~10% of data).
        Rep i: test=fold 2i, valid=fold 2i+1, train=remaining 8 folds (~80%).
        Output: <group_splits>/benchmark_final_group_splits_rep{i}.csv
        Columns: Unique_identifier, row_index, group_split  (0=train, 1=valid, 2=test)

    Extreme OOD split (--split-type extreme_ood):
        5 reps with fold-specific OOD test sets and ID-biased train/valid pools.
        Output: <extreme_ood_splits>/benchmark_final_extreme_ood_splits_rep{i}.csv
        Columns: Unique_identifier, row_index, extreme_ood  (0=train, 1=valid, 2=test)

    Optional (--save-full-csvs):
        Also saves full-column per-partition CSVs.
    """
    parser = argparse.ArgumentParser(description="Split reaction data into train/valid/test sets.")
    parser.add_argument('--data-dir-path', type=str,
                        default='data/Template_curated/benchmark/',
                        help='Directory containing benchmark_final.csv and legacy split CSVs.')
    parser.add_argument('--csv-path', type=str, default='benchmark_final.csv',
                        help='Benchmark CSV filename (default: benchmark_final.csv)')
    parser.add_argument('--splits-dir', type=str, default=None,
                        help='Output directory for canonical split index files. '
                            'Defaults to <data-dir-path>/<split-type>_splits/')
    parser.add_argument('--split-type', type=str, choices=['random', 'group', 'extreme_ood'],
                        default='random', help='Type of data split (default: random)')
    parser.add_argument('--n-reps', type=int, default=5,
                        help='Number of independent repetitions (default: 5). '
                             'For --split-type group this equals the number of folds '
                             'in the group-splits file and is set automatically.')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Base random seed; rep i uses seed base_seed+i (default: 42). '
                             'Not used for --split-type group.')
    parser.add_argument('--group-splits-csv', type=str, default=None,
                        help='Path to rxn_groups CSV for --split-type group. '
                            'Default: <data-dir-path>/rxn_groups/'
                             'benchmark_final_group_splits_RI_t055.csv')
    parser.add_argument('--save-full-csvs', action='store_true', default=False,
                        help='Also save full-column split CSVs alongside the index files.')
    args = parser.parse_args()

    csv_path   = os.path.join(args.data_dir_path, args.csv_path)
    splits_dir = args.splits_dir or os.path.join(args.data_dir_path,
                                                  f'{args.split_type}_splits')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    print(f"Loading benchmark from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    uid_to_row = {uid: i for i, uid in enumerate(df['Unique_identifier'])}
    total = len(df)
    print(f"  {total:,} reactions loaded")

    os.makedirs(splits_dir, exist_ok=True)

    columns_to_keep = [
        "Unique_identifier", "IsBalanced", "Balance",
        "MissingMw", "MissingNrAtoms", "MissingNrCarbons",
        "rxn_class", "1kTPLLabel", "FlowER_ID",
        "TargetReaction_with_stereo", "cluster_id",
        "InputReaction_harmonized", "input_target_diff_harmonized",
    ]
    columns_to_keep = [c for c in columns_to_keep if c in df.columns]

        # ── Group/OOD split ───────────────────────────────────────────────────────
    if args.split_type == 'group':
        group_csv = args.group_splits_csv or os.path.join(
            args.data_dir_path,
            'rxn_groups',
            'benchmark_final_group_splits_RI_t055.csv',
        )
        if not os.path.exists(group_csv):
            raise FileNotFoundError(f"Group splits CSV not found: {group_csv}")
        print(f"\nUsing group splits from {group_csv}")

        # Load fold assignments with Unique_identifier for subsampling
        _header = pd.read_csv(group_csv, nrows=0).columns.tolist()
        _fold_cols_in_file = [c for c in _header if c.startswith('ood_fold')]
        _tmp = pd.read_csv(group_csv, usecols=['Unique_identifier'] + _fold_cols_in_file)
        _fold_col = _fold_cols_in_file[0]
        folds = sorted(f for f in _tmp[_fold_col].unique() if f != -1)
        fold_sizes = {f: int((_tmp[_fold_col] == f).sum()) for f in folds}
        total_n = len(_tmp)
        target_n = total_n * 0.10   # target 10% per fold

        # Step 1: pick 5 test folds closest to 10%
        n_reps = 5
        ranked = sorted(folds, key=lambda f: abs(fold_sizes[f] / total_n - 0.10))
        test_folds = ranked[:n_reps]

        # Step 2: for each rep pick valid fold closest to 10% (any fold except
        # the test fold); subsample it down to target_n if it exceeds that
        pairs = []   # (test_fold, valid_fold, subsample_n_or_None)
        for tf in test_folds:
            remaining = [f for f in folds if f != tf]
            vf = min(remaining, key=lambda f: abs(fold_sizes[f] / total_n - 0.10))
            ss = int(round(target_n)) if fold_sizes[vf] > int(round(target_n)) else None
            pairs.append((tf, vf, ss))

        print(f"  Fold sizes: { {int(f): sz for f, sz in fold_sizes.items()} }")
        print(f"  {'Rep':>3}  {'test':>5}  {'test%':>6}  "
              f"{'valid_fold':>10}  {'valid_used':>10}  {'valid%':>6}  {'train%':>7}")
        for i, (tf, vf, ss) in enumerate(pairs):
            pt = fold_sizes[tf] / total_n * 100
            pv = (ss if ss else fold_sizes[vf]) / total_n * 100
            print(f"  {i:>3}  {int(tf):>5}  {pt:>5.1f}%  "
                  f"  {int(vf):>8}  {(ss if ss else fold_sizes[vf]):>10,}  "
                  f"{pv:>5.1f}%  {100-pt-pv:>6.1f}%")

        rng_valid = np.random.default_rng(42)
        all_reps_meta = []

        for rep, (test_fold, valid_fold, subsample_n) in enumerate(pairs):
            test_uids = set(_tmp.loc[_tmp[_fold_col] == test_fold, 'Unique_identifier'])
            valid_uids_all = _tmp.loc[_tmp[_fold_col] == valid_fold, 'Unique_identifier'].tolist()

            if subsample_n is not None:
                chosen_idx = rng_valid.choice(len(valid_uids_all), size=subsample_n, replace=False)
                valid_uids = set(np.array(valid_uids_all)[chosen_idx])
                leftover_uids = set(valid_uids_all) - valid_uids
            else:
                valid_uids    = set(valid_uids_all)
                leftover_uids = set()

            # Train = all folds except test + used valid reactions
            other_uids = set(_tmp.loc[
                ~_tmp[_fold_col].isin([test_fold, valid_fold]) & (_tmp[_fold_col] != -1),
                'Unique_identifier'
            ])
            train_uids = other_uids | leftover_uids

            test_df  = df[df['Unique_identifier'].isin(test_uids)].reset_index(drop=True)
            valid_df = df[df['Unique_identifier'].isin(valid_uids)].reset_index(drop=True)
            train_df = df[df['Unique_identifier'].isin(train_uids)].reset_index(drop=True)

            print(f"\nRep {rep}  test_fold={int(test_fold)}  valid_fold={int(valid_fold)}"
                  + (f"  (valid subsampled to {subsample_n:,})" if subsample_n else ""))
            print(f"  train: {len(train_df):,}  valid: {len(valid_df):,}  test: {len(test_df):,}"
                  f"  ({len(train_df)/total:.1%} / {len(valid_df)/total:.1%} / {len(test_df)/total:.1%})")

            canonical = build_canonical_split_df(train_df, valid_df, test_df, uid_to_row)
            canonical = canonical.rename(columns={'random_split': 'group_split'})
            out_name  = f'benchmark_final_group_splits_rep{rep}.csv'
            out_path  = os.path.join(splits_dir, out_name)
            canonical.to_csv(out_path, index=False)
            print(f"  Saved canonical split → {out_path}")

            all_reps_meta.append({
                'rep': rep,
                'test_fold':  int(test_fold),
                'valid_fold': int(valid_fold),
                'valid_subsampled': subsample_n is not None,
                'valid_n_used': len(valid_df),
                'train_folds_full': [int(f) for f in folds if f not in (test_fold, valid_fold)],
                'n_train': len(train_df),
                'n_valid': len(valid_df),
                'n_test':  len(test_df),
                'pct_train': round(len(train_df) / total * 100, 2),
                'pct_valid': round(len(valid_df) / total * 100, 2),
                'pct_test':  round(len(test_df)  / total * 100, 2),
            })

            if args.save_full_csvs:
                suffix = f'_group_rep{rep}_test{int(test_fold)}_valid{int(valid_fold)}'
                for name, part_df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
                    part_df[columns_to_keep].to_csv(
                        os.path.join(args.data_dir_path, f'{name}_split{suffix}.csv'),
                        index=False,
                    )
                print(f"  Saved full-column split CSVs with suffix '{suffix}'")

        # Save single metadata JSON covering all reps
        meta = {
            'source_file': os.path.abspath(group_csv),
            'fold_column': _fold_col,
            'fold_sizes': {str(int(f)): sz for f, sz in fold_sizes.items()},
            'reps': all_reps_meta,
        }
        meta_path = os.path.join(splits_dir, 'benchmark_final_group_splits_meta.json')
        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=2)
        print(f"\nSaved metadata → {meta_path}")
        print(f"\nDone. {n_reps} group-split repetitions saved to {splits_dir}")
        return

    # ── Random ────────────────────────────────────────────────
    elif args.split_type == 'random':
        for rep in range(args.n_reps):
            seed = args.base_seed + rep
            print(f"\nRepetition {rep}  (seed={seed})")

            if args.split_type == 'random':
                train_df, valid_df, test_df = split_data_random(df, random_state=seed)

            print(f"  train: {len(train_df):,}  valid: {len(valid_df):,}  test: {len(test_df):,}"
                f"  ({len(train_df)/total:.1%} / {len(valid_df)/total:.1%} / {len(test_df)/total:.1%})")

            # ── Canonical index file ──────────────────────────────────────────────
            canonical = build_canonical_split_df(train_df, valid_df, test_df, uid_to_row)
            out_name = f'benchmark_final_{args.split_type}_splits_rep{rep}_seed{seed}.csv'
            out_path = os.path.join(splits_dir, out_name)
            canonical.to_csv(out_path, index=False)
            print(f"  Saved canonical split → {out_path}")

            # ── Full-column CSVs (optional) ───────────────────────────────────────
            # this requires a lot of disk space, so it's optional and only saved if --save-full-csvs is set.
            if args.save_full_csvs:
                suffix = f'_{args.split_type}_rep{rep}_seed{seed}'
                for name, part_df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
                    part_df[columns_to_keep].to_csv(
                        os.path.join(args.data_dir_path, f'{name}_split{suffix}.csv'),
                        index=False,
                    )
                print(f"  Saved full-column split CSVs with suffix '{suffix}'")

        print(f"\nDone. {args.n_reps} repetition(s) saved to {splits_dir}")
    
    elif args.split_type == 'extreme_ood':
        # --- EXTREME OOD MISSING STATS SPLIT: 5 reps, each with a different test fold ---
        ood_splits_csv = args.group_splits_csv or os.path.join(
            args.data_dir_path,
            'ood_folds_missing_stats',
            'benchmark_final_ood_splits_RI_t055.csv',
        )
        group_summary_csv = os.path.join(
            args.data_dir_path,
            'ood_folds_missing_stats',
            'benchmark_final_ood_group_summary_RI_t055.csv',
        )
        if not os.path.exists(ood_splits_csv):
            raise FileNotFoundError(f"OOD splits CSV not found: {ood_splits_csv}")
        if not os.path.exists(group_summary_csv):
            raise FileNotFoundError(f"Group summary CSV not found: {group_summary_csv}")
        splits_df = pd.read_csv(ood_splits_csv)
        group_df = pd.read_csv(group_summary_csv)
        fold_col = [c for c in splits_df.columns if c.startswith('ood_fold')][0]
        group_col = [c for c in splits_df.columns if c.startswith('leakage_group_id')][0]
        # Identify the 5 largest/most populated test folds
        all_folds = [f for f in splits_df[fold_col].unique() if f != -1]
        fold_sizes = {f: (splits_df[fold_col] == f).sum() for f in all_folds}
        test_folds = sorted(all_folds, key=lambda f: -fold_sizes[f])[:5]
        print(f"Extreme OOD: using test folds {test_folds}")
        # Prepare group isolation mapping
        group_summary_col = [c for c in group_df.columns if c.startswith('group_id')][0]
        nn_dist_col = 'nn_dist'
        group_isolation = group_df.set_index(group_summary_col)[nn_dist_col].to_dict()
        # For each rep, use one test fold, rest for train/valid
        for rep, test_fold in enumerate(test_folds):
            print(f"\nRep {rep}: test_fold={test_fold}")
            test_uids = set(splits_df[splits_df[fold_col] == test_fold]['Unique_identifier'])
            # Remaining data (fold == -1 or not in test_fold)
            remaining_mask = (splits_df[fold_col] != test_fold)
            remaining_df = splits_df[remaining_mask].copy()
            # Only use train-only (fold == -1) for IID selection
            train_only_mask = remaining_df[fold_col] == -1
            train_only_df = remaining_df[train_only_mask].copy()
            train_only_df['nn_dist'] = train_only_df[group_col].map(group_isolation)
            train_only_df = train_only_df.sort_values('nn_dist', ascending=True)
            n_test = len(test_uids)
            n_train = int(0.8 * total)
            n_valid = total - n_train - n_test
            # Ensure we don't select more train than available
            n_train = min(n_train, len(train_only_df))
            n_valid = max(0, len(train_only_df) - n_train)
            train_uids = set(train_only_df['Unique_identifier'].iloc[:n_train])
            valid_uids = set(train_only_df['Unique_identifier'].iloc[n_train:n_train+n_valid])
            # Build DataFrames
            train_df = df[df['Unique_identifier'].isin(train_uids)].reset_index(drop=True)
            valid_df = df[df['Unique_identifier'].isin(valid_uids)].reset_index(drop=True)
            test_df = df[df['Unique_identifier'].isin(test_uids)].reset_index(drop=True)
            print(f"  train: {len(train_df):,}  valid: {len(valid_df):,}  test: {len(test_df):,}")
            canonical = build_canonical_split_df(train_df, valid_df, test_df, uid_to_row)
            canonical = canonical.rename(columns={'random_split': 'extreme_ood'})
            out_name = f'benchmark_final_extreme_ood_splits_rep{rep}.csv'
            out_path = os.path.join(splits_dir, out_name)
            canonical.to_csv(out_path, index=False)
            print(f"  Saved extreme OOD split → {out_path}")
            if args.save_full_csvs:
                suffix = f'_extreme_ood_rep{rep}'
                for name, part_df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
                    part_df[columns_to_keep].to_csv(
                        os.path.join(args.data_dir_path, f'{name}_split{suffix}.csv'),
                        index=False,
                    )
                print(f"  Saved full-column split CSVs with suffix '{suffix}'")
        print(f"\nDone. Extreme OOD splits for 5 reps saved to {splits_dir}")
        return


if __name__ == '__main__':
    main()

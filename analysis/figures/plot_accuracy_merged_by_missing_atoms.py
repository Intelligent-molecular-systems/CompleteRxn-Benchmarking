#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot top-1 accuracy as a function of the number of missing atoms.

Edit the configuration block below and run this file directly.
Use MODE = 'multi_fold' to pool all repetitions of one split type and compare
experiments (e.g., CBS_PT vs BS_PT) as lines with shaded ±1 std bands.
Use MODE = 'single' for a single fold stacked-bar chart.
"""

import os
import re
import sys
import math
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.plot_style import apply_neurips_style


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
MODE = 'multi_fold'   # 'single' or 'multi_fold'

SINGLE_CONFIG = {
    'top1_stats':        'results/main/predictions/crb/random/random_seed42_rep0/top1_predictions_with_stats.txt',
    'benchmark_final':   'data/Template_curated/benchmark/benchmark_final.csv',
    'split_csv':         'data/Template_curated/benchmark/random_splits/benchmark_final_random_splits_rep0_seed42.csv',
    'split_col':         'random_split',
    'split_label':       2,
    'missing_atoms_col': 'MissingNrCarbons',
    'max_missing':       10,
    'accuracy_metric_col': 'matches_selected_metric',
    'output':            'results/main/figures/accuracy_by_missing_atoms_single.png',
}

split_type = 'extreme_ood'  # fallback when split_types is not provided
MULTI_FOLD_CONFIG = {
    'results_dir':       'results/main/predictions',
    'experiments':       ['CBS_PT', 'BS_PT'],
    # Optional subset for plotting, e.g. ['CBS_PT'] for CRB-only figure.
    'experiments_to_plot': ['CBS_PT', 'BS_PT'],
    'split_type':        split_type,        # 'random', 'group', or 'extreme_ood'
    'benchmark_final':   'data/Template_curated/benchmark/benchmark_final.csv',
    'splits_dir':        'data/Template_curated/benchmark',
    'missing_atoms_col': 'MissingNrCarbons',
    'accuracy_metric_col': 'matches_selected_metric',
    'max_missing':       16,
    'split_types':       ['random', 'group', 'extreme_ood'],
    'summary_experiment': 'CBS_PT',
    'binning_mode':      'user',         # 'fixed', 'quantile', 'min_count', or 'user'
    'quantile_n_bins':   8,
    'min_bin_size':      500,
    # Only used when binning_mode='user' or second_figure_mode='user'.
    # Each tuple is an inclusive [low, high] range after capping at max_missing.
    'user_bin_ranges':   [(0, 0), (1, 1), (2, 2), (3,3), (4, 4), (5, 5), (6,6), (7, 7), (8, 8), (9,9), (10,11), (12, 13), (14, 15), (16, 16)],
    'second_figure_mode': 'min_count',
    'second_figure_output': 'results/main/figures/accuracy_random_group_by_missing_atoms_multifold_mincount.png',
    'fraction_output':   'results/main/figures/accuracy_random_group_by_missing_atoms_multifold_fractions.png',
    'lower_panel_mode':  'counts_log10',   # kept for backwards compatibility
    'experiment_display_names': {
        'BS_PT': 'RB',
        'CBS_PT': 'CRB',
    },
    'split_display_names': {
        'random': 'Random',
        'group': 'Group',
        'extreme_ood': 'Extreme OOD',
    },
    'split_linestyles': {
        'random': '-',
        'group': '--',
        'extreme_ood': ':',
    },
    'report_pvalues': True,
    'report_pvalues_per_bin': False,
    'min_bin_n_for_pvalue': 30,
    'output':            'results/main/figures/accuracy_random_group_by_missing_atoms_multifold.png',
}

# Colors per experiment for the multi-fold line plot
EXPERIMENT_COLORS = {
    'CBS_PT': '#009B77',
    'BS_PT':  '#EC6842',
}

COLORS = {
    'Accurate':               '#009B77',
    'Balanced, not accurate': '#EC6842',
    'Imbalanced':             '#A50034',
}
CATEGORY_ORDER = ['Accurate', 'Balanced, not accurate', 'Imbalanced']

# Split type metadata: (result_dir_pattern, split_subdir, split_csv_pattern, split_col)
SPLIT_META = {
    'random':      (r'^random_seed(\d+)_rep(\d+)$',  'random_splits',      'benchmark_final_random_splits_rep{rep}_seed{seed}.csv', 'random_split'),
    'group':       (r'^group_rep(\d+)$',              'group_splits',       'benchmark_final_group_splits_rep{rep}.csv',             'group_split'),
    'extreme_ood': (r'^extreme_ood_rep(\d+)$',        'extreme_ood_splits', 'benchmark_final_extreme_ood_splits_rep{rep}.csv',       'extreme_ood'),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        {'true': True, 'false': False, '1': True, '0': False}
    )


def load_stats(top1_stats_path: str) -> pd.DataFrame:
    df = pd.read_csv(top1_stats_path, sep='\t')
    from utils.scoring import overlay_scores
    df = overlay_scores(df,top1_stats_path)
    if 'matches_ground_truth' in df.columns:
        df['matches_ground_truth'] = parse_bool_series(df['matches_ground_truth'])
    if 'matches_selected_metric' in df.columns:
        df['matches_selected_metric'] = parse_bool_series(df['matches_selected_metric'])
    df['is_balanced'] = parse_bool_series(df['is_balanced'])
    return df


def load_benchmark_subset(benchmark_path: str, split_csv_path: str,
                           split_col: str, split_label: int) -> pd.DataFrame:
    bench = pd.read_csv(benchmark_path, low_memory=False)
    split_df = pd.read_csv(split_csv_path)
    uids = split_df.loc[split_df[split_col] == split_label, 'Unique_identifier']
    return bench[bench['Unique_identifier'].isin(uids)].reset_index(drop=True)


def assign_category(df: pd.DataFrame) -> pd.Series:
    cats = pd.Series('Imbalanced', index=df.index)
    cats[df['is_balanced']] = 'Balanced, not accurate'
    cats[df['matches_ground_truth']] = 'Accurate'
    return cats


def fraction_accurate_per_bin(df: pd.DataFrame, missing_col: str, max_m: int) -> pd.Series:
    """Return fraction of accurate predictions per missing-atom bin (index = bin value)."""
    df = df.copy()
    df['missing_bin'] = df[missing_col].clip(upper=max_m)
    df['accurate'] = df['matches_ground_truth'].astype(float)
    return df.groupby('missing_bin')['accurate'].mean()


def build_bin_spec(
    reference_values: pd.Series,
    mode: str,
    max_m: int,
    quantile_n_bins: int,
    min_bin_size: int,
    user_bin_ranges,
) -> dict:
    """Create bin specification with ordered bin ids and labels."""
    capped = reference_values.fillna(0).astype(float).clip(upper=max_m)

    if mode == 'fixed':
        labels = [str(i) for i in range(max_m)] + [f'>= {max_m}']
        return {
            'mode': 'fixed',
            'max_missing': max_m,
            'bin_ids': list(range(max_m + 1)),
            'labels': labels,
            'edges': None,
            'ranges': [(i, i) for i in range(max_m)] + [(max_m, max_m)],
        }

    if mode == 'min_count':
        value_counts = capped.astype(int).value_counts().sort_index()
        if value_counts.empty:
            return build_bin_spec(capped, 'fixed', max_m, quantile_n_bins, min_bin_size, user_bin_ranges)

        ranges = []
        start = None
        running = 0
        for value, count in value_counts.items():
            if start is None:
                start = int(value)
                running = 0
            running += int(count)
            if running >= min_bin_size:
                ranges.append((start, int(value)))
                start = None
                running = 0

        if start is not None:
            if ranges:
                prev_start, _ = ranges[-1]
                ranges[-1] = (prev_start, int(value_counts.index.max()))
            else:
                ranges = [(int(value_counts.index.min()), int(value_counts.index.max()))]

        labels = []
        for lo, hi in ranges:
            if lo == hi:
                labels.append(str(lo))
            elif hi >= max_m:
                labels.append(f'>{lo}')
            else:
                labels.append(f'{lo}-{hi}')

        return {
            'mode': 'min_count',
            'max_missing': max_m,
            'bin_ids': list(range(len(ranges))),
            'labels': labels,
            'edges': None,
            'ranges': ranges,
            'min_bin_size': min_bin_size,
        }

    if mode == 'user':
        if not user_bin_ranges:
            raise ValueError("user_bin_ranges must be provided when binning_mode='user'.")

        cleaned_ranges = []
        for item in user_bin_ranges:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each user bin must be a 2-item tuple/list: (low, high).")
            lo_raw, hi_raw = item
            lo = int(lo_raw)
            hi = max_m if hi_raw is None else int(hi_raw)
            if lo > hi:
                raise ValueError(f"Invalid user bin range ({lo}, {hi}): low must be <= high.")
            lo = max(0, lo)
            hi = min(max_m, hi)
            cleaned_ranges.append((lo, hi))

        cleaned_ranges = sorted(cleaned_ranges, key=lambda r: (r[0], r[1]))
        merged_ranges = []
        for lo, hi in cleaned_ranges:
            if not merged_ranges:
                merged_ranges.append((lo, hi))
                continue
            prev_lo, prev_hi = merged_ranges[-1]
            if lo <= prev_hi:
                raise ValueError("user_bin_ranges overlap. Please provide non-overlapping inclusive ranges.")
            merged_ranges.append((lo, hi))

        if merged_ranges[0][0] > 0:
            raise ValueError("user_bin_ranges must start at 0 or below.")
        if merged_ranges[-1][1] < max_m:
            raise ValueError(f"user_bin_ranges must cover values up to max_missing={max_m}.")

        labels = []
        for lo, hi in merged_ranges:
            if lo == hi:
                labels.append(str(lo))
            elif hi >= max_m:
                labels.append(f'{lo}->= {max_m}')
            else:
                labels.append(f'{lo}-{hi}')

        return {
            'mode': 'user',
            'max_missing': max_m,
            'bin_ids': list(range(len(merged_ranges))),
            'labels': labels,
            'edges': None,
            'ranges': merged_ranges,
        }

    if mode != 'quantile':
        raise ValueError("binning_mode must be 'fixed', 'quantile', 'min_count', or 'user'.")

    n_unique = int(capped.nunique())
    n_bins = int(min(max(2, quantile_n_bins), n_unique))
    if n_bins < 2:
        return build_bin_spec(capped, 'fixed', max_m, quantile_n_bins, min_bin_size, user_bin_ranges)

    _, edges = pd.qcut(capped, q=n_bins, retbins=True, duplicates='drop')
    if len(edges) < 3:
        return build_bin_spec(capped, 'fixed', max_m, quantile_n_bins, min_bin_size, user_bin_ranges)

    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        left_i = int(round(left))
        right_i = int(round(right))
        labels.append(str(left_i) if left_i == right_i else f'{left_i}-{right_i}')

    return {
        'mode': 'quantile',
        'max_missing': max_m,
        'bin_ids': list(range(len(edges) - 1)),
        'labels': labels,
        'edges': edges,
        'ranges': None,
    }


def assign_bins(values: pd.Series, bin_spec: dict) -> pd.Series:
    """Assign each value to a bin id according to bin_spec."""
    capped = values.fillna(0).astype(float).clip(upper=bin_spec['max_missing'])

    if bin_spec['mode'] == 'fixed':
        return capped.astype(int)

    if bin_spec['mode'] in ('min_count', 'user'):
        out = pd.Series(0, index=capped.index, dtype='int64')
        for bin_id, (lo, hi) in enumerate(bin_spec['ranges']):
            mask = (capped >= lo) & (capped <= hi)
            out.loc[mask] = bin_id
        return out

    bins = pd.cut(capped, bins=bin_spec['edges'], labels=False, include_lowest=True, right=True)
    bins = bins.fillna(0).astype(int)
    return bins


def two_proportion_pvalue(success_1: int, total_1: int, success_2: int, total_2: int) -> float:
    """Two-sided z-test p-value for two proportions."""
    if total_1 <= 0 or total_2 <= 0:
        return float('nan')
    p1 = success_1 / total_1
    p2 = success_2 / total_2
    pooled = (success_1 + success_2) / (total_1 + total_2)
    se = math.sqrt(max(pooled * (1.0 - pooled) * (1.0 / total_1 + 1.0 / total_2), 0.0))
    if se == 0.0:
        return 1.0
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2.0))


def report_significance_tests(cfg: dict, all_fold_dfs: dict, bin_spec: dict) -> None:
    """Print significance comparisons between experiments by split type."""
    if not cfg.get('report_pvalues', True):
        return

    split_types = cfg.get('split_types') or [cfg['split_type']]
    selected_experiments = cfg.get('experiments_to_plot') or cfg.get('experiments', [])
    split_display_names = cfg.get('split_display_names', {})
    exp_display = cfg.get('experiment_display_names', {})
    accuracy_metric_col = cfg.get('accuracy_metric_col', 'matches_selected_metric')
    missing_col = cfg['missing_atoms_col']
    min_bin_n = int(cfg.get('min_bin_n_for_pvalue', 30))

    print('\nSignificance tests (two-proportion z-test):')
    print('Overall pooled across folds per split:')
    for split_name in split_types:
        if split_name not in all_fold_dfs:
            continue
        experiments = [e for e, dfs in all_fold_dfs[split_name].items() if dfs and e in selected_experiments]
        if len(experiments) < 2:
            continue
        for exp_a, exp_b in itertools.combinations(experiments, 2):
            df_a = pd.concat(all_fold_dfs[split_name][exp_a], ignore_index=True)
            df_b = pd.concat(all_fold_dfs[split_name][exp_b], ignore_index=True)
            s1 = int(df_a[accuracy_metric_col].sum())
            n1 = int(len(df_a))
            s2 = int(df_b[accuracy_metric_col].sum())
            n2 = int(len(df_b))
            p = two_proportion_pvalue(s1, n1, s2, n2)
            a_name = exp_display.get(exp_a, exp_a)
            b_name = exp_display.get(exp_b, exp_b)
            split_label = split_display_names.get(split_name, split_name)
            acc1 = (s1 / n1) if n1 else float('nan')
            acc2 = (s2 / n2) if n2 else float('nan')
            print(
                f"  {split_label}: {a_name} vs {b_name} | "
                f"acc={acc1:.4f} vs {acc2:.4f}, n={n1} vs {n2}, p={p:.3e}"
            )

    if not cfg.get('report_pvalues_per_bin', False):
        return

    print(f"Per-bin pooled tests (only bins with n >= {min_bin_n} for both models):")
    for split_name in split_types:
        if split_name not in all_fold_dfs:
            continue
        experiments = [e for e, dfs in all_fold_dfs[split_name].items() if dfs and e in selected_experiments]
        if len(experiments) < 2:
            continue
        for exp_a, exp_b in itertools.combinations(experiments, 2):
            df_a = pd.concat(all_fold_dfs[split_name][exp_a], ignore_index=True).copy()
            df_b = pd.concat(all_fold_dfs[split_name][exp_b], ignore_index=True).copy()
            df_a['missing_bin'] = assign_bins(df_a[missing_col], bin_spec)
            df_b['missing_bin'] = assign_bins(df_b[missing_col], bin_spec)

            grp_a = df_a.groupby('missing_bin')[accuracy_metric_col].agg(['sum', 'count']).reindex(bin_spec['bin_ids']).fillna(0)
            grp_b = df_b.groupby('missing_bin')[accuracy_metric_col].agg(['sum', 'count']).reindex(bin_spec['bin_ids']).fillna(0)
            a_name = exp_display.get(exp_a, exp_a)
            b_name = exp_display.get(exp_b, exp_b)
            split_label = split_display_names.get(split_name, split_name)

            for bin_id, label in zip(bin_spec['bin_ids'], bin_spec['labels']):
                s1 = int(grp_a.loc[bin_id, 'sum'])
                n1 = int(grp_a.loc[bin_id, 'count'])
                s2 = int(grp_b.loc[bin_id, 'sum'])
                n2 = int(grp_b.loc[bin_id, 'count'])
                if n1 < min_bin_n or n2 < min_bin_n:
                    continue
                p = two_proportion_pvalue(s1, n1, s2, n2)
                print(f"  {split_label}, bin={label}: {a_name} vs {b_name} | p={p:.3e}")


def render_multi_fold_plot(cfg: dict, all_fold_dfs: dict, bin_mode: str, out_path: str) -> None:
    """Render a three-panel multi-fold plot using a specified binning mode."""
    split_types = cfg.get('split_types') or [cfg['split_type']]
    max_m = cfg['max_missing']
    missing_col = cfg['missing_atoms_col']
    accuracy_metric_col = cfg.get('accuracy_metric_col', 'matches_selected_metric')
    quantile_n_bins = cfg.get('quantile_n_bins', 8)
    min_bin_size = cfg.get('min_bin_size', 5000)
    user_bin_ranges = cfg.get('user_bin_ranges')
    experiment_display_names = cfg.get('experiment_display_names', {})
    selected_experiments = cfg.get('experiments_to_plot') or cfg.get('experiments', [])
    split_display_names = cfg.get('split_display_names', {})
    split_linestyles = cfg.get('split_linestyles', {})

    summary_experiment = cfg.get('summary_experiment')
    non_empty_splits = [
        split_name for split_name in split_types
        if split_name in all_fold_dfs and any(all_fold_dfs[split_name].values())
    ]
    if not non_empty_splits:
        raise ValueError('No fold data available to plot.')

    if summary_experiment not in all_fold_dfs.get(non_empty_splits[0], {}):
        non_empty = []
        for split_name in non_empty_splits:
            non_empty.extend([exp for exp, dfs in all_fold_dfs[split_name].items() if dfs])
        if not non_empty:
            raise ValueError('No fold data available to plot.')
        summary_experiment = non_empty[0]

    reference_values = pd.concat(
        [
            fold_df[missing_col]
            for split_name in non_empty_splits
            for fold_df in all_fold_dfs[split_name].get(summary_experiment, [])
        ],
        ignore_index=True,
    )
    bin_spec = build_bin_spec(
        reference_values,
        bin_mode,
        max_m,
        quantile_n_bins,
        min_bin_size,
        user_bin_ranges,
    )
    all_bins = bin_spec['bin_ids']
    x_labels = bin_spec['labels']

    if out_path == cfg.get('output'):
        report_significance_tests(cfg, all_fold_dfs, bin_spec)

    all_fractions = {}
    summary_counts_by_split = {}
    summary_totals_by_split = {}
    summary_fractions_by_split = {}
    for split_name in non_empty_splits:
        split_fraction_data = {}
        summary_rows = []
        for experiment, fold_dfs in all_fold_dfs[split_name].items():
            if experiment not in selected_experiments:
                continue
            exp_series = []
            for fold_df in fold_dfs:
                binned = fold_df.copy()
                binned['missing_bin'] = assign_bins(binned[missing_col], bin_spec)
                frac = binned.groupby('missing_bin')[accuracy_metric_col].mean().reindex(all_bins)
                exp_series.append(frac)
                if experiment == summary_experiment:
                    summary_rows.append(binned[['missing_bin', 'category']].copy())
            split_fraction_data[experiment] = exp_series
        all_fractions[split_name] = split_fraction_data

        summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame(columns=['missing_bin', 'category'])
        n_reps = len(all_fold_dfs[split_name][summary_experiment])
        if len(summary_df):
            summary_counts = (
                summary_df.groupby(['missing_bin', 'category'])
                .size()
                .unstack(fill_value=0)
                .reindex(index=all_bins, fill_value=0)
                .reindex(columns=CATEGORY_ORDER, fill_value=0)
                / n_reps
            )
        else:
            summary_counts = pd.DataFrame(0, index=all_bins, columns=CATEGORY_ORDER)
        summary_totals = summary_counts.sum(axis=1)
        summary_fractions = summary_counts.div(summary_totals.where(summary_totals > 0), axis=0).fillna(0.0)
        summary_counts_by_split[split_name] = summary_counts
        summary_totals_by_split[split_name] = summary_totals
        summary_fractions_by_split[split_name] = summary_fractions

    apply_neurips_style()
    plt.rcParams['axes.axisbelow'] = True
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.4, 4.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={'height_ratios': [1.0, 0.9, 0.9], 'hspace': 0.04},
    )
    ax_top, ax_mid, ax_bottom = axes

    x = list(range(len(all_bins)))
    n_splits = len(non_empty_splits)
    experiment_order = [e for e in selected_experiments if any(all_fractions[s].get(e, []) for s in non_empty_splits)]
    n_experiments = max(len(experiment_order), 1)
    split_offsets_top = {
        split_name: (idx - (n_splits - 1) / 2) * 0.22
        for idx, split_name in enumerate(non_empty_splits)
    }
    experiment_offsets_top = {
        exp: (idx - (n_experiments - 1) / 2) * 0.10
        for idx, exp in enumerate(experiment_order)
    }
    split_point_colors = {
        'random': '#4C78A8',
        'group': '#F58518',
        'extreme_ood': '#54A24B',
    }
    experiment_markers = {
        'CBS_PT': 'o',
        'BS_PT': 'D',
    }
    experiment_fillstyle = {
        'CBS_PT': 'full',
        'BS_PT': 'none',
    }

    for split_name in non_empty_splits:
        for experiment, fold_series in all_fractions[split_name].items():
            if not fold_series:
                continue
            aligned = pd.DataFrame({i: s.reindex(all_bins) for i, s in enumerate(fold_series)}, index=all_bins)
            mean = aligned.mean(axis=1)
            q25 = aligned.quantile(0.25, axis=1)
            q75 = aligned.quantile(0.75, axis=1)
            lower_err = (mean - q25).clip(lower=0)
            upper_err = (q75 - mean).clip(lower=0)

            color = split_point_colors.get(split_name, '#4C78A8')
            display_name = experiment_display_names.get(experiment, experiment)
            split_label = split_display_names.get(split_name, split_name)
            marker = experiment_markers.get(experiment, 'o')
            fillstyle = experiment_fillstyle.get(experiment, 'full')
            x_offset = split_offsets_top.get(split_name, 0.0) + experiment_offsets_top.get(experiment, 0.0)
            x_pos = [xi + x_offset for xi in x]
            ax_top.errorbar(
                x_pos,
                mean.values,
                yerr=[lower_err.values, upper_err.values],
                linewidth=0.0,
                linestyle='none',
                marker=marker,
                fillstyle=fillstyle,
                markersize=4.2,
                capsize=2.8,
                elinewidth=1.1,
                markeredgewidth=1.0,
                label=None,
                color=color,
            )

    n_splits = len(non_empty_splits)
    bar_group_width = 0.82
    bar_width = min(0.34, bar_group_width / max(n_splits, 1))
    offsets = [
        (idx - (n_splits - 1) / 2) * bar_width
        for idx in range(n_splits)
    ]
    split_bar_colors = {
        'random': '#6E6E6E',
        'group': '#B0B0B0',
        'extreme_ood': '#D0D0D0',
    }
    split_hatches = {
        'random': '',
        'group': '//',
        'extreme_ood': 'xx',
    }

    max_total = max(
        [int(summary_totals_by_split[split_name].max()) for split_name in non_empty_splits] or [1]
    )
    for idx, split_name in enumerate(non_empty_splits):
        positions = [xi + offsets[idx] for xi in x]
        totals = summary_totals_by_split[split_name].reindex(all_bins).values
        ax_mid.bar(
            positions,
            totals,
            color=split_bar_colors.get(split_name, '#7A7A7A'),
            width=bar_width,
            edgecolor='white',
            linewidth=0.3,
            hatch=split_hatches.get(split_name, ''),
            label=split_display_names.get(split_name, split_name),
        )
        for pos, n in zip(positions, totals):
            if n > 0:
                ax_mid.text(
                    pos,
                    n * 1.08,
                    f'n={int(n)}',
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=5.5,
                    color='#444444',
                )

    ax_mid.set_yscale('log')
    ax_mid.set_ylim(10, max(10, max_total * 20))

    # Bottom panel: category fractions (linear scale).
    for idx, split_name in enumerate(non_empty_splits):
        positions = [xi + offsets[idx] for xi in x]
        bottom_frac = pd.Series(0.0, index=all_bins)
        split_fractions = summary_fractions_by_split[split_name]
        for cat in CATEGORY_ORDER:
            ax_bottom.bar(
                positions,
                split_fractions[cat].values,
                bottom=bottom_frac.values,
                color=COLORS[cat],
                label=cat if idx == 0 else None,
                width=bar_width,
                edgecolor='white',
                linewidth=0.3,
                hatch=split_hatches.get(split_name, ''),
            )
            bottom_frac = bottom_frac + split_fractions[cat].values

    ax_top.set_ylabel('Top-1\naccuracy')
    ax_top.set_ylim(0.4, 1.03)
    ax_top.yaxis.grid(True, linestyle='-', alpha=0.5)
    exp_handles = []
    for exp in experiment_order:
        marker = experiment_markers.get(exp, 'o')
        fill = experiment_fillstyle.get(exp, 'full')
        display = experiment_display_names.get(exp, exp)
        markerface = 'black' if fill != 'none' else 'none'
        exp_handles.append(
            Line2D([0], [0], marker=marker, linestyle='none', color='black', markerfacecolor=markerface,
                   markeredgecolor='black', markeredgewidth=1.0, markersize=5, label=display)
        )
    split_handles = [
        Line2D([0], [0], marker='o', linestyle='none', color=split_point_colors.get(s, '#4C78A8'),
               markerfacecolor=split_point_colors.get(s, '#4C78A8'), markersize=5,
               label=split_display_names.get(s, s))
        for s in non_empty_splits
    ]
    if len(MULTI_FOLD_CONFIG.get('experiments_to_plot', [])) > 1:
        legend_handles = split_handles + exp_handles
    else:
        legend_handles = split_handles
    top_legend = ax_top.legend(
        handles=legend_handles,
        loc='lower left',
        frameon=True,
        framealpha=1.0,
        facecolor='white',
        fontsize=7,
        handlelength=1.0,
        ncol=2 if len(legend_handles) > 3 else 1,
    )
    ax_top.add_artist(top_legend)

    ax_mid.set_ylabel('Avg count per rep')
    ax_mid.yaxis.grid(True, linestyle='-', alpha=0.5)
    if n_splits > 1:
        ax_mid.legend(loc='lower left', frameon=True, framealpha=1.0, facecolor='white', fontsize=7, handlelength=1.2)

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(x_labels)
    ax_bottom.set_xlabel(f'Number of missing {missing_col.replace("MissingNr", "").lower()}')
    ax_bottom.set_ylabel('Prediction Outcome Fraction')
    ax_bottom.set_ylim(0, 1.12)
    ax_bottom.yaxis.grid(True, linestyle='-', alpha=0.5)
    ax_bottom.legend(loc='lower left', frameon=True, framealpha=1.0, facecolor='white', fontsize=7, handlelength=1.0)

    # Keep top panel title empty; manuscript caption will carry context.
    ax_top.set_title('')

    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


def discover_folds(results_dir: str, experiment: str, split_type: str):
    """Return list of dicts with fold metadata for one experiment + split type."""
    pattern_str, split_subdir, csv_template, split_col = SPLIT_META[split_type]
    pattern = re.compile(pattern_str)
    exp_dir = os.path.join(results_dir, {'BS_PT': 'rb', 'CBS_PT': 'crb'}[experiment], split_type)
    folds = []
    for name in sorted(os.listdir(exp_dir)):
        m = pattern.match(name)
        if m is None:
            continue
        rep = int(m.group(2) if split_type == 'random' else m.group(1))
        seed = int(m.group(1)) if split_type == 'random' else None
        top1_stats = os.path.join(exp_dir, name, 'top1_predictions_with_stats.txt')
        if not os.path.isfile(top1_stats):
            print(f"Warning: missing {top1_stats}")
            continue
        folds.append({'rep': rep, 'seed': seed, 'top1_stats': top1_stats,
                      'split_col': split_col, 'split_subdir': split_subdir,
                      'csv_template': csv_template})
    return folds


# ---------------------------------------------------------------------------
# Single-fold mode
# ---------------------------------------------------------------------------

def run_single(cfg: dict) -> None:
    df = load_stats(cfg['top1_stats'])
    bench_subset = load_benchmark_subset(
        cfg['benchmark_final'], cfg['split_csv'],
        cfg['split_col'], cfg['split_label'],
    )

    if 'original_index' not in df.columns:
        raise ValueError("'original_index' column missing from top1_stats.")

    mapped = bench_subset.iloc[df['original_index'].astype(int).values].reset_index(drop=True)
    missing_col = cfg['missing_atoms_col']
    df['missing_atoms'] = mapped[missing_col].fillna(0).astype(int).values
    df['category'] = assign_category(df)

    max_m = cfg['max_missing']
    df['missing_bin'] = df['missing_atoms'].clip(upper=max_m)
    bin_labels = {i: str(i) for i in range(max_m + 1)}
    bin_labels[max_m] = f'>= {max_m}'

    counts = (
        df.groupby(['missing_bin', 'category'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_ORDER, fill_value=0)
    )
    totals = counts.sum(axis=1)
    fractions = counts.div(totals, axis=0)

    apply_neurips_style()
    plt.rcParams['axes.axisbelow'] = True
    fig, ax = plt.subplots(figsize=(12, 3.5))
    bottom = pd.Series(0.0, index=fractions.index)
    x = list(range(len(fractions)))
    x_labels = [bin_labels[b] for b in fractions.index]

    for cat in CATEGORY_ORDER:
        ax.bar(x, fractions[cat], bottom=bottom, color=COLORS[cat],
               label=cat, width=0.65, edgecolor='white', linewidth=0.3)
        bottom = bottom + fractions[cat]

    for i, (n, _) in enumerate(zip(totals, fractions.index)):
        ax.text(i, 1.01, f'n={n}', ha='center', va='bottom', fontsize=6.5, color='#444444')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(f'Number of missing {missing_col.replace("MissingNr", "").lower()}')
    ax.set_ylabel('Fraction of reactions')
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, linestyle='-', alpha=0.5)
    ax.legend(loc='lower left', frameon=True, framealpha=1.0, facecolor='white', fontsize=7, handlelength=1.0, handleheight=0.8)
    ax.set_title('Top-1 accuracy by number of missing atoms')
    plt.tight_layout()

    out = cfg['output']
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved to {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Multi-fold mode
# ---------------------------------------------------------------------------

def run_multi_fold(cfg: dict) -> None:
    missing_col = cfg['missing_atoms_col']
    accuracy_metric_col = cfg.get('accuracy_metric_col', 'matches_selected_metric')
    selected_experiments = cfg.get('experiments_to_plot') or cfg.get('experiments', [])
    split_types = cfg.get('split_types') or [cfg['split_type']]
    split_types = [s for s in split_types if s in SPLIT_META]
    if not split_types:
        raise ValueError("No valid split_types configured. Use any of: random, group, extreme_ood.")
    if not selected_experiments:
        raise ValueError("Set at least one experiment in experiments_to_plot or experiments.")
    bench = pd.read_csv(cfg['benchmark_final'], low_memory=False)

    all_fold_dfs = {}

    for split_type in split_types:
        all_fold_dfs[split_type] = {}
        for experiment in selected_experiments:
            folds = discover_folds(cfg['results_dir'], experiment, split_type)
            if not folds:
                print(f"Warning: no folds found for {experiment} / {split_type}")
                all_fold_dfs[split_type][experiment] = []
                continue
            print(f"{experiment} / {split_type}: found {len(folds)} fold(s)")

            fold_dfs = []
            for fold in folds:
                rep = fold['rep']
                seed = fold['seed']
                csv_name = fold['csv_template'].format(rep=rep, seed=seed)
                split_csv = os.path.join(cfg['splits_dir'], fold['split_subdir'], csv_name)

                if not os.path.isfile(split_csv):
                    print(f"  Warning: split CSV not found: {split_csv}")
                    continue

                split_df = pd.read_csv(split_csv)
                split_col = fold['split_col']
                uids = split_df.loc[split_df[split_col] == 2, 'Unique_identifier']
                bench_sub = bench[bench['Unique_identifier'].isin(uids)].reset_index(drop=True)

                df = load_stats(fold['top1_stats'])
                if 'original_index' not in df.columns:
                    print(f"  Warning: no original_index in {fold['top1_stats']}, skipping")
                    continue
                if accuracy_metric_col not in df.columns:
                    print(f"  Warning: no {accuracy_metric_col} in {fold['top1_stats']}, skipping")
                    continue

                mapped = bench_sub.iloc[df['original_index'].astype(int).values].reset_index(drop=True)
                df[missing_col] = mapped[missing_col].fillna(0).astype(int).values
                df['category'] = assign_category(df)
                fold_dfs.append(df[[accuracy_metric_col, missing_col, 'category']].copy())
                print(f"  rep{rep}: {len(df)} reactions")

            all_fold_dfs[split_type][experiment] = fold_dfs

    primary_mode = cfg.get('binning_mode', 'fixed')
    render_multi_fold_plot(cfg, all_fold_dfs, primary_mode, cfg.get('output'))

    second_out = cfg.get('second_figure_output')
    if second_out:
        second_mode = cfg.get('second_figure_mode')
        if not second_mode:
            second_mode = 'quantile' if primary_mode == 'fixed' else 'fixed'
        render_multi_fold_plot(cfg, all_fold_dfs, second_mode, second_out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    release=os.environ.get("COMPLETERXN_RESULTS_ROOT")
    if release:
        for name in ['SINGLE_CONFIG', 'MULTI_FOLD_CONFIG']:
            config=globals().get(name,{})
            for key,value in list(config.items()):
                if isinstance(value,str) and value.startswith("results/main/figures/"):
                    config[key]=os.path.join(release,value[len("results/"):])

    if MODE == 'multi_fold':
        run_multi_fold(MULTI_FOLD_CONFIG)
    elif MODE == 'single':
        run_single(SINGLE_CONFIG)
    else:
        raise ValueError("MODE must be 'single' or 'multi_fold'.")


if __name__ == '__main__':
    main()

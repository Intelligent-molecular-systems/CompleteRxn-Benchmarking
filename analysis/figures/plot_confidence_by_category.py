#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot prediction confidence for correct, incorrect, and unbalanced cases.

Edit the configuration block below and run this file directly.
Use MODE = 'multi_fold' for the pooled BS/CBS panel + single figures,
or MODE = 'single' for one log_probs/top1_stats pair.
"""

import os
import re
import sys
from types import SimpleNamespace
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.plot_style import apply_neurips_style


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
MODE = 'multi_fold'  # 'single' or 'multi_fold'
N_BEST = 5

MULTI_FOLD_CONFIG = {
    'results_dir': 'results/main/predictions',
    'experiments': ['BS_PT', 'CBS_PT'],
    'split_types': ['random', 'group', 'extreme_ood'],
    'output': 'results/main/figures/confidence_all_folds_bs_cbs.png',
    'multi_summary_csv': 'results/main/figures/confidence_all_folds_bs_cbs_summary.csv',
    'multi_output_dir': '',
}

SINGLE_FOLD_CONFIG = {
    'log_probs': 'results/main/predictions/crb/extreme_ood/extreme_ood_rep0/beamSize5_nBest5.txt_log_probs',
    'top1_stats': 'results/main/predictions/crb/extreme_ood/extreme_ood_rep0/top1_predictions_with_stats.txt',
    'output': 'results/main/figures/confidence_CBS_PT_extreme_ood_rep0.png',
    'output_missing': 'results/main/figures/confidence_CBS_PT_extreme_ood_rep0_missing.png',
    'benchmark_final': 'data/Template_curated/benchmark/benchmark_final.csv',
    'split_csv': 'data/Template_curated/benchmark/extreme_ood_splits/benchmark_final_extreme_ood_splits_rep0.csv',
    'split_col': 'extreme_ood',
    'split_label': 2,
    'missing_atoms_col': 'MissingNrAtoms',
    'missing_carbons_col': 'MissingNrCarbons',
    'identifier_col': 'Unique_identifier',
    'gt_col': 'TargetReaction_with_stereo',
    'input_col': 'InputReaction_harmonized',
    'high_conf_threshold': 0.8,
    'high_conf_output': 'results/main/predictions/crb/extreme_ood/extreme_ood_rep0/high_conf_incorrect.csv',
    'high_conf_balanced_only': False,
    'low_conf_threshold': 0.2,
    'low_conf_output': 'results/main/predictions/crb/extreme_ood/extreme_ood_rep0/low_conf_balanced_incorrect.csv',
}

COLORS = {
    'Accurate':               '#009B77',
    'Balanced, not accurate': '#EC6842',
    'Imbalanced':             '#A50034',
}
CATEGORY_ORDER = ['Accurate', 'Balanced, not accurate', 'Imbalanced']


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Read boolean values stored as numbers or text."""
    return series.astype(str).str.strip().str.lower().map(
        {'true': True, 'false': False, '1': True, '0': False}
    )


def extract_missing_counts(balance_info: str) -> tuple[int, int]:
    """Extract missing atom and carbon counts from balance_info RHS.

    For imbalanced reactions, score script emits strings like:
      "$L:1O $R:2C1H $Charge:0"
    Missing atoms/carbons are interpreted as RHS counts.
    """
    if not isinstance(balance_info, str):
        return 0, 0

    rhs_match = re.search(r'\$R:([^$]*)', balance_info)
    if not rhs_match:
        return 0, 0

    rhs = rhs_match.group(1).strip()
    if not rhs:
        return 0, 0

    atom_matches = re.findall(r'(\d+)([A-Z][a-z]?)', rhs)
    missing_atoms = 0
    missing_carbons = 0
    for count_str, atom in atom_matches:
        count = int(count_str)
        missing_atoms += count
        if atom == 'C':
            missing_carbons += count
    return missing_atoms, missing_carbons


def resolve_split_column(split_df: pd.DataFrame, user_split_col: str) -> str:
    if user_split_col:
        if user_split_col not in split_df.columns:
            raise ValueError(f"Split column '{user_split_col}' not found in split CSV")
        return user_split_col
    for col in ('random_split', 'group_split', 'extreme_ood'):
        if col in split_df.columns:
            return col
    raise ValueError("Could not detect split column. Use --split-col explicitly.")


def _merge_side(base_side, add_side):
    base = [s for s in str(base_side).split('.') if s] if pd.notna(base_side) else []
    add = [s for s in str(add_side).split('.') if s] if pd.notna(add_side) else []
    return '.'.join(base + add)


def compose_completed_reaction(input_reaction, predicted_diff):
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


def build_category_dataframe(df: pd.DataFrame, prob) -> pd.DataFrame:
    return pd.concat([
        pd.DataFrame({'probability': prob[df['matches_ground_truth'].values == True],
                      'category': 'Accurate'}),
        pd.DataFrame({'probability': prob[(df['is_balanced'].values == True) & (df['matches_ground_truth'].values == False)],
                      'category': 'Balanced, not accurate'}),
        pd.DataFrame({'probability': prob[df['is_balanced'].values == False],
                      'category': 'Imbalanced'}),
    ], ignore_index=True)


def load_probs_and_stats(log_probs_path: str, top1_stats_path: str, n_best: int):
    log_probs = pd.read_csv(log_probs_path, header=None, names=['log_prob'])
    df = pd.read_csv(top1_stats_path, sep='\t')
    from utils.scoring import overlay_scores
    df=overlay_scores(df,top1_stats_path)

    if 'original_index' in df.columns:
        prob = 10 ** log_probs.iloc[df['original_index'].values * n_best]['log_prob'].values
    else:
        expected = len(df) * n_best
        if len(log_probs) != expected:
            raise ValueError(
                f"log_probs has {len(log_probs)} lines but expected {expected} "
                f"({len(df)} rows × n_best={n_best}). "
                "If OOV filtering was used, re-run scoring to get a stats file with 'original_index'."
            )
        prob = 10 ** log_probs.iloc[::n_best]['log_prob'].values

    df['matches_ground_truth'] = parse_bool_series(df['matches_ground_truth'])
    df['is_balanced'] = parse_bool_series(df['is_balanced'])
    return df, prob


def parse_fold_from_dirname(dirname: str):
    m_random = re.match(r"random_seed(\d+)_rep(\d+)$", dirname)
    if m_random:
        return "random", int(m_random.group(2)), int(m_random.group(1))
    m_group = re.match(r"group_rep(\d+)$", dirname)
    if m_group:
        return "group", int(m_group.group(1)), None
    m_extreme = re.match(r"extreme_ood_rep(\d+)$", dirname)
    if m_extreme:
        return "extreme_ood", int(m_extreme.group(1)), None
    return None, None, None


def discover_fold_inputs(results_dir: str, experiments, split_types):
    discovered = []
    for experiment_type in experiments:
        exp_dir = os.path.join(results_dir, {'BS_PT': 'rb', 'CBS_PT': 'crb'}[experiment_type])
        if not os.path.isdir(exp_dir):
            print(f"Warning: missing experiment directory: {exp_dir}")
            continue

        for split_type_dir in sorted(os.listdir(exp_dir)):
            if split_type_dir not in split_types:
                continue
            for split_suffix in sorted(os.listdir(os.path.join(exp_dir, split_type_dir))):
                fold_dir = os.path.join(exp_dir, split_type_dir, split_suffix)
                if not os.path.isdir(fold_dir):
                    continue

                split_type, fold, seed = parse_fold_from_dirname(split_suffix)
                if split_type is None or split_type != split_type_dir:
                    continue

                top1_stats = os.path.join(fold_dir, 'top1_predictions_with_stats.txt')
                if not os.path.isfile(top1_stats):
                    print(f"Warning: missing top1 stats file in {fold_dir}")
                    continue

                all_files = sorted(os.listdir(fold_dir))
                log_prob_candidates = [f for f in all_files if f.endswith('_log_probs') or 'log_probs' in f]
                if not log_prob_candidates:
                    print(f"Warning: missing log_probs file in {fold_dir}")
                    continue
                log_probs = os.path.join(fold_dir, log_prob_candidates[0])

                discovered.append({
                    'experiment_type': experiment_type,
                    'split_type': split_type,
                    'fold': fold,
                    'seed': seed,
                    'split_suffix': split_suffix,
                    'log_probs': log_probs,
                    'top1_stats': top1_stats,
                })
    return discovered


def run_multi_fold_mode(opt):
    experiments = [s.strip() for s in opt.experiments.split(',') if s.strip()]
    split_types = [s.strip() for s in opt.split_types.split(',') if s.strip()]
    discovered = discover_fold_inputs(opt.results_dir, experiments, split_types)

    if not discovered:
        raise FileNotFoundError(
            f"No fold inputs found under {opt.results_dir}. "
            "Check --results-dir, --experiments, and --split-types."
        )

    print(f"Discovered {len(discovered)} fold directories")

    all_data = []
    for item in discovered:
        df, prob = load_probs_and_stats(item['log_probs'], item['top1_stats'], opt.n_best)
        fold_data = build_category_dataframe(df, prob)
        fold_data['experiment_type'] = item['experiment_type']
        fold_data['split_type'] = item['split_type']
        fold_data['fold'] = item['fold']
        fold_data['split_suffix'] = item['split_suffix']
        all_data.append(fold_data)

    data = pd.concat(all_data, ignore_index=True)
    data['category'] = pd.Categorical(data['category'], categories=CATEGORY_ORDER, ordered=True)

    row_order = [e for e in experiments if e in set(data['experiment_type'])]
    col_order = [s for s in split_types if s in set(data['split_type'])]

    # Apply seaborn style first, then overwrite typography with our publication settings.
    sns.set_theme(style='whitegrid')
    apply_neurips_style()

    counts = data.groupby(['experiment_type', 'split_type', 'category']).size().reset_index(name='n')
    fold_medians = data.groupby(
        ['experiment_type', 'split_type', 'fold', 'category'], as_index=False
    )['probability'].median()

    if opt.multi_output_dir:
        output_dir = opt.multi_output_dir
    elif opt.output:
        output_dir = os.path.dirname(opt.output) or '.'
    else:
        output_dir = os.path.join('figures', 'ood_folds')
    os.makedirs(output_dir, exist_ok=True)

    if opt.output:
        base_stem = os.path.splitext(os.path.basename(opt.output))[0]
    else:
        base_stem = 'confidence_multifold'

    saved_paths = []
    display_labels = {
        'Accurate': 'Acc.',
        'Balanced, not accurate': 'Bal. not acc.',
        'Imbalanced': 'Imbal.',
    }
    split_title_labels = {
        'random': 'Random',
        'group': 'Group',
        'extreme_ood': 'Extreme OOD',
    }

    if opt.output:
        from matplotlib.patches import Patch

        exp_display = {
            'BS_PT': 'RB',
            'CBS_PT': 'CRB',
        }
        exp_hatches = {
            'BS_PT': '',
            'CBS_PT': '///',
        }
        exp_offsets = {
            'BS_PT': -0.18,
            'CBS_PT': 0.18,
        }

        for j, split_name in enumerate(col_order):
            fig, ax = plt.subplots(figsize=(2.35, 2.7))#, constrained_layout=True)
            tick_labels = [display_labels[cat] for cat in CATEGORY_ORDER]
            # Force identical margins across all plots
            fig.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.95)

            split_exp_totals = (
                counts[counts['split_type'] == split_name]
                .groupby('experiment_type')['n']
                .sum()
                .to_dict()
            )

            n_total_single = None
            if row_order:
                n_total_single = int(split_exp_totals.get(row_order[0], 0))

            for ci, cat in enumerate(CATEGORY_ORDER):
                for exp_name in row_order:
                    subset = data[
                        (data['split_type'] == split_name)
                        & (data['experiment_type'] == exp_name)
                        & (data['category'] == cat)
                    ]
                    if subset.empty:
                        continue

                    x_pos = ci + exp_offsets.get(exp_name, 0.0)
                    vp = ax.violinplot(
                        subset['probability'].values,
                        positions=[x_pos],
                        widths=0.32,
                        showmeans=False,
                        showmedians=True,
                        showextrema=False,
                    )
                    for body in vp['bodies']:
                        body.set_facecolor(COLORS[cat])
                        body.set_edgecolor('#222222')
                        body.set_alpha(0.72)
                        body.set_linewidth(0.55)
                        body.set_hatch(exp_hatches.get(exp_name, ''))
                    if 'cmedians' in vp:
                        vp['cmedians'].set_color('#222222')
                        vp['cmedians'].set_linewidth(0.8)

                    n_cat = len(subset)
                    n_total = int(split_exp_totals.get(exp_name, 0))
                    pct = (100.0 * n_cat / n_total) if n_total else 0.0

                    text_dx = -0.07 if exp_offsets.get(exp_name, 0.0) < 0 else 0.07
                    ha = 'right' if exp_offsets.get(exp_name, 0.0) < 0 else 'left'

                    ax.text(
                        x_pos + text_dx,
                        0.5,
                        f"n={n_cat} ({pct:.1f}%)",
                        rotation=90,
                        ha=ha,
                        va='center',
                        fontsize=7.8,
                        color='#222222',
                        zorder=5,
                    )

            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([k / 10 for k in range(11)])
            ax.yaxis.grid(True, linestyle='-', alpha=0.6)
            ax.set_xticks(range(len(CATEGORY_ORDER)))
            ax.set_xticklabels(tick_labels)
            ax.tick_params(axis='x', pad=1)
            ax.set_xlabel('')
            ax.set_title('')

            # only first panel gets y-axis label and tick labels
            if j == 0:
                ax.set_ylabel('Prediction Probability')
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)

            #ax.text(
            #    0.5,
            #    -0.185,
            #    f"n_total={n_total_single}",
            #    transform=ax.transAxes,
            #    ha='center',
            #    va='top',
            #    fontsize=7.2,
            #    clip_on=False,
            #)

            out_path = os.path.join(output_dir, f'{base_stem}_{split_name}.png')
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved split panel to {out_path}")
            plt.close(fig)

        # save legend separately if useful
        fig_leg, ax_leg = plt.subplots(figsize=(4.6, 0.8))
        ax_leg.axis('off')

        category_handles = [
            Patch(facecolor=COLORS[cat], edgecolor='#222222', linewidth=0.5,
                  label=display_labels[cat].replace('\n', ' '))
            for cat in CATEGORY_ORDER
        ]
        experiment_handles = [
            Patch(facecolor='white', edgecolor='#222222', linewidth=0.7,
                  hatch=exp_hatches.get(exp, ''), label=exp_display.get(exp, exp))
            for exp in row_order
        ]
        all_handles = category_handles + experiment_handles

        fig_leg.legend(
            handles=all_handles,
            loc='center',
            ncol=len(all_handles),
            frameon=True,
            framealpha=0.84,
            facecolor='white',
            fontsize=7,
            handlelength=1.4,
            columnspacing=1.0,
            handletextpad=0.45,
        )
        legend_path = os.path.join(output_dir, f'{base_stem}_legend.png')
        fig_leg.savefig(legend_path, dpi=300, bbox_inches='tight')
        print(f"Saved legend to {legend_path}")
        plt.close(fig_leg)
    
    for exp_name in row_order:
        for split_name in col_order:
            sub = data[(data['experiment_type'] == exp_name) & (data['split_type'] == split_name)]
            if sub.empty:
                continue

            csub = counts[
                (counts['experiment_type'] == exp_name)
                & (counts['split_type'] == split_name)
            ]
            tick_labels = []
            for cat in CATEGORY_ORDER:
                n_cat = int(csub.loc[csub['category'] == cat, 'n'].sum())
                tick_labels.append(f"{display_labels[cat]}\n(n={n_cat})")

            fig, ax = plt.subplots(figsize=(3.4, 2.6))
            sns.violinplot(
                ax=ax,
                data=sub,
                x='category',
                y='probability',
                hue='category',
                order=CATEGORY_ORDER,
                palette=COLORS,
                inner='quartile',
                cut=0,
                linewidth=0.45,
                legend=False,
            )

            msub = fold_medians[
                (fold_medians['experiment_type'] == exp_name)
                & (fold_medians['split_type'] == split_name)
            ]
            if len(msub):
                sns.stripplot(
                    ax=ax,
                    data=msub,
                    x='category',
                    y='probability',
                    order=CATEGORY_ORDER,
                    color='black',
                    size=2.2,
                    jitter=0.10,
                    alpha=0.85,
                )

            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([k / 10 for k in range(11)])
            ax.yaxis.grid(True, linestyle='-', alpha=0.6)
            ax.set_xticks(range(len(CATEGORY_ORDER)))
            ax.set_xticklabels(tick_labels)
            ax.tick_params(axis='x', pad=1)
            ax.set_xlabel('')
            ax.set_ylabel('Prediction Probability')
            ax.set_title('')
            plt.tight_layout()

            out_path = os.path.join(output_dir, f'{base_stem}_{exp_name}_{split_name}.png')
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            saved_paths.append(out_path)
            plt.close(fig)

    print(f"Saved {len(saved_paths)} multi-fold plot files to {output_dir}")
    for p in saved_paths:
        print(f"  - {p}")

    summary = data.groupby(['experiment_type', 'split_type', 'category'], as_index=False).agg(
        pooled_n=('probability', 'size'),
        pooled_mean=('probability', 'mean'),
        pooled_median=('probability', 'median'),
        pooled_q25=('probability', lambda s: s.quantile(0.25)),
        pooled_q75=('probability', lambda s: s.quantile(0.75)),
    )
    fold_summary = fold_medians.groupby(['experiment_type', 'split_type', 'category'], as_index=False).agg(
        fold_count=('fold', 'nunique'),
        fold_median_mean=('probability', 'mean'),
        fold_median_std=('probability', 'std'),
        fold_median_min=('probability', 'min'),
        fold_median_max=('probability', 'max'),
    )
    summary = summary.merge(fold_summary, on=['experiment_type', 'split_type', 'category'], how='left')

    if opt.multi_summary_csv:
        summary.to_csv(opt.multi_summary_csv, index=False)
        print(f"Saved fold summary table to {opt.multi_summary_csv}")

    print('Fold coverage by experiment/split:')
    coverage = pd.DataFrame(discovered).groupby(['experiment_type', 'split_type'])['fold'].nunique()
    for (exp_name, split_name), n_folds in coverage.items():
        print(f"  {exp_name} / {split_name}: {int(n_folds)} folds")


def main():
    release=os.environ.get("COMPLETERXN_RESULTS_ROOT")
    if release:
        for name in ['MULTI_FOLD_CONFIG', 'SINGLE_FOLD_CONFIG']:
            config=globals().get(name,{})
            for key,value in list(config.items()):
                if isinstance(value,str) and value.startswith("results/main/figures/"):
                    config[key]=os.path.join(release,value[len("results/"):])

    if MODE == 'multi_fold':
        opt = SimpleNamespace(
            n_best=N_BEST,
            experiments=','.join(MULTI_FOLD_CONFIG['experiments']),
            split_types=','.join(MULTI_FOLD_CONFIG['split_types']),
            results_dir=MULTI_FOLD_CONFIG['results_dir'],
            output=MULTI_FOLD_CONFIG['output'],
            multi_summary_csv=MULTI_FOLD_CONFIG['multi_summary_csv'],
            multi_output_dir=MULTI_FOLD_CONFIG['multi_output_dir'],
        )
        run_multi_fold_mode(opt)
        return

    if MODE != 'single':
        raise ValueError("MODE must be either 'single' or 'multi_fold'.")

    opt = SimpleNamespace(
        n_best=N_BEST,
        log_probs=SINGLE_FOLD_CONFIG['log_probs'],
        top1_stats=SINGLE_FOLD_CONFIG['top1_stats'],
        output=SINGLE_FOLD_CONFIG['output'],
        output_missing=SINGLE_FOLD_CONFIG['output_missing'],
        benchmark_final=SINGLE_FOLD_CONFIG['benchmark_final'],
        split_csv=SINGLE_FOLD_CONFIG['split_csv'],
        split_col=SINGLE_FOLD_CONFIG['split_col'],
        split_label=SINGLE_FOLD_CONFIG['split_label'],
        missing_atoms_col=SINGLE_FOLD_CONFIG['missing_atoms_col'],
        missing_carbons_col=SINGLE_FOLD_CONFIG['missing_carbons_col'],
        identifier_col=SINGLE_FOLD_CONFIG['identifier_col'],
        gt_col=SINGLE_FOLD_CONFIG['gt_col'],
        input_col=SINGLE_FOLD_CONFIG['input_col'],
        high_conf_threshold=SINGLE_FOLD_CONFIG['high_conf_threshold'],
        high_conf_output=SINGLE_FOLD_CONFIG['high_conf_output'],
        high_conf_balanced_only=SINGLE_FOLD_CONFIG['high_conf_balanced_only'],
        low_conf_threshold=SINGLE_FOLD_CONFIG['low_conf_threshold'],
        low_conf_output=SINGLE_FOLD_CONFIG['low_conf_output'],
    )

    if not opt.log_probs or not opt.top1_stats:
        raise ValueError("In MODE='single', set SINGLE_FOLD_CONFIG['log_probs'] and ['top1_stats'].")

    # Load and join
    df, prob = load_probs_and_stats(opt.log_probs, opt.top1_stats, opt.n_best)
    data = build_category_dataframe(df, prob)

    n_per_cat = data.groupby('category').size().reindex(CATEGORY_ORDER, fill_value=0)
    total_n = int(n_per_cat.sum())
    print("Counts per category:")
    for cat, n in n_per_cat.items():
        print(f"  {cat}: {n}")

    apply_neurips_style()
    plt.rcParams['axes.axisbelow'] = True

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.violinplot(
        ax=ax,
        data=data,
        x='category',
        hue='category',
        y='probability',
        order=CATEGORY_ORDER,
        palette=COLORS,
        inner='quartile',
        cut=0,
        legend=False,
    )

    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.yaxis.grid(True, linestyle='-', alpha=0.7)
    tick_labels = []
    for cat in CATEGORY_ORDER:
        n = int(n_per_cat[cat])
        pct = (100.0 * n / total_n) if total_n else 0.0
        tick_labels.append(f"{cat}\n(n={n}, {pct:.1f}%)")
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('')
    ax.set_ylabel('Prediction Probability')
    ax.set_title('Confidence by prediction category')
    plt.tight_layout()

    if opt.output:
        plt.savefig(opt.output, dpi=300, bbox_inches='tight')
        print(f"Saved to {opt.output}")
    else:
        plt.show()

    mapped_benchmark = None
    # Preferred source: benchmark_final columns (if provided).
    if opt.benchmark_final and opt.split_csv and 'original_index' in df.columns:
        bench = pd.read_csv(opt.benchmark_final, low_memory=False)
        split_df = pd.read_csv(opt.split_csv)
        split_col = resolve_split_column(split_df, opt.split_col)
        if 'Unique_identifier' not in split_df.columns or 'Unique_identifier' not in bench.columns:
            raise ValueError("Both split CSV and benchmark_final must contain 'Unique_identifier'.")

        uids = split_df.loc[split_df[split_col] == opt.split_label, 'Unique_identifier']
        bench_subset = bench[bench['Unique_identifier'].isin(uids)].reset_index(drop=True)

        if df['original_index'].max() >= len(bench_subset):
            raise ValueError(
                f"original_index max={int(df['original_index'].max())} but benchmark test subset has only "
                f"{len(bench_subset)} rows. Check --split-csv / --split-label."
            )

        mapped_benchmark = bench_subset.iloc[df['original_index'].astype(int).values].reset_index(drop=True)

    # Additional plot: balanced but not accurate confidence vs missing atoms/carbons.
    bna_mask = (df['is_balanced'] == True) & (df['matches_ground_truth'] == False)
    bna_df = df.loc[bna_mask, ['balance_info']].copy()
    bna_df['probability'] = prob[bna_mask.values]

    if mapped_benchmark is not None:
        if opt.missing_atoms_col not in mapped_benchmark.columns or opt.missing_carbons_col not in mapped_benchmark.columns:
            raise ValueError(
                f"Missing columns in benchmark subset. Need '{opt.missing_atoms_col}' and '{opt.missing_carbons_col}'."
            )

        mapped = mapped_benchmark
        bna_df['missing_atoms'] = mapped.loc[bna_mask.values, opt.missing_atoms_col].fillna(0).astype(int).values
        bna_df['missing_carbons'] = mapped.loc[bna_mask.values, opt.missing_carbons_col].fillna(0).astype(int).values
    else:
        # Fallback: parse from balance_info emitted by score script.
        bna_df[['missing_atoms', 'missing_carbons']] = bna_df['balance_info'].apply(
            lambda s: pd.Series(extract_missing_counts(s))
        )

    # Optional exports for manual inspection/plotting.
    if opt.high_conf_output or opt.low_conf_output:
        out_df = pd.DataFrame({
            'original_index': df['original_index'] if 'original_index' in df.columns else range(len(df)),
            'prediction_probability': prob,
            'is_balanced': df['is_balanced'].values,
            'matches_ground_truth': df['matches_ground_truth'].values,
            'top1_prediction_diff': df['top1_prediction'].values if 'top1_prediction' in df.columns else [''] * len(df),
        })

        if mapped_benchmark is not None:
            if opt.identifier_col in mapped_benchmark.columns:
                out_df['identifier'] = mapped_benchmark[opt.identifier_col].values
            else:
                out_df['identifier'] = ''
            out_df['ground_truth_reaction'] = (
                mapped_benchmark[opt.gt_col].values if opt.gt_col in mapped_benchmark.columns else ''
            )
            out_df['input_reaction'] = (
                mapped_benchmark[opt.input_col].values if opt.input_col in mapped_benchmark.columns else ''
            )
            out_df['predicted_completed_reaction'] = out_df.apply(
                lambda r: compose_completed_reaction(r['input_reaction'], r['top1_prediction_diff']), axis=1
            )
            if opt.missing_atoms_col in mapped_benchmark.columns:
                out_df['missing_atoms_benchmark'] = mapped_benchmark[opt.missing_atoms_col].values
            if opt.missing_carbons_col in mapped_benchmark.columns:
                out_df['missing_carbons_benchmark'] = mapped_benchmark[opt.missing_carbons_col].values
        else:
            out_df['identifier'] = ''
            out_df['ground_truth_reaction'] = ''
            out_df['input_reaction'] = ''
            out_df['predicted_completed_reaction'] = ''

        incorrect_mask = (df['matches_ground_truth'] == False)
        balanced_incorrect_mask = incorrect_mask & (df['is_balanced'] == True)

        if opt.high_conf_output:
            high_base_mask = incorrect_mask
            if opt.high_conf_balanced_only:
                high_base_mask = balanced_incorrect_mask
            high_conf_mask = high_base_mask & (prob >= opt.high_conf_threshold)
            high_df = out_df.loc[high_conf_mask.values].copy()
            high_df = high_df.sort_values('prediction_probability', ascending=False)
            high_df.to_csv(opt.high_conf_output, index=False)
            print(
                f"Saved {len(high_df)} high-confidence incorrect predictions "
                f"(threshold >= {opt.high_conf_threshold:.3f}) to {opt.high_conf_output}"
            )

        if opt.low_conf_output:
            low_conf_mask = balanced_incorrect_mask & (prob <= opt.low_conf_threshold)
            low_df = out_df.loc[low_conf_mask.values].copy()
            low_df = low_df.sort_values('prediction_probability', ascending=True)
            low_df.to_csv(opt.low_conf_output, index=False)
            print(
                f"Saved {len(low_df)} low-confidence balanced incorrect predictions "
                f"(threshold <= {opt.low_conf_threshold:.3f}) to {opt.low_conf_output}"
            )

    print(f"Balanced, not accurate rows: {len(bna_df)}")
    print(
        "Missing atoms summary (balanced, not accurate): "
        f"min={bna_df['missing_atoms'].min() if len(bna_df) else 0}, "
        f"max={bna_df['missing_atoms'].max() if len(bna_df) else 0}"
    )
    print(
        "Missing carbons summary (balanced, not accurate): "
        f"min={bna_df['missing_carbons'].min() if len(bna_df) else 0}, "
        f"max={bna_df['missing_carbons'].max() if len(bna_df) else 0}"
    )

    fig2, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    sns.stripplot(
        ax=axes[0],
        data=bna_df,
        x='missing_atoms',
        y='probability',
        color='#EC6842',
        alpha=0.6,
        jitter=0.2,
    )
    axes[0].set_title('Balanced, not accurate')
    axes[0].set_xlabel('Missing atoms')
    axes[0].set_ylabel('Prediction Probability')
    axes[0].yaxis.grid(True, linestyle='-', alpha=0.7)

    sns.stripplot(
        ax=axes[1],
        data=bna_df,
        x='missing_carbons',
        y='probability',
        color='#A50034',
        alpha=0.6,
        jitter=0.2,
    )
    axes[1].set_title('Balanced, not accurate')
    axes[1].set_xlabel('Missing carbons')
    axes[1].set_ylabel('')
    axes[1].yaxis.grid(True, linestyle='-', alpha=0.7)

    for ax_i in axes:
        ax_i.set_ylim(-0.02, 1.02)
        ax_i.set_yticks([i / 10 for i in range(11)])

    plt.tight_layout()

    missing_output = opt.output_missing
    if not missing_output and opt.output:
        if opt.output.lower().endswith('.png'):
            missing_output = opt.output[:-4] + '_missing.png'
        else:
            missing_output = opt.output + '_missing.png'

    if missing_output:
        plt.savefig(missing_output, dpi=300, bbox_inches='tight')
        print(f"Saved to {missing_output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()

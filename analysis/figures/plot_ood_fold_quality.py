#!/usr/bin/env python3
"""
Plot how similar each OOD test fold is to its training data.

Produces two figures:

Figure 1 — CDF comparison (pooled across all folds):
  - OOD test → nearest train   (solid orange)
  - OOD test → nearest in-fold (solid purple)
  - Random test → nearest train (solid green)
  - Train → nearest train       (dashed grey, reference)

Figure 2 — Per-fold violin plot:
  - OOD → train and random → train side-by-side per fold
  (intra-fold violin removed as uninformative for manuscript)

Works with all three split types:
  - ood_folds/               training = fold -1
    - rxn_groups/              training = all other folds
  - ood_folds_missing_stats/ training = fold -1
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.plot_style import apply_neurips_style

# ============ CONFIGURATION ============
#SPLITS_CSV = 'data/Template_curated/benchmark/rxn_groups/benchmark_final_group_splits_RI_t055.csv'
SPLITS_CSV = 'data/Template_curated/benchmark/ood_folds_missing_stats/benchmark_final_ood_splits_RI_t055.csv'
#SPLITS_CSV = 'data/Template_curated/benchmark/ood_folds/benchmark_final_ood_splits_RI_t055.csv'
# or directly use the group splits
# Directory containing canonical rep files produced by data_split.py --split-type group.
# (files: benchmark_final_group_splits_rep*.csv + benchmark_final_group_splits_meta.json)
# Set to None to fall back to raw-fold analysis from SPLITS_CSV.
#GROUP_SPLITS_DIR = 'data/Template_curated/benchmark/group_splits'
GROUP_SPLITS_DIR = None  # fallback to legacy raw fold CSV

FPS_PATH = 'data/Template_curated/benchmark/benchmark_final_fps_CanonicalReaction.npy'

# Random split files used for the baseline comparison
RANDOM_TEST_CSV  = 'data/Template_curated/benchmark/test_split_random.csv'
RANDOM_TRAIN_CSV = 'data/Template_curated/benchmark/train_split_random.csv'

# Max reactions sampled per fold for distributions. Set to None to use all.
MAX_SAMPLE = 10000

# Tanimoto threshold above which a test reaction is considered "leaked" to training
LEAKAGE_THRESHOLD = 0.6

# Number of training reactions sampled for the train→train reference curve
TRAIN_TRAIN_SAMPLE = 20000

# Number of nearest neighbours whose similarity is averaged per query reaction
KNN = 5

OUTPUT_DIR = 'figures'

# Reuse precomputed similarity arrays if available.
USE_PRECOMPUTED_IF_AVAILABLE = True
# Optional explicit paths. Leave as None to use split-derived default paths.
PRECOMPUTED_CACHE_PATH = None
PRECOMPUTED_SUMMARY_PATH = None
# =======================================


def tanimoto_topk(query_fps, fps_mmap, pool_idx, k=5, chunk=1024):
    """Return (n, k) array of the top-k Tanimoto similarities for each query row.

    Only one chunk of pool rows is in memory at a time.
    Callers can derive top-1 via .max(axis=1) and mean(top-k) via .mean(axis=1).
    """
    q = np.asarray(query_fps, dtype=np.float32)
    n = len(q)
    q_cnt = q.sum(axis=1)                          # (n,)
    topk  = np.full((n, k), -1.0, dtype=np.float32)
    for start in range(0, len(pool_idx), chunk):
        end   = min(start + chunk, len(pool_idx))
        p     = np.asarray(fps_mmap[pool_idx[start:end]], dtype=np.float32)
        ab    = q @ p.T
        p_cnt = p.sum(axis=1)
        sim   = ab / (q_cnt[:, None] + p_cnt[None, :] - ab + 1e-8)
        combined = np.concatenate([topk, sim], axis=1)  # (n, k+c)
        idx  = np.argpartition(combined, -k, axis=1)[:, -k:]
        topk = combined[np.arange(n)[:, None], idx]
    return topk


def plot_cdf(ax, values, **kwargs):
    """Plot empirical CDF of values on ax."""
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, **kwargs)


def main():
    # Slightly larger typography for camera-ready readability.
    apply_neurips_style(font_size=12.0, tick_size=10.0, line_width=0.9, save_dpi=300, figure_dpi=300)

    def _safe_fold_key(fold_id):
        return re.sub(r'[^A-Za-z0-9_\-]+', '_', str(fold_id))

    use_group_mode = GROUP_SPLITS_DIR is not None and os.path.isdir(GROUP_SPLITS_DIR)
    if use_group_mode:
        split_name = 'group_splits_canonical'
        split_subdir = 'group_splits'
    else:
        split_name = os.path.splitext(os.path.basename(SPLITS_CSV))[0]
        split_subdir = os.path.basename(os.path.dirname(SPLITS_CSV))

    out_dir = os.path.join(OUTPUT_DIR, split_subdir)
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = PRECOMPUTED_SUMMARY_PATH or os.path.join(out_dir, f'ood_fold_quality_summary_{split_name}.csv')
    cache_path = PRECOMPUTED_CACHE_PATH or os.path.join(out_dir, f'ood_fold_quality_computation_{split_name}.npz')

    loaded_from_cache = False
    if USE_PRECOMPUTED_IF_AVAILABLE and os.path.exists(cache_path):
        print(f"Loading precomputed cache from {cache_path}...")
        npz = np.load(cache_path, allow_pickle=True)
        test_folds = [str(f) for f in npz['fold_ids'].tolist()]

        train_train_nn1 = np.asarray(npz['train_train_nn1'], dtype=np.float32)
        train_train_nn = np.asarray(npz['train_train_nn'], dtype=np.float32)
        ood1_pooled = np.asarray(npz['ood1_pooled'], dtype=np.float32)
        rnd1_pooled = np.asarray(npz['rnd1_pooled'], dtype=np.float32)
        ood_pooled = np.asarray(npz['ood_pooled'], dtype=np.float32)
        rnd_pooled = np.asarray(npz['rnd_pooled'], dtype=np.float32)
        intra1_pooled = np.asarray(npz['intra1_pooled'], dtype=np.float32) if 'intra1_pooled' in npz else None
        intra_pooled = np.asarray(npz['intra_pooled'], dtype=np.float32) if 'intra_pooled' in npz else None

        def _load_fold_dict(prefix):
            out = {}
            for fold_id in test_folds:
                k = f"{prefix}__{_safe_fold_key(fold_id)}"
                if k in npz:
                    out[fold_id] = np.asarray(npz[k], dtype=np.float32)
            return out

        all_ood_nn1 = _load_fold_dict('ood_nn1')
        all_ood_nn = _load_fold_dict('ood_nnk')
        all_rnd_nn1 = _load_fold_dict('rnd_nn1')
        all_rnd_nn = _load_fold_dict('rnd_nnk')
        all_intra_nn1 = _load_fold_dict('intra_nn1')
        all_intra_nn = _load_fold_dict('intra_nnk')

        if os.path.exists(summary_csv):
            summary = pd.read_csv(summary_csv)
        else:
            rows = []
            for fold_id in test_folds:
                if fold_id not in all_ood_nn or fold_id not in all_rnd_nn:
                    continue
                ood_nn = all_ood_nn[fold_id]
                rnd_nn = all_rnd_nn[fold_id]
                ood_nn1 = all_ood_nn1.get(fold_id)
                rnd_nn1 = all_rnd_nn1.get(fold_id)
                leak_ood = float((ood_nn1 > LEAKAGE_THRESHOLD).mean()) if ood_nn1 is not None else np.nan
                leak_rnd = float((rnd_nn1 > LEAKAGE_THRESHOLD).mean()) if rnd_nn1 is not None else np.nan
                rows.append({
                    'fold': fold_id,
                    'n_test': int(len(ood_nn)),
                    'ood_nn_median': float(np.median(ood_nn)),
                    'random_nn_median': float(np.median(rnd_nn)),
                    'distribution_shift': float(np.median(rnd_nn) - np.median(ood_nn)),
                    f'leakage_ood_t{LEAKAGE_THRESHOLD}': leak_ood,
                    f'leakage_rnd_t{LEAKAGE_THRESHOLD}': leak_rnd,
                })
            summary = pd.DataFrame(rows).sort_values('distribution_shift', ascending=False)

        print(f"Loaded precomputed arrays for {len(test_folds)} fold(s).")
        loaded_from_cache = True
        
    if not loaded_from_cache and use_group_mode:
        # ── Mode A: canonical rep files from data_split.py ───────────────────
        import glob
        import json as _json

        rep_files = sorted(glob.glob(
            os.path.join(GROUP_SPLITS_DIR, 'benchmark_final_group_splits_rep[0-9]*.csv')
        ))
        rep_files = [f for f in rep_files if '_meta' not in f]
        if not rep_files:
            raise FileNotFoundError(f"No rep files found in {GROUP_SPLITS_DIR}")
        print(f"Found {len(rep_files)} rep file(s) in {GROUP_SPLITS_DIR}")

        # Load fold labels from metadata JSON
        meta_path = os.path.join(GROUP_SPLITS_DIR, 'benchmark_final_group_splits_meta.json')
        rep_meta_by_rep = {}
        if os.path.exists(meta_path):
            with open(meta_path) as _fh:
                _meta = _json.load(_fh)
            for _rm in _meta.get('reps', []):
                rep_meta_by_rep[_rm['rep']] = _rm

        # uid_to_pos from first rep file (row_index = position in benchmark_final)
        _first_rep_df = pd.read_csv(rep_files[0], usecols=['Unique_identifier', 'row_index'])
        uid_to_pos = dict(zip(_first_rep_df['Unique_identifier'], _first_rep_df['row_index']))

        print(f"Loading fingerprints from {FPS_PATH}...")
        fps = np.load(FPS_PATH, mmap_mode='r')

        print(f"Loading random split from {RANDOM_TEST_CSV} / {RANDOM_TRAIN_CSV}...")
        rand_test_df  = pd.read_csv(RANDOM_TEST_CSV,  usecols=['Unique_identifier'])
        rand_train_df = pd.read_csv(RANDOM_TRAIN_CSV, usecols=['Unique_identifier'])
        rand_test_idx  = np.array([uid_to_pos[u] for u in rand_test_df['Unique_identifier']  if u in uid_to_pos])
        rand_train_idx = np.array([uid_to_pos[u] for u in rand_train_df['Unique_identifier'] if u in uid_to_pos])
        print(f"Random split — test: {len(rand_test_idx):,}  train: {len(rand_train_idx):,}")

        rng = np.random.default_rng(42)

        print(f"Computing train→train reference (n={TRAIN_TRAIN_SAMPLE})...")
        perm            = rng.permutation(len(rand_train_idx))
        tt_query_idx    = rand_train_idx[perm[:TRAIN_TRAIN_SAMPLE]]
        tt_pool_idx     = rand_train_idx[perm[TRAIN_TRAIN_SAMPLE:]]
        F_tt_query      = np.asarray(fps[tt_query_idx], dtype=np.float32)
        topk_tt         = tanimoto_topk(F_tt_query, fps, tt_pool_idx, k=KNN)
        train_train_nn1 = topk_tt.max(axis=1)
        train_train_nn  = topk_tt.mean(axis=1)

        rows       = []
        all_ood_nn1 = {};  all_ood_nn  = {}
        all_rnd_nn1 = {};  all_rnd_nn  = {}
        all_intra_nn1 = {}; all_intra_nn = {}
        all_ood_nn1_pooled  = [];  all_ood_nn_pooled  = []
        all_rnd_nn1_pooled  = [];  all_rnd_nn_pooled  = []
        all_intra_nn1_pooled = []; all_intra_nn_pooled = []
        test_folds = []

        for rep_file in rep_files:
            rep_num = int(
                os.path.basename(rep_file)
                .replace('benchmark_final_group_splits_rep', '')
                .replace('.csv', '')
            )
            rep_df = pd.read_csv(rep_file, usecols=['row_index', 'group_split'])
            test_idx  = rep_df.loc[rep_df['group_split'] == 2, 'row_index'].values
            train_idx = rep_df.loc[rep_df['group_split'] == 0, 'row_index'].values
            rm = rep_meta_by_rep.get(rep_num, {})
            fold_id = f"r{rep_num}_t{rm['test_fold']}" if 'test_fold' in rm else f"rep{rep_num}"

            if len(test_idx) == 0 or len(train_idx) == 0:
                print(f"  {fold_id}: skipped (empty partition)")
                continue

            test_folds.append(fold_id)

            n_sample    = len(test_idx) if MAX_SAMPLE is None else min(MAX_SAMPLE, len(test_idx))
            ood_sample  = rng.choice(test_idx, size=n_sample, replace=False)
            F_ood       = np.asarray(fps[ood_sample], dtype=np.float32)

            n_rand      = min(n_sample, len(rand_test_idx))
            rand_sample = rng.choice(rand_test_idx, size=n_rand, replace=False)
            F_rand      = np.asarray(fps[rand_sample], dtype=np.float32)

            topk_ood = tanimoto_topk(F_ood,  fps, train_idx,      k=KNN)
            topk_rnd = tanimoto_topk(F_rand, fps, rand_train_idx, k=KNN)
            ood_nn1 = topk_ood.max(axis=1);  ood_nn = topk_ood.mean(axis=1)
            rnd_nn1 = topk_rnd.max(axis=1);  rnd_nn = topk_rnd.mean(axis=1)

            # Intra-fold: pool = full fold (includes queries); request k+1 so we
            # can drop the self-match (always top similarity). Pool size = len(test_idx)
            # regardless of n_sample, avoiding the pool-shrinkage bias.
            if len(test_idx) >= 2:
                topk_intra = tanimoto_topk(F_ood, fps, test_idx, k=KNN + 1)
                # drop the top-1 column (self-match), keep the next k
                sorted_intra = np.sort(topk_intra, axis=1)[:, ::-1]
                intra_nn1 = sorted_intra[:, 1]          # 2nd-best = true NN
                intra_nn  = sorted_intra[:, 1:KNN+1].mean(axis=1)  # mean of next k
                all_intra_nn1[fold_id] = intra_nn1
                all_intra_nn[fold_id] = intra_nn
                all_intra_nn1_pooled.append(intra_nn1)
                all_intra_nn_pooled.append(intra_nn)

            all_ood_nn1[fold_id] = ood_nn1;  all_ood_nn[fold_id] = ood_nn
            all_rnd_nn1[fold_id] = rnd_nn1;  all_rnd_nn[fold_id] = rnd_nn
            all_ood_nn1_pooled.append(ood_nn1);  all_ood_nn_pooled.append(ood_nn)
            all_rnd_nn1_pooled.append(rnd_nn1);  all_rnd_nn_pooled.append(rnd_nn)

            leak_ood = float((ood_nn1 > LEAKAGE_THRESHOLD).mean())
            leak_rnd = float((rnd_nn1 > LEAKAGE_THRESHOLD).mean())

            n_train = int((rep_df['group_split'] == 0).sum())
            rows.append({
                'fold':               fold_id,
                'n_test':             len(test_idx),
                'ood_nn_median':      float(np.median(ood_nn)),
                'random_nn_median':   float(np.median(rnd_nn)),
                'distribution_shift': float(np.median(rnd_nn) - np.median(ood_nn)),
                f'leakage_ood_t{LEAKAGE_THRESHOLD}':  leak_ood,
                f'leakage_rnd_t{LEAKAGE_THRESHOLD}':  leak_rnd,
            })
            print(f"  {fold_id}: n_test={len(test_idx):>6}  n_train={n_train:>7}  "
                  f"ood_nn={rows[-1]['ood_nn_median']:.3f}  "
                  f"rnd_nn={rows[-1]['random_nn_median']:.3f}  "
                  f"shift={rows[-1]['distribution_shift']:+.3f}  "
                  f"leakage(OOD)={leak_ood:.1%}  leakage(rnd)={leak_rnd:.1%}")

    elif not loaded_from_cache:
        # ── Mode B: raw fold CSV (legacy) ─────────────────────────────────────
        print(f"Loading splits from {SPLITS_CSV}...")
        splits_df = pd.read_csv(SPLITS_CSV)

        fold_cols = [c for c in splits_df.columns if c.startswith('ood_fold')]
        if not fold_cols:
            raise ValueError(f"No 'ood_fold_*' column found in {SPLITS_CSV}")
        fold_col  = fold_cols[0]
        fold_vals = sorted(splits_df[fold_col].unique())
        has_neg1  = -1 in fold_vals

        # Build test fold list and paired valid fold per rep.
        if has_neg1:
            test_folds    = [f for f in fold_vals if f != -1]
            fold_to_valid = {}   # empty → use all non-test as train
        else:
            all_folds = [f for f in fold_vals if f != -1]
            fold_to_valid = {
                all_folds[i]: all_folds[(i + 1) % len(all_folds)]
                for i in range(len(all_folds))
            }
            test_folds = list(fold_to_valid.keys())
            total_n    = len(splits_df)
            print(f"  {'test':>5}  {'test%':>6}  {'valid':>5}  {'valid%':>6}  {'train%':>7}")
            for tf, vf in fold_to_valid.items():
                pt = (splits_df[fold_col] == tf).sum() / total_n * 100
                pv = (splits_df[fold_col] == vf).sum() / total_n * 100
                print(f"  {tf:>5}  {pt:>5.1f}%  {vf:>5}  {pv:>5.1f}%  {100-pt-pv:>6.1f}%")

        print(f"Fold column: '{fold_col}' | Test folds: {test_folds}")
        print(f"Training partition: {'fold -1' if has_neg1 else 'all other folds'}")

        # -- Figure 3: Distribution of missing atoms/carbons (if available) ─────────────────────────# --- Plot train vs test distributions for MissingNrAtoms and MissingNrCarbons ---
        # Only if SPLITS_CSV and benchmark_final.csv exist and have Unique_identifier
        final_csv = 'data/Template_curated/benchmark/benchmark_final.csv'
        df_final = pd.read_csv(final_csv, usecols=['Unique_identifier', 'MissingNrAtoms', 'MissingNrCarbons'])
        if 'Unique_identifier' in splits_df.columns and 'Unique_identifier' in df_final.columns:
            merged = splits_df[['Unique_identifier', fold_col]].merge(df_final, on='Unique_identifier', how='left')
            train_mask = merged[fold_col] == -1
            test_mask = merged[fold_col] >= 0
            for col, label in [('MissingNrAtoms', 'Missing Atoms'), ('MissingNrCarbons', 'Missing Carbons')]:
                plt.figure(figsize=(5, 4))
                plt.hist(merged.loc[train_mask, col].dropna(), bins=50, alpha=0.7, label='Train', color='tab:blue', density=True)
                plt.hist(merged.loc[test_mask, col].dropna(), bins=50, alpha=0.7, label='Test (OOD)', color='tab:orange', density=True)
                plt.xlabel(label)
                plt.ylabel('Density')
                plt.title(f'Distribution of {label}: Train vs Test (OOD)')
                plt.legend()
                plt.tight_layout()
                out_path = os.path.join(OUTPUT_DIR, f'{col.lower()}_train_vs_test.png')
                plt.savefig(out_path)
                print(f'Saved {out_path}')
                # Print statistics: Mean and median for train and test
                train_mean = merged.loc[train_mask, col].mean()
                train_median = merged.loc[train_mask, col].median()
                test_mean = merged.loc[test_mask, col].mean()
                test_median = merged.loc[test_mask, col].median()
                print(f"  {label}: Train mean={train_mean:.2f}, median={train_median:.2f} | Test mean={test_mean:.2f}, median={test_median:.2f}")
                plt.close()
        
        print(f"Loading fingerprints from {FPS_PATH}...")
        fps_raw = np.load(FPS_PATH, mmap_mode='r')

        if 'row_index' in splits_df.columns:
            uid_to_pos = dict(zip(splits_df['Unique_identifier'], splits_df['row_index']))
        else:
            uid_to_pos = {uid: i for i, uid in enumerate(splits_df['Unique_identifier'])}

        fps = fps_raw  # keep as mmap — pools are sliced chunk-by-chunk inside tanimoto_nn_to_pool

        # ── Random split baseline ────────────────────────────────────────────────
        print(f"Loading random split from {RANDOM_TEST_CSV} / {RANDOM_TRAIN_CSV}...")
        rand_test_df  = pd.read_csv(RANDOM_TEST_CSV,  usecols=['Unique_identifier'])
        rand_train_df = pd.read_csv(RANDOM_TRAIN_CSV, usecols=['Unique_identifier'])

        rand_test_idx  = np.array([uid_to_pos[u] for u in rand_test_df['Unique_identifier']  if u in uid_to_pos])
        rand_train_idx = np.array([uid_to_pos[u] for u in rand_train_df['Unique_identifier'] if u in uid_to_pos])
        print(f"Random split — test: {len(rand_test_idx):,}  train: {len(rand_train_idx):,}")

        rng = np.random.default_rng(42)

        # ── Train→train reference (sampled) ─────────────────────────────────────
        print(f"Computing train→train reference (n={TRAIN_TRAIN_SAMPLE})...")
        perm           = rng.permutation(len(rand_train_idx))
        tt_query_idx   = rand_train_idx[perm[:TRAIN_TRAIN_SAMPLE]]
        tt_pool_idx    = rand_train_idx[perm[TRAIN_TRAIN_SAMPLE:]]   # rest of training set
        F_tt_query     = np.asarray(fps[tt_query_idx], dtype=np.float32)
        topk_tt        = tanimoto_topk(F_tt_query, fps, tt_pool_idx, k=KNN)
        train_train_nn1 = topk_tt.max(axis=1)    # top-1
        train_train_nn  = topk_tt.mean(axis=1)   # mean(top-k)

        rows       = []
        all_ood_nn1 = {};  all_ood_nn  = {}   # top-1 and mean(top-k) per fold
        all_rnd_nn1 = {};  all_rnd_nn  = {}
        all_intra_nn1 = {}; all_intra_nn = {}
        all_ood_nn1_pooled  = [];  all_ood_nn_pooled  = []
        all_rnd_nn1_pooled  = [];  all_rnd_nn_pooled  = []
        all_intra_nn1_pooled = []; all_intra_nn_pooled = []

        for fold_id in test_folds:
            test_uids = splits_df.loc[splits_df[fold_col] == fold_id, 'Unique_identifier'].tolist()
            if has_neg1:
                train_uids = splits_df.loc[splits_df[fold_col] == -1, 'Unique_identifier'].tolist()
            else:
                # Exclude both test fold and its paired valid fold (matches model training)
                exclude = {fold_id}
                if fold_id in fold_to_valid:
                    exclude.add(fold_to_valid[fold_id])
                train_uids = splits_df.loc[~splits_df[fold_col].isin(exclude), 'Unique_identifier'].tolist()

            test_idx  = np.array([uid_to_pos[u] for u in test_uids  if u in uid_to_pos])
            train_idx = np.array([uid_to_pos[u] for u in train_uids if u in uid_to_pos])

            if len(test_idx) == 0 or len(train_idx) == 0:
                print(f"  Fold {fold_id}: skipped (empty partition)")
                continue

            n_sample    = len(test_idx) if MAX_SAMPLE is None else min(MAX_SAMPLE, len(test_idx))
            ood_sample  = rng.choice(test_idx, size=n_sample, replace=False)
            F_ood       = np.asarray(fps[ood_sample], dtype=np.float32)

            n_rand      = min(n_sample, len(rand_test_idx))
            rand_sample = rng.choice(rand_test_idx, size=n_rand, replace=False)
            F_rand      = np.asarray(fps[rand_sample], dtype=np.float32)

            # Compute top-k matrix once; extract top-1 and mean(top-k) from it
            topk_ood = tanimoto_topk(F_ood,  fps, train_idx,      k=KNN)
            topk_rnd = tanimoto_topk(F_rand, fps, rand_train_idx, k=KNN)
            ood_nn1 = topk_ood.max(axis=1);  ood_nn = topk_ood.mean(axis=1)
            rnd_nn1 = topk_rnd.max(axis=1);  rnd_nn = topk_rnd.mean(axis=1)

            # Intra-fold: pool = full fold (includes queries); request k+1 so we
            # can drop the self-match (always top similarity). Pool size = len(test_idx)
            # regardless of n_sample, avoiding the pool-shrinkage bias.
            if len(test_idx) >= 2:
                topk_intra = tanimoto_topk(F_ood, fps, test_idx, k=KNN + 1)
                # drop the top-1 column (self-match), keep the next k
                sorted_intra = np.sort(topk_intra, axis=1)[:, ::-1]
                intra_nn1 = sorted_intra[:, 1]          # 2nd-best = true NN
                intra_nn  = sorted_intra[:, 1:KNN+1].mean(axis=1)  # mean of next k
                all_intra_nn1[fold_id] = intra_nn1
                all_intra_nn[fold_id] = intra_nn
                all_intra_nn1_pooled.append(intra_nn1)
                all_intra_nn_pooled.append(intra_nn)

            all_ood_nn1[fold_id] = ood_nn1;  all_ood_nn[fold_id] = ood_nn
            all_rnd_nn1[fold_id] = rnd_nn1;  all_rnd_nn[fold_id] = rnd_nn
            all_ood_nn1_pooled.append(ood_nn1);  all_ood_nn_pooled.append(ood_nn)
            all_rnd_nn1_pooled.append(rnd_nn1);  all_rnd_nn_pooled.append(rnd_nn)

            leak_ood = float((ood_nn1 > LEAKAGE_THRESHOLD).mean())
            leak_rnd = float((rnd_nn1 > LEAKAGE_THRESHOLD).mean())

            rows.append({
                'fold':               fold_id,
                'n_test':             len(test_idx),
                'ood_nn_median':      float(np.median(ood_nn)),
                'random_nn_median':   float(np.median(rnd_nn)),
                'distribution_shift': float(np.median(rnd_nn) - np.median(ood_nn)),
                f'leakage_ood_t{LEAKAGE_THRESHOLD}':  leak_ood,
                f'leakage_rnd_t{LEAKAGE_THRESHOLD}':  leak_rnd,
            })
            print(f"  Fold {fold_id}: n={len(test_idx):>6}  "
                  f"ood_nn={rows[-1]['ood_nn_median']:.3f}  "
                  f"rnd_nn={rows[-1]['random_nn_median']:.3f}  "
                  f"shift={rows[-1]['distribution_shift']:+.3f}  "
                  f"leakage(OOD)={leak_ood:.1%}  leakage(rnd)={leak_rnd:.1%}")

    # ── Shared: summary + figures ─────────────────────────────────────────────
    if not loaded_from_cache:
        summary = pd.DataFrame(rows).sort_values('distribution_shift', ascending=False)
    print("\nSummary (sorted by distribution shift):")
    print(summary.to_string(index=False))

    # Pool across folds
    if not loaded_from_cache:
        ood1_pooled   = np.concatenate(all_ood_nn1_pooled)
        rnd1_pooled   = np.concatenate(all_rnd_nn1_pooled)
        ood_pooled    = np.concatenate(all_ood_nn_pooled)
        rnd_pooled    = np.concatenate(all_rnd_nn_pooled)
        intra1_pooled = np.concatenate(all_intra_nn1_pooled) if all_intra_nn1_pooled else None
        intra_pooled  = np.concatenate(all_intra_nn_pooled)  if all_intra_nn_pooled  else None

    # Split-aware manuscript styling:
    # - OOD: Ex. OOD in green (solid/dashed)
    # - Group: Group in orange (solid/dashed)
    # - Random stays blue
    is_group_split = (split_subdir == 'group_splits')
    TEST_LABEL = 'Group' if is_group_split else 'Ex. OOD'
    COLOR_OOD   = '#F58518' if is_group_split else '#54A24B'
    COLOR_RND   = '#4C78A8'
    COLOR_INTRA_LINE = COLOR_OOD
    COLOR_INTRA_VIOLIN = tuple(
        np.asarray(to_rgb(COLOR_OOD)) + (1.0 - np.asarray(to_rgb(COLOR_OOD))) * 0.45
    )
    COLOR_TT    = '#6E6E6E'
    CDF_AXIS_LABEL_FS = 12
    CDF_TICK_FS = 11
    CDF_LEGEND_FS = 9
    VIOLIN_AXIS_LABEL_FS = 12
    VIOLIN_TICK_FS = 11
    VIOLIN_XTICK_FS = 9
    VIOLIN_LEGEND_FS = 9

    # ── Local helpers ────────────────────────────────────────────────────────
    def _draw_cdf(ax, ood, rnd, intra, tt, xlabel, panel_title):
        plot_cdf(ax, ood, color=COLOR_OOD, lw=2,
                label=f'{TEST_LABEL} test vs. train')
        if intra is not None:
            plot_cdf(ax, intra, color=COLOR_INTRA_LINE, lw=2, ls='--',
                    label=f'{TEST_LABEL} test vs. same test fold')
        plot_cdf(ax, rnd, color=COLOR_RND, lw=2,
                label='Random test vs. train')
        plot_cdf(ax, tt, color=COLOR_TT, lw=1.5, ls='--',
                label='Train vs. train reference')
        ax.set_xlabel(xlabel, fontsize=CDF_AXIS_LABEL_FS)
        ax.set_ylabel('Cumulative fraction of reactions', fontsize=CDF_AXIS_LABEL_FS)
        ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
        ax.tick_params(axis='both', labelsize=CDF_TICK_FS)
        ax.yaxis.grid(True, linestyle='-', alpha=0.5)
        legend_handles = [
            Line2D([0], [0], color=COLOR_OOD, lw=2, ls='-',
                label=f'{TEST_LABEL} test vs. train'),
        ]
        if intra is not None:
            legend_handles.append(
                Line2D([0], [0], color=COLOR_INTRA_LINE, lw=2, ls='--',
                    label=f'{TEST_LABEL} test vs. same test fold')
            )
        legend_handles.extend([
            Line2D([0], [0], color=COLOR_RND, lw=2, ls='-',
                label='Random test vs. train'),
            Line2D([0], [0], color=COLOR_TT, lw=1.5, ls='--',
                label='Train vs. train reference'),
        ])
        ax.legend(handles=legend_handles, fontsize=CDF_LEGEND_FS, frameon=True, framealpha=1.0,
                  facecolor='white', loc='upper left', handlelength=2.4)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title('')

    def _draw_violins(axes, ood_dict, rnd_dict, intra_dict=None, show_fold_title=False):
        test_short = 'G' if TEST_LABEL == 'Group' else 'OOD'
        fold_counter = 0
        for ax, fold_id in zip(axes, test_folds):
            fold_counter += 1
            if fold_id not in ood_dict:
                ax.set_visible(False); continue
            has_intra = intra_dict is not None and fold_id in intra_dict
            if has_intra:
                violin_data = [ood_dict[fold_id], intra_dict[fold_id], rnd_dict[fold_id]]
                positions = [0, 1, 2]
                labels = [f'{test_short}(te.) →\n{test_short} tr',
                          f'{test_short}(te.) →\nin-fold',
                          'R(te.) →\nR tr']
                colors = [COLOR_OOD, COLOR_INTRA_VIOLIN, COLOR_RND]
            else:
                violin_data = [ood_dict[fold_id], rnd_dict[fold_id]]
                positions = [0, 1]
                labels = [f'{test_short}(te.) →\n{test_short} tr',
                          'R(te.) →\nR tr']
                colors = [COLOR_OOD, COLOR_RND]
            parts = ax.violinplot(violin_data, positions=positions,
                                  showmedians=True, widths=0.7, showextrema=False)
            for body, color in zip(parts['bodies'], colors):
                body.set_facecolor(color); body.set_alpha(0.75)
            parts['cmedians'].set_colors(['white'] * len(positions))
            parts['cmedians'].set_linewidth(2)
            ax.set_xticks(positions)
            #ax.set_xticklabels(labels, fontsize=VIOLIN_XTICK_FS)
            ax.tick_params(axis='y', labelsize=VIOLIN_TICK_FS)
            if show_fold_title:
                ax.set_title(f'Test fold {fold_counter}', fontsize=VIOLIN_TICK_FS, pad=4)
            ax.yaxis.grid(True, linestyle='-', alpha=0.5)
            ax.spines[['top', 'right']].set_visible(False)

    # Labels
    xlabel1    = 'Nearest-neighbor similarity'
    xlabelmean = 'Average nearest-neighbor similarity'
    ylabel1    = 'Nearest-neighbor similarity'
    ymean      = 'Average nearest-neighbor similarity'
    n_folds    = len(test_folds)

    # Change None to intra1_pooled or intra_pooled if you want to include the intra-fold curve in the CDF plots
    # ── Figure 1a: CDF top-1 ─────────────────────────────────────────────────
    fig_c1, ax_c1 = plt.subplots(figsize=(6, 4), dpi=150)
    _draw_cdf(ax_c1, ood1_pooled, rnd1_pooled, None, train_train_nn1,
              xlabel1, f'Top-1 similarity CDF\n{split_name}')
    plt.tight_layout()
    p = os.path.join(out_dir, f'ood_fold_cdf_top1_{split_name}.png')
    fig_c1.savefig(p, dpi=200, bbox_inches='tight');  print(f"Saved {p}")

    # ── Figure 1b: CDF mean(top-k) ───────────────────────────────────────────
    fig_ck, ax_ck = plt.subplots(figsize=(6, 4), dpi=150)
    _draw_cdf(ax_ck, ood_pooled, rnd_pooled, None, train_train_nn,
              xlabelmean, f'Mean top-{KNN} similarity CDF\n{split_name}')
    plt.tight_layout()
    p = os.path.join(out_dir, f'ood_fold_cdf_avgk_{split_name}.png')
    fig_ck.savefig(p, dpi=200, bbox_inches='tight');  print(f"Saved {p}")

    # ── Figure 1c: CDF combined (2 panels) ───────────────────────────────────
    fig_cc, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4), dpi=150, sharey=True)
    _draw_cdf(ax_l, ood1_pooled, rnd1_pooled, None, train_train_nn1,
              xlabel1, 'Top-1')
    _draw_cdf(ax_r, ood_pooled, rnd_pooled, None, train_train_nn,
              xlabelmean, f'Mean top-{KNN}')
    ax_r.set_ylabel('')  # shared y-axis
    fig_cc.suptitle('')
    plt.tight_layout()
    p = os.path.join(out_dir, f'ood_fold_cdf_combined_{split_name}.png')
    fig_cc.savefig(p, dpi=200, bbox_inches='tight');  print(f"Saved {p}")

    # ── Figure 2a: violin top-1 ───────────────────────────────────────────────
    fig_v1, axes_v1 = plt.subplots(1, n_folds, figsize=(max(6, 2.5 * n_folds), 4),
                                   sharey=True, dpi=150)
    if n_folds == 1: axes_v1 = [axes_v1]
    _draw_violins(axes_v1, all_ood_nn1, all_rnd_nn1, all_intra_nn1, show_fold_title=True)
    axes_v1[0].set_ylabel(ylabel1, fontsize=VIOLIN_AXIS_LABEL_FS)
    # Hide x-axis tick labels since they don't fit well and the legend already explains the categories
    for ax in axes_v1:
        ax.set_xticklabels([])
    fig_v1.legend(handles=[
        mpatches.Patch(color=COLOR_OOD, alpha=0.75, label=f'{TEST_LABEL} (test) → nearest train'),
        mpatches.Patch(color=COLOR_INTRA_VIOLIN, alpha=0.75, label=f'{TEST_LABEL} (test)→ nearest in-fold'),
        mpatches.Patch(color=COLOR_RND, alpha=0.75, label='Random (test) → nearest train'),
     ], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3,
       frameon=True, framealpha=1.0, facecolor='white', fontsize=VIOLIN_LEGEND_FS, handlelength=1.4)
    fig_v1.suptitle('')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out_dir, f'ood_fold_violin_top1_{split_name}.png')
    fig_v1.savefig(p, dpi=200, bbox_inches='tight');  print(f"Saved {p}")

    # ── Figure 2b: violin mean(top-k) ────────────────────────────────────────
    fig_vk, axes_vk = plt.subplots(1, n_folds, figsize=(max(6, 2.5 * n_folds), 4),
                                   sharey=True, dpi=150)
    if n_folds == 1: axes_vk = [axes_vk]
    _draw_violins(axes_vk, all_ood_nn, all_rnd_nn, all_intra_nn, show_fold_title=True)
    axes_vk[0].set_ylabel(ymean, fontsize=VIOLIN_AXIS_LABEL_FS)
    # Hide x-axis tick labels since they don't fit well and the legend already explains the categories
    for ax in axes_vk:
        ax.set_xticklabels([])
    fig_vk.legend(handles=[
        mpatches.Patch(color=COLOR_OOD, alpha=0.75, label=f'{TEST_LABEL} (test) → mean top-{KNN} train'),
        mpatches.Patch(color=COLOR_INTRA_VIOLIN, alpha=0.75, label=f'{TEST_LABEL} (test) → mean top-{KNN} in-fold'),
        mpatches.Patch(color=COLOR_RND, alpha=0.75, label=f'Random (test) → mean top-{KNN} train'),
     ], loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3,
       frameon=True, framealpha=1.0, facecolor='white', fontsize=VIOLIN_LEGEND_FS, handlelength=1.4)
    fig_vk.suptitle('')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out_dir, f'ood_fold_violin_avgk_{split_name}.png')
    fig_vk.savefig(p, dpi=200, bbox_inches='tight');  print(f"Saved {p}")

    # ── Figure 2c: violin combined (2 rows: top-1 / mean-k) ──────────────────
    fig_vc, axes_vc = plt.subplots(2, n_folds, figsize=(max(6, 2.5 * n_folds), 7),
                                   sharey='row', dpi=150)
    if n_folds == 1:
        axes_vc = axes_vc.reshape(2, 1)
    _draw_violins(axes_vc[0], all_ood_nn1, all_rnd_nn1, all_intra_nn1, show_fold_title=True)
    _draw_violins(axes_vc[1], all_ood_nn,  all_rnd_nn,  all_intra_nn, show_fold_title=False)
    axes_vc[0, 0].set_ylabel(f'{ylabel1}', fontsize=VIOLIN_AXIS_LABEL_FS)
    axes_vc[1, 0].set_ylabel(f'{ymean}', fontsize=VIOLIN_AXIS_LABEL_FS)
    # Hide x-axis tick labels on the top row
    for ax in axes_vc[0]:
        ax.set_xticklabels([])
    # Hide x-axis tick labels on the bottom row
    for ax in axes_vc[1]:
        ax.set_xticklabels([])
    fig_vc.legend(handles=[
        mpatches.Patch(color=COLOR_OOD, alpha=0.75, label=f'{TEST_LABEL} → train'),
        mpatches.Patch(color=COLOR_INTRA_VIOLIN, alpha=0.75, label=f'{TEST_LABEL} → in-fold'),
        mpatches.Patch(color=COLOR_RND, alpha=0.75, label='Random (test) → train'),
     ], loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3,
       frameon=True, framealpha=1.0, facecolor='white', fontsize=VIOLIN_LEGEND_FS, handlelength=1.4)
    fig_vc.suptitle('')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out_dir, f'ood_fold_violin_combined_{split_name}.png')
    fig_vc.savefig(p, dpi=200, bbox_inches='tight');  print(f"Saved {p}")

    summary.to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv}")

    # Save computed arrays so reruns can reuse/analyze outputs without re-reading plots.
    cache_payload = {
        'fold_ids': np.array([str(f) for f in test_folds], dtype=object),
        'train_train_nn1': np.asarray(train_train_nn1, dtype=np.float32),
        'train_train_nn': np.asarray(train_train_nn, dtype=np.float32),
        'ood1_pooled': np.asarray(ood1_pooled, dtype=np.float32),
        'rnd1_pooled': np.asarray(rnd1_pooled, dtype=np.float32),
        'ood_pooled': np.asarray(ood_pooled, dtype=np.float32),
        'rnd_pooled': np.asarray(rnd_pooled, dtype=np.float32),
        'knn': np.array([KNN], dtype=np.int32),
        'max_sample': np.array([-1 if MAX_SAMPLE is None else MAX_SAMPLE], dtype=np.int32),
    }
    if intra1_pooled is not None:
        cache_payload['intra1_pooled'] = np.asarray(intra1_pooled, dtype=np.float32)
    if intra_pooled is not None:
        cache_payload['intra_pooled'] = np.asarray(intra_pooled, dtype=np.float32)

    for fold_id, arr in all_ood_nn1.items():
        cache_payload[f"ood_nn1__{_safe_fold_key(fold_id)}"] = np.asarray(arr, dtype=np.float32)
    for fold_id, arr in all_ood_nn.items():
        cache_payload[f"ood_nnk__{_safe_fold_key(fold_id)}"] = np.asarray(arr, dtype=np.float32)
    for fold_id, arr in all_rnd_nn1.items():
        cache_payload[f"rnd_nn1__{_safe_fold_key(fold_id)}"] = np.asarray(arr, dtype=np.float32)
    for fold_id, arr in all_rnd_nn.items():
        cache_payload[f"rnd_nnk__{_safe_fold_key(fold_id)}"] = np.asarray(arr, dtype=np.float32)
    for fold_id, arr in all_intra_nn1.items():
        cache_payload[f"intra_nn1__{_safe_fold_key(fold_id)}"] = np.asarray(arr, dtype=np.float32)
    for fold_id, arr in all_intra_nn.items():
        cache_payload[f"intra_nnk__{_safe_fold_key(fold_id)}"] = np.asarray(arr, dtype=np.float32)

    np.savez_compressed(cache_path, **cache_payload)
    print(f"Saved {cache_path}")

    plt.show()


if __name__ == '__main__':
    main()

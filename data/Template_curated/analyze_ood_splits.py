#!/usr/bin/env python3
"""
Analyze OOD quality of reaction splits based on DRFP fingerprints.

For each fold computes exact nearest-train Tanimoto similarity (max over the
full train set) and compares against a single global random-split baseline of
the same size.

Memory-bounded exact computation via batched matrix multiply — no approximate
NN index required (peak ~100 MB per fold instead of several GB for NNDescent).
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from drfp import DrfpEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_reagents(reactions: list[str], column_name: str) -> list[str]:
    if column_name != "CanonicalReaction":
        return reactions
    return [
        r.split(">")[0] + ">>" + r.split(">")[2] if r.count(">") == 2 else r
        for r in reactions
    ]


def _tanimoto_topk_batched(fps_query: np.ndarray, fps_ref: np.ndarray,
                            k: int = 5,
                            batch_q: int = 512, batch_r: int = 8192,
                            label: str = "") -> tuple[np.ndarray, np.ndarray]:
    """
    Compute, for each query, the max Tanimoto (top-1) and the mean Tanimoto
    over the k nearest reference fingerprints — in a single pass.

    Keeps a (n_query, k) running buffer of the highest similarities seen so
    far, updating it chunk by chunk.  Peak memory ≈ batch_q×batch_r×12 B.

    Returns:
        max_tan  : (n_query,) float32  — similarity to nearest train reaction
        mean_topk: (n_query,) float32  — mean similarity to k nearest train reactions
    """
    n_q   = fps_query.shape[0]
    topk  = np.zeros((n_q, k), dtype=np.float32)   # running top-k buffer

    q_f      = fps_query.astype(np.float32)
    counts_q = q_f.sum(axis=1)

    n_ref        = fps_ref.shape[0]
    n_outer      = (n_ref + batch_r - 1) // batch_r
    report_every = max(1, n_outer // 4)

    for oi, rs in enumerate(range(0, n_ref, batch_r)):
        if label and oi % report_every == 0:
            print(f"    {label}: {100 * rs / n_ref:.0f}%", flush=True)
        re  = min(rs + batch_r, n_ref)
        rb  = fps_ref[rs:re].astype(np.float32)   # (br, D)
        crb = rb.sum(axis=1)                       # (br,)

        for qs in range(0, n_q, batch_q):
            qe  = min(qs + batch_q, n_q)
            qb  = q_f[qs:qe]                       # (bq, D)
            inter = qb @ rb.T                      # (bq, br)
            union = counts_q[qs:qe, None] + crb[None, :] - inter
            tan   = np.where(union > 0, inter / union, 0.0)  # (bq, br)

            # Merge new chunk into running top-k: concatenate and partial-sort
            combined = np.concatenate([topk[qs:qe], tan], axis=1)  # (bq, k+br)
            # argpartition is O(n) — much faster than full sort
            idx = np.argpartition(combined, -k, axis=1)[:, -k:]
            topk[qs:qe] = combined[np.arange(qe - qs)[:, None], idx]

    max_tan   = topk.max(axis=1)
    mean_topk = topk.mean(axis=1)
    return max_tan, mean_topk


def _summarize(sim: np.ndarray, prefix: str, thresholds: list[float]) -> dict:
    out = {
        f"{prefix}_mean":   float(np.mean(sim)),
        f"{prefix}_median": float(np.median(sim)),
        f"{prefix}_p90":    float(np.percentile(sim, 90)),
        f"{prefix}_p95":    float(np.percentile(sim, 95)),
    }
    for t in thresholds:
        key = str(t).replace(".", "p")
        out[f"{prefix}_frac_ge_{key}"] = float(np.mean(sim >= t))
    return out


TOPK = 5   # number of nearest train neighbours for mean-topk metric


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",       required=True)
    ap.add_argument("--column",       required=True, help="Reaction SMILES column")
    ap.add_argument("--fold_column",  required=True, help="Fold IDs (-1, 0..N-1)")
    ap.add_argument("--group_column", default=None,
                    help="Optional leakage-group column for group count stats")
    ap.add_argument("--outfile", default=None,
                    help="Metrics CSV (default: <infile_dir>/<infile_stem>_fold_metrics.csv)")
    ap.add_argument("--per_reaction_out", default=None,
                    help="Optional CSV for per-reaction nearest-train Tanimoto "
                         "(default: <infile_dir>/<infile_stem>_per_reaction_metrics.csv); "
                         "pass empty string to disable")
    ap.add_argument("--fps_file", default="",
                    help="Path to precomputed .npy fingerprint file. "
                         "Defaults to <infile>_fps_<column>.npy")

    ap.add_argument("--dim",    type=int, default=1024)
    ap.add_argument("--radius", type=int, default=3)

    ap.add_argument("--similarity_thresholds", default="0.40,0.60,0.70,0.80",
                    help="Tanimoto thresholds for leakage-fraction stats")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    thresholds = [float(x) for x in args.similarity_thresholds.split(",") if x.strip()]

    # Auto-derive output paths if not given
    _indir  = os.path.dirname(os.path.abspath(args.infile))
    _stem   = os.path.splitext(os.path.basename(args.infile))[0]
    if args.outfile is None:
        args.outfile = os.path.join(_indir, f"{_stem}_fold_metrics.csv")
    if args.per_reaction_out is None:
        args.per_reaction_out = os.path.join(_indir, f"{_stem}_per_reaction_metrics.csv")
    # Pass --per_reaction_out "" explicitly to disable per-reaction output

    df = pd.read_csv(args.infile, low_memory=False)
    # Fold column is always required; reaction SMILES column only needed
    # to recompute fps (skip check when fps_file is provided explicitly).
    required_cols = [args.fold_column]
    if not args.fps_file:
        required_cols = [args.column] + required_cols
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")

    # ------------------------------------------------------------------
    # Fingerprints (memory-mapped when pre-computed)
    # ------------------------------------------------------------------
    fps_file = args.fps_file or args.infile.replace(".csv", f"_fps_{args.column}.npy")
    if os.path.exists(fps_file):
        print(f"Loading fingerprints from {fps_file}")
        fps = np.load(fps_file, mmap_mode="r")
    else:
        rxns = _remove_reagents(df[args.column].astype(str).tolist(), args.column)
        fps  = DrfpEncoder.encode(rxns, n_folded_length=args.dim,
                                  radius=args.radius, show_progress_bar=True)
        np.save(fps_file, fps)

    fps_bin = np.asarray(fps, dtype=np.uint8)   # view of mmap if dtype matches

    # If the split CSV is lightweight (Unique_identifier + fold columns only),
    # it stores row_index = position in the source fps file.  Reindex fps to
    # match df row order so all downstream boolean masks work correctly.
    if "row_index" in df.columns:
        fps_bin = fps_bin[df["row_index"].values]

    n       = fps_bin.shape[0]

    folds = sorted(int(x) for x in pd.unique(df[args.fold_column]) if int(x) >= 0)
    if not folds:
        raise ValueError("No non-negative fold IDs found.")

    rng = np.random.default_rng(args.random_state)

    # ------------------------------------------------------------------
    # Global random baseline (computed once for the size of the first fold)
    # ------------------------------------------------------------------
    # For splits with a permanent train pool (fold == -1): baseline = random
    # reactions drawn from that pool.
    # For splits where every reaction is assigned a fold (group_splits): baseline
    # = random reactions from the complement of fold 0.
    has_train_only = (df[args.fold_column].values == -1).any()
    if has_train_only:
        actual_train_mask = df[args.fold_column].values == -1
    else:
        actual_train_mask = df[args.fold_column].values != folds[0]
    n_train_total     = int(actual_train_mask.sum())
    first_fold_size   = int((df[args.fold_column].values == folds[0]).sum())
    rand_idx          = rng.choice(n_train_total, size=first_fold_size, replace=False)
    rand_mask         = np.zeros(n, dtype=bool)
    rand_mask[np.flatnonzero(actual_train_mask)[rand_idx]] = True
    print(f"\nComputing global random baseline (n_test={first_fold_size:,})...")
    rand_max_tan, rand_mean_topk = _tanimoto_topk_batched(
        fps_bin[rand_mask], fps_bin[actual_train_mask & ~rand_mask],
        k=TOPK, label="random baseline")
    rand_summary = _summarize(rand_max_tan, "random_tanimoto_to_train", thresholds)
    rand_summary.update(_summarize(rand_mean_topk, f"random_mean_top{TOPK}_to_train", thresholds))
    print(f"  Random baseline  max median={rand_summary['random_tanimoto_to_train_median']:.3f}  "
          f"mean-top{TOPK} median={rand_summary[f'random_mean_top{TOPK}_to_train_median']:.3f}")

    # ------------------------------------------------------------------
    # Per-fold analysis
    # ------------------------------------------------------------------
    metrics_rows, per_rxn_rows, fold_distributions = [], [], []

    for fold_id in folds:
        print(f"\n=== Fold {fold_id} ===")
        test_mask  = df[args.fold_column].values == fold_id
        # Use permanent train pool (fold == -1) when available; otherwise
        # use leave-one-fold-out (all other folds serve as train).
        if has_train_only:
            train_mask = df[args.fold_column].values == -1
        else:
            train_mask = df[args.fold_column].values != fold_id
        n_test, n_train = int(test_mask.sum()), int(train_mask.sum())
        if n_test == 0 or n_train == 0:
            print("  Skipping empty fold.")
            continue

        fps_test  = fps_bin[test_mask]
        fps_train = fps_bin[train_mask]

        print(f"  n_test={n_test:,}  n_train={n_train:,}")
        print(f"  Computing exact Tanimoto ({n_test:,} × {n_train:,})...")
        top_tan, mean_topk = _tanimoto_topk_batched(fps_test, fps_train,
                                                     k=TOPK, label=f"fold {fold_id}")

        row = {
            "fold_id":       fold_id,
            "n_test":        n_test,
            "n_train":       n_train,
            "test_fraction": round(n_test / n, 4),
        }
        if args.group_column and args.group_column in df.columns:
            row["n_test_groups"] = int(df.loc[test_mask, args.group_column].nunique())

        row.update(_summarize(top_tan, "tanimoto_to_train", thresholds))
        row.update(_summarize(mean_topk, f"mean_top{TOPK}_to_train", thresholds))
        row.update(rand_summary)
        row["delta_median_vs_random"] = (
            row["tanimoto_to_train_median"]
            - rand_summary["random_tanimoto_to_train_median"]
        )
        row[f"delta_mean_top{TOPK}_median_vs_random"] = (
            row[f"mean_top{TOPK}_to_train_median"]
            - rand_summary[f"random_mean_top{TOPK}_to_train_median"]
        )

        metrics_rows.append(row)
        fold_distributions.append({"fold_id": fold_id, "top_tan": top_tan,
                                    "mean_topk": mean_topk})

        print(f"  Tanimoto to train  max median={row['tanimoto_to_train_median']:.3f}  "
              f"mean-top{TOPK} median={row[f'mean_top{TOPK}_to_train_median']:.3f}  "
              f"p90={row['tanimoto_to_train_p90']:.3f}")
        print(f"  Random baseline    max median={rand_summary['random_tanimoto_to_train_median']:.3f}  "
              f"mean-top{TOPK} median={rand_summary[f'random_mean_top{TOPK}_to_train_median']:.3f}")
        print(f"  Delta vs random (max):       {row['delta_median_vs_random']:+.3f}")
        print(f"  Delta vs random (mean-top{TOPK}): {row[f'delta_mean_top{TOPK}_median_vs_random']:+.3f}")
        for t in thresholds:
            k = str(t).replace(".", "p")
            print(f"  Tanimoto >= {t:.2f}:  "
                  f"OOD={row[f'tanimoto_to_train_frac_ge_{k}']:.2%}  "
                  f"random={rand_summary[f'random_tanimoto_to_train_frac_ge_{k}']:.2%}")

        if args.per_reaction_out != "":
            per_rxn_rows.append(pd.DataFrame({
                "row_index":                    np.flatnonzero(test_mask),
                "fold_id":                      fold_id,
                "max_tanimoto_to_train":         top_tan,
                f"mean_top{TOPK}_tanimoto_to_train": mean_topk,
            }))

    pd.DataFrame(metrics_rows).to_csv(args.outfile, index=False)
    print(f"\nSaved fold metrics -> {args.outfile}")

    if args.per_reaction_out != "" and per_rxn_rows:
        pd.concat(per_rxn_rows, ignore_index=True).to_csv(
            args.per_reaction_out, index=False)
        print(f"Saved per-reaction metrics -> {args.per_reaction_out}")

    # ------------------------------------------------------------------
    # Distribution plot: OOD test→train (blue) vs random baseline (gray)
    # ------------------------------------------------------------------
    if fold_distributions:
        n_folds_plot = len(fold_distributions)
        bins = np.linspace(0, 1, 60)
        fig, axes = plt.subplots(
            2, n_folds_plot,
            figsize=(4 * n_folds_plot, 7),
            sharey="row", dpi=130,
        )
        if n_folds_plot == 1:
            axes = axes.reshape(2, 1)

        rand_median_max  = float(np.median(rand_max_tan))
        rand_median_topk = float(np.median(rand_mean_topk))
        row_configs = [
            (0, "top_tan",   rand_max_tan,   rand_median_max,
             "Max Tanimoto (top-1)",  "steelblue", "Max Tanimoto similarity"),
            (1, "mean_topk", rand_mean_topk, rand_median_topk,
             f"Mean Tanimoto (top-{TOPK})", "darkorange", f"Mean top-{TOPK} Tanimoto similarity"),
        ]
        for row_i, fd_key, rand_vals, rand_med, title_prefix, color, xlabel in row_configs:
            for col_i, fd in enumerate(fold_distributions):
                ax = axes[row_i][col_i]
                fid = fd["fold_id"]
                ax.hist(rand_vals, bins=bins, alpha=0.55,
                        color="gray", label="random", density=True)
                ax.hist(fd[fd_key], bins=bins, alpha=0.70,
                        color=color, label="OOD test→train", density=True)
                ax.axvline(rand_med, color="gray", lw=1.5, ls="--")
                ax.axvline(float(np.median(fd[fd_key])), color=color, lw=1.5, ls="--")
                ax.set_xlabel(xlabel, fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                if col_i == 0:
                    ax.set_ylabel("Density", fontsize=8)
                    ax.legend(frameon=False, fontsize=7)
                if row_i == 0:
                    ax.set_title(f"Fold {fid}", fontsize=10)

        fig.suptitle(
            "Tanimoto distributions: OOD test→train vs random baseline\n"
            "Row 1: max (top-1)  |  Row 2: mean top-5  |  Dashed = medians",
            y=1.01, fontsize=10,
        )
        plt.tight_layout()
        plot_path = args.outfile.replace(".csv", "_tanimoto_distributions.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved distribution plot -> {plot_path}")

    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

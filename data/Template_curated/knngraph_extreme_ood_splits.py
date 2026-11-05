#!/usr/bin/env python3
"""Build extreme OOD folds using reaction similarity and missing-atom counts.

Compute or load DRFP fingerprints and find approximate nearest neighbors.
Join neighbors whose exact Tanimoto similarity reaches --leakage_tanimoto
(default 0.55). Connected reactions form groups that must stay together.

Reduce the fingerprints with TruncatedSVD, then measure group isolation.
Select groups using size, isolation, and missing-atom or missing-carbon cutoffs.
Cluster their normalized centroids with K-means to form distinct test folds.
Groups not selected for testing remain in training.

Write a reaction CSV and a group summary. ood_fold=-1 means training only.
Values from 0 to n_folds-1 identify the held-out fold.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize

try:
    from pynndescent import NNDescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_reagents(reactions: list[str], column_name: str) -> list[str]:
    if column_name != "CanonicalReaction":
        return reactions
    print("Removing reagents from CanonicalReaction before encoding...")
    return [
        r.split(">")[0] + ">>" + r.split(">")[2] if r.count(">") == 2 else r
        for r in reactions
    ]


def _group_col(column_name: str) -> str:
    if column_name == "CanonicalReaction":
        return "leakage_group_id_RI"
    if column_name == "InputReaction_canonical":
        return "leakage_group_id_noRI"
    return "leakage_group_id"


def _fold_col(column_name: str) -> str:
    if column_name == "CanonicalReaction":
        return "ood_fold_RI"
    if column_name == "InputReaction_canonical":
        return "ood_fold_noRI"
    return "ood_fold"


def _build_knn(X: np.ndarray, k: int, backend: str, random_state: int):
    """Return (indices, distances) arrays of shape (N, k), self excluded."""
    use_pynd = backend == "pynndescent" or (backend == "auto" and HAS_PYNNDESCENT)

    if use_pynd:
        print(f"  PyNNDescent kNN: k={k}")
        idx = NNDescent(X, n_neighbors=k + 1, metric="cosine",
                        random_state=random_state, n_jobs=-1)
        indices, distances = idx.neighbor_graph
    else:
        print(f"  sklearn kNN: k={k}")
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine",
                                algorithm="auto", n_jobs=-1)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)

    # Remove self reliably: self should appear in column 0 but isn't guaranteed.
    n = X.shape[0]
    self_col = np.where(indices == np.arange(n)[:, None])
    has_self = np.unique(self_col[0])
    if len(has_self) > n * 0.5:
        # Most rows have self -> strip it
        mask = indices != np.arange(n)[:, None]
        indices   = np.vstack([indices[i][mask[i]][:k]   for i in range(n)])
        distances = np.vstack([distances[i][mask[i]][:k] for i in range(n)])
    else:
        indices   = indices[:, :k]
        distances = distances[:, :k]

    return indices.astype(np.int32), distances.astype(np.float32)


def _tanimoto_from_knn(fps_bin: np.ndarray, knn_indices: np.ndarray,
                        batch: int = 128) -> np.ndarray:
    """
    Compute Tanimoto similarity for every (i, knn_indices[i, j]) pair.
    Batches rows of size `batch` to cut Python loop overhead ~batch× vs.
    the per-row loop.  Memory per batch: batch * k * D * 4 bytes (~50 MB
    for defaults batch=128, k=100, D=1024).
    Returns a flat array of length n * k.
    """
    n, k   = knn_indices.shape
    out    = np.empty(n * k, dtype=np.float32)
    counts = fps_bin.sum(axis=1).astype(np.uint16)
    for s in range(0, n, batch):
        e   = min(s + batch, n)
        b   = e - s
        idx = knn_indices[s:e].reshape(-1)                        # (b*k,)
        A   = fps_bin[s:e].astype(np.float32)                     # (b, D)
        B   = fps_bin[idx].reshape(b, k, -1).astype(np.float32)   # (b, k, D)
        inter = (B * A[:, None, :]).sum(axis=-1)                   # (b, k)
        c_a   = counts[s:e].astype(np.float32)                    # (b,)
        c_b   = counts[idx].reshape(b, k).astype(np.float32)      # (b, k)
        union = c_a[:, None] + c_b - inter
        out[s * k:e * k] = np.where(union > 0, inter / union, 0.0).reshape(-1)
    return out


def _plot_group_knn_map(
    group_centroids: np.ndarray,
    group_sizes: np.ndarray,
    assigned_group_to_fold: np.ndarray,
    eligible_mask: np.ndarray,
    knn_backend: str,
    n_folds: int,
    random_state: int,
    out_path: str,
) -> None:
    """Plot global and local views of group placement in graph space."""
    n_groups = len(group_centroids)
    if n_groups == 0:
        return

    if n_groups == 1:
        layout = np.zeros((1, 2), dtype=np.float32)
    else:
        if HAS_UMAP:
            n_neighbors = max(5, min(30, n_groups - 1))
            reducer = UMAP(
                n_components=2,
                metric="cosine",
                n_neighbors=n_neighbors,
                min_dist=0.35,
                random_state=random_state,
            )
            layout = reducer.fit_transform(group_centroids).astype(np.float32)
            layout_method = f"UMAP (cosine, n_neighbors={n_neighbors})"
        else:
            layout = PCA(n_components=2, random_state=random_state).fit_transform(group_centroids).astype(np.float32)
            layout_method = "PCA fallback"

    map_k = max(2, min(10, n_groups - 1)) if n_groups > 1 else 0
    if map_k >= 2:
        group_knn, _ = _build_knn(group_centroids, map_k, knn_backend, random_state)
    else:
        group_knn = np.empty((n_groups, 0), dtype=np.int32)

    x = layout[:, 0]
    y = layout[:, 1]
    node_sizes = 8 + 6 * np.log1p(group_sizes)
    selected_mask = assigned_group_to_fold >= 0
    cmap = plt.get_cmap("tab10")

    fig, (ax_global, ax_focus) = plt.subplots(1, 2, figsize=(13, 6), dpi=180)

    train_mask = ~selected_mask
    ax_global.scatter(x[train_mask], y[train_mask], s=np.clip(node_sizes[train_mask] * 0.5, 4, 18),
                      c="#d0d0d0", alpha=0.18, linewidths=0, rasterized=True,
                      label="Train-only groups", zorder=1)

    if np.any(eligible_mask):
        ax_global.scatter(x[eligible_mask], y[eligible_mask],
                          s=np.clip(node_sizes[eligible_mask] + 12, 18, 80),
                          facecolors="none", edgecolors="#333333", linewidths=0.5,
                          alpha=0.55, label="Eligible groups", zorder=2)

    for f in range(n_folds):
        fold_mask = assigned_group_to_fold == f
        if np.any(fold_mask):
            ax_global.scatter(x[fold_mask], y[fold_mask],
                              s=np.clip(node_sizes[fold_mask] + 18, 24, 120),
                              color=cmap(f % 10), alpha=0.95, linewidths=0,
                              label=f"Test fold {f}", zorder=3)

    focus_nodes = set(np.flatnonzero(selected_mask).tolist())
    for src in list(focus_nodes):
        if src < len(group_knn):
            focus_nodes.update(int(dst) for dst in group_knn[src])
    focus_nodes = np.array(sorted(focus_nodes), dtype=int)
    focus_mask = np.zeros(n_groups, dtype=bool)
    focus_mask[focus_nodes] = True

    edges_seen = set()
    for src in focus_nodes:
        for dst in group_knn[src]:
            dst = int(dst)
            if not focus_mask[dst]:
                continue
            a, b = (src, dst) if src < dst else (dst, src)
            if a == b or (a, b) in edges_seen:
                continue
            edges_seen.add((a, b))
            ax_focus.plot([x[a], x[b]], [y[a], y[b]], color="#c9c9c9", lw=0.6,
                          alpha=0.5, zorder=1)

    local_bg_mask = focus_mask & train_mask
    if np.any(local_bg_mask):
        ax_focus.scatter(x[local_bg_mask], y[local_bg_mask],
                         s=np.clip(node_sizes[local_bg_mask] * 0.8, 8, 28),
                         c="#bdbdbd", alpha=0.55, linewidths=0,
                         label="1-hop neighbors", zorder=2)

    for f in range(n_folds):
        fold_mask = assigned_group_to_fold == f
        if np.any(fold_mask):
            ax_focus.scatter(x[fold_mask], y[fold_mask],
                             s=np.clip(node_sizes[fold_mask] + 26, 30, 140),
                             color=cmap(f % 10), alpha=0.98, linewidths=0,
                             label=f"Selected fold {f}", zorder=3)

    for ax in (ax_global, ax_focus):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_global.set_title(f"All groups in 2D layout\n{layout_method}")
    ax_focus.set_title("Selected groups with local kNN neighborhood")
    ax_global.set_xlabel("Layout axis 1")
    ax_global.set_ylabel("Layout axis 2")
    ax_focus.set_xlabel("Layout axis 1")

    handles_global, labels_global = ax_global.get_legend_handles_labels()
    handles_focus, labels_focus = ax_focus.get_legend_handles_labels()
    legend_items = dict(zip(labels_global + labels_focus, handles_global + handles_focus))
    fig.legend(legend_items.values(), legend_items.keys(), frameon=False,
               fontsize=8, loc="upper center", ncol=min(4, len(legend_items)))
    fig.suptitle("Group-level OOD selection map", fontsize=11, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",  required=True, help="Input CSV file")
    ap.add_argument("--column",  required=True, help="Reaction SMILES column")
    ap.add_argument("--outfile", default=None,
                    help="Output CSV (default: <infile_dir>/ood_folds/<stem>_ood_splits_<RI|noRI>.csv)")
    ap.add_argument("--group_summary_out", default=None,
                    help="Group summary CSV (default: <infile_dir>/ood_folds/<stem>_ood_group_summary_<RI|noRI>.csv)")
    ap.add_argument("--meta_out", default=None,
                    help="Metadata JSON (default: alongside --outfile with suffix _meta.json)")

    # Fingerprint
    ap.add_argument("--dim",    type=int, default=1024, help="Folded DRFP length")
    ap.add_argument("--radius", type=int, default=3,    help="DRFP radius")
    ap.add_argument("--svd",    type=int, default=100,  help="SVD dims for centroid scoring")

    # kNN
    ap.add_argument("--knn_k",   type=int, default=100,
                    help="Neighbours per reaction for leakage graph")
    ap.add_argument("--knn_backend", choices=["auto", "pynndescent", "sklearn"],
                    default="auto")

    # Leakage threshold — now in Tanimoto space (raw fingerprints)
    ap.add_argument("--leakage_tanimoto", type=float, default=0.55,
                    help="Tanimoto similarity threshold for merging reactions into a "
                         "leakage group (operates on raw binary DRFP, not SVD space).")

    # Fold construction
    ap.add_argument("--n_folds",           type=int,   default=5)
    ap.add_argument("--max_group_fraction", type=float, default=0.03,
                    help="Groups larger than this fraction are excluded from test folds.")
    ap.add_argument("--min_isolation_pct",  type=float, default=67.0,
                    help="Only groups with centroid isolation above this percentile "
                         "of the isolation-score distribution are eligible for test "
                         "folds. Default 67 keeps the top 33%% most isolated groups. "
                         "Raise towards 80-90 for stricter OOD (lower test-train "
                         "similarity); lower towards 25-40 for larger folds.")

    # Completion-burden filters
    ap.add_argument("--min_missing_atoms_pct", type=float, default=80.0,
                    help="Only groups with median MissingNrAtoms above this percentile "
                         "are eligible for ExtremeOOD.")
    ap.add_argument("--min_missing_carbons_pct", type=float, default=80.0,
                    help="Only groups with median MissingNrCarbons above this percentile "
                         "are eligible for ExtremeOOD.")
    ap.add_argument("--burden_mode", choices=["or", "and"], default="or",
                    help="'or': group must be high in missing atoms OR missing carbons; "
                         "'and': group must be high in both. Default: or (stricter).")

    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--plot_group_map", action="store_true",
                    help="Save a group-level kNN map showing selected test-fold groups.")
    ap.add_argument("--group_map_out", default=None,
                    help="Output PNG for group-level map (default: <outdir>/<stem>_group_knn_map_<suffix><tag>.png)")
    args = ap.parse_args()

    if args.knn_backend == "pynndescent" and not HAS_PYNNDESCENT:
        raise ImportError("pynndescent is not installed.")

    _thresh_tag = f"_t{int(round(args.leakage_tanimoto * 100)):03d}"
    _suffix = {"CanonicalReaction": "RI", "InputReaction_canonical": "noRI"}.get(
        args.column, args.column)
    _indir  = os.path.dirname(os.path.abspath(args.infile))
    _stem   = os.path.splitext(os.path.basename(args.infile))[0]
    _outdir = os.path.join(_indir, "ood_folds_missing_stats")
    os.makedirs(_outdir, exist_ok=True)
    if args.outfile is None:
        args.outfile = os.path.join(_outdir, f"{_stem}_ood_splits_{_suffix}{_thresh_tag}.csv")
    if args.group_summary_out is None:
        args.group_summary_out = os.path.join(_outdir, f"{_stem}_ood_group_summary_{_suffix}{_thresh_tag}.csv")
    if args.meta_out is None:
        args.meta_out = os.path.splitext(args.outfile)[0] + "_meta.json"
    if args.group_map_out is None:
        args.group_map_out = os.path.join(_outdir, f"{_stem}_group_knn_map_{_suffix}{_thresh_tag}.png")
    print(f"Output -> {args.outfile}")

    # ------------------------------------------------------------------
    # 1. Load reactions
    # ------------------------------------------------------------------
    required_cols = ["Unique_identifier", args.column, "MissingNrAtoms", "MissingNrCarbons"]
    df_rxn = pd.read_csv(args.infile, usecols=required_cols, low_memory=False)

    if args.column not in df_rxn.columns:
        raise ValueError(f"Column '{args.column}' not found.")
    for col in ["MissingNrAtoms", "MissingNrCarbons"]:
        if col not in df_rxn.columns:
            raise ValueError(f"Column '{col}' not found.")

    df_rxn["MissingNrAtoms"] = pd.to_numeric(df_rxn["MissingNrAtoms"], errors="coerce")
    df_rxn["MissingNrCarbons"] = pd.to_numeric(df_rxn["MissingNrCarbons"], errors="coerce")
    if df_rxn["MissingNrAtoms"].isna().any():
        raise ValueError("MissingNrAtoms contains missing or non-numeric values.")
    if df_rxn["MissingNrCarbons"].isna().any():
        raise ValueError("MissingNrCarbons contains missing or non-numeric values.")

    # ------------------------------------------------------------------
    # 2. Fingerprints
    # ------------------------------------------------------------------
    fps_file = args.infile.replace(".csv", f"_fps_{args.column}.npy")
    if os.path.exists(fps_file):
        print(f"Loading fingerprints from {fps_file}")
        fps = np.load(fps_file, mmap_mode="r")
    else:
        rxns = _remove_reagents(df_rxn[args.column].astype(str).tolist(), args.column)
        fps  = DrfpEncoder.encode(rxns, n_folded_length=args.dim,
                                  radius=args.radius, show_progress_bar=True)
        np.save(fps_file, fps)

    fps_bin  = np.asarray(fps, dtype=np.uint8)
    n        = fps_bin.shape[0]
    print(f"Reactions: {n}")

    # ------------------------------------------------------------------
    # 3. kNN graph in SVD space, Tanimoto edges in fingerprint space
    # ------------------------------------------------------------------
    # X_svd, knn_indices and tan_flat depend only on --column/--svd/--knn_k/
    # --random_state, NOT on --leakage_tanimoto, so they are cached and
    # reused across all threshold values and all three split scripts.
    _base       = args.infile.replace(".csv", f"_{args.column}_svd{args.svd}_k{args.knn_k}_r{args.random_state}")
    _xsvd_cache = _base + "_xsvd.npy"
    _knn_cache  = _base + "_knn.npy"
    _tan_cache  = _base + "_tan.npy"

    svd_dim = max(2, min(args.svd, n - 1, fps_bin.shape[1] - 1))
    knn_k   = max(2, min(args.knn_k, n - 1))

    if os.path.exists(_xsvd_cache):
        print(f"Loading cached X_svd from {_xsvd_cache}")
        X_svd = np.load(_xsvd_cache)
    else:
        X_norm = normalize(sparse.csr_matrix(fps_bin, dtype=np.float32))
        print(f"TruncatedSVD: {svd_dim} components")
        X_svd  = normalize(TruncatedSVD(n_components=svd_dim,
                                        random_state=args.random_state).fit_transform(X_norm))
        np.save(_xsvd_cache, X_svd)

    if os.path.exists(_knn_cache):
        print(f"Loading cached kNN indices from {_knn_cache}")
        knn_indices = np.load(_knn_cache)
    else:
        print(f"Building kNN graph (k={knn_k})...")
        knn_indices, _ = _build_knn(X_svd, knn_k, args.knn_backend, args.random_state)
        np.save(_knn_cache, knn_indices)

    rows_flat = np.repeat(np.arange(n), knn_k)
    cols_flat = knn_indices.reshape(-1)

    if os.path.exists(_tan_cache):
        print(f"Loading cached Tanimoto edges from {_tan_cache}")
        tan_flat = np.load(_tan_cache)
    else:
        print("Computing Tanimoto similarities for kNN edges...")
        tan_flat = _tanimoto_from_knn(fps_bin, knn_indices)
        np.save(_tan_cache, tan_flat)

    # Diagnostic
    print(f"\nTanimoto distribution across {len(tan_flat):,} kNN edges:")
    for pct in [50, 75, 90, 95, 99]:
        print(f"  p{pct}: {np.percentile(tan_flat, pct):.3f}")
    print(f"  >= {args.leakage_tanimoto:.2f}: "
          f"{(tan_flat >= args.leakage_tanimoto).mean():.2%} of edges\n")

    # ------------------------------------------------------------------
    # 4. Connected components = leakage groups
    # ------------------------------------------------------------------
    keep     = tan_flat >= args.leakage_tanimoto
    r_keep   = np.concatenate([rows_flat[keep], cols_flat[keep]])
    c_keep   = np.concatenate([cols_flat[keep], rows_flat[keep]])
    leak_graph = sparse.csr_matrix(
        (np.ones(len(r_keep), dtype=np.uint8), (r_keep, c_keep)),
        shape=(n, n),
    )
    leak_graph.setdiag(1)
    leak_graph.eliminate_zeros()

    n_groups, group_labels = connected_components(leak_graph, directed=False,
                                                  return_labels=True)
    group_sizes = np.bincount(group_labels, minlength=n_groups)
    print(f"Leakage groups: {n_groups}  (median size {np.median(group_sizes):.0f}, "
          f"max {group_sizes.max()})")

    # ------------------------------------------------------------------
    # 5. Centroid isolation score (in SVD cosine space)
    # ------------------------------------------------------------------
    print("Computing group centroids and isolation scores...")
    group_sums = np.zeros((n_groups, svd_dim), dtype=np.float32)
    np.add.at(group_sums, group_labels, X_svd)
    group_centroids = normalize(group_sums / group_sizes[:, None])

    if n_groups > 1:
        c_idx, c_dist = _build_knn(group_centroids, 1, args.knn_backend, args.random_state)
        nn_group = c_idx[:, 0]
        nn_dist  = c_dist[:, 0]
    else:
        nn_group = np.full(n_groups, -1, dtype=int)
        nn_dist  = np.zeros(n_groups, dtype=np.float32)

    # ------------------------------------------------------------------
    # 5b. Group-level completion burden summaries
    # ------------------------------------------------------------------
    tmp = pd.DataFrame({
        "group_id": group_labels,
        "MissingNrAtoms": df_rxn["MissingNrAtoms"].values,
        "MissingNrCarbons": df_rxn["MissingNrCarbons"].values,
    })
    burden_df = tmp.groupby("group_id", sort=True).agg(
        group_missing_atoms_median=("MissingNrAtoms", "median"),
        group_missing_atoms_mean=("MissingNrAtoms", "mean"),
        group_missing_carbons_median=("MissingNrCarbons", "median"),
        group_missing_carbons_mean=("MissingNrCarbons", "mean"),
    ).reset_index()

    # ------------------------------------------------------------------
    # 6. K-means clustering → one fold per chemical-space region
    # ------------------------------------------------------------------
    max_group_size   = int(args.max_group_fraction * n)
    isolation_thresh = float(np.percentile(nn_dist, args.min_isolation_pct))
    atoms_thresh     = float(np.percentile(
        burden_df["group_missing_atoms_median"].values,
        args.min_missing_atoms_pct
    ))
    carbons_thresh   = float(np.percentile(
        burden_df["group_missing_carbons_median"].values,
        args.min_missing_carbons_pct
    ))

    group_df = pd.DataFrame({
        "group_id":   np.arange(n_groups, dtype=int),
        "group_size": group_sizes,
        "nn_group":   nn_group,
        "nn_dist":    nn_dist,
    }).merge(burden_df, on="group_id", how="left")

    # Eligible: small enough AND sufficiently isolated AND high completion burden
    isolated_mask = (
        (group_df["group_size"] <= max_group_size) &
        (group_df["nn_dist"] >= isolation_thresh)
    ).values

    high_atoms_mask = (
        group_df["group_missing_atoms_median"] >= atoms_thresh
    ).values
    high_carbons_mask = (
        group_df["group_missing_carbons_median"] >= carbons_thresh
    ).values

    if args.burden_mode == "and":
        burden_mask = high_atoms_mask & high_carbons_mask
    else:
        burden_mask = high_atoms_mask | high_carbons_mask

    eligible_mask = isolated_mask & burden_mask
    eligible_gids = np.flatnonzero(eligible_mask)
    eligible_centroids = group_centroids[eligible_gids]

    # Drop zero-norm centroids — cosine distance is undefined for them
    norms = np.linalg.norm(eligible_centroids, axis=1)
    nonzero = norms > 1e-12
    eligible_gids = eligible_gids[nonzero]
    eligible_centroids = eligible_centroids[nonzero]
    n_eligible = len(eligible_gids)

    print(f"\nExtremeOOD thresholds:")
    print(f"  isolation >= p{args.min_isolation_pct:.0f} = {isolation_thresh:.4f}")
    print(f"  missing atoms median >= p{args.min_missing_atoms_pct:.0f} = {atoms_thresh:.4f}")
    print(f"  missing carbons median >= p{args.min_missing_carbons_pct:.0f} = {carbons_thresh:.4f}")
    print(f"  burden mode = {args.burden_mode}")

    print(f"\nEligible groups for test folds: {n_eligible:,}  "
          f"(size <= {max_group_size:,})")
    print(f"  isolated+size filter: {isolated_mask.sum():,}")
    print(f"  burden filter:        {burden_mask.sum():,}")
    print(f"  combined eligible:    {eligible_mask.sum():,}")

    if n_eligible < args.n_folds:
        raise ValueError(
            f"Only {n_eligible} eligible groups for {args.n_folds} folds. "
            f"Relax thresholds or reduce n_folds."
        )

    print(f"K-means clustering into {args.n_folds} folds (spherical, on normalized centroids)...")
    km = KMeans(
        n_clusters=args.n_folds,
        n_init=10,
        random_state=args.random_state,
    )
    cluster_labels = km.fit_predict(eligible_centroids)

    assigned_group_to_fold = np.full(n_groups, -1, dtype=int)
    for i, gid in enumerate(eligible_gids):
        assigned_group_to_fold[gid] = int(cluster_labels[i])

    reaction_fold = assigned_group_to_fold[group_labels]

    # Compute fold sizes after assignment
    fold_sizes = np.array([
        int((reaction_fold == f).sum()) for f in range(args.n_folds)
    ])

    if args.plot_group_map:
        print(f"Saving group-level kNN map to {args.group_map_out} ...")
        _plot_group_knn_map(
            group_centroids=group_centroids,
            group_sizes=group_sizes,
            assigned_group_to_fold=assigned_group_to_fold,
            eligible_mask=eligible_mask,
            knn_backend=args.knn_backend,
            n_folds=args.n_folds,
            random_state=args.random_state,
            out_path=args.group_map_out,
        )

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    group_df["eligible_extremeood"] = eligible_mask
    group_df["assigned_ood_fold"] = assigned_group_to_fold
    group_df.to_csv(args.group_summary_out, index=False)

    df_out = pd.DataFrame({
        "Unique_identifier":      df_rxn["Unique_identifier"].values,
        "row_index":              np.arange(n),
        _group_col(args.column):  group_labels,
        _fold_col(args.column):   reaction_fold,
    })
    df_out.to_csv(args.outfile, index=False)

    meta = {
        "input": {
            "infile": os.path.abspath(args.infile),
            "column": args.column,
        },
        "outputs": {
            "outfile": os.path.abspath(args.outfile),
            "group_summary_out": os.path.abspath(args.group_summary_out),
            "meta_out": os.path.abspath(args.meta_out),
        },
        "args": vars(args),
        "derived_thresholds": {
            "max_group_size": int(max_group_size),
            "isolation_threshold": float(isolation_thresh),
            "missing_atoms_threshold": float(atoms_thresh),
            "missing_carbons_threshold": float(carbons_thresh),
        },
        "counts": {
            "n_reactions": int(n),
            "n_groups": int(n_groups),
            "n_eligible_groups": int(n_eligible),
            "isolated_size_groups": int(isolated_mask.sum()),
            "burden_groups": int(burden_mask.sum()),
            "combined_eligible_groups": int(eligible_mask.sum()),
            "train_only_reactions": int((reaction_fold == -1).sum()),
        },
        "fold_sizes": {
            str(f): int((reaction_fold == f).sum()) for f in range(args.n_folds)
        },
    }
    with open(args.meta_out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nFold sizes (no fixed target — each covers a distinct chemical-space region):")
    for f in range(args.n_folds):
        n_f = fold_sizes[f]
        n_g = int((assigned_group_to_fold == f).sum())
        print(f"  Fold {f}: {n_f:,} reactions ({n_f/n:.1%})  {n_g} groups")
    always_train = int((reaction_fold == -1).sum())
    print(f"  Train-only: {always_train:,} ({always_train/n:.1%})")
    print(f"\n-> {args.outfile}\n-> {args.group_summary_out}\n-> {args.meta_out}")


if __name__ == "__main__":
    main()

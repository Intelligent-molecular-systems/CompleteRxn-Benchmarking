#!/usr/bin/env python3
"""Build group-based splits using reaction fingerprint similarity.

Compute or load DRFP fingerprints and find approximate nearest neighbors.
Use exact Tanimoto similarities to identify edges above --leakage_tanimoto.
Connected reactions form groups that stay in the same fold.
See the command-line options for group selection and fold assignment.
"""

import argparse
import os

import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

try:
    from pynndescent import NNDescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_reagents(reactions: list[str], column_name: str) -> list[str]:
    """
    Remove the reagent field from reaction SMILES only for CanonicalReaction,
    converting reactants>reagents>products -> reactants>>products.
    """
    if column_name != "CanonicalReaction":
        return reactions

    print("Removing reagents from CanonicalReaction before encoding...")
    out = []
    for r in reactions:
        if r.count(">") == 2:
            left, _, right = r.split(">")
            out.append(f"{left}>>{right}")
        else:
            out.append(r)
    return out


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
    """
    Return (indices, distances) arrays of shape (N, k), self excluded.
    Distances are cosine distances.
    """
    use_pynd = backend == "pynndescent" or (backend == "auto" and HAS_PYNNDESCENT)

    if use_pynd:
        print(f"  PyNNDescent kNN: k={k}")
        idx = NNDescent(
            X,
            n_neighbors=k + 1,
            metric="cosine",
            random_state=random_state,
            n_jobs=-1,
        )
        indices, distances = idx.neighbor_graph
    else:
        print(f"  sklearn kNN: k={k}")
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(
            n_neighbors=k + 1,
            metric="cosine",
            algorithm="auto",
            n_jobs=-1,
        )
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)

    n = X.shape[0]

    # Remove self robustly. Usually self is present, but not always in a fixed column.
    mask = indices != np.arange(n)[:, None]

    # If self is present for most rows, strip it row-wise.
    if mask.sum() < indices.size:
        if (mask.sum(axis=1) < k).any():
            raise RuntimeError(
                "After self-removal, some rows have fewer than k neighbors. "
                "Try reducing --knn_k."
            )
        indices = np.vstack([indices[i][mask[i]][:k] for i in range(n)])
        distances = np.vstack([distances[i][mask[i]][:k] for i in range(n)])
    else:
        indices = indices[:, :k]
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Input CSV file")
    ap.add_argument("--column", required=True, help="Reaction SMILES column")
    ap.add_argument("--outfile", default=None,
                    help="Output CSV (default: <infile_dir>/rxn_groups/<stem>_group_splits_<RI|noRI>.csv)")
    ap.add_argument("--group_summary_out", default=None,
                    help="Group summary CSV (default: <infile_dir>/rxn_groups/<stem>_group_summary_<RI|noRI>.csv)")

    # Fingerprint
    ap.add_argument("--dim", type=int, default=1024, help="Folded DRFP length")
    ap.add_argument("--radius", type=int, default=3, help="DRFP radius")
    ap.add_argument("--svd", type=int, default=100, help="SVD dimensions for group centroids")

    # kNN for leakage graph construction
    ap.add_argument(
        "--knn_k",
        type=int,
        default=100,
        help="Neighbors per reaction for leakage graph construction",
    )
    ap.add_argument(
        "--knn_backend",
        choices=["auto", "pynndescent", "sklearn"],
        default="auto",
    )

    # Leakage threshold in raw binary DRFP Tanimoto space
    ap.add_argument(
        "--leakage_tanimoto",
        type=float,
        default=0.55,
        help=(
            "Tanimoto similarity threshold for merging reactions into a leakage group "
            "(operates on raw binary DRFP, not SVD space)"
        ),
    )

    # Fold construction
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--random_state", type=int, default=42)

    args = ap.parse_args()

    if args.knn_backend == "pynndescent" and not HAS_PYNNDESCENT:
        raise ImportError("pynndescent is not installed.")

    _thresh_tag = f"_t{int(round(args.leakage_tanimoto * 100)):03d}"
    _suffix = {"CanonicalReaction": "RI", "InputReaction_canonical": "noRI"}.get(
        args.column, args.column)
    _indir  = os.path.dirname(os.path.abspath(args.infile))
    _stem   = os.path.splitext(os.path.basename(args.infile))[0]
    _outdir = os.path.join(_indir, "rxn_groups")
    os.makedirs(_outdir, exist_ok=True)
    if args.outfile is None:
        args.outfile = os.path.join(_outdir, f"{_stem}_group_splits_{_suffix}{_thresh_tag}.csv")
    if args.group_summary_out is None:
        args.group_summary_out = os.path.join(_outdir, f"{_stem}_group_summary_{_suffix}{_thresh_tag}.csv")
    print(f"Output -> {args.outfile}")

    # ------------------------------------------------------------------
    # 1. Load reactions
    # ------------------------------------------------------------------
    df_rxn = pd.read_csv(args.infile, usecols=["Unique_identifier", args.column],
                         low_memory=False)
    if args.column not in df_rxn.columns:
        raise ValueError(f"Column '{args.column}' not found.")

    # ------------------------------------------------------------------
    # 2. Fingerprints
    # ------------------------------------------------------------------
    fps_file = args.infile.replace(".csv", f"_fps_{args.column}.npy")
    if os.path.exists(fps_file):
        print(f"Loading fingerprints from {fps_file}")
        fps = np.load(fps_file, mmap_mode="r")
    else:
        rxns = _remove_reagents(df_rxn[args.column].astype(str).tolist(), args.column)
        fps = DrfpEncoder.encode(
            rxns,
            n_folded_length=args.dim,
            radius=args.radius,
            show_progress_bar=True,
        )
        np.save(fps_file, fps)

    fps_bin = np.asarray(fps, dtype=np.uint8)
    n = fps_bin.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 reactions.")
    print(f"Reactions: {n:,}")

    # ------------------------------------------------------------------
    # 3. Build approximate kNN in reduced space, then exact Tanimoto on edges
    # ------------------------------------------------------------------
    # X_svd, knn_indices and tan_flat depend only on --column/--svd/--knn_k/
    # --random_state, NOT on --leakage_tanimoto, so they are cached and
    # reused across all threshold values and all three split scripts.
    _base       = args.infile.replace(".csv", f"_{args.column}_svd{args.svd}_k{args.knn_k}_r{args.random_state}")
    _xsvd_cache = _base + "_xsvd.npy"
    _knn_cache  = _base + "_knn.npy"
    _tan_cache  = _base + "_tan.npy"

    svd_dim = max(2, min(args.svd, n - 1, fps_bin.shape[1] - 1))
    knn_k   = max(1, min(args.knn_k, n - 1))

    if os.path.exists(_xsvd_cache):
        print(f"Loading cached X_svd from {_xsvd_cache}")
        X_svd = np.load(_xsvd_cache)
    else:
        X_norm = normalize(sparse.csr_matrix(fps_bin, dtype=np.float32))
        print(f"TruncatedSVD: {svd_dim} components")
        X_svd  = normalize(
            TruncatedSVD(n_components=svd_dim,
                         random_state=args.random_state).fit_transform(X_norm)
        )
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
    print(
        f"  >= {args.leakage_tanimoto:.2f}: "
        f"{(tan_flat >= args.leakage_tanimoto).mean():.2%} of edges\n"
    )

    # ------------------------------------------------------------------
    # 4. Connected components = leakage groups
    # ------------------------------------------------------------------
    keep = tan_flat >= args.leakage_tanimoto

    r_keep = np.concatenate([rows_flat[keep], cols_flat[keep]])
    c_keep = np.concatenate([cols_flat[keep], rows_flat[keep]])

    leak_graph = sparse.csr_matrix(
        (np.ones(len(r_keep), dtype=np.uint8), (r_keep, c_keep)),
        shape=(n, n),
    )
    leak_graph.setdiag(1)
    leak_graph.eliminate_zeros()

    n_groups, group_labels = connected_components(
        leak_graph,
        directed=False,
        return_labels=True,
    )
    group_sizes = np.bincount(group_labels, minlength=n_groups)

    print(
        f"Leakage groups: {n_groups:,}  "
        f"(median size {np.median(group_sizes):.0f}, max {group_sizes.max()})"
    )

    # ------------------------------------------------------------------
    # 5. Compute one centroid per leakage group
    # ------------------------------------------------------------------
    print("Computing group centroids...")
    group_sums = np.zeros((n_groups, svd_dim), dtype=np.float32)
    np.add.at(group_sums, group_labels, X_svd)
    group_centroids = group_sums / group_sizes[:, None]

    # Normalize for cosine/spherical k-means
    group_centroids = normalize(group_centroids)

    group_df = pd.DataFrame({
        "group_id": np.arange(n_groups, dtype=int),
        "group_size": group_sizes,
    })

    # Drop degenerate centroids if any
    norms = np.linalg.norm(group_centroids, axis=1)
    valid_mask = norms > 1e-12
    valid_gids = np.flatnonzero(valid_mask)
    valid_centroids = group_centroids[valid_gids]
    n_valid = len(valid_gids)

    print(f"Valid leakage groups for clustering: {n_valid:,}")
    if n_valid < args.n_folds:
        raise ValueError(
            f"Need at least {args.n_folds} valid leakage groups, got {n_valid}."
        )

    # ------------------------------------------------------------------
    # 6. Cluster all leakage-group centroids into OOD folds
    # ------------------------------------------------------------------
    print(
        f"K-means clustering all group centroids into {args.n_folds} folds "
        "(spherical/cosine on normalized centroids)..."
    )
    km = KMeans(
        n_clusters=args.n_folds,
        n_init=10,
        random_state=args.random_state,
    )
    cluster_labels = km.fit_predict(valid_centroids)

    assigned_group_to_fold = np.full(n_groups, -1, dtype=int)
    assigned_group_to_fold[valid_gids] = cluster_labels

    # Assign any zero-norm (degenerate) groups to the smallest fold so far
    degenerate_gids = np.flatnonzero(assigned_group_to_fold < 0)
    if len(degenerate_gids) > 0:
        print(f"  Assigning {len(degenerate_gids)} zero-norm group(s) to smallest fold.")
        for gid in degenerate_gids:
            fold_counts = np.bincount(
                assigned_group_to_fold[assigned_group_to_fold >= 0], minlength=args.n_folds
            )
            assigned_group_to_fold[gid] = int(np.argmin(fold_counts))

    reaction_fold = assigned_group_to_fold[group_labels]

    fold_sizes = np.array([
        int((reaction_fold == f).sum()) for f in range(args.n_folds)
    ])
    fold_group_counts = np.array([
        int((assigned_group_to_fold == f).sum()) for f in range(args.n_folds)
    ])

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    # Lightweight output: Unique_identifier + row_index + group/fold columns only.
    # Join back to benchmark_final.csv via Unique_identifier to get reaction data.
    group_df["ood_fold"] = assigned_group_to_fold
    group_df.to_csv(args.group_summary_out, index=False)

    df_out = pd.DataFrame({
        "Unique_identifier":      df_rxn["Unique_identifier"].values,
        "row_index":              np.arange(n),
        _group_col(args.column):  group_labels,
        _fold_col(args.column):   reaction_fold,
    })
    df_out.to_csv(args.outfile, index=False)

    print("\nFold sizes:")
    for f in range(args.n_folds):
        n_f = fold_sizes[f]
        n_g = fold_group_counts[f]
        print(f"  Fold {f}: {n_f:,} reactions ({n_f / n:.1%})  {n_g} groups")

    print(f"\n-> {args.outfile}")
    print(f"-> {args.group_summary_out}")


if __name__ == "__main__":
    main()
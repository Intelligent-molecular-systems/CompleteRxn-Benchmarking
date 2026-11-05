#!/usr/bin/env python3
"""
DRFP clustering with HDBSCAN

Input:  CSV file with reaction data
Output: CSV with original columns + cluster_id column
(-1 means noise / unassigned)
"""

import argparse
import os
import numpy as np
import pandas as pd

from drfp import DrfpEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import hdbscan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Input CSV file")
    ap.add_argument("--column", required=True, help="Column name containing reaction SMILES for fingerprint encoding")
    ap.add_argument("--outfile", default="hdbscan_clusters.csv")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--svd", type=int, default=64)
    ap.add_argument("--min_cluster_size", type=int, default=25)
    ap.add_argument("--sample_size", type=int, default=50000, help="Sample size for clustering (default: 10000). Set to None to cluster on full dataset.")
    args = ap.parse_args()

    # load CSV file
    df_input = pd.read_csv(args.infile)

    # check if the specified column exists
    if args.column not in df_input.columns:
        raise ValueError(f"Column '{args.column}' not found in CSV. Available columns: {list(df_input.columns)}")
    # Check if column name is either "CanonicalReaction" or "InputReaction_canonical" for proper cluster_id column naming in output
    if args.column not in ["CanonicalReaction", "InputReaction_canonical"]:
        print(f"Warning: Column '{args.column}' is not recognized as a standard reaction column. Output cluster_id column will be generically named 'cluster_id'.")
    # Check if fps already exist for infile_fps_{column}.npy, if so load them instead of recomputing
    fps_file = args.infile.replace(".csv", f"_fps_{args.column}.npy")
    if os.path.exists(fps_file):
        print(f"Loading precomputed fingerprints from {fps_file}...")
        fps = np.load(fps_file)
    else:
        # extract reaction SMILES from specified column
        rxns = df_input[args.column].astype(str).tolist()

        # For CanonicalReaction, remove reagents (middle part of reactants>reagents>products)
        # so DRFP focuses on the main transformation only
        if args.column == "CanonicalReaction":
            print("Removing reagents from CanonicalReaction before encoding...")
            rxns = [r.split(">")[0] + ">>" + r.split(">")[2] if r.count(">") == 2 else r for r in rxns]

        # DRFP encoding
        fps = DrfpEncoder.encode(
            rxns,
            n_folded_length=args.dim,
            radius=args.radius,
            show_progress_bar=True,
        )
        # store fingerprints for later use
        np.save(fps_file, fps)

    X = np.asarray(fps, dtype=np.float32)

    # normalize + dimensionality reduction (critical for HDBSCAN)
    X = normalize(X)
    svd_dim = min(args.svd, X.shape[0] - 1)
    Z = TruncatedSVD(n_components=svd_dim, random_state=0).fit_transform(X)

    # cluster
    # Sample-based clustering for large datasets
    sample_size = args.sample_size
    if sample_size and len(Z) > sample_size:
        print(f"Clustering on {sample_size} sample, then propagating labels...")
        np.random.seed(42)
        sample_idx = np.random.choice(len(Z), sample_size, replace=False)
        Z_sample = Z[sample_idx]
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=-1
        )
        sample_labels = clusterer.fit_predict(Z_sample)
        
        # Propagate to full dataset
        labels = np.full(len(Z), -1, dtype=int)
        labels[sample_idx] = sample_labels
        
        # Use KNN to assign remaining points
        clustered = sample_labels != -1
        if clustered.sum() > 0:
            knn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
            knn.fit(Z_sample[clustered])
            
            remaining = np.setdiff1d(np.arange(len(Z)), sample_idx)
            if len(remaining) > 0:
                distances, indices = knn.kneighbors(Z[remaining])
                # Assign to most common cluster among 5 nearest neighbors
                neighbor_labels = sample_labels[clustered][indices]
                from scipy import stats
                labels[remaining] = stats.mode(neighbor_labels, axis=1, keepdims=False)[0]
    else:
        print("Clustering on full dataset...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=-1
        )
        labels = clusterer.fit_predict(Z)

    # save results - add cluster_id to the original dataframe
    # column name depends on which reaction column was used
    if args.column == "CanonicalReaction":
        cluster_col = "cluster_id_RI"
    elif args.column == "InputReaction_canonical":
        cluster_col = "cluster_id_noRI"
    else:
        print(f"Warning: Unrecognized column '{args.column}', using generic 'cluster_id' for output.")
        cluster_col = "cluster_id"

    df_output = df_input.copy()
    df_output[cluster_col] = labels
    df_output.to_csv(args.outfile, index=False)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100

    print(f"Clusters found: {n_clusters}")
    print(f"Noise reactions: {noise_pct:.1f}%")
    print(f"Saved → {args.outfile}")


if __name__ == "__main__":
    main()

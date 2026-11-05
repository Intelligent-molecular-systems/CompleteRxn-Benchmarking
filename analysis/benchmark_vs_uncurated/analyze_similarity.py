#!/usr/bin/env python3
"""Compare reaction similarity within the benchmark and against uncurated USPTO.

For each sampled USPTO reaction, find its nearest benchmark reaction using
DRFP reaction fingerprints. Compare those similarities with the nearest-neighbor
similarities within the benchmark, excluding each reaction's match to itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from pynndescent import NNDescent


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data/Template_curated/benchmark/benchmark_final.csv"
BENCHMARK_FPS = ROOT / (
    "data/Template_curated/benchmark/"
    "benchmark_final_fps_CanonicalReaction.npy"
)
UNCURATED = ROOT / "data/USPTO_uncurated/uspto_uncurated_processed.csv"
DEFAULT_OUT = ROOT / "results/additional/benchmark_vs_uncurated"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--validate-queries", type=int, default=10)
    parser.add_argument(
        "--benchmark-limit",
        type=int,
        default=0,
        help="Use only the first N benchmark rows for a smoke test (0 = all).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def remove_reagents(reaction):
    """Match the split fingerprint: reactants>reagents>products -> reactants>>products."""
    try:
        left, _, right = str(reaction).split(">")
        return f"{left}>>{right}"
    except ValueError:
        return ""


def deterministic_sample(path, n, seed):
    """Select the same random sample for a given seed without loading the full CSV."""
    kept = []
    for chunk in pd.read_csv(
        path,
        usecols=["Unique_identifier", "CanonicalReaction"],
        chunksize=200_000,
        low_memory=False,
    ):
        keys = chunk["Unique_identifier"].astype(str) + f"|{seed}"
        chunk["_sample_hash"] = pd.util.hash_pandas_object(
            keys, index=False
        ).to_numpy(dtype=np.uint64)
        kept.append(chunk.nsmallest(n, "_sample_hash"))
        kept = [pd.concat(kept, ignore_index=True).nsmallest(n, "_sample_hash")]
    return kept[0].drop(columns="_sample_hash").reset_index(drop=True)


def tanimoto_to_candidates(query_fps, reference_fps, candidate_indices):
    best_similarity = np.empty(len(query_fps), dtype=np.float32)
    best_index = np.empty(len(query_fps), dtype=np.int64)
    for i, (query, candidates) in enumerate(
        zip(query_fps, candidate_indices)
    ):
        candidates = np.unique(candidates)
        references = np.asarray(reference_fps[candidates], dtype=np.uint8)
        intersection = np.bitwise_and(references, query).sum(axis=1)
        union = np.bitwise_or(references, query).sum(axis=1)
        similarity = np.divide(
            intersection,
            union,
            out=np.zeros(len(candidates), dtype=np.float32),
            where=union > 0,
        )
        best = int(np.argmax(similarity))
        best_similarity[i] = similarity[best]
        best_index[i] = candidates[best]
    return best_similarity, best_index


def paired_tanimoto(left_fps, right_fps):
    intersection = np.bitwise_and(left_fps, right_fps).sum(axis=1)
    union = np.bitwise_or(left_fps, right_fps).sum(axis=1)
    return np.divide(
        intersection,
        union,
        out=np.zeros(len(left_fps), dtype=np.float32),
        where=union > 0,
    )


def internal_similarity(index):
    """Find the nearest other reaction using the approximate DRFP similarity graph."""
    indices, distances = index.neighbor_graph
    rows = np.arange(len(indices))[:, None]
    nonself_distances = np.where(indices != rows, distances, np.inf)
    return 1 - nonself_distances.min(axis=1)


def brute_force_max(query_fps, reference_fps, n):
    """Check a small set of nearest-neighbor results against the full reference set."""
    n = min(n, len(query_fps))
    if n == 0:
        return np.array([], dtype=np.float32)
    lookup = np.unpackbits(
        np.arange(256, dtype=np.uint8)[:, None], axis=1
    ).sum(axis=1)
    reference_packed = np.packbits(reference_fps, axis=1)
    query_packed = np.packbits(query_fps[:n], axis=1)
    maxima = []
    for query in query_packed:
        intersection = lookup[np.bitwise_and(reference_packed, query)].sum(axis=1)
        union = lookup[np.bitwise_or(reference_packed, query)].sum(axis=1)
        maxima.append(
            np.divide(
                intersection,
                union,
                out=np.zeros(len(reference_packed), dtype=np.float32),
                where=union > 0,
            ).max()
        )
    return np.asarray(maxima, dtype=np.float32)


def distribution_row(name, values):
    quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "population": name,
        "n": len(values),
        "mean": values.mean(),
        "p10": quantiles[0],
        "p25": quantiles[1],
        "median": quantiles[2],
        "p75": quantiles[3],
        "p90": quantiles[4],
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = pd.read_csv(
        BENCHMARK,
        usecols=["Unique_identifier", "CanonicalReaction"],
        low_memory=False,
    )
    all_fps = np.load(BENCHMARK_FPS, mmap_mode="r")
    limit = args.benchmark_limit or len(benchmark)
    benchmark = benchmark.iloc[:limit].reset_index(drop=True)
    reference_fps = np.asarray(all_fps[:limit], dtype=np.uint8)
    if len(reference_fps) < 2:
        raise ValueError("The benchmark reference needs at least two reactions.")

    uncurated = deterministic_sample(
        UNCURATED, args.sample_size, args.seed
    )
    query_reactions = [
        remove_reagents(reaction)
        for reaction in uncurated["CanonicalReaction"]
    ]
    query_fps = np.asarray(
        DrfpEncoder.encode(
            query_reactions,
            n_folded_length=reference_fps.shape[1],
            radius=3,
            show_progress_bar=True,
        ),
        dtype=np.uint8,
    )

    print(
        f"Building label-free DRFP index: {len(reference_fps):,} CompleteRXN "
        f"reactions; {len(query_fps):,} uncurated queries"
    )
    n_candidates = min(args.candidates, len(reference_fps) - 1)
    index = NNDescent(
        reference_fps,
        n_neighbors=min(max(30, n_candidates // 2), len(reference_fps) - 1),
        metric="jaccard",
        random_state=42,
        n_jobs=-1,
        low_memory=True,
    )
    candidate_indices, _ = index.query(
        query_fps,
        k=n_candidates,
    )
    cross_similarity, nearest_index = tanimoto_to_candidates(
        query_fps, reference_fps, candidate_indices
    )
    benchmark_internal = internal_similarity(index)
    rng = np.random.default_rng(args.seed)
    random_indices = rng.integers(0, len(reference_fps), size=len(query_fps))
    random_pair_similarity = paired_tanimoto(
        query_fps,
        reference_fps[random_indices],
    )

    exact_validation = brute_force_max(
        query_fps, reference_fps, args.validate_queries
    )
    approximate_validation = cross_similarity[: len(exact_validation)]
    recall = (
        np.isclose(exact_validation, approximate_validation, atol=1e-7).mean()
        if len(exact_validation)
        else float("nan")
    )

    per_query = uncurated.copy()
    per_query["nearest_tanimoto"] = cross_similarity
    per_query["nearest_benchmark_uid"] = benchmark.iloc[
        nearest_index
    ]["Unique_identifier"].to_numpy()
    per_query["nearest_benchmark_reaction"] = benchmark.iloc[
        nearest_index
    ]["CanonicalReaction"].to_numpy()
    per_query["random_pair_tanimoto"] = random_pair_similarity
    per_query.to_csv(args.out_dir / "per_query.csv", index=False)

    summary = pd.DataFrame(
        [
            distribution_row(
                "CompleteRXN leave-one-out", np.asarray(benchmark_internal)
            ),
            distribution_row(
                "Uncurated USPTO to CompleteRXN", cross_similarity
            ),
            distribution_row(
                "Random uncurated-CompleteRXN pairs", random_pair_similarity
            ),
        ]
    )
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Candidate-search recall on {len(exact_validation)} exact checks: {100 * recall:.1f}%")
    print(f"Saved summary.csv and per_query.csv to {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot model accuracy and reaction counts against the number of missing atoms.

The default figure compares CRB and SynRBL using the current filtered scores.
Choose models and atom counts in CONFIG. Error bars show the 25th to 75th
percentiles across folds. The lower panel shows counts on a logarithmic scale.
An optional middle panel shows SynRBL's confident MCS-based solved fraction,
which is not accuracy against the reference.
"""

import os
import re
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis.figures.plot_accuracy_merged_by_missing_atoms import assign_bins, build_bin_spec
from utils.plot_style import apply_neurips_style


CONFIG = {
    "full_results_csv": "results/main/scores/synrbl_evaluated.csv",
    "benchmark_final_csv": "data/Template_curated/benchmark/benchmark_final.csv",
    "splits_dir": "data/Template_curated/benchmark",
    "crb_results_dir": "results/main/predictions",
    "rb_experiment": "BS_PT",
    "crb_experiment": "CBS_PT",
    "selected_models": ["crb", "synrbl"],
    "split_types": ["random", "group", "extreme_ood"],
    "missing_atoms_col": "MissingNrCarbons",
    "max_missing": 16,
    "crb_metric_col": "matches_selected_metric",
    "synrbl_metric_mode": "deep_pragmatic_equivalence_match",  # or synrbl_self_reported_acc
    "synrbl_min_confidence": 0.5,
    "include_synrbl_self_panel": False,
    "synrbl_deep_use_only_solved_rows": False,
    "binning_mode": "min_count",  # fixed, quantile, min_count, or user
    "quantile_n_bins": 8,
    "min_bin_size": 500,
    "user_bin_ranges": [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        (16, 16),
    ],
    "errorbar_alpha": 0.45,
    "marker_size_top": 3.3,
    "marker_size_mid": 3.2,
    "split_jitter_step": 0.24,
    "model_jitter_step": 0.14,
    "output": "results/main/figures/crb_synrbl_equivalence_by_missing_atoms_top2.png",
    "split_display_names": {
        "random": "Random",
        "group": "Group",
        "extreme_ood": "Extreme OOD",
    },
    "model_display_names": {
        "rb": "RB",
        "crb": "CRB",
        "synrbl": "SynRBL",
    },
}

SPLIT_CSV_META = {
    "random": {
        "subdir": "random_splits",
        "filename_re": r"^benchmark_final_random_splits_rep(\d+)_seed\d+\.csv$",
        "split_col": "random_split",
    },
    "group": {
        "subdir": "group_splits",
        "filename_re": r"^benchmark_final_group_splits_rep(\d+)\.csv$",
        "split_col": "group_split",
    },
    "extreme_ood": {
        "subdir": "extreme_ood_splits",
        "filename_re": r"^benchmark_final_extreme_ood_splits_rep(\d+)\.csv$",
        "split_col": "extreme_ood",
    },
}


def _abs_path(path: str) -> str:
    release = os.environ.get("COMPLETERXN_RESULTS_ROOT")
    if release and (path.startswith("results/main/scores/") or path.startswith("results/main/figures/")):
        path = os.path.join(release,path[len("results/"):])
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0) != 0
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _mix_with_white(hex_color: str, white_fraction: float) -> str:
    white_fraction = min(max(float(white_fraction), 0.0), 1.0)
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"#{hex_color}"

    channels = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
    mixed = [
        int(round(channel * (1.0 - white_fraction) + 255 * white_fraction))
        for channel in channels
    ]
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def _top1_path(cfg: dict, split_type: str, rep: int, split_fname: str, experiment: str) -> str:
    if split_type == "random":
        m_seed = re.search(r"seed(\d+)", split_fname)
        if not m_seed:
            raise ValueError(f"Could not parse random seed from split filename: {split_fname}")
        seed = int(m_seed.group(1))
        fold_dir = f"random_seed{seed}_rep{rep}"
    elif split_type == "group":
        fold_dir = f"group_rep{rep}"
    elif split_type == "extreme_ood":
        fold_dir = f"extreme_ood_rep{rep}"
    else:
        raise ValueError(f"Unsupported split type: {split_type}")

    model_dir = {"BS_PT": "rb", "CBS_PT": "crb"}[experiment]
    return _abs_path(os.path.join(cfg["crb_results_dir"], model_dir, split_type, fold_dir, "top1_predictions_with_stats.txt"))


def _map_top1_rows_to_uids(benchmark_sub: pd.DataFrame, top1_df: pd.DataFrame) -> pd.DataFrame:
    if "original_index" not in top1_df.columns:
        raise KeyError("Column 'original_index' missing in CRB top1 stats file.")

    idx = pd.to_numeric(top1_df["original_index"], errors="coerce").dropna().astype(int)
    idx = idx[(idx >= 0) & (idx < len(benchmark_sub))]
    if idx.empty:
        return pd.DataFrame(columns=["Unique_identifier"])

    return benchmark_sub.iloc[idx.values].reset_index(drop=True)


def _compute_synrbl_metric(df_bin: pd.DataFrame, cfg: dict) -> float:
    mode = cfg["synrbl_metric_mode"]
    if mode == "deep_pragmatic_equivalence_match":
        work = df_bin
        if cfg.get("synrbl_deep_use_only_solved_rows", False) and "solved" in work.columns:
            work = work[_safe_bool(work["solved"])]
        if work.empty:
            return float("nan")
        if "deep_pragmatic_equivalence_match" not in work.columns:
            raise KeyError("Column 'deep_pragmatic_equivalence_match' missing in SynRBL full results.")
        return float(_safe_bool(work["deep_pragmatic_equivalence_match"]).mean())

    if mode == "synrbl_self_reported_acc":
        solved = _safe_bool(df_bin.get("solved", pd.Series(False, index=df_bin.index)))
        solved_by = df_bin.get("solved_by", pd.Series("", index=df_bin.index)).astype(str).str.strip().str.lower()
        conf = pd.to_numeric(df_bin.get("confidence", 0.0), errors="coerce").fillna(0.0)
        confident_mcs = solved & (solved_by == "mcs-based") & (conf >= float(cfg["synrbl_min_confidence"]))
        return float(confident_mcs.sum() / len(df_bin)) if len(df_bin) else float("nan")

    raise ValueError("synrbl_metric_mode must be deep_pragmatic_equivalence_match or synrbl_self_reported_acc")


def _top_ylabel(cfg: dict) -> str:
    syn_mode = cfg["synrbl_metric_mode"]
    if syn_mode == "synrbl_self_reported_acc":
        syn_name = "SynRBL self-reported acc"
    else:
        syn_name = "SynRBL deep pragmatic acc"
    return "Accuracy (equiv.)"


def _mid_ylabel(cfg: dict) -> str:
    return f"SynRBL self-reported acc\n(conf >= {float(cfg['synrbl_min_confidence']):.2f})"


def _collect_fold_data(cfg: dict) -> Dict[str, Dict[str, List[pd.DataFrame]]]:
    """Load the selected scores and check that all models use the same test reactions."""
    bench = pd.read_csv(_abs_path(cfg["benchmark_final_csv"]), low_memory=False)
    syn = pd.read_csv(_abs_path(cfg["full_results_csv"]), low_memory=False)
    missing_col = cfg["missing_atoms_col"]
    metadata = bench[["Unique_identifier", missing_col]]
    out = {s: {"rb": [], "crb": [], "synrbl": [], "counts": []} for s in cfg["split_types"]}
    for split in cfg["split_types"]:
        for rep in range(5):
            fold = f"random_seed{42+rep}_rep{rep}" if split == "random" else f"{split}_rep{rep}"
            loaded = {}
            for model, key in [("RB", "rb"), ("CRB", "crb"), ("SynRBL", "synrbl")]:
                path = _abs_path(f"results/main/scores/per_reaction/{model}_{fold}.csv")
                rows = pd.read_csv(path)
                if not rows["retained"].eq(1).all():
                    raise ValueError(f"Unfiltered rows in main results: {path}")
                rows = rows.merge(metadata, on="Unique_identifier", validate="one_to_one")
                rows["metric"] = rows["equivalence"].astype(float)
                rows["is_balanced"] = rows["balanced"].astype(bool)
                loaded[key] = rows
            for key in ["rb", "synrbl"]:
                if loaded[key]["Unique_identifier"].tolist() != loaded["crb"]["Unique_identifier"].tolist():
                    raise ValueError(f"Mismatched model populations: {fold}")
            loaded["synrbl"] = loaded["synrbl"].merge(
                syn[["Unique_identifier", "solved", "solved_by", "confidence"]],
                on="Unique_identifier", validate="one_to_one")
            loaded["synrbl"]["deep_pragmatic_equivalence_match"] = loaded["synrbl"]["equivalence"].astype(bool)
            for key in ["rb", "crb", "synrbl"]:
                out[split][key].append(loaded[key])
            out[split]["counts"].append(loaded["crb"][[missing_col]].copy())
    return out


def run(cfg: dict) -> None:
    allowed_models = ["rb", "crb", "synrbl"]
    selected_models = [m for m in cfg.get("selected_models", allowed_models) if m in allowed_models]
    if not selected_models:
        raise ValueError("selected_models must contain at least one of: rb, crb, synrbl")

    include_mid_panel = bool(cfg.get("include_synrbl_self_panel", True)) and ("synrbl" in selected_models)

    all_folds = _collect_fold_data(cfg)
    split_types = [
        s
        for s in cfg["split_types"]
        if all_folds.get(s) and all(all_folds[s].get(model_key) for model_key in selected_models)
    ]
    if not split_types:
        raise ValueError("No valid split folds found for CRB/SynRBL overlay plot.")

    missing_col = cfg["missing_atoms_col"]
    reference_values = pd.concat(
        [
            fold_df[missing_col]
            for split_name in split_types
            for fold_df in all_folds[split_name]["counts"]
        ],
        ignore_index=True,
    )

    bin_spec = build_bin_spec(
        reference_values=reference_values,
        mode=cfg["binning_mode"],
        max_m=int(cfg["max_missing"]),
        quantile_n_bins=int(cfg["quantile_n_bins"]),
        min_bin_size=int(cfg["min_bin_size"]),
        user_bin_ranges=cfg.get("user_bin_ranges"),
    )
    all_bins = bin_spec["bin_ids"]
    x_labels = bin_spec["labels"]

    curves = {
        s: {**{model_key: [] for model_key in selected_models}, "synrbl_self": ([] if include_mid_panel else None)}
        for s in split_types
    }
    totals_by_split = {}

    for split_name in split_types:
        if "rb" in selected_models:
            for rb_fold in all_folds[split_name]["rb"]:
                binned = rb_fold.copy()
                binned["missing_bin"] = assign_bins(binned[missing_col], bin_spec)
                series = binned.groupby("missing_bin")["metric"].mean().reindex(all_bins)
                curves[split_name]["rb"].append(series)

        if "crb" in selected_models:
            for crb_fold in all_folds[split_name]["crb"]:
                binned = crb_fold.copy()
                binned["missing_bin"] = assign_bins(binned[missing_col], bin_spec)
                series = binned.groupby("missing_bin")["metric"].mean().reindex(all_bins)
                curves[split_name]["crb"].append(series)

        if "synrbl" in selected_models:
            for syn_fold in all_folds[split_name]["synrbl"]:
                binned = syn_fold.copy()
                binned["missing_bin"] = assign_bins(binned[missing_col], bin_spec)
                series = (
                    binned.groupby("missing_bin")
                    .apply(lambda g: _compute_synrbl_metric(g, cfg), include_groups=False)
                    .reindex(all_bins)
                )
                curves[split_name]["synrbl"].append(series)

                if include_mid_panel:
                    self_cfg = dict(cfg)
                    self_cfg["synrbl_metric_mode"] = "synrbl_self_reported_acc"
                    self_series = (
                        binned.groupby("missing_bin")
                        .apply(lambda g: _compute_synrbl_metric(g, self_cfg), include_groups=False)
                        .reindex(all_bins)
                    )
                    curves[split_name]["synrbl_self"].append(self_series)

        count_rows = []
        for count_fold in all_folds[split_name]["counts"]:
            count_binned = count_fold.copy()
            count_binned["missing_bin"] = assign_bins(count_binned[missing_col], bin_spec)
            count_rows.append(count_binned[["missing_bin"]].copy())

        joined = pd.concat(count_rows, ignore_index=True) if count_rows else pd.DataFrame(columns=["missing_bin"])
        n_reps = len(all_folds[split_name]["crb"])
        totals_by_split[split_name] = (joined.groupby("missing_bin").size() / n_reps).reindex(all_bins, fill_value=0)

    apply_neurips_style()
    plt.rcParams["axes.axisbelow"] = True
    if include_mid_panel:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(6.4, 4.55),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 0.95], "hspace": 0.04},
        )
        ax_top, ax_mid, ax_bot = axes
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(6.4, 3.25),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0, 0.9], "hspace": 0.04},
        )
        ax_top, ax_bot = axes
        ax_mid = None

    x = list(range(len(all_bins)))
    n_splits = len(split_types)
    split_jitter_step = float(cfg.get("split_jitter_step", 0.24))
    model_jitter_step = float(cfg.get("model_jitter_step", 0.14))
    marker_size_top = float(cfg.get("marker_size_top", 3.3))
    marker_size_mid = float(cfg.get("marker_size_mid", 3.2))
    split_offsets = {
        split_name: (idx - (n_splits - 1) / 2) * split_jitter_step
        for idx, split_name in enumerate(split_types)
    }
    model_offsets = {
        model_key: (idx - (len(selected_models) - 1) / 2) * model_jitter_step
        for idx, model_key in enumerate(selected_models)
    }

    split_colors = {
        "random": "#4C78A8",
        "group": "#F58518",
        "extreme_ood": "#54A24B",
    }
    model_shade_mix = {"rb": 0.0, "crb": 0.0, "synrbl": 0.38}
    model_markers = {"rb": "s", "crb": "o", "synrbl": "D"}
    model_fillstyle = {"rb": "none", "crb": "full", "synrbl": "none"}

    for split_name in split_types:
        base_color = split_colors.get(split_name, "#4C78A8")
        for model_key in selected_models:
            fold_series = curves[split_name][model_key]
            if not fold_series:
                continue

            aligned = pd.DataFrame({i: s.reindex(all_bins) for i, s in enumerate(fold_series)}, index=all_bins)
            mean = aligned.mean(axis=1)
            q25 = aligned.quantile(0.25, axis=1)
            q75 = aligned.quantile(0.75, axis=1)
            lower = (mean - q25).clip(lower=0)
            upper = (q75 - mean).clip(lower=0)

            x_offset = split_offsets.get(split_name, 0.0) + model_offsets[model_key]
            x_pos = [xi + x_offset for xi in x]
            color = _mix_with_white(base_color, model_shade_mix.get(model_key, 0.0))

            ax_top.errorbar(
                x_pos,
                mean.values,
                yerr=[lower.values, upper.values],
                linewidth=0.0,
                linestyle="none",
                marker=model_markers[model_key],
                fillstyle=model_fillstyle[model_key],
                markersize=marker_size_top,
                capsize=2.8,
                elinewidth=0.8,
                markeredgewidth=0.9,
                color=color,
#                alpha=float(cfg.get("errorbar_alpha", 0.45)),
            )

    if include_mid_panel and ax_mid is not None:
        for split_name in split_types:
            base_color = split_colors.get(split_name, "#4C78A8")
            color = _mix_with_white(base_color, model_shade_mix.get("synrbl", 0.0))
            fold_series = curves[split_name]["synrbl_self"]
            if not fold_series:
                continue

            aligned = pd.DataFrame({i: s.reindex(all_bins) for i, s in enumerate(fold_series)}, index=all_bins)
            mean = aligned.mean(axis=1)
            q25 = aligned.quantile(0.25, axis=1)
            q75 = aligned.quantile(0.75, axis=1)
            lower = (mean - q25).clip(lower=0)
            upper = (q75 - mean).clip(lower=0)

            x_offset = split_offsets.get(split_name, 0.0)
            x_pos = [xi + x_offset for xi in x]

            ax_mid.errorbar(
                x_pos,
                mean.values,
                yerr=[lower.values, upper.values],
                linewidth=0.0,
                linestyle="none",
                marker="D",
                fillstyle="none",
                markersize=marker_size_mid,
                capsize=2.5,
                elinewidth=0.75,
                markeredgewidth=0.9,
                color=color,
            )

    bar_group_width = 0.82
    bar_width = min(0.34, bar_group_width / max(n_splits, 1))
    offsets = [(idx - (n_splits - 1) / 2) * bar_width for idx in range(n_splits)]
    split_bar_colors = split_colors
    # mix with white for aesthatics
    split_bar_colors = {s: _mix_with_white(c, 0.3) for s, c in split_bar_colors.items()}
    #split_hatches = {
    #    "random": "",
    #    "group": "//",
    #    "extreme_ood": "xx",
    #}

    max_total = max([int(totals_by_split[s].max()) for s in split_types] or [1])
    for idx, split_name in enumerate(split_types):
        positions = [xi + offsets[idx] for xi in x]
        totals = totals_by_split[split_name].reindex(all_bins).values
        ax_bot.bar(
            positions,
            totals,
            color=split_bar_colors.get(split_name, "#7A7A7A"),
            width=bar_width,
            edgecolor="white",
            linewidth=0.3,
            #hatch=split_hatches.get(split_name, ""),
            label=cfg["split_display_names"].get(split_name, split_name),
        )
        for pos, n in zip(positions, totals):
            if n > 0:
                ax_bot.text(
                    pos,
                    n * 1.08,
                    f"n={int(n)}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=5.5,
                    color="#444444",
                )

    ax_bot.set_yscale("log")
    ax_bot.set_ylim(10, max(10, max_total * 20))

    split_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=split_colors.get(s, "#4C78A8"),
            markerfacecolor=split_colors.get(s, "#4C78A8"),
            markersize=5,
            label=cfg["split_display_names"].get(s, s),
        )
        for s in split_types
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=model_markers[m],
            linestyle="none",
            color="black",
            markerfacecolor=("black" if model_fillstyle[m] != "none" else "none"),
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=5,
            label=cfg["model_display_names"].get(m, m),
        )
        for m in selected_models
    ]

    ax_top.set_ylabel(_top_ylabel(cfg))
    ax_top.set_ylim(0.0, 1.03)
    ax_top.yaxis.grid(True, linestyle="-", alpha=0.5)
    ax_top.legend(
        handles=split_handles + model_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.82,
        facecolor="white",
        fontsize=7,
        handlelength=1.0,
        ncol=2,
    )

    if include_mid_panel and ax_mid is not None:
        ax_mid.set_ylabel(_mid_ylabel(cfg))
        ax_mid.yaxis.grid(True, linestyle="-", alpha=0.5)
        ax_mid.set_ylim(0.0, 1.03)

    ax_bot.set_ylabel("Avg count per rep")
    ax_bot.yaxis.grid(True, linestyle="-", alpha=0.5)
    if n_splits > 1:
        ax_bot.legend(loc="upper right", frameon=True, framealpha=0.82, facecolor="white", fontsize=7, handlelength=1.2)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(x_labels)
    ax_bot.set_xlabel(f"Number of missing {missing_col.replace('MissingNr', '').lower()}")
    ax_top.set_title("")

    out = _abs_path(cfg["output"])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


def main() -> None:
    run(dict(CONFIG))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare CRB and SynRBL outcomes on the same test reactions.

Fractions per method and missing-carbon bin:
- Accurate
- Balanced, not accurate
- Imbalanced (independently checked for every model)
"""

import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis.figures.plot_crb_synrbl_top2_by_missing_atoms import (  # noqa: E402
    CONFIG as BASE_CONFIG,
    _abs_path,
    _collect_fold_data,
    _safe_bool,
)
from analysis.figures.plot_accuracy_merged_by_missing_atoms import assign_bins, build_bin_spec
from utils.plot_style import apply_neurips_style  # noqa: E402


CONFIG = {
    "split_types": ["random", "extreme_ood"],
    "sanity_missing_value": 5,
    "output": "results/main/figures/synrbl_vs_crb_outcome_fractions.png",
}


def _fractions_for_crb(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"accurate": float("nan"), "balanced_not_accurate": float("nan"), "imbalanced": float("nan")}

    accurate = _safe_bool(df["metric"])
    balanced = _safe_bool(df.get("is_balanced", pd.Series(False, index=df.index)))
    balanced_not_accurate = balanced & (~accurate)
    imbalanced = ~balanced

    n = float(len(df))
    return {
        "accurate": float(accurate.sum() / n),
        "balanced_not_accurate": float(balanced_not_accurate.sum() / n),
        "imbalanced": float(imbalanced.sum() / n),
    }


def _fractions_for_synrbl(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"accurate": float("nan"), "balanced_not_accurate": float("nan"), "imbalanced": float("nan")}

    solved = _safe_bool(df["is_balanced"])
    accurate = solved & _safe_bool(df.get("deep_pragmatic_equivalence_match", pd.Series(False, index=df.index)))
    balanced_not_accurate = solved & (~accurate)
    imbalanced = ~solved

    n = float(len(df))
    return {
        "accurate": float(accurate.sum() / n),
        "balanced_not_accurate": float(balanced_not_accurate.sum() / n),
        "imbalanced": float(imbalanced.sum() / n),
    }


def _collect_aligned_rows(cfg: dict):
    """Match CRB and SynRBL rows by split, and count the outcomes across repetitions."""
    all_folds = _collect_fold_data(cfg)
    missing_col = cfg["missing_atoms_col"]

    per_split = {}
    all_count_rows = []
    n_reps_by_split = {}
    for split_name in cfg["split_types"]:
        if split_name not in all_folds:
            continue
        crb_rows, synrbl_rows = [], []
        n_pairs = min(len(all_folds[split_name]["crb"]), len(all_folds[split_name]["synrbl"]))
        for i in range(n_pairs):
            crb_rows.append(all_folds[split_name]["crb"][i])
            synrbl_rows.append(all_folds[split_name]["synrbl"][i])
        for fold_df in all_folds[split_name]["counts"]:
            all_count_rows.append(fold_df)

        per_split[split_name] = {
            "crb": pd.concat(crb_rows, ignore_index=True) if crb_rows else pd.DataFrame(
                columns=["metric", "is_balanced", missing_col]
            ),
            "synrbl": pd.concat(synrbl_rows, ignore_index=True) if synrbl_rows else pd.DataFrame(
                columns=["solved", "deep_pragmatic_equivalence_match", missing_col]
            ),
        }
        n_reps_by_split[split_name] = n_pairs

    count_df = pd.concat(all_count_rows, ignore_index=True) if all_count_rows else pd.DataFrame(columns=[missing_col])
    return {"per_split": per_split, "counts": count_df, "n_reps": n_reps_by_split}


def _fractions_by_bin(df: pd.DataFrame, method: str, bin_col: str, all_bins: list) -> pd.DataFrame:
    rows = []
    for b in all_bins:
        sub = df[df[bin_col] == b]
        if method == "crb":
            fr = _fractions_for_crb(sub)
        else:
            fr = _fractions_for_synrbl(sub)
        fr["n"] = float(len(sub))
        fr[bin_col] = b
        rows.append(fr)

    out = pd.DataFrame(rows).set_index(bin_col)
    return out.reindex(all_bins)


def _absolute_counts(df: pd.DataFrame, method: str) -> Dict[str, int]:
    if df.empty:
        return {"accurate": 0, "balanced_not_accurate": 0, "imbalanced": 0, "n": 0}

    if method == "crb":
        accurate = _safe_bool(df["metric"])
        balanced = _safe_bool(df.get("is_balanced", pd.Series(False, index=df.index)))
        balanced_not_accurate = balanced & (~accurate)
        imbalanced = ~balanced
    else:
        solved = _safe_bool(df["is_balanced"])
        accurate = solved & _safe_bool(df.get("deep_pragmatic_equivalence_match", pd.Series(False, index=df.index)))
        balanced_not_accurate = solved & (~accurate)
        imbalanced = ~solved

    return {
        "accurate": int(accurate.sum()),
        "balanced_not_accurate": int(balanced_not_accurate.sum()),
        "imbalanced": int(imbalanced.sum()),
        "n": int(len(df)),
    }


def _draw_fraction_panel(
    ax,
    tables: dict,
    methods: list,
    all_bins: list,
    categories: list,
    colors: dict,
    method_hatch: dict,
    offsets: dict,
    bar_width: float,
    panel_abs_counts: dict,
    target_missing_value: int,
    target_bin_label: str,
) -> None:
    x = np.arange(len(all_bins), dtype=float)
    for method in methods:
        bottom = np.zeros(len(all_bins), dtype=float)
        xpos = x + offsets[method]
        for cat in categories:
            vals = tables[method][cat].to_numpy(dtype=float)
            ax.bar(
                xpos,
                vals,
                bottom=bottom,
                width=bar_width,
                color=colors[cat],
                edgecolor="white",
                linewidth=0.45,
                hatch=method_hatch[method],
            )
            bottom += np.nan_to_num(vals, nan=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction")
    ax.yaxis.grid(True, linestyle="-", alpha=0.45)


def run(cfg: dict) -> None:
    plot_cfg = dict(BASE_CONFIG)
    # Restrict to the requested split types only.
    split_types = cfg.get("split_types", ["random", "extreme_ood"])
    plot_cfg["split_types"] = split_types

    data = _collect_aligned_rows(plot_cfg)
    per_split = data["per_split"]
    n_reps_by_split = data.get("n_reps", {})
    missing_col = plot_cfg["missing_atoms_col"]

    # Build bin spec from pooled count rows (consistent min_count binning).
    bin_spec = build_bin_spec(
        reference_values=data["counts"][missing_col],
        mode=plot_cfg["binning_mode"],
        max_m=int(plot_cfg["max_missing"]),
        quantile_n_bins=int(plot_cfg["quantile_n_bins"]),
        min_bin_size=int(plot_cfg["min_bin_size"]),
        user_bin_ranges=plot_cfg.get("user_bin_ranges"),
    )
    all_bins = bin_spec["bin_ids"]
    x_labels = bin_spec["labels"]
    target_missing_value = int(cfg.get("sanity_missing_value", 5))
    target_bin = assign_bins(pd.Series([target_missing_value]), bin_spec).iloc[0]
    target_bin_label = "NA"
    if pd.notna(target_bin):
        try:
            target_bin_idx = all_bins.index(target_bin)
            target_bin_label = str(x_labels[target_bin_idx])
        except ValueError:
            target_bin_label = str(target_bin)

    # Compute per-split fraction tables.
    split_tables = {}
    split_counts = {}
    split_abs_counts = {}
    for split_name, split_data in per_split.items():
        crb_df = split_data["crb"].copy()
        synrbl_df = split_data["synrbl"].copy()
        crb_df["missing_bin"] = assign_bins(crb_df[missing_col], bin_spec)
        synrbl_df["missing_bin"] = assign_bins(synrbl_df[missing_col], bin_spec)
        split_tables[split_name] = {
            "crb": _fractions_by_bin(crb_df, "crb", "missing_bin", all_bins),
            "synrbl": _fractions_by_bin(synrbl_df, "synrbl", "missing_bin", all_bins),
        }
        n_reps = n_reps_by_split.get(split_name, 1)
        split_counts[split_name] = (crb_df.groupby("missing_bin").size() / n_reps).reindex(all_bins, fill_value=0)
        crb_target = crb_df[crb_df["missing_bin"] == target_bin].copy()
        synrbl_target = synrbl_df[synrbl_df["missing_bin"] == target_bin].copy()
        split_abs_counts[split_name] = {
            "crb": _absolute_counts(crb_target, "crb"),
            "synrbl": _absolute_counts(synrbl_target, "synrbl"),
        }

    active_splits = [s for s in split_types if s in split_tables]
    n_fraction_panels = len(active_splits)

    categories = ["accurate", "balanced_not_accurate", "imbalanced"]
    category_labels = {
        "accurate": "Accurate",
        "balanced_not_accurate": "Balanced, not accurate",
        "imbalanced": "Imbalanced",
    }
    colors = {
        "accurate": "#009B77",
        "balanced_not_accurate": "#EC6842",
        "imbalanced": "#A50034",
    }
    methods = ["crb", "synrbl"]
    bar_width = 0.36
    offsets = {"crb": -bar_width / 2.0, "synrbl": bar_width / 2.0}
    method_hatch = {"crb": "", "synrbl": "//"}

    split_bar_colors = {"random": "#6E6E6E", "extreme_ood": "#B0B0B0"}
    split_bar_hatches = {"random": "", "extreme_ood": "xx"}
    split_display_names = plot_cfg.get("split_display_names", {})

    apply_neurips_style()
    plt.rcParams["axes.axisbelow"] = True

    height_ratios = [1.0] * n_fraction_panels + [0.65]
    fig, axes = plt.subplots(
        n_fraction_panels + 1,
        1,
        figsize=(7.0, 2.6 * n_fraction_panels + 1.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.04},
    )
    fraction_axes = axes[:n_fraction_panels]
    ax_bot = axes[n_fraction_panels]
    x = np.arange(len(all_bins), dtype=float)

    # ── fraction panels ──────────────────────────────────────────────────────
    category_handles = [
        Patch(facecolor=colors[cat], edgecolor="white", linewidth=0.45, label=category_labels[cat])
        for cat in categories
    ]
    method_handles = [
        Patch(facecolor="#DDDDDD", edgecolor="#777777", hatch=method_hatch[m], label=plot_cfg["model_display_names"][m])
        for m in methods
    ]

    for ax_frac, split_name in zip(fraction_axes, active_splits):
        _draw_fraction_panel(
            ax_frac, split_tables[split_name], methods, all_bins,
            categories, colors, method_hatch, offsets, bar_width,
            split_abs_counts[split_name], target_missing_value, target_bin_label,
        )
        display_name = split_display_names.get(split_name, split_name)
        ax_frac.set_title(display_name, fontsize=8, loc="left", pad=3)

    print(
        f"\nAbsolute outcome counts for missing={target_missing_value} "
        f"(bin label: {target_bin_label}) across all folds:"
    )
    for split_name in active_splits:
        for method in methods:
            c = split_abs_counts[split_name][method]
            print(
                f"  split={split_name:12s} method={method:6s} "
                f"acc={c['accurate']:7d} bal_only={c['balanced_not_accurate']:7d} "
                f"imbal={c['imbalanced']:7d} n={c['n']:7d}"
            )

    # Legends on first fraction panel only.
    leg1 = fraction_axes[0].legend(
        handles=category_handles,
        loc="lower left",
        frameon=True, framealpha=0.85, facecolor="white",
        fontsize=7, title="Outcome", title_fontsize=7,
    )
    fraction_axes[0].add_artist(leg1)
    fraction_axes[0].legend(
        handles=method_handles,
        loc="upper right",
        frameon=True, framealpha=0.85, facecolor="white",
        fontsize=7, title="Method", title_fontsize=7,
    )

    # ── counts panel ─────────────────────────────────────────────────────────
    n_splits = len(active_splits)
    count_bar_width = min(0.34, 0.82 / max(n_splits, 1))
    count_offsets = [(idx - (n_splits - 1) / 2.0) * count_bar_width for idx in range(n_splits)]
    max_total = 1
    for split_name in active_splits:
        m = int(split_counts[split_name].max())
        if m > max_total:
            max_total = m

    for idx, split_name in enumerate(active_splits):
        totals = split_counts[split_name].reindex(all_bins).values
        xpos = x + count_offsets[idx]
        ax_bot.bar(
            xpos, totals,
            width=count_bar_width,
            color=split_bar_colors.get(split_name, "#888888"),
            edgecolor="white", linewidth=0.3,
            hatch=split_bar_hatches.get(split_name, ""),
            label=split_display_names.get(split_name, split_name),
        )
        for pos, n in zip(xpos, totals):
            if n > 0:
                ax_bot.text(pos, n * 1.08, f"n={int(n)}", ha="center", va="bottom",
                            rotation=90, fontsize=5.5, color="#444444")

    ax_bot.set_yscale("log")
    ax_bot.set_ylim(10, max(10, max_total * 20))
    ax_bot.set_ylabel("Avg count per rep")
    ax_bot.yaxis.grid(True, linestyle="-", alpha=0.45)
    if n_splits > 1:
        ax_bot.legend(loc="lower left", frameon=True, framealpha=0.85,
                      facecolor="white", fontsize=7, handlelength=1.2)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(x_labels)
    ax_bot.set_xlabel(f"Number of missing {missing_col.replace('MissingNr', '').lower()}")

    out = _abs_path(cfg["output"])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


def main() -> None:
    run(dict(CONFIG))


if __name__ == "__main__":
    main()

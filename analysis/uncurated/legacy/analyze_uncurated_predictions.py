#!/usr/bin/env python3
"""Analyze SynRBL vs CRB predictions on USPTO uncurated chunks.

Outputs:
1) Percent balanced reactions for SynRBL and CRB.
2) SynRBL confidence thresholds (>0.50 and >0.90).
3) CRB top-1 sequence-probability threshold (>0.95).
4) Unique_identifier overlap between model outputs. (SynRBL does not output predictions for all rows, so this is a subset of the total dataset.)
5) Prediction agreement on overlapping IDs using utils.metrics.compare_reactions.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
import re
import sys
import warnings
from functools import lru_cache
from typing import Dict, List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from tqdm.auto import tqdm


_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
	sys.path.insert(0, _REPO_ROOT)



warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
tqdm.pandas()


# One-off full-file run configuration.
SYNRBL_DIR = "results/SynRBL/uspto_uncurated_CanonicalReaction_results"
CRB_MERGED_PRED_PATH = "results/main/predictions/crb/uncurated/merged/beamSize5_nBest5_all.txt"
CRB_MERGED_LOGP_PATH = "results/main/predictions/crb/uncurated/merged/beamSize5_nBest5_all.txt_log_probs"
CRB_MERGED_SRC_PATH = "results/main/predictions/crb/uncurated/merged/test_src_all.txt"
CRB_MERGED_MANIFEST_PATH = "results/main/predictions/crb/uncurated/merged/merge_manifest.csv"
CRB_INDEX_DIR = "models/constrained_reaction_balancer/USPTO_uncurated/chunks"
CRB_RETRY_INDEX_DIR = "results/main/predictions/crb/uncurated/tail_retry/chunks_tail_retry"
RETRY_START_CHUNK = 44
N_BEST = 5
SYN_CONF_THR = 0.90
CRB_CONF_THR = 0.95
OUTPUT_SUMMARY_CSV = "results/archive/metric_v1/additional/uncurated_consensus_summary.csv"
OUTPUT_UIDS_DIR = "results/archive/metric_v1/additional/uncurated_consensus_uids"
CRB_CACHE_CSV = "results/main/predictions/crb/uncurated/loaded_balanced.csv"
CRB_TAIL_CACHE_CSV = "results/main/predictions/crb/uncurated/tail_rows.csv"
# Set to a positive int to limit analysis to the first N rows of the CRB data
# (by Unique_identifier order in the merged file). Use None for the full run.
DEV_SUBSET: int | None = None


def _safe_bool(series: pd.Series) -> pd.Series:
	if pd.api.types.is_bool_dtype(series):
		return series.fillna(False)
	if pd.api.types.is_numeric_dtype(series):
		return pd.to_numeric(series, errors="coerce").fillna(0) != 0
	lowered = series.astype(str).str.strip().str.lower()
	return lowered.isin({"1", "true", "t", "yes", "y"})


def _mol_with_hs_from_smiles(smiles: str):
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return None
	return Chem.AddHs(mol)


@lru_cache(maxsize=None)
def _reaction_valid_and_balanced(reaction: str) -> Tuple[bool, bool]:
	"""Return (is_valid_prediction, is_balanced) for a completed reaction."""
	if not isinstance(reaction, str) or ">" not in reaction:
		return False, False
	try:
		reactants, _, products = reaction.split(">")
	except ValueError:
		return False, False

	reactants_list = [s for s in reactants.split(".") if s]
	products_list = [s for s in products.split(".") if s]
	if not reactants_list or not products_list:
		return False, False

	reactant_mols = [_mol_with_hs_from_smiles(smi) for smi in reactants_list]
	product_mols = [_mol_with_hs_from_smiles(smi) for smi in products_list]
	if None in reactant_mols or None in product_mols:
		return False, False

	atom_types = set()
	for mol in reactant_mols + product_mols:
		for atom in mol.GetAtoms():
			atom_types.add(atom.GetSymbol())

	left = {k: 0 for k in atom_types}
	right = {k: 0 for k in atom_types}
	left["charge"] = 0
	right["charge"] = 0

	for mol in reactant_mols:
		for atom in mol.GetAtoms():
			symbol = atom.GetSymbol()
			left[symbol] = left.get(symbol, 0) + 1
			left["charge"] += atom.GetFormalCharge()

	for mol in product_mols:
		for atom in mol.GetAtoms():
			symbol = atom.GetSymbol()
			right[symbol] = right.get(symbol, 0) + 1
			right["charge"] += atom.GetFormalCharge()

	return True, left == right


def _is_reaction_balanced(reaction: str) -> bool:
	return _reaction_valid_and_balanced(reaction)[1]


def dpe_match(prediction: str, other: str) -> bool:
	from src.completerxn.legacy.equivalence_v1 import compare_reactions
	try:
		return bool(compare_reactions(prediction, other).deep_pragmatic_equivalence_match)
	except Exception:
		return False


def _merge_side(base_side: str, add_side: str) -> str:
	base = [s for s in (base_side or "").split(".") if s]
	add = [s for s in (add_side or "").split(".") if s]
	return ".".join(base + add)


def compose_completed_reaction(input_reaction: str, predicted_diff: str) -> str:
	"""Compose full completed reaction by adding predicted diff to input side-wise."""
	if not isinstance(input_reaction, str) or not isinstance(predicted_diff, str):
		return ""
	try:
		in_r, in_rg, in_p = input_reaction.split(">")
		d_r, d_rg, d_p = predicted_diff.split(">")
	except ValueError:
		return ""

	out_r = _merge_side(in_r, d_r)
	out_rg = _merge_side(in_rg, d_rg)
	out_p = _merge_side(in_p, d_p)
	return f"{out_r}>{out_rg}>{out_p}"


def _chunk_id_from_synrbl_name(name: str) -> str:
	m = re.search(r"chunk_(\d+)_results\.csv$", name)
	if m is None:
		return ""
	return m.group(1)


def _chunk_id_from_crb_dir(name: str) -> str:
	m = re.match(r"chunk_(\d+)$", name)
	if m is None:
		return ""
	return m.group(1)


def _detokenize_prediction(tokenized: str) -> str:
	# CRB output is tokenized with spaces; remove all whitespace to recover SMILES text.
	return "".join((tokenized or "").strip().split())


def _prob_from_log_score(log_score: float) -> float:
	# Match analysis/figures/plot_confidence_by_category.py: probability = 10 ** log_prob.
	if pd.isna(log_score):
		return float("nan")
	return 10.0 ** float(log_score)


def _unknown_tokens_in_line(tokenized_line: str, vocab_tokens: set) -> List[str]:
	toks = tokenized_line.strip().split()
	return [t for t in toks if t not in vocab_tokens]


def load_synrbl_results(synrbl_results_dir: str, chunk_filter: str | None = None) -> pd.DataFrame:
	rows: List[pd.DataFrame] = []

	for fname in sorted(os.listdir(synrbl_results_dir)):
		if not fname.endswith("_results.csv"):
			continue
		chunk_id = _chunk_id_from_synrbl_name(fname)
		if not chunk_id:
			continue
		if chunk_filter is not None and chunk_id != chunk_filter:
			continue
		fpath = os.path.join(synrbl_results_dir, fname)
		df = pd.read_csv(fpath, low_memory=False)

		required = {"Unique_identifier", "Reaction"}
		missing = required.difference(df.columns)
		if missing:
			raise KeyError(f"SynRBL file {fpath} missing required columns: {sorted(missing)}")

		out = pd.DataFrame(
			{
				"Unique_identifier": df["Unique_identifier"].astype(str),
				"prediction_reaction": df["Reaction"].astype(str),
				"synrbl_solved": _safe_bool(df.get("solved", pd.Series(False, index=df.index))).astype(bool),
				"synrbl_confidence": pd.to_numeric(df.get("confidence", float("nan")), errors="coerce"),
				"chunk_id": chunk_id,
			}
		)
		rows.append(out)

	if not rows:
		raise FileNotFoundError(f"No SynRBL result chunk CSVs found in: {synrbl_results_dir}")

	syn = pd.concat(rows, ignore_index=True)
	syn["prediction_reaction"] = syn["prediction_reaction"].replace({"nan": "", "None": "", "none": ""})

	dup_count = int(syn["Unique_identifier"].duplicated().sum())
	if dup_count:
		print(f"[warn] SynRBL has {dup_count} duplicate Unique_identifier rows; keeping first occurrence.")
		syn = syn.drop_duplicates(subset=["Unique_identifier"], keep="first").copy()

	# SynRBL convention: solved=True means reaction was balanced/completed.
	syn["is_balanced"] = syn["synrbl_solved"].astype(bool)
	return syn


def load_crb_results(
	crb_results_dir: str,
	crb_index_dir: str,
	vocab_tokens: set,
	n_best: int,
	chunk_filter: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Return (crb_df, tail_df).

	crb_df  - all decoded rows (including unk-token rows), with columns
	           has_unk_tokens (bool) and unknown_tokens (str).
	tail_df - rows never decoded due to incomplete chunk execution, with
	           column src_line for writing a retry src file.
	"""
	rows: List[pd.DataFrame] = []
	tail_rows: List[dict] = []

	index_ids = {}
	for name in sorted(os.listdir(crb_index_dir)):
		m = re.match(r"test_index_chunk_(\d+)$", name)
		if m is not None:
			index_ids[m.group(1)] = os.path.join(crb_index_dir, name)

	for chunk_dir_name in sorted(os.listdir(crb_results_dir)):
		chunk_id = _chunk_id_from_crb_dir(chunk_dir_name)
		if not chunk_id or chunk_id not in index_ids:
			continue
		if chunk_filter is not None and chunk_id != chunk_filter:
			continue

		pred_path = os.path.join(crb_results_dir, chunk_dir_name, "beamSize5_nBest5.txt")
		logp_path = os.path.join(crb_results_dir, chunk_dir_name, "beamSize5_nBest5.txt_log_probs")
		if not (os.path.isfile(pred_path) and os.path.isfile(logp_path)):
			continue

		idx_df = pd.read_csv(index_ids[chunk_id], low_memory=False)
		required = {"Unique_identifier", "input_reaction"}
		missing = required.difference(idx_df.columns)
		if missing:
			raise KeyError(f"CRB index file {index_ids[chunk_id]} missing columns: {sorted(missing)}")

		src_path = os.path.join(crb_index_dir, f"test_src_chunk_{chunk_id}")
		if not os.path.isfile(src_path):
			raise FileNotFoundError(f"Missing tokenized input file for chunk {chunk_id}: {src_path}")
		with open(src_path, "r", encoding="utf-8") as f:
			src_lines = [line.rstrip("\n") for line in f]
		if len(src_lines) != len(idx_df):
			raise ValueError(
				f"Chunk {chunk_id}: tokenized source length ({len(src_lines)}) does not match "
				f"index rows ({len(idx_df)})."
			)

		with open(pred_path, "r", encoding="utf-8") as f:
			pred_lines = [line.rstrip("\n") for line in f]
		with open(logp_path, "r", encoding="utf-8") as f:
			logp_lines = [line.rstrip("\n") for line in f]
		if len(pred_lines) != len(logp_lines):
			common = min(len(pred_lines), len(logp_lines))
			print(
				f"[warn] Chunk {chunk_id}: pred/log line mismatch "
				f"(pred={len(pred_lines)}, log={len(logp_lines)}). Truncating to {common}."
			)
			pred_lines = pred_lines[:common]
			logp_lines = logp_lines[:common]

		local_n_best = int(n_best)
		if local_n_best <= 0:
			raise ValueError(f"Invalid n_best={n_best}; must be >= 1")

		n_orig = len(src_lines)  # equals original len(idx_df)
		available_samples = len(pred_lines) // local_n_best
		remainder = len(pred_lines) % local_n_best
		if remainder:
			print(
				f"[warn] Chunk {chunk_id}: {remainder} trailing beam/log lines do not form a full "
				f"sample block (n_best={local_n_best}); dropping them."
			)

		if available_samples == 0:
			print(f"[warn] Chunk {chunk_id}: no complete decoded samples; skipping chunk.")
			continue

		# Compute unk-token status for every original row.
		has_unk: List[bool] = []
		unk_token_strs: List[str] = []
		for line in src_lines:
			unk = _unknown_tokens_in_line(line, vocab_tokens)
			has_unk.append(bool(unk))
			unk_token_strs.append("|".join(sorted(set(unk))) if unk else "")

		# Classify rows:
		#   j < available_samples  → prediction exists; keep all (unk rows get has_unk_tokens=True).
		#   j >= available_samples → never decoded; save for retry with original src line.
		valid_orig_indices: List[int] = []
		for j in range(n_orig):
			if j >= available_samples:
				tail_rows.append({
					"Unique_identifier": str(idx_df.iloc[j]["Unique_identifier"]),
					"input_reaction": str(idx_df.iloc[j]["input_reaction"]),
					"src_line": src_lines[j],
					"chunk_id": chunk_id,
				})
			else:
				valid_orig_indices.append(j)

		n_unk_valid = sum(has_unk[j] for j in valid_orig_indices)
		n_tail = n_orig - len(valid_orig_indices) - sum(
			1 for j in range(n_orig) if j < available_samples and has_unk[j]
		)
		if n_unk_valid:
			print(
				f"[info] Chunk {chunk_id}: {n_unk_valid} decoded rows have unk tokens "
				f"(unconstrained beam search); included with has_unk_tokens=True."
			)
		if n_tail > 0:
			print(
				f"[warn] Chunk {chunk_id}: decoded {available_samples}/{n_orig} inputs; "
				f"{n_tail} tail rows saved for retry."
			)

		if not valid_orig_indices:
			print(f"[warn] Chunk {chunk_id}: no valid rows remaining; skipping chunk.")
			continue

		# Extract predictions using the ORIGINAL position j in the pred/logp files.
		top1_pred = [_detokenize_prediction(pred_lines[j * local_n_best]) for j in valid_orig_indices]
		top1_logp = pd.to_numeric(
			pd.Series([logp_lines[j * local_n_best] for j in valid_orig_indices]), errors="coerce"
		)
		top1_prob = top1_logp.map(_prob_from_log_score)

		# Filter idx_df to valid rows (preserving original row order).
		idx_df = idx_df.iloc[valid_orig_indices].reset_index(drop=True)

		completed = [
			compose_completed_reaction(inp, pred)
			for inp, pred in zip(idx_df["input_reaction"].astype(str), top1_pred)
		]

		out = pd.DataFrame(
			{
				"Unique_identifier": idx_df["Unique_identifier"].astype(str),
				"prediction_reaction": completed,
				"crb_top1_diff": top1_pred,
				"crb_top1_log_score": top1_logp.values,
				"crb_sequence_probability": top1_prob.values,
				"has_unk_tokens": [has_unk[j] for j in valid_orig_indices],
				"unknown_tokens": [unk_token_strs[j] for j in valid_orig_indices],
				"chunk_id": chunk_id,
			}
		)
		rows.append(out)

	if not rows:
		raise FileNotFoundError(
			"No usable CRB chunk pairs found. Checked CRB result chunks and test_index_chunk_* files."
		)

	crb = pd.concat(rows, ignore_index=True)
	dup_count = int(crb["Unique_identifier"].duplicated().sum())
	if dup_count:
		print(f"[warn] CRB has {dup_count} duplicate Unique_identifier rows; keeping first occurrence.")
		crb = crb.drop_duplicates(subset=["Unique_identifier"], keep="first").copy()

	if tail_rows:
		tail_df = pd.DataFrame(tail_rows)
	else:
		tail_df = pd.DataFrame(columns=["Unique_identifier", "input_reaction", "src_line", "chunk_id"])

	return crb, tail_df


def load_crb_results_from_merged_files(
	merged_pred_path: str,
	merged_logp_path: str,
	merged_src_path: str,
	merged_manifest_path: str,
	crb_index_dir: str,
	retry_index_dir: str,
	n_best: int,
	retry_start_chunk: int,
	cache_csv_path: str | None = None,
	tail_cache_csv_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Return (crb_df, tail_df) using merged global CRB files.

	This path enforces row alignment by consuming rows in the exact merged order:
	- one src line per sample
	- top-1 prediction at every n_best-th line of merged predictions
	- Unique_identifier and chunk provenance from merge_manifest + index files
	"""
	if cache_csv_path and os.path.isfile(cache_csv_path):
		print(f"Loading CRB cache: {cache_csv_path}")
		crb = pd.read_csv(cache_csv_path, low_memory=False)
		for col in ("is_valid_prediction", "is_balanced", "has_unk_tokens"):
			if col in crb.columns:
				crb[col] = _safe_bool(crb[col])

		if tail_cache_csv_path and os.path.isfile(tail_cache_csv_path):
			tail_df = pd.read_csv(tail_cache_csv_path, low_memory=False)
		else:
			tail_df = pd.DataFrame(columns=["Unique_identifier", "input_reaction", "src_line", "chunk_id"])
		return crb, tail_df

	if not os.path.isfile(merged_pred_path):
		raise FileNotFoundError(f"Merged prediction file not found: {merged_pred_path}")
	if not os.path.isfile(merged_logp_path):
		raise FileNotFoundError(f"Merged log-prob file not found: {merged_logp_path}")
	if not os.path.isfile(merged_src_path):
		raise FileNotFoundError(f"Merged src file not found: {merged_src_path}")
	if not os.path.isfile(merged_manifest_path):
		raise FileNotFoundError(f"Merged manifest file not found: {merged_manifest_path}")

	manifest = pd.read_csv(merged_manifest_path, low_memory=False)
	required_manifest_cols = {"chunk_id", "source_kind", "src_lines_used", "src_tail_dropped"}
	missing_manifest = required_manifest_cols.difference(manifest.columns)
	if missing_manifest:
		raise KeyError(
			f"Merged manifest missing required columns: {sorted(missing_manifest)}"
		)

	with open(merged_src_path, "r", encoding="utf-8") as f:
		src_lines = [line.rstrip("\n") for line in f]
	with open(merged_pred_path, "r", encoding="utf-8") as f:
		pred_lines = [line.rstrip("\n") for line in f]
	with open(merged_logp_path, "r", encoding="utf-8") as f:
		logp_lines = [line.rstrip("\n") for line in f]

	if len(pred_lines) != len(logp_lines):
		common = min(len(pred_lines), len(logp_lines))
		print(
			f"[warn] merged pred/log line mismatch (pred={len(pred_lines)}, log={len(logp_lines)}). "
			f"Truncating to {common}."
		)
		pred_lines = pred_lines[:common]
		logp_lines = logp_lines[:common]

	local_n_best = int(n_best)
	if local_n_best <= 0:
		raise ValueError(f"Invalid n_best={n_best}; must be >= 1")
	if len(pred_lines) % local_n_best != 0:
		remainder = len(pred_lines) % local_n_best
		usable = len(pred_lines) - remainder
		print(
			f"[warn] merged prediction lines are not divisible by n_best={local_n_best}; "
			f"dropping {remainder} trailing lines."
		)
		pred_lines = pred_lines[:usable]
		logp_lines = logp_lines[:usable]

	n_samples = len(pred_lines) // local_n_best
	if n_samples > len(src_lines):
		raise ValueError(
			f"Merged files inconsistent: decoded samples={n_samples} exceeds src lines={len(src_lines)}"
		)
	if n_samples < len(src_lines):
		print(
			f"[warn] merged src has {len(src_lines) - n_samples} extra lines; truncating src to {n_samples}."
		)
		src_lines = src_lines[:n_samples]

	uid_rows: List[pd.DataFrame] = []
	tail_rows: List[dict] = []
	for _, row in manifest.iterrows():
		chunk_id = int(row["chunk_id"])
		source_kind = str(row["source_kind"]).strip().lower()
		src_lines_used = int(pd.to_numeric(row["src_lines_used"], errors="coerce"))
		src_tail_dropped = int(pd.to_numeric(row["src_tail_dropped"], errors="coerce"))

		if source_kind == "retry":
			retry_idx = chunk_id - int(retry_start_chunk)
			index_path = os.path.join(retry_index_dir, f"test_index_chunk_{retry_idx:04d}")
			src_path = os.path.join(retry_index_dir, f"test_src_chunk_{retry_idx:04d}")
		else:
			index_path = os.path.join(crb_index_dir, f"test_index_chunk_{chunk_id:04d}")
			src_path = os.path.join(crb_index_dir, f"test_src_chunk_{chunk_id:04d}")

		if not os.path.isfile(index_path):
			raise FileNotFoundError(f"Missing index file for chunk {chunk_id:04d}: {index_path}")
		idx_df = pd.read_csv(index_path, low_memory=False)
		required = {"Unique_identifier", "input_reaction"}
		missing = required.difference(idx_df.columns)
		if missing:
			raise KeyError(f"Index file {index_path} missing columns: {sorted(missing)}")

		if src_lines_used > len(idx_df):
			raise ValueError(
				f"Chunk {chunk_id:04d}: src_lines_used={src_lines_used} exceeds index rows={len(idx_df)}"
			)

		uid_part = idx_df.iloc[:src_lines_used].copy()
		uid_part["chunk_id"] = f"{chunk_id:04d}"
		uid_rows.append(uid_part[["Unique_identifier", "input_reaction", "chunk_id"]])

		if src_tail_dropped > 0:
			if not os.path.isfile(src_path):
				raise FileNotFoundError(f"Missing src file for dropped-tail extraction: {src_path}")
			with open(src_path, "r", encoding="utf-8") as fin:
				full_src = [line.rstrip("\n") for line in fin]
			tail_start = src_lines_used
			tail_end = min(len(full_src), len(idx_df))
			for j in range(tail_start, tail_end):
				tail_rows.append(
					{
						"Unique_identifier": str(idx_df.iloc[j]["Unique_identifier"]),
						"input_reaction": str(idx_df.iloc[j]["input_reaction"]),
						"src_line": full_src[j],
						"chunk_id": f"{chunk_id:04d}",
					}
				)

	if not uid_rows:
		raise FileNotFoundError("No chunk rows loaded from merged manifest/index files.")

	uid_df = pd.concat(uid_rows, ignore_index=True)
	if len(uid_df) != n_samples:
		raise ValueError(
			f"Merged index/sample mismatch: uid rows={len(uid_df)} vs decoded samples={n_samples}"
		)

	input_reactions: List[str] = []
	for line in src_lines:
		input_reactions.append(_detokenize_prediction(line))

	index_inputs = uid_df["input_reaction"].astype(str).tolist()
	input_match_mask = [a == b for a, b in zip(index_inputs, input_reactions)]
	n_mismatch = int(sum(1 for ok in input_match_mask if not ok))
	if n_mismatch:
		sample_bad = [i for i, ok in enumerate(input_match_mask) if not ok][:3]
		raise ValueError(
			"Merged src/pred ordering does not align with Unique_identifier mapping. "
			f"Found {n_mismatch} input-reaction mismatches. Sample row ids: {sample_bad}"
		)

	top1_pred = [_detokenize_prediction(pred_lines[j * local_n_best]) for j in range(n_samples)]
	top1_logp = pd.to_numeric(pd.Series([logp_lines[j * local_n_best] for j in range(n_samples)]), errors="coerce")
	top1_prob = top1_logp.map(_prob_from_log_score)

	completed = [
		compose_completed_reaction(inp, pred)
		for inp, pred in zip(input_reactions, top1_pred)
	]

	crb = pd.DataFrame(
		{
			"Unique_identifier": uid_df["Unique_identifier"].astype(str).values,
			"input_reaction": input_reactions,
			"prediction_reaction": completed,
			"crb_top1_diff": top1_pred,
			"crb_top1_log_score": top1_logp.values,
			"crb_sequence_probability": top1_prob.values,
			"has_unk_tokens": False,
			"unknown_tokens": "",
			"chunk_id": uid_df["chunk_id"].astype(str).values,
		}
	)

	dup_count = int(crb["Unique_identifier"].duplicated().sum())
	if dup_count:
		print(f"[warn] CRB has {dup_count} duplicate Unique_identifier rows; keeping first occurrence.")
		crb = crb.drop_duplicates(subset=["Unique_identifier"], keep="first").copy()

	status = crb["prediction_reaction"].progress_map(_reaction_valid_and_balanced)
	crb["is_valid_prediction"] = status.map(lambda x: bool(x[0]))
	crb["is_balanced"] = status.map(lambda x: bool(x[1]))

	if tail_rows:
		tail_df = pd.DataFrame(tail_rows)
	else:
		tail_df = pd.DataFrame(columns=["Unique_identifier", "input_reaction", "src_line", "chunk_id"])

	if cache_csv_path:
		cache_dir = os.path.dirname(cache_csv_path)
		if cache_dir:
			os.makedirs(cache_dir, exist_ok=True)
		crb.to_csv(cache_csv_path, index=False)
		print(f"Saved CRB cache: {cache_csv_path}")
	if tail_cache_csv_path:
		tail_cache_dir = os.path.dirname(tail_cache_csv_path)
		if tail_cache_dir:
			os.makedirs(tail_cache_dir, exist_ok=True)
		tail_df.to_csv(tail_cache_csv_path, index=False)
		print(f"Saved CRB tail cache: {tail_cache_csv_path}")

	return crb, tail_df


def _pct(numerator: int, denominator: int) -> float:
	if denominator <= 0:
		return float("nan")
	return 100.0 * float(numerator) / float(denominator)


def _print_section(title: str) -> None:
	print("\n" + "=" * 88)
	print(title)
	print("=" * 88)


def main() -> None:
	parser=argparse.ArgumentParser(description="Historical v1 uncurated report; current agreement uses uncurated_stereo_free_agreement/recompute.py")
	parser.add_argument("--output-dir", type=Path, required=True)
	args=parser.parse_args()
	if args.output_dir.exists() and any(args.output_dir.iterdir()):
		parser.error("Choose a new output directory")
	def _abs(p: str) -> str:
		return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)

	synrbl_dir = _abs(SYNRBL_DIR)
	crb_merged_pred_path = _abs(CRB_MERGED_PRED_PATH)
	crb_merged_logp_path = _abs(CRB_MERGED_LOGP_PATH)
	crb_merged_src_path = _abs(CRB_MERGED_SRC_PATH)
	crb_merged_manifest_path = _abs(CRB_MERGED_MANIFEST_PATH)
	crb_index_dir = _abs(CRB_INDEX_DIR)
	crb_retry_index_dir = _abs(CRB_RETRY_INDEX_DIR)
	output_summary_csv = str(args.output_dir/"uncurated_consensus_summary.csv")
	output_uids_dir = str(args.output_dir/"uncurated_consensus_uids")
	crb_cache_csv = _abs(CRB_CACHE_CSV)
	crb_tail_cache_csv = _abs(CRB_TAIL_CACHE_CSV)

	_print_section("Load Data")
	syn = load_synrbl_results(synrbl_dir)
	crb, tail_df = load_crb_results_from_merged_files(
		merged_pred_path=crb_merged_pred_path,
		merged_logp_path=crb_merged_logp_path,
		merged_src_path=crb_merged_src_path,
		merged_manifest_path=crb_merged_manifest_path,
		crb_index_dir=crb_index_dir,
		retry_index_dir=crb_retry_index_dir,
		n_best=N_BEST,
		retry_start_chunk=RETRY_START_CHUNK,
		cache_csv_path=crb_cache_csv,
		tail_cache_csv_path=crb_tail_cache_csv,
	)
	print(f"SynRBL rows: {len(syn):,}")
	print(f"CRB rows (total decoded): {len(crb):,}")
	print(f"CRB tail rows still missing predictions: {len(tail_df):,}")

	if DEV_SUBSET is not None:
		print(f"[DEV] Subsetting to first {DEV_SUBSET:,} CRB rows by merged-file order.")
		subset_ids = set(crb["Unique_identifier"].iloc[:DEV_SUBSET])
		crb = crb[crb["Unique_identifier"].isin(subset_ids)].copy()
		syn = syn[syn["Unique_identifier"].isin(subset_ids)].copy()
		print(f"[DEV] After subset — SynRBL: {len(syn):,}, CRB: {len(crb):,}")

	# Per-method solved/balanced coverage.
	syn_total_n = len(syn)
	crb_total_n = len(crb)
	syn_solved_n = int(_safe_bool(syn.get("is_balanced", pd.Series(False, index=syn.index))).sum())
	crb_balanced_n = int(_safe_bool(crb.get("is_balanced", pd.Series(False, index=crb.index))).sum())
	crb_valid_n = int(_safe_bool(crb.get("is_valid_prediction", pd.Series(False, index=crb.index))).sum())

	_print_section("Balanced/Solved Coverage By Method")
	print(
		f"SynRBL solved/balanced: {syn_solved_n:,}/{syn_total_n:,} "
		f"({_pct(syn_solved_n, syn_total_n):.2f}%)"
	)
	print(
		f"CRB valid predictions:  {crb_valid_n:,}/{crb_total_n:,} "
		f"({_pct(crb_valid_n, crb_total_n):.2f}%)"
	)
	print(
		f"CRB balanced:           {crb_balanced_n:,}/{crb_total_n:,} "
		f"({_pct(crb_balanced_n, crb_total_n):.2f}%)"
	)

	crb_parseable = crb[crb["prediction_reaction"].astype(str).str.contains(">")]

	_print_section("CRB UID Linkage Check")
	print("Merged-file mapping check passed: each CRB decoded line block is linked to a Unique_identifier via manifest+index files.")
	print("Input reaction check passed: detokenized merged src lines match index input_reaction rows exactly.")

	_print_section("CRB Parseability")
	print(f"CRB reactions containing '>': {len(crb_parseable):,}/{crb_total_n:,} ({_pct(len(crb_parseable), crb_total_n):.2f}%)")

	_print_section("Model Confidence Coverage")
	syn_conf = pd.to_numeric(syn["synrbl_confidence"], errors="coerce").fillna(0.0)
	syn_gt50 = int((syn_conf > 0.50).sum())
	syn_gt90 = int((syn_conf > 0.90).sum())
	print(f"SynRBL confidence > 0.50: {syn_gt50:,}/{len(syn):,} ({_pct(syn_gt50, len(syn)):.2f}%)")
	print(f"SynRBL confidence > 0.90: {syn_gt90:,}/{len(syn):,} ({_pct(syn_gt90, len(syn)):.2f}%)")
	crb_prob = pd.to_numeric(crb["crb_sequence_probability"], errors="coerce").fillna(0.0)
	for thresh in (0.80, 0.90, 0.95):
		n_above = int((crb_prob > thresh).sum())
		print(
			f"CRB top1 prob > {thresh:.2f}: "
			f"{n_above:,}/{crb_total_n:,} ({_pct(n_above, crb_total_n):.2f}%)"
		)

	_print_section("Unique Identifier Overlap")
	syn_ids = set(syn["Unique_identifier"].dropna().astype(str))
	crb_ids = set(crb["Unique_identifier"].dropna().astype(str))
	overlap_ids = syn_ids.intersection(crb_ids)
	union_ids = syn_ids.union(crb_ids)
	print(f"SynRBL unique IDs: {len(syn_ids):,}")
	print(f"CRB unique IDs:    {len(crb_ids):,}")
	print(f"Overlap IDs:       {len(overlap_ids):,}")
	print(f"Overlap % of total dataset (union): {_pct(len(overlap_ids), len(union_ids)):.2f}%")
	print(f"Overlap % of SynRBL: {_pct(len(overlap_ids), len(syn_ids)):.2f}%")
	print(f"Overlap % of CRB:    {_pct(len(overlap_ids), len(crb_ids)):.2f}%")

	_print_section("Consensus And Agreement On Overlap")
	overlap_df = syn[["Unique_identifier", "prediction_reaction"]].merge(
		crb[["Unique_identifier", "prediction_reaction", "crb_sequence_probability"]],
		on="Unique_identifier",
		suffixes=("_synrbl", "_crb"),
		how="inner",
	)
	overlap_df = overlap_df.merge(
		syn[["Unique_identifier", "synrbl_confidence"]],
		on="Unique_identifier",
		how="left",
	)
	overlap_df = overlap_df[
		overlap_df["prediction_reaction_synrbl"].astype(str).str.contains(">")
		& overlap_df["prediction_reaction_crb"].astype(str).str.contains(">")
	].copy()

	# Require both predictions to be atom-balanced before running DPE.
	n_overlap_syntax = len(overlap_df)
	syn_bal = overlap_df["prediction_reaction_synrbl"].map(_is_reaction_balanced)
	crb_bal = overlap_df["prediction_reaction_crb"].map(_is_reaction_balanced)
	overlap_valid = overlap_df[syn_bal & crb_bal].copy()
	n_dropped_unbalanced = n_overlap_syntax - len(overlap_valid)
	print(
		f"Overlap rows with valid syntax: {n_overlap_syntax:,}  |  "
		f"Both balanced: {len(overlap_valid):,}  |  "
		f"Dropped (unbalanced): {n_dropped_unbalanced:,}"
	)

	# Consensus-style uncurated summary: no ground truth, so report coverage + confidence.
	overlap_valid["models_agree_dpe"] = [
		dpe_match(a, b)
		for a, b in tqdm(
			zip(
				overlap_valid["prediction_reaction_synrbl"].astype(str),
				overlap_valid["prediction_reaction_crb"].astype(str),
			),
			total=len(overlap_valid),
			desc="Consensus agreement (DPE)",
		)
	]

	overlap_valid["synrbl_confidence"] = pd.to_numeric(overlap_valid["synrbl_confidence"], errors="coerce").fillna(0.0)
	overlap_valid["crb_sequence_probability"] = pd.to_numeric(
		overlap_valid["crb_sequence_probability"], errors="coerce"
	).fillna(0.0)
	both_confident = (overlap_valid["synrbl_confidence"] >= SYN_CONF_THR) & (
		overlap_valid["crb_sequence_probability"] >= CRB_CONF_THR
	)
	agree = overlap_valid["models_agree_dpe"]
	confident_agree = both_confident & agree

	n_overlap_valid = len(overlap_valid)
	n_agree = int(agree.sum())
	n_both_confident = int(both_confident.sum())
	n_confident_agree = int(confident_agree.sum())

	print(f"Agreement (DPE):                  {n_agree:,}/{n_overlap_valid:,} ({_pct(n_agree, n_overlap_valid):.2f}%)")
	print(
		f"Both confident (Syn>={SYN_CONF_THR:.2f}, CRB>={CRB_CONF_THR:.2f}): "
		f"{n_both_confident:,}/{n_overlap_valid:,} ({_pct(n_both_confident, n_overlap_valid):.2f}%)"
	)
	print(
		f"Confident + agreement (DPE):      {n_confident_agree:,}/{n_overlap_valid:,} "
		f"({_pct(n_confident_agree, n_overlap_valid):.2f}%)"
	)

	os.makedirs(output_uids_dir, exist_ok=True)
	overlap_valid.loc[confident_agree, ["Unique_identifier"]].to_csv(
		os.path.join(output_uids_dir, "confident_agree_uids.csv"), index=False
	)
	overlap_valid.loc[agree & (~both_confident), ["Unique_identifier"]].to_csv(
		os.path.join(output_uids_dir, "agree_not_confident_uids.csv"), index=False
	)

	summary_row = {
		"syn_rows": len(syn),
		"syn_solved_balanced_n": syn_solved_n,
		"syn_solved_balanced_pct": _pct(syn_solved_n, syn_total_n),
		"crb_rows": len(crb),
		"crb_valid_prediction_n": crb_valid_n,
		"crb_valid_prediction_pct": _pct(crb_valid_n, crb_total_n),
		"crb_balanced_n": crb_balanced_n,
		"crb_balanced_pct": _pct(crb_balanced_n, crb_total_n),
		"crb_parseable_rows": len(crb_parseable),
		"crb_tail_missing_rows": len(tail_df),
		"overlap_syntax_rows": n_overlap_syntax,
		"overlap_both_balanced_rows": len(overlap_valid),
		"overlap_dropped_unbalanced": n_dropped_unbalanced,
		"agreement_dpe_n": n_agree,
		"agreement_dpe_pct_overlap": _pct(n_agree, n_overlap_valid),
		"both_confident_n": n_both_confident,
		"both_confident_pct_overlap": _pct(n_both_confident, n_overlap_valid),
		"confident_agreement_n": n_confident_agree,
		"confident_agreement_pct_overlap": _pct(n_confident_agree, n_overlap_valid),
		"syn_conf_thr": SYN_CONF_THR,
		"crb_conf_thr": CRB_CONF_THR,
	}
	os.makedirs(os.path.dirname(output_summary_csv), exist_ok=True)
	pd.DataFrame([summary_row]).to_csv(output_summary_csv, index=False)
	print(f"Wrote summary CSV: {output_summary_csv}")
	print(f"Wrote UID lists in: {output_uids_dir}")


if __name__ == "__main__":
	main()
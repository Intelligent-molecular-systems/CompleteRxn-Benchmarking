import pandas as pd
import os
import re
import argparse
from collections import Counter

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def extract_difference(src_reaction, target_reaction):
    """
    Extract the difference between the source reaction and the target reaction in terms of molecules.
    The difference is the molecules in the target reaction that are not in the source reaction.
    Uses rdkit and compares the molecules based on their canonical SMILES representation to ensure accurate comparison.

    Args:
        src_reaction (str): The source reaction (InputReaction).
        target_reaction (str): The target reaction (rxn_flower).

    Returns:
        str: The difference as a string of molecules.
    """

    # Split reactions into reactants and products
    src_reactants, src_reagents, src_products = src_reaction.split('>')
    target_reactants, target_reagents, target_products = target_reaction.split('>')

    if len(src_reactants) == 0 or len(src_products) == 0 or len(target_reactants) == 0 or len(target_products) == 0:
        raise ValueError("Reactions must be in the format 'reactants>reagents>products'")

    # Get all molecules with their multiplicity and figure out which part is not contained in the source reaction
    # Assume that the molecules have been canonicalized already. 
    src_reactants_list = src_reactants.split('.')
    src_reagents_list = src_reagents.split('.')
    src_products_list = src_products.split('.')
    target_reactants_list = target_reactants.split('.')
    target_reagents_list = target_reagents.split('.')
    target_products_list = target_products.split('.')
    # Find molecules in target that are not in source, should take into account multiplicity (e.g., if source has 2 of a molecule and target has 3, the difference should include 1 of that molecule)
    diff_reactants = []
    diff_reagents = []
    diff_products = []
    # throw a warning if any of the source part lengths is bigger than target part lengths, as that would indicate a potential issue with the data (e.g., missing reagents in the target reaction)
    if len(src_reactants_list) > len(target_reactants_list):
        print(f"Warning: Source reaction has more reactants than target reaction. Source: {src_reactants_list}, Target: {target_reactants_list}")
    if len(src_reagents_list) > len(target_reagents_list):
        print(f"Warning: Source reaction has more reagents than target reaction. Source: {src_reagents_list}, Target: {target_reagents_list}")
    if len(src_products_list) > len(target_products_list):
        print(f"Warning: Source reaction has more products than target reaction. Source: {src_products_list}, Target: {target_products_list}")

    # Warn if source contains molecule identities (or extra multiplicity) not present in target.
    def warn_source_not_in_target(src_list, tgt_list, side_name):
        src_only = Counter(src_list) - Counter(tgt_list)
        if src_only:
            missing_in_target = [mol for mol, count in src_only.items() for _ in range(count)]
            print(
                f"Warning: Source {side_name} contains molecules not present in target {side_name}. "
                f"Missing in target: {missing_in_target}; Source: {src_list}; Target: {tgt_list}"
            )
            return False
        else: 
            return True

    valid_react = warn_source_not_in_target(src_reactants_list, target_reactants_list, "reactants")
    valid_reag = warn_source_not_in_target(src_reagents_list, target_reagents_list, "reagents")
    valid_prod = warn_source_not_in_target(src_products_list, target_products_list, "products")

    for mol in target_reactants_list:
        if mol not in src_reactants_list:
            diff_reactants.append(mol)
        else:
            src_reactants_list.remove(mol)  # Remove one instance to account for multiplicity
    for mol in target_reagents_list:
        if mol not in src_reagents_list:
            diff_reagents.append(mol)
        else:
            src_reagents_list.remove(mol)
    for mol in target_products_list:
        if mol not in src_products_list:
            diff_products.append(mol)
        else:
            src_products_list.remove(mol)
    # Return a combined string of the differences in the same format as the reactions (reactants>reagents>products)
    diff_reaction = f"{'.'.join(diff_reactants)}>{'.'.join(diff_reagents)}>{'.'.join(diff_products)}"

    return diff_reaction

def prepare_files(input_csv, output_dir, split_name, stereo_resolved=True, input_col = "InputReaction", target_col = "TargetReaction_with_stereo"):
    """
    Prepare train_src.txt and train_tgt.txt files for a specific split.

    Args:
        input_csv (str): Path to the input CSV file.
        output_dir (str): Directory to save the output files.
        split_name (str): Name of the split (e.g., 'train', 'val', 'test').
        stereo_resolved (bool): Whether the input CSV contains stereo-resolved reactions. If True, will use target column TargetReaction_with_stereo instead of TargetReaction for extracting the difference. Default is True.
        input_col (str): Name of the input column in the CSV file. Default is "InputReaction".
        target_col (str): Name of the target column in the CSV file. Default is "TargetReaction_with_stereo".
    Returns:
        None
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)
    print(input_csv)

    # Ensure required columns exist
    if stereo_resolved: 
        if input_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Input CSV must contain '{input_col}' and '{target_col}' columns for stereo-resolved data.")
    else:
        if input_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Input CSV must contain '{input_col}' and '{target_col}' columns for non-stereo-resolved data.")

    # Prepare source and target data
    df['src'] = df[input_col]
    # progress bar for applying the extract_difference function to each row, with a warning if the function throws an error for any row (e.g., due to unexpected reaction format)
    from tqdm import tqdm
    tqdm.pandas(desc=f"Processing {split_name} split")
    if stereo_resolved:
        df['tgt'] = df.progress_apply(lambda row: extract_difference(row[input_col], row[target_col]), axis=1)
    else:
        df['tgt'] = df.progress_apply(lambda row: extract_difference(row[input_col], row[target_col]), axis=1)

    # Save to files
    os.makedirs(output_dir, exist_ok=True)
    src_file = os.path.join(output_dir, f"{split_name}_src.txt")
    tgt_file = os.path.join(output_dir, f"{split_name}_tgt.txt")

    # Apply tokenization to the source and target columns before saving
    df['src'] = df['src'].apply(smi_tokenizer)
    df['tgt'] = df['tgt'].apply(smi_tokenizer)

    df['src'].to_csv(src_file, index=False, header=False)
    df['tgt'].to_csv(tgt_file, index=False, header=False)

    print(f"Files for {split_name} split saved to {output_dir}:")
    print(f"  - {src_file}")
    print(f"  - {tgt_file}")


def build_split_suffix(split_type, seed=None, repetition=None):
    """Create a suffix used for file discovery and output directory naming."""
    suffix = split_type
    if seed is not None:
        suffix += f"_seed{seed}"
    if repetition is not None:
        suffix += f"_rep{repetition}"
    return suffix


def resolve_split_file(input_dir, split_name, split_type, seed=None, repetition=None):
    """Find the split assignment file for the requested partition.

    First try the shared split file containing reaction IDs and assignments.
    Fall back to older files that store full rows for each partition.
    """
    # ── New canonical format (single file covering all partitions) ──────────
    # The canonical file lives in a sub-directory named after the split type.
    canonical_dir = os.path.join(input_dir, f'{split_type}_splits')
    canonical_candidates = []
    if split_type == 'group':
        # Group files are named benchmark_final_group_splits_rep{N}.csv
        if repetition is not None:
            canonical_candidates.append(f'benchmark_final_group_splits_rep{repetition}.csv')
        # Fallback: rep0
        canonical_candidates.append('benchmark_final_group_splits_rep0.csv')
    elif split_type == 'extreme_ood':
        # Extreme OOD files are named benchmark_final_extreme_ood_splits_rep{N}.csv
        if repetition is not None:
            canonical_candidates.append(f'benchmark_final_extreme_ood_splits_rep{repetition}.csv')
        # Fallback: rep0
        canonical_candidates.append('benchmark_final_extreme_ood_splits_rep0.csv')
    elif seed is not None and repetition is not None:
        canonical_candidates.append(
            f'benchmark_final_{split_type}_splits_rep{repetition}_seed{seed}.csv')
    if repetition is not None:
        # Any seed: glob-style search not available, but try common base seeds
        for s in range(repetition, repetition + 20):   # seed >= rep by convention
            canonical_candidates.append(
                f'benchmark_final_{split_type}_splits_rep{repetition}_seed{s}.csv')
    if seed is not None:
        canonical_candidates.append(f'benchmark_final_{split_type}_splits_seed{seed}.csv')
    canonical_candidates.append(f'benchmark_final_{split_type}_splits.csv')

    for filename in canonical_candidates:
        path = os.path.join(canonical_dir, filename)
        if os.path.exists(path):
            return path, 'canonical'

    # ── Legacy per-partition format ──────────────────────────────────────────
    legacy_candidates = []
    if seed is not None and repetition is not None:
        legacy_candidates.append(f"{split_name}_split_{split_type}_seed{seed}_rep{repetition}.csv")
    if seed is not None:
        legacy_candidates.append(f"{split_name}_split_{split_type}_seed{seed}.csv")
    if repetition is not None:
        legacy_candidates.append(f"{split_name}_split_{split_type}_rep{repetition}.csv")
    legacy_candidates.append(f"{split_name}_split_{split_type}.csv")

    for filename in legacy_candidates:
        path = os.path.join(input_dir, filename)
        if os.path.exists(path):
            return path, 'v1'

    raise FileNotFoundError(
        f"Could not find split file for split='{split_name}', type='{split_type}', "
        f"seed='{seed}', repetition='{repetition}'.\n"
        f"  Canonical tried in: {canonical_dir}\n"
        f"  Legacy tried in:    {input_dir}"
    )


SPLIT_LABEL = {'train': 0, 'valid': 1, 'test': 2}


def load_split_csv(path, file_format, split_name, benchmark_csv=None):
    """Read one train, validation, or test partition from either split format.

    For a shared assignment file, select the requested partition and match its
    reaction IDs to the full benchmark. Older per-partition files already contain
    the full columns and are returned directly.
    """
    print(f"Loading split CSV from {path} (format={file_format})")
    if file_format == 'v1':
        return pd.read_csv(path)

    # ── Canonical ────────────────────────────────────────────────────────────
    if benchmark_csv is None or not os.path.exists(benchmark_csv):
        raise ValueError(
            f"Canonical split file requires --benchmark-csv to join against, "
            f"but '{benchmark_csv}' does not exist."
        )
    label = SPLIT_LABEL[split_name]
    index_df = pd.read_csv(path)
    split_col = next((c for c in ['random_split', 'group_split', 'extreme_ood'] if c in index_df.columns), None)
    if split_col is None:
        raise ValueError(f"No recognised split column in {path}. Expected 'random_split', 'group_split', or 'extreme_ood'.")
    uids = index_df.loc[index_df[split_col] == label, 'Unique_identifier']
    bench = pd.read_csv(benchmark_csv)
    return bench[bench['Unique_identifier'].isin(uids)].reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CRB train/valid/test src/tgt files from benchmark split CSVs.")
    parser.add_argument("--input-dir", default="data/Template_curated/benchmark/", help="Directory containing split CSV files.")
    parser.add_argument("--benchmark-csv", default="data/Template_curated/benchmark/benchmark_final.csv",
                        help="Path to benchmark_final.csv — required when using canonical split index files.")
    parser.add_argument("--output-dir", default="models/constrained_reaction_balancer/Template_curated/benchmark/prepared/", help="Base output directory for generated txt files.")
    parser.add_argument("--split-type", choices=["random", "group", "extreme_ood"], default="random", help="Split type to process.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed identifier used in split filenames and output folder naming.")
    parser.add_argument("--repetition", type=int, default=None, help="Optional repetition identifier used in split filenames and output folder naming.")
    parser.add_argument("--stereo-resolved", action="store_true", default=True, help="Use stereo-resolved target column.")
    parser.add_argument("--no-stereo-resolved", dest="stereo_resolved", action="store_false", help="Disable stereo-resolved mode.")
    parser.add_argument("--input-col", default="InputReaction_harmonized", help="Source reaction column in split CSV files.")
    parser.add_argument("--target-col", default="TargetReaction_with_stereo", help="Target reaction column in split CSV files.")
    return parser.parse_args()

def main():
    """
    Main function to prepare train, val, and test files.
    """
    args = parse_args()

    split_suffix = build_split_suffix(args.split_type, seed=args.seed, repetition=args.repetition)
    output_dir = os.path.join(args.output_dir, split_suffix)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve each partition file (canonical index or legacy full-column CSV)
    splits = {}
    for split_name in ("train", "valid", "test"):
        path, fmt = resolve_split_file(
            args.input_dir, split_name, args.split_type,
            seed=args.seed, repetition=args.repetition,
        )
        splits[split_name] = (path, fmt)

    print(f"Preparing split set: {split_suffix}")
    print(f"Input directory:  {args.input_dir}")
    print(f"Benchmark CSV:    {args.benchmark_csv}")
    print(f"Output directory: {output_dir}")

    # Process each split
    for split_name, (input_path, fmt) in splits.items():
        print(f"\n  [{split_name}]  format={fmt}  file={input_path}")
        try:
            df = load_split_csv(input_path, fmt, split_name, benchmark_csv=args.benchmark_csv)
            # Write to a temporary CSV so prepare_files can read it
            tmp_csv = os.path.join(output_dir, f"_tmp_{split_name}.csv")
            df.to_csv(tmp_csv, index=False)
            prepare_files(
                tmp_csv,
                output_dir,
                split_name,
                stereo_resolved=args.stereo_resolved,
                input_col=args.input_col,
                target_col=args.target_col,
            )
            os.remove(tmp_csv)
        except Exception as e:
            print(f"Error processing {split_name} split: {e}")

if __name__ == '__main__':
    src_test = "CC(=O)c1ccc(Oc2ncccc2C(=O)Cl)cc1.O.O=[N+]([O-])c1ccccc1.[Al+3].[Cl-].[Cl-].[Cl-]>>CC(=O)c1ccc2oc3ncccc3c(=O)c2c1"
    tgt_test = "CC(=O)c1ccc(Oc2ncccc2C(=O)Cl)cc1.O.O=[N+]([O-])c1ccccc1.[Al+3].[Al+3].[Cl-].[Cl-].[Cl-].[Cl-].[Cl-].[Cl-]>>CC(=O)c1ccc2oc3ncccc3c(=O)c2c1.Cl.Cl[Al](Cl)Cl.Cl[Al](Cl)Cl.O.O=[N+]([O-])c1ccccc1"
    test_difference = extract_difference(src_test, tgt_test)
    print(f"Test difference: {test_difference}")  
    assert test_difference == "[Al+3].[Cl-].[Cl-].[Cl-]>>Cl.Cl[Al](Cl)Cl.Cl[Al](Cl)Cl.O.O=[N+]([O-])c1ccccc1", "Test case failed for extract_difference function"
    #test_difference_2 = extract_difference(tgt_test, src_test) # Should throw warning
    # Another test case with stereochemistry ambigouity
    src_test_2 = 'CC(=O)O.CC(=O)[O-].CCCCCCCC/C=C/CCCCl.CCCCCCCC/C=C\CCCCl.[K+]>>CCCCCCCCC=CCCCOC(C)=O'
    tgt_test_2 = 'CC(=O)O.CC(=O)[O-].CCCCCCCCC=CCCCCl.CCCCCCCCC=CCCCCl.[K+]>>CC(=O)O.CCCCCCCCC=CCCCCl.CCCCCCCCC=CCCCOC(C)=O.[Cl-].[K+]'
    test_difference_2 = extract_difference(src_test_2, tgt_test_2)
    print(f"Test difference 2: {test_difference_2}")

    main()
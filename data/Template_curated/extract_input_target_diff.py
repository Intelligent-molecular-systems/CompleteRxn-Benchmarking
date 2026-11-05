"""Find the molecules needed to turn each input reaction into its target.

When the input contains extra copies of a molecule, try removing duplicates
while keeping at least one copy. Write the adjusted input and missing species.
"""

import argparse
import pandas as pd
import os
import re
from collections import Counter


def _expand_counter_in_reference_order(counter_obj, reference_list):
    """Expand a Counter into a list while preserving the reference list order."""
    out = []
    remaining = Counter(counter_obj)
    for mol in reference_list:
        if remaining[mol] > 0:
            out.append(mol)
            remaining[mol] -= 1
    return out


def _remove_extra_multiplicity_from_source(src_list, tgt_list, side_name, verbose=True):
    """
    Remove only duplicate molecules from source when source has extra multiplicity.
    Keeps at least one copy of each source-only molecule.
    """
    src_counter = Counter(src_list)
    tgt_counter = Counter(tgt_list)
    src_only = src_counter - tgt_counter

    if verbose and src_only:
        missing_in_target = _expand_counter_in_reference_order(src_only, src_list)
        print(
            f"Warning: Source {side_name} contains molecules not present in target {side_name}. "
            f"Missing in target: {missing_in_target}; Source: {src_list}; Target: {tgt_list}"
        )

    # Remove only the surplus multiplicity and never remove the last copy of a molecule.
    to_remove = {}
    for mol, extra_count in src_only.items():
        removable = max(0, src_counter[mol] - 1)
        n_remove = min(extra_count, removable)
        if n_remove > 0:
            to_remove[mol] = n_remove

    if not to_remove:
        return list(src_list), False, len(src_only) == 0, src_only

    updated = []
    removed_counter = Counter()
    for mol in src_list:
        if removed_counter[mol] < to_remove.get(mol, 0):
            removed_counter[mol] += 1
            continue
        updated.append(mol)

    remaining_src_only = Counter(updated) - tgt_counter
    resolved = len(remaining_src_only) == 0
    if verbose and not resolved:
        still_missing = _expand_counter_in_reference_order(remaining_src_only, updated)
        print(
            f"Warning: Could not resolve all discrepancies between source and target {side_name} after attempted removal. "
            f"Remaining missing in target: {still_missing}; Source: {updated}; Target: {tgt_list}"
        )

    return updated, True, resolved, remaining_src_only

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def extract_difference(
    src_reaction,
    target_reaction,
    attempt_src_multiplicity_fix=False,
    return_harmonized_source=False,
    verbose=True,
):
    """Find target molecules missing from the input on each reaction side.

    Compare canonical SMILES and preserve molecule counts. Where allowed,
    remove extra duplicate input molecules before computing the difference.
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
    original_src_reactants_list = list(src_reactants_list)
    original_src_reagents_list = list(src_reagents_list)
    original_src_products_list = list(src_products_list)
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

    if attempt_src_multiplicity_fix:
        src_reactants_list, _, _, rem_reactants = _remove_extra_multiplicity_from_source(
            src_reactants_list,
            target_reactants_list,
            "reactants",
            verbose=verbose,
        )
        src_reagents_list, _, _, rem_reagents = _remove_extra_multiplicity_from_source(
            src_reagents_list,
            target_reagents_list,
            "reagents",
            verbose=verbose,
        )
        src_products_list, _, _, rem_products = _remove_extra_multiplicity_from_source(
            src_products_list,
            target_products_list,
            "products",
            verbose=verbose,
        )
    else:
        rem_reactants = Counter(src_reactants_list) - Counter(target_reactants_list)
        rem_reagents = Counter(src_reagents_list) - Counter(target_reagents_list)
        rem_products = Counter(src_products_list) - Counter(target_products_list)
        if verbose:
            for side_name, src_list, tgt_list, rem in (
                ("reactants", src_reactants_list, target_reactants_list, rem_reactants),
                ("reagents", src_reagents_list, target_reagents_list, rem_reagents),
                ("products", src_products_list, target_products_list, rem_products),
            ):
                if rem:
                    missing = _expand_counter_in_reference_order(rem, src_list)
                    print(
                        f"Warning: Source {side_name} contains molecules not present in target {side_name}. "
                        f"Missing in target: {missing}; Source: {src_list}; Target: {tgt_list}"
                    )

    harmonized_src_reaction = (
        f"{'.'.join(src_reactants_list)}>{'.'.join(src_reagents_list)}>{'.'.join(src_products_list)}"
    )

    # Compute target-minus-source difference with multiplicity, while preserving target order.
    diff_reactants = _expand_counter_in_reference_order(
        Counter(target_reactants_list) - Counter(src_reactants_list),
        target_reactants_list,
    )
    diff_reagents = _expand_counter_in_reference_order(
        Counter(target_reagents_list) - Counter(src_reagents_list),
        target_reagents_list,
    )
    diff_products = _expand_counter_in_reference_order(
        Counter(target_products_list) - Counter(src_products_list),
        target_products_list,
    )
    # Return a combined string of the differences in the same format as the reactions (reactants>reagents>products)
    diff_reaction = f"{'.'.join(diff_reactants)}>{'.'.join(diff_reagents)}>{'.'.join(diff_products)}"

    if not return_harmonized_source:
        return diff_reaction

    removed_reactants = _expand_counter_in_reference_order(
        Counter(original_src_reactants_list) - Counter(src_reactants_list),
        original_src_reactants_list,
    )
    removed_reagents = _expand_counter_in_reference_order(
        Counter(original_src_reagents_list) - Counter(src_reagents_list),
        original_src_reagents_list,
    )
    removed_products = _expand_counter_in_reference_order(
        Counter(original_src_products_list) - Counter(src_products_list),
        original_src_products_list,
    )

    unresolved_reactants = _expand_counter_in_reference_order(rem_reactants, src_reactants_list)
    unresolved_reagents = _expand_counter_in_reference_order(rem_reagents, src_reagents_list)
    unresolved_products = _expand_counter_in_reference_order(rem_products, src_products_list)
    has_source_not_in_target = bool(unresolved_reactants or unresolved_reagents or unresolved_products)

    report = {
        "removed_from_source": {
            "reactants": removed_reactants,
            "reagents": removed_reagents,
            "products": removed_products,
        },
        "source_not_in_target": {
            "has_any": has_source_not_in_target,
            "reactants": unresolved_reactants,
            "reagents": unresolved_reagents,
            "products": unresolved_products,
        },
    }
    return diff_reaction, harmonized_src_reaction, report


def harmonize_input_dataframe(
    df,
    src_col="InputReaction_canonical",
    tgt_col="TargetReaction_with_stereo",
    attempt_src_multiplicity_fix=False,
    use_harmonized_source=False,
    out_src_col="InputReaction_harmonized",
    out_diff_col="input_target_diff_harmonized",
    out_modified_col="input_harmonized_modified",
    out_invalid_col="source_not_in_target",
    exclude_source_not_in_target=False,
):
    """Prepare consistent input reactions and missing-species targets for a DataFrame."""
    out_df = df.copy()

    def _row_apply(row):
        diff_rxn, harmonized_src, report = extract_difference(
            row[src_col],
            row[tgt_col],
            attempt_src_multiplicity_fix=attempt_src_multiplicity_fix,
            return_harmonized_source=True,
            verbose=False,
        )
        return pd.Series(
            {
                out_src_col: harmonized_src,
                out_diff_col: diff_rxn,
                out_modified_col: bool(
                    report["removed_from_source"]["reactants"]
                    or report["removed_from_source"]["reagents"]
                    or report["removed_from_source"]["products"]
                ),
                out_invalid_col: report["source_not_in_target"]["has_any"],
            }
        )

    out_df[[out_src_col, out_diff_col, out_modified_col, out_invalid_col]] = out_df.apply(_row_apply, axis=1)

    if exclude_source_not_in_target:
        out_df = out_df.loc[~out_df[out_invalid_col]].copy()

    if use_harmonized_source:
        out_df[src_col] = out_df[out_src_col]
    return out_df


def process_input_target_csv(
    input_csv,
    output_csv,
    src_col="InputReaction_canonical",
    tgt_col="TargetReaction_with_stereo",
    attempt_src_multiplicity_fix=False,
    use_harmonized_source=False,
    exclude_source_not_in_target=False,
    excluded_output_csv=None,
):
    """Read a CSV, adjust input reactions, and save their missing-species targets."""
    df = pd.read_csv(input_csv)
    # Subselect based on unique identifier column USPTO_US03960961_1004419
    required_cols = {src_col, tgt_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    full_df = harmonize_input_dataframe(
        df,
        src_col=src_col,
        tgt_col=tgt_col,
        attempt_src_multiplicity_fix=attempt_src_multiplicity_fix,
        use_harmonized_source=use_harmonized_source,
        exclude_source_not_in_target=False,
    )

    if exclude_source_not_in_target:
        invalid_df = full_df.loc[full_df["source_not_in_target"]].copy()
        out_df = full_df.loc[~full_df["source_not_in_target"]].copy()
        if excluded_output_csv is None:
            base, ext = os.path.splitext(output_csv)
            ext = ext if ext else ".csv"
            excluded_output_csv = f"{base}_excluded_source_not_in_target{ext}"
        if excluded_output_csv is not None:
            invalid_df.to_csv(excluded_output_csv, index=False)
        print(f"Excluded {len(invalid_df)} rows with source-only molecules not present in target.")
        if excluded_output_csv is not None:
            print(f"Saved excluded rows to: {excluded_output_csv}")
    else:
        out_df = full_df

    out_df.to_csv(output_csv, index=False)
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Input-target difference harmonization.")
    parser.add_argument("--input-csv", help="Input CSV path.")
    parser.add_argument("--output-csv", help="Output CSV path.")
    parser.add_argument("--src-col", default="InputReaction_canonical", help="Source reaction column name.")
    parser.add_argument("--tgt-col", default="TargetReaction_with_stereo", help="Target reaction column name.")
    parser.add_argument(
        "--attempt-src-multiplicity-fix",
        action="store_true",
        help="Attempt to remove source-only multiplicity before difference extraction.",
    )
    parser.add_argument(
        "--use-harmonized-source",
        action="store_true",
        help="Replace source column with harmonized source in output CSV.",
    )
    parser.add_argument(
        "--exclude-source-not-in-target",
        action="store_true",
        help="Exclude rows where source contains molecules not present in target.",
    )
    parser.add_argument(
        "--excluded-output-csv",
        default=None,
        help="Optional path to save excluded rows when --exclude-source-not-in-target is set.",
    )
    args = parser.parse_args()

    if not args.input_csv or not args.output_csv:
        print("No --input-csv/--output-csv provided. Skipping CSV processing in main().")
        return

    process_input_target_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        src_col=args.src_col,
        tgt_col=args.tgt_col,
        attempt_src_multiplicity_fix=args.attempt_src_multiplicity_fix,
        use_harmonized_source=args.use_harmonized_source,
        exclude_source_not_in_target=args.exclude_source_not_in_target,
        excluded_output_csv=args.excluded_output_csv,
    )
    print(f"Saved harmonized dataframe to: {args.output_csv}")


if __name__ == '__main__':
    """ src_test = "CC(=O)c1ccc(Oc2ncccc2C(=O)Cl)cc1.O.O=[N+]([O-])c1ccccc1.[Al+3].[Cl-].[Cl-].[Cl-]>>CC(=O)c1ccc2oc3ncccc3c(=O)c2c1"
    tgt_test = "CC(=O)c1ccc(Oc2ncccc2C(=O)Cl)cc1.O.O=[N+]([O-])c1ccccc1.[Al+3].[Al+3].[Cl-].[Cl-].[Cl-].[Cl-].[Cl-].[Cl-]>>CC(=O)c1ccc2oc3ncccc3c(=O)c2c1.Cl.Cl[Al](Cl)Cl.Cl[Al](Cl)Cl.O.O=[N+]([O-])c1ccccc1"
    test_difference = extract_difference(src_test, tgt_test, attempt_src_multiplicity_fix=True, verbose=True)
    print(f"Test difference: {test_difference}")  
    assert test_difference == "[Al+3].[Cl-].[Cl-].[Cl-]>>Cl.Cl[Al](Cl)Cl.Cl[Al](Cl)Cl.O.O=[N+]([O-])c1ccccc1", "Test case failed for extract_difference function"
    #test_difference_2 = extract_difference(tgt_test, src_test) # Should throw warning
    # Another test case with stereochemistry ambigouity
    src_test_2 = 'CC(=O)O.CC(=O)[O-].CCCCCCCC/C=C/CCCCl.CCCCCCCC/C=C\CCCCl.[K+]>>CCCCCCCCC=CCCCOC(C)=O'
    tgt_test_2 = 'CC(=O)O.CC(=O)[O-].CCCCCCCCC=CCCCCl.CCCCCCCCC=CCCCCl.[K+]>>CC(=O)O.CCCCCCCCC=CCCCCl.CCCCCCCCC=CCCCOC(C)=O.[Cl-].[K+]'
    test_difference_2 = extract_difference(src_test_2, tgt_test_2, attempt_src_multiplicity_fix=True, verbose=True)
    print(f"Test difference 2: {test_difference_2}")
    # Test one more case with a molecule that has multiplicity of 2 in source and 1 in target, which should be resolved by the function
    src_test_3 = 'CC(=O)O.CC(=O)[O-].CCCCCCCCC=CCCCCl.CCCCCCCCCC=CCCCCl.[K+].[K+]>>CC(=O)O.CCCCCCCCCC=CCCCCl.CCCCCCCCCC=CCCCCl.[K+]'
    tgt_test_3 = 'CC(=O)O.CC(=O)[O-].CCCCCCCCC=CCCCCl.CCCCCCCCCC=CCCCCl.[K+]>>CC(=O)O.CCCCCCCCCC=CCCCCl.CCCCCCCCCC=CCCCCl.[K+]'
    test_difference_3 = extract_difference(src_test_3, tgt_test_3, attempt_src_multiplicity_fix=True, return_harmonized_source=True, verbose=True)
    print(f"Test difference 3: {test_difference_3[0]}, Harmonized source: {test_difference_3[1]}, Report: {test_difference_3[2]}") """

    # test harmonization with example
    src = "CN(C)C=O.Cc1ccc(F)cc1[C@H]1CC[C@H](N)CC1.Cl.O=C(CCCCl)c1ccc(F)cc1.[H-].[I-].[K+].[K].[Na+]>>Cc1ccc(F)cc1[C@H]1CC[C@H](NCCCC(=O)c2ccc(F)cc2)CC1.Cl.Cl"
    tgt = "CN(C)C=O.Cc1ccc(F)cc1[C@H]1CC[C@H](N)CC1.Cl.O=C(CCCCl)c1ccc(F)cc1.[H-].[I-].[K+].[K].[Na+]>>CN(C)C=O.Cc1ccc(F)cc1[C@H]1CC[C@H](NCCCC(=O)c2ccc(F)cc2)CC1.Cl.[Cl-].[H][H].[I-].[K+].[K].[Na+]"

    res = extract_difference(src, tgt, attempt_src_multiplicity_fix=True, return_harmonized_source=True, verbose=True) 
    # try for unique identifier: USPTO_US03960961_1004419

    print(res)

    main()
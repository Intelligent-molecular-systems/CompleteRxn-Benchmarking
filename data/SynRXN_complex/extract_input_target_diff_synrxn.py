"""
Extract the difference between input (rxn) and target (ground_truth) reaction SMILES.

Usage in a pandas DataFrame::

    from extract_input_target_diff_synrxn import extract_difference

    df["input_target_diff"] = df.apply(
        lambda row: extract_difference(row["rxn"], row["ground_truth"]), axis=1
    )
"""

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
    Return the difference between target and source as a reaction SMILES string
    (reactants>reagents>products), containing only the molecules present in
    target but not in source (accounting for multiplicity).
    """
    src_reactants, src_reagents, src_products = src_reaction.split('>')
    target_reactants, target_reagents, target_products = target_reaction.split('>')

    src_r   = src_reactants.split('.')  if src_reactants  else []
    src_rg  = src_reagents.split('.')   if src_reagents   else []
    src_p   = src_products.split('.')   if src_products   else []
    tgt_r   = target_reactants.split('.') if target_reactants else []
    tgt_rg  = target_reagents.split('.')  if target_reagents  else []
    tgt_p   = target_products.split('.')  if target_products  else []

    diff_r  = _expand_counter_in_reference_order(Counter(tgt_r)  - Counter(src_r),  tgt_r)
    diff_rg = _expand_counter_in_reference_order(Counter(tgt_rg) - Counter(src_rg), tgt_rg)
    diff_p  = _expand_counter_in_reference_order(Counter(tgt_p)  - Counter(src_p),  tgt_p)

    return f"{'.' .join(diff_r)}>{'.' .join(diff_rg)}>{'.' .join(diff_p)}"

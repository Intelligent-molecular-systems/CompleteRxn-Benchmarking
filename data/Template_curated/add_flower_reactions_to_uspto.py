#!/usr/bin/env python3
"""
Join FlowER reactions with their one-to-one matched USPTO records.

This script reads the uspto_flower_one_to_one.csv and flower_reagents_identified.tsv,
matches each USPTO reaction to its corresponding FlowER reaction, and adds columns:
- 'rxn_flower': Raw FlowER reaction SMILES
- 'rxn_reagents_identified_flower': FlowER reaction with identified reagents
- 'moved_metadata': Metadata about molecules moved to reagents

"""

import ast
import pandas as pd
from tqdm import tqdm

def add_flower_reactions_to_uspto(uspto_path, flower_path, output_path):
    """
    Add flower-curated reactions to USPTO dataframe.
    
    Args:
        uspto_path (str): Path to uspto_flower_one_to_one.csv
        flower_path (str): Path to flower_reagents_identified.tsv
        output_path (str): Path to save the output CSV
    """
    print("Loading USPTO dataframe...")
    uspto_df = pd.read_csv(uspto_path, low_memory=False)
    uspto_df['FlowER_ID_parsed'] = uspto_df['FlowER_ID_parsed'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    print("Loading FlowER dataframe...")
    flower_df = pd.read_csv(flower_path, sep='\t', low_memory=False)
    
    # Create lookup dictionary for fast O(1) access
    print("Building lookup index...")
    flower_df['reaction_id'] = flower_df['reaction_id'].astype(str)
    
    # Create a composite key (subset, reaction_id) for fast lookup
    flower_lookup = {}
    for idx, row in flower_df.iterrows():
        key = (row['subset'], row['reaction_id'])
        flower_lookup[key] = {
            'rxn': row.get('rxn'),
            'rxn_reagents_identified': row.get('rxn_reagents_identified'),
            'moved_metadata': row.get('moved_metadata')
        }
    
    print(f"Built lookup index with {len(flower_lookup)} entries")
    
    # Create new columns using vectorized lookup
    print("Matching reactions...")
    def lookup_flower_reaction(flower_ids):
        """Extract flower reaction and metadata from parsed IDs."""
        if not flower_ids or len(flower_ids) == 0:
            return None, None, None
        
        try:
            flower_id = flower_ids[0]
            flower_subset, flower_idx = flower_id.split('_')[1], flower_id.split('_')[2]
            
            key = ("flower_" + flower_subset, flower_idx)
            if key in flower_lookup:
                result = flower_lookup[key]
                return result['rxn'], result['rxn_reagents_identified'], result['moved_metadata']
        except Exception as e:
            pass
        
        return None, None, None
    
    # Apply lookup function
    print("Looking up reactions...")
    results = uspto_df['FlowER_ID_parsed'].apply(lookup_flower_reaction)
    uspto_df['rxn_flower'] = results.apply(lambda x: x[0])
    uspto_df['rxn_reagents_identified_flower'] = results.apply(lambda x: x[1])
    uspto_df['moved_metadata'] = results.apply(lambda x: x[2])
    
    # Save to file
    print(f"\nSaving to {output_path}...")
    uspto_df.to_csv(output_path, index=False)
    
    # Print summary
    total = len(uspto_df)
    matched = sum(1 for x in uspto_df['rxn_reagents_identified_flower'] if x is not None)
    print(f"\nSummary:")
    print(f"  Total USPTO reactions: {total}")
    print(f"  Successfully matched to FlowER: {matched} ({100*matched/total:.1f}%)")
    print(f"  Could not match: {total - matched} ({100*(total-matched)/total:.1f}%)")
    
    return uspto_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Add flower-curated reactions to USPTO one-to-one mapping'
    )
    parser.add_argument(
        '--uspto',
        default='data/Template_curated/processed/uspto_flower_one_to_one.csv',
        help='Path to USPTO CSV file (default: data/Template_curated/processed/uspto_flower_one_to_one.csv)'
    )
    parser.add_argument(
        '--flower',
        default='data/Template_curated/processed/flower_reagents_identified.tsv',
        help='Path to FlowER TSV file (default: data/Template_curated/processed/flower_reagents_identified.tsv)'
    )
    parser.add_argument(
        '--output',
        default='data/Template_curated/processed/one_to_one_final.csv',
        help='Output CSV file path (default: data/Template_curated/processed/one_to_one_final.csv)'
    )
    
    args = parser.parse_args()
    
    add_flower_reactions_to_uspto(args.uspto, args.flower, args.output)

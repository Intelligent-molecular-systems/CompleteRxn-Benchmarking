"""
Find matching FlowER and USPTO records and save each match type.

Creates three subsets based on the mapping between USPTO and FlowER records:
1. one_to_one: USPTO records with exactly one FlowER_ID that appears in no other USPTO record
2. many_to_one: Multiple USPTO records mapping to the same single FlowER_ID
3. one_to_many: USPTO records with multiple FlowER_IDs
"""

import pandas as pd
import ast
from pathlib import Path
from collections import Counter


def parse_flower_id_list(flower_id_str):
    """Parse FlowER_ID column from string representation of list."""
    if pd.isna(flower_id_str):
        return []
    try:
        return ast.literal_eval(flower_id_str)
    except:
        return []


def load_and_filter_uspto(uspto_path):
    """Load USPTO data and filter for non-empty FlowER_ID."""
    uspto = pd.read_csv(uspto_path)
    
    # Parse FlowER_ID lists
    uspto['FlowER_ID_parsed'] = uspto['FlowER_ID'].apply(parse_flower_id_list)
    
    # Filter out empty FlowER_ID lists
    uspto_filtered = uspto[uspto['FlowER_ID_parsed'].apply(len) > 0].copy()
    
    print(f"Total USPTO records: {len(uspto)}")
    print(f"USPTO records with FlowER mapping: {len(uspto_filtered)}")
    
    return uspto_filtered


def create_subsets(uspto_filtered):
    """Create three subsets based on mapping patterns."""
    
    # Count how many times each FlowER_ID appears across all USPTO records
    flower_id_counter = Counter()
    for flower_ids in uspto_filtered['FlowER_ID_parsed']:
        for fid in flower_ids:
            flower_id_counter[fid] += 1
    
    # Classify each USPTO record
    one_to_one_records = []
    many_to_one_records = []
    one_to_many_records = []
    
    for idx, row in uspto_filtered.iterrows():
        flower_ids = row['FlowER_ID_parsed']
        
        if len(flower_ids) == 1:
            # Single FlowER_ID
            fid = flower_ids[0]
            if flower_id_counter[fid] == 1:
                # This FlowER_ID appears in only this USPTO record -> 1:1
                one_to_one_records.append(idx)
            else:
                # This FlowER_ID appears in multiple USPTO records -> Many:1
                many_to_one_records.append(idx)
        else:
            # Multiple FlowER_IDs in one USPTO record -> 1:Many
            one_to_many_records.append(idx)
    
    # Create subset dataframes
    one_to_one = uspto_filtered.loc[one_to_one_records].copy()
    many_to_one = uspto_filtered.loc[many_to_one_records].copy()
    one_to_many = uspto_filtered.loc[one_to_many_records].copy()
    
    # Sort many_to_one by FlowER_ID to group records mapping to same FlowER reaction
    if len(many_to_one) > 0:
        many_to_one['FlowER_ID_sort'] = many_to_one['FlowER_ID_parsed'].apply(lambda x: x[0])
        many_to_one = many_to_one.sort_values('FlowER_ID_sort').drop(columns=['FlowER_ID_sort'])
    
    return one_to_one, many_to_one, one_to_many


def main(uspto_path, flower_path, output_dir):
    """Main processing pipeline."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and filter USPTO data
    uspto_filtered = load_and_filter_uspto(uspto_path)
    
    # Create subsets
    one_to_one, many_to_one, one_to_many = create_subsets(uspto_filtered)
    
    # Print statistics
    print(f"\nSubset statistics:")
    print(f"  1:1 matching: {len(one_to_one)} records")
    print(f"  Many:1 matching: {len(many_to_one)} records")
    print(f"  1:Many matching: {len(one_to_many)} records")
    print(f"  Total: {len(one_to_one) + len(many_to_one) + len(one_to_many)}")
    
    # Save subsets
    one_to_one.to_csv(output_path / 'uspto_flower_one_to_one.csv', index=False)
    many_to_one.to_csv(output_path / 'uspto_flower_many_to_one.csv', index=False)
    one_to_many.to_csv(output_path / 'uspto_flower_one_to_many.csv', index=False)
    
    # Save combined filtered dataset
    uspto_filtered.to_csv(output_path / 'uspto_flower_all_matched.csv', index=False)
    
    print(f"\nFiles saved to {output_dir}/")
    print("  - uspto_flower_one_to_one.csv")
    print("  - uspto_flower_many_to_one.csv")
    print("  - uspto_flower_one_to_many.csv")
    print("  - uspto_flower_all_matched.csv")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process USPTO-FlowER matching datasets')
    parser.add_argument('--uspto', required=True, help='Path to USPTO CSV file')
    parser.add_argument('--flower', required=True, help='Path to FlowER TSV file')
    parser.add_argument('--output', default='data/processed', help='Output directory')
    
    args = parser.parse_args()
    
    main(args.uspto, args.flower, args.output)

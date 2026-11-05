"""
Prepare benchmark reactions for SynRBL.
Removes reagents, solvents, catalysts, and conditions from reactions.
"""

import pandas as pd
import re
import sys
from pathlib import Path
from typing import Optional


def remove_reagents_and_conditions(reaction: str) -> str:
    """
    Remove reagents, solvents, catalysts, and conditions from reaction SMILES
    
    Args:
        reaction: Reaction SMILES string (e.g., "reactants>reagents>products")
        
    Returns:
        Cleaned reaction string with only reactants and products (e.g., "reactants>>products")
    """
    # Remove pipe-enclosed information (metadata) first
    reaction = re.sub(r'\|.*?\|', '', reaction)
    
    # Split by '>' and keep only first (reactants) and last (products) parts
    parts = reaction.split('>')
    if len(parts) >= 2:
        reactants = parts[0]
        products = parts[-1]
        reaction = f"{reactants}>>{products}"
    
    return reaction


def preprocess_dataset(
    input_file: str,
    output_file: str,
    reaction_col: str = 'InputReaction',
    input_sep: str = ',',
    output_sep: str = ','
) -> pd.DataFrame:
    """
    Preprocess benchmark dataset for SynRBL
    
    Keeps all original columns and adds:
    - 'Reaction': Cleaned reaction (reagents/conditions removed)
    - 'ExpectedReaction': Empty column for reference (if needed)
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output file (if None, saves with SynRBL_ prefix)
        reaction_col: Name of column containing original reactions
        input_sep: Delimiter for input file (default: ',')
        output_sep: Delimiter for output file (default: ',')
        
    Returns:
        Preprocessed DataFrame
    """
    # Read input data
    data = pd.read_csv(input_file, sep=input_sep)
    
    print(f"Loaded {len(data)} reactions from {input_file}")
    print(f"Columns: {list(data.columns)}")
    
    # Create cleaned reaction column (In case there are still reagents/conditions, this will remove them)
    data['Reaction'] = data[reaction_col].apply(remove_reagents_and_conditions)
    
    # Add empty ExpectedReaction column
    data['ExpectedReaction'] = ''
    
    # Display sample
    print(f"\nSample reactions:")
    print(f"Original: {data[reaction_col].iloc[0]}")
    print(f"Cleaned:  {data['Reaction'].iloc[0]}")

    
    # Save preprocessed data
    data.to_csv(output_file, index=False, sep=output_sep)
    print(f"\nPreprocessed data saved to {output_file}")
    
    return data


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <input_file> <output_file> [--reaction-col COLUMN]")
        print("       [--input-sep SEP] [--output-sep SEP]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Parse optional arguments
    reaction_col = 'InputReaction'
    input_sep = ','
    output_sep = ','
    
    for i, arg in enumerate(sys.argv[2:]):
        if arg == '--reaction-col' and i+3 < len(sys.argv):
            reaction_col = sys.argv[i+3]
        elif arg == '--input-sep' and i+3 < len(sys.argv):
            input_sep = sys.argv[i+3]
        elif arg == '--output-sep' and i+3 < len(sys.argv):
            output_sep = sys.argv[i+3]
    
    preprocess_dataset(
        input_file=input_file,
        output_file=output_file,
        reaction_col=reaction_col,
        input_sep=input_sep,
        output_sep=output_sep
    )

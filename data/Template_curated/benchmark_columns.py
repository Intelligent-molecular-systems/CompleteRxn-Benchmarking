import pandas as pd
import os
from pathlib import Path

# Read the input CSV
input_file = "data/Template_curated/processed/one_to_one_final.csv"
df = pd.read_csv(input_file)

def move_reagents_to_reactants(reaction_smiles):
    """
    Move all reagents (molecules between >) to reactants side.
    Sort the reactant side alphabetically.
    """
    if pd.isna(reaction_smiles):
        return reaction_smiles
    
    # Split reaction into parts: reactants > reagents > products
    parts = reaction_smiles.split(">")
    
    if len(parts) != 3:
        return reaction_smiles
    
    reactants = parts[0]
    reagents = parts[1]
    products = parts[2]
    
    # Combine reactants and reagents
    if reagents.strip():
        # Parse molecules from both sides
        reactant_mols = [mol.strip() for mol in reactants.split(".") if mol.strip()]
        reagent_mols = [mol.strip() for mol in reagents.split(".") if mol.strip()]
        
        # Combine and sort (also solves the problem that the order might give away what are reagents vs reactants)
        all_reactants = reactant_mols + reagent_mols
        all_reactants.sort()
        
        # Reconstruct the reaction
        new_reactants = ".".join(all_reactants)
        return f"{new_reactants}>>{products}"
    else:
        return reaction_smiles

# Apply the transformation
df["InputReaction"] = df["CanonicalOriginalReaction"].apply(move_reagents_to_reactants)
df["TargetReaction"] = df["rxn_flower"].copy()  # the flower dataset reactions are balanced
# Create benchmark directory if it doesn't exist
os.makedirs("data/Template_curated/benchmark", exist_ok=True)

# Save the processed dataframe
output_file = "data/Template_curated/benchmark/benchmark.csv"
df.to_csv(output_file, index=False)

print(f"Benchmark saved to {output_file}")

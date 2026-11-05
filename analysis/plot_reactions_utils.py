#!/usr/bin/env python3
"""
Utility functions for plotting chemical reactions using RDKit.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import re
import math


def _combine_mols(mols):
    """Combine multiple molecule objects into one."""
    mols = [m for m in mols if m is not None]
    if not mols:
        return None
    combo = mols[0]
    for m in mols[1:]:
        combo = Chem.CombineMols(combo, m)
    return combo


def _merge_agents_into_one(rxn):
    """Merge all agent molecules in a reaction into a single agent."""
    agents = list(rxn.GetAgents())
    if len(agents) <= 1:
        return rxn
    combined = _combine_mols(agents)
    if combined is None:
        return rxn
    rdDepictor.Compute2DCoords(combined)
    rxn2 = rdChemReactions.ChemicalReaction(rxn)
    rxn2.RemoveAgentTemplates()
    rxn2.AddAgentTemplate(combined)
    return rxn2


def plot_reaction(rxn_smiles, width=1600, height=400):
    """Generate SVG depiction of a reaction from its SMILES representation.
    
    Args:
        rxn_smiles (str): Reaction SMILES string
        width (int): Width of the SVG
        height (int): Height of the SVG
    
    Returns:
        str: SVG string of the reaction
    """
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)
    rxn = _merge_agents_into_one(rxn)

    d2d = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = d2d.drawOptions()
    opts.scaleBondWidth = True
    opts.bondLineWidth = 2
    opts.fixedBondLength = 30

    d2d.DrawReaction(rxn)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()

    return svg


def extract_svg_inner_content(svg_text):
    """Extract the inner content of an SVG, removing XML declaration and outer <svg> tags.
    
    Args:
        svg_text (str): Full SVG string
    
    Returns:
        str: Inner SVG content without outer tags
    """
    # Remove XML declaration
    svg_text = re.sub(r'<\?xml[^?]*\?>', '', svg_text).strip()
    
    # Remove the opening <svg ...> tag and closing </svg> tag
    svg_text = re.sub(r'^<svg[^>]*>', '', svg_text, flags=re.MULTILINE)
    svg_text = re.sub(r'</svg>\s*$', '', svg_text, flags=re.MULTILINE)
    
    # Clean up namespace prefixes that may be added during XML parsing
    svg_text = re.sub(r'ns\d+:', '', svg_text)
    
    return svg_text.strip()


def load_reaction_data(csv_path):
    """Load reaction data from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    return pd.read_csv(csv_path, low_memory=False)


def plot_reaction_grid(reaction_data, output_path, max_reactions=None, ncols=2, 
                      reaction_width=800, reaction_height=400, padding=30, label_height=20):
    """Plot multiple reactions in a grid layout.
    
    Args:
        reaction_data (list): List of tuples (rxn_smiles, label) where label is the reaction title/identifier
        output_path (str): Path to save the output SVG file
        max_reactions (int, optional): Maximum number of reactions to plot. If None, plot all.
        ncols (int): Number of columns in the grid
        reaction_width (int): Width of each reaction in pixels
        reaction_height (int): Height of each reaction in pixels
        padding (int): Padding between reactions in pixels
        label_height (int): Height reserved for labels in pixels
    
    Returns:
        None
    """
    if max_reactions:
        reaction_data = reaction_data[:max_reactions]
    
    n_reactions = len(reaction_data)
    if n_reactions == 0:
        print("No reactions to plot")
        return
        
    nrows = math.ceil(n_reactions / ncols)
    
    # Generate SVGs for all reactions
    svg_contents = []
    for rxn_smiles, label in reaction_data:
        try:
            svg = plot_reaction(rxn_smiles)
            svg_content = extract_svg_inner_content(svg)
            svg_contents.append((svg_content, label))
        except Exception as e:
            print(f"Error plotting reaction '{label}': {e}")
            svg_contents.append((None, label))
    
    # Calculate dimensions
    outer_padding = 10  # Minimal outer padding
    inner_padding = padding  # Padding between reactions
    total_width = ncols * reaction_width + (ncols - 1) * inner_padding + 2 * outer_padding
    total_height = nrows * (reaction_height + label_height) + (nrows - 1) * inner_padding + 2 * outer_padding
    
    # Build combined SVG
    svg_parts = [f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">']
    
    for idx, (svg_content, label) in enumerate(svg_contents):
        row = idx // ncols
        col = idx % ncols
        
        x = col * (reaction_width + inner_padding) + outer_padding
        y = row * (reaction_height + label_height + inner_padding) + outer_padding
        
        # Add label
        svg_parts.append(f'<text x="{x}" y="{y}" font-size="14" font-family="Arial" font-weight="bold">{label}</text>')
        
        # Add reaction
        if svg_content:
            svg_parts.append(f'''<svg x="{x}" y="{y + label_height}" width="{reaction_width}" height="{reaction_height}" 
                viewBox="0 0 1600 400" preserveAspectRatio="xMidYMid meet">
                {svg_content}
            </svg>''')
        else:
            svg_parts.append(f'<text x="{x}" y="{y + label_height + 200}" font-size="16" fill="red">Error rendering reaction</text>')
    
    svg_parts.append('</svg>')
    
    combined_svg = '\n'.join(svg_parts)
    with open(output_path, 'w') as f:
        f.write(combined_svg)
    
    print(f"Saved {n_reactions} reactions to {output_path}")


def create_combined_svg(svg_contents_with_labels, output_path, width=1700, height=900, 
                       reaction_height=400, padding=50):
    """Create a combined SVG with multiple labeled reactions stacked vertically.
    
    Args:
        svg_contents_with_labels (list): List of tuples (svg_content, label)
        output_path (str): Path to save the output SVG file
        width (int): Total width of the combined SVG
        height (int): Total height of the combined SVG
        reaction_height (int): Height allocated for each reaction
        padding (int): Padding between elements
    
    Returns:
        None
    """
    svg_parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    
    y_offset = padding
    for svg_content, label in svg_contents_with_labels:
        svg_parts.append(f'<g transform="translate({padding},{y_offset})">')
        svg_parts.append(f'<text x="0" y="0" font-size="24" font-family="Arial">{label}</text>')
        svg_parts.append(f'<g transform="translate(0,30)" width="1600" height="{reaction_height}">')
        svg_parts.append(svg_content)
        svg_parts.append('</g>')
        svg_parts.append('</g>')
        y_offset += reaction_height + 80
    
    svg_parts.append('</svg>')
    
    combined_svg = '\n'.join(svg_parts)
    with open(output_path, 'w') as f:
        f.write(combined_svg)
    
    print(f"Saved combined reactions to {output_path}")

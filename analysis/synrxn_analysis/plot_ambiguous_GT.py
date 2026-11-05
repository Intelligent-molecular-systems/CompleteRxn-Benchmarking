#!/usr/bin/env python3
"""Draw the original SynRXN review cases that still do not match the reference."""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit.Chem import Draw, rdChemReactions

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / "results/additional/synrxn_analysis"
ARCHIVE = ROOT / "results/archive/metric_v1/additional/synrxn_analysis"


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        spec = importlib.util.find_spec("matplotlib")
        if spec and spec.origin:
            path = Path(spec.origin).parent/"mpl-data/fonts/ttf/DejaVuSans.ttf"
            if path.is_file():
                return ImageFont.truetype(str(path), size)
        return ImageFont.load_default()


def reaction_panel(smiles, title):
    reaction = rdChemReactions.ReactionFromSmarts(str(smiles), useSmiles=True)
    if reaction is None:
        raise ValueError(f"RDKit could not parse {title}: {smiles}")
    drawing = Draw.ReactionToImage(reaction, subImgSize=(250, 190))
    panel = Image.new("RGB", (max(1100, drawing.width), drawing.height + 48), "white")
    panel.paste(drawing, (0, 48))
    ImageDraw.Draw(panel).text((15, 8), title, fill="black", font=font(22))
    return panel


def case_figure(reaction_id, row, judgment):
    panels = [reaction_panel(row[column], title) for column, title in (
        ("input_reaction", "Incomplete input"), ("ground_truth", "SynRXN reference"),
        ("completed_prediction", "CRB completion"))]
    figure = Image.new("RGB", (max(p.width for p in panels),
                               100 + sum(p.height + 12 for p in panels)), "white")
    draw = ImageDraw.Draw(figure)
    draw.text((15, 10), f"{reaction_id} | Existing judgment: {judgment}", fill="black", font=font(28))
    y = 100
    for panel in panels:
        figure.paste(panel, (0, y))
        y += panel.height + 12
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", help="Explicit historical audit IDs; default: cases still unmatched by the benchmark metric")
    parser.add_argument("--data", type=Path, default=CURRENT/"per_reaction.csv")
    parser.add_argument("--audit", type=Path, default=ARCHIVE/"audit_sample.csv")
    parser.add_argument("--output-dir", type=Path, default=CURRENT/"review")
    parser.add_argument("--output", type=Path, help="Optional combined PNG, in addition to individual plots")
    args = parser.parse_args()
    current = pd.read_csv(args.data).set_index("r_id", verify_integrity=True)
    audit = pd.read_csv(args.audit).set_index("r_id", verify_integrity=True)
    if not audit.index.isin(current.index).all():
        parser.error("Some historical audit IDs are missing from current scores")
    status = audit[["audit_label", "audit_note", "target_status"]].join(
        current[["prediction_balanced", "target_balanced", "equivalent_match"]], validate="one_to_one")
    # Require actual boolean CSV columns: strings such as 'False' must not count as True.
    for column in ("prediction_balanced", "target_balanced", "equivalent_match"):
        if status[column].dtype != bool:
            parser.error(f"Expected boolean column: {column}")
    remaining = status.prediction_balanced & status.target_balanced & ~status.equivalent_match
    ids = args.ids if args.ids else status.index[remaining].tolist()
    if not ids or not set(ids).issubset(status.index):
        parser.error("Choose at least one ID from the historical audit")
    figures = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for reaction_id in ids:
        figure = case_figure(reaction_id, current.loc[reaction_id], status.loc[reaction_id, "audit_label"])
        figure.save(args.output_dir/f"{reaction_id}.png")
        figures.append(figure)
    status.to_csv(args.output_dir/"audit_status.csv")
    if args.output:
        combined = Image.new("RGB", (max(p.width for p in figures), sum(p.height for p in figures)), "white")
        y = 0
        for figure in figures:
            combined.paste(figure, (0, y))
            y += figure.height
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined.save(args.output)
    print(f"Saved {len(figures)} cases and the 12-case status table to {args.output_dir}")


if __name__ == "__main__":
    main()

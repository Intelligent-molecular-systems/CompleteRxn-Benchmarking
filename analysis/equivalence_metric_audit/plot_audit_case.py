#!/usr/bin/env python3
"""Draw one review case, hiding which prediction came from which source by default."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import RDLogger
from rdkit.Chem import Draw, rdChemReactions


HERE = Path(__file__).resolve().parents[2] / "results/archive/metric_v1/additional/equivalence_metric_audit"
KEY = HERE / "audit_key.csv"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def panel(reaction_smiles, title, high_resolution=False):
    reaction = rdChemReactions.ReactionFromSmarts(reaction_smiles, useSmiles=True)
    if reaction is None:
        raise ValueError("Could not parse {}".format(title))
    size = (440, 300) if high_resolution else (220, 150)
    header = 68 if high_resolution else 34
    font = (
        ImageFont.truetype(str(FONT), 30)
        if high_resolution
        else ImageFont.load_default()
    )
    image = Draw.ReactionToImage(reaction, subImgSize=size).convert("RGB")
    out = Image.new("RGB", (image.width, image.height + header), "white")
    out.paste(image, (0, header))
    ImageDraw.Draw(out).text(
        (16 if high_resolution else 8, 14 if high_resolution else 8),
        title,
        fill="black",
        font=font,
    )
    return out


def stack(panels):
    width = max(image.width for image in panels)
    height = sum(image.height for image in panels)
    out = Image.new("RGB", (width, height), "white")
    y = 0
    for image in panels:
        out.paste(image, (0, y))
        y += image.height
    return out


def load_case(path, audit_id):
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["audit_id"]) == audit_id:
                return row
    raise KeyError("audit_id {} not found".format(audit_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-id", required=True, type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reveal-labels",
        action="store_true",
        help="Reveal which completion is FlowER and which is SynRBL",
    )
    parser.add_argument(
        "--high-resolution",
        action="store_true",
        help="Render at twice the default structural and label resolution",
    )
    args = parser.parse_args()

    audit_row = load_case(HERE / "audit_sample_blinded.csv", args.audit_id)
    prefix = "Audit {:02d} - ".format(args.audit_id)
    images = [
        panel(
            audit_row["input_reaction"],
            prefix + "Incomplete input",
            args.high_resolution,
        )
    ]
    if args.reveal_labels:
        row = load_case(KEY, args.audit_id)
        images.extend([
            panel(
                row["flower_target"],
                prefix + "FlowER completion",
                args.high_resolution,
            ),
            panel(
                row["synrbl_completion"],
                prefix + "SynRBL completion",
                args.high_resolution,
            ),
        ])
        suffix = "revealed"
    else:
        images.extend([
            panel(
                audit_row["completion_a"],
                prefix + "Completion A",
                args.high_resolution,
            ),
            panel(
                audit_row["completion_b"],
                prefix + "Completion B",
                args.high_resolution,
            ),
        ])
        suffix = "blinded"

    output = args.output or HERE / "plots" / "audit_{}_{}.png".format(args.audit_id, suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    stack(images).save(output, dpi=(200, 200))
    print("Saved {}".format(output))


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    main()

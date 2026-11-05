"""Draw a reaction SMILES as a wrapped scheme with one bond length for all molecules.

Identical molecules on one side are drawn once with a coefficient. Agents are
drawn smaller above the arrow. Molecules listed in ``shade`` (canonical SMILES
per side) get a coloured background, e.g. to mark differences from a reference.
"""

from collections import Counter
from io import BytesIO

import numpy as np

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


BOND = 50            # pixels per bond
MAX_WIDTH = 2600     # wrap lines wider than this
GAP = 18             # space around operators
SHADE = (1.0, 0.86, 0.66, 1.0)
AGENT_SCALE = 0.6

rdDepictor.SetPreferCoordGen(True)


def molecule_key(smiles, stereo=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid molecule: {!r}".format(smiles))
    if not stereo:
        Chem.RemoveStereochemistry(mol)
    if all(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        mol = Chem.AddHs(mol)  # [HH] and [H][H] are the same molecule
    return Chem.MolToSmiles(mol)


def split_reaction(reaction):
    sides = reaction.strip().split(">")
    if len(sides) != 3:
        raise ValueError("Expected reactants>agents>products: {!r}".format(reaction))
    return [[s for s in side.split(".") if s] for side in sides]


def side_counts(molecules, stereo=True):
    return Counter(molecule_key(s, stereo) for s in molecules)


def _font(size):
    return ImageFont.load_default(size=size)


def _molecule(smiles, bond, shaded):
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    points = mol.GetConformer().GetPositions()
    if np.isnan(points).any():  # CoordGen fails on H-only species such as [H][H]
        rdDepictor.SetPreferCoordGen(False)
        rdDepictor.Compute2DCoords(mol)
        rdDepictor.SetPreferCoordGen(True)
        points = mol.GetConformer().GetPositions()
    scale = bond / 1.5  # RDKit depictions use a bond length of 1.5
    margin = int(0.9 * bond)
    width = int(np.ptp(points[:, 0]) * scale) + 2 * margin
    height = int(np.ptp(points[:, 1]) * scale) + 2 * margin
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.fixedBondLength = bond
    options.padding = 0.0
    options.scalingFactor = scale
    options.fixedFontSize = int(0.55 * bond)
    if shaded:
        options.setBackgroundColour(SHADE)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")


def _text(text, size):
    font = _font(size)
    box = font.getbbox(text)
    image = Image.new("RGB", (box[2] - box[0] + 2 * GAP, size * 2), "white")
    ImageDraw.Draw(image).text((GAP - box[0], size // 2 - box[1] // 2), text, fill="black", font=font)
    return image


def _hstack(images, gap=0):
    width = sum(i.width for i in images) + gap * max(len(images) - 1, 0)
    height = max(i.height for i in images)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        out.paste(image, (x, (height - image.height) // 2))
        x += image.width + gap
    return out


def _arrow(agents):
    length = max(4 * BOND, (agents.width if agents else 0) + 2 * GAP)
    top = agents.height + 8 if agents else 2 * BOND
    image = Image.new("RGB", (length + 2 * GAP, 2 * top), "white")
    draw = ImageDraw.Draw(image)
    y = top
    draw.line((GAP, y, GAP + length - 20, y), fill="black", width=5)
    draw.polygon([(GAP + length, y), (GAP + length - 28, y - 12), (GAP + length - 28, y + 12)], fill="black")
    if agents:
        image.paste(agents, (GAP + (length - agents.width) // 2, 0))
    return image


def _side(molecules, shade, bond, order=None):
    counts = Counter(molecules)
    unique = list(dict.fromkeys(molecules))
    if order:
        # Shared molecules follow the reference order; others go last.
        rank = {key: i for i, key in enumerate(order)}
        unique.sort(key=lambda s: rank.get(molecule_key(s), len(rank)))
    items = []
    for smiles in unique:
        picture = _molecule(smiles, bond, molecule_key(smiles) in shade)
        if counts[smiles] > 1:
            picture = _hstack([_text(str(counts[smiles]), int(0.9 * bond)), picture])
        items.append(picture)
    return items


def draw_reaction(reaction, shade=((), ()), reference=None, max_width=MAX_WIDTH):
    """Return a PIL image.

    ``shade`` holds canonical SMILES to highlight per side. With a ``reference``
    reaction, molecules shared with it are drawn in the reference's order.
    """
    reactants, agents, products = split_reaction(reaction)
    order = (None, None)
    if reference:
        ref_reactants, _, ref_products = split_reaction(reference)
        order = ([molecule_key(s) for s in ref_reactants], [molecule_key(s) for s in ref_products])
    left = _side(reactants, set(shade[0]), BOND, order[0])
    right = _side(products, set(shade[1]), BOND, order[1])
    agent_row = None
    if agents:
        agent_row = _hstack(_side(agents, set(), int(AGENT_SCALE * BOND)), gap=GAP)

    plus = _text("+", BOND)
    tokens = []
    for i, item in enumerate(left):
        tokens += ([plus] if i else []) + [item]
    tokens.append(_arrow(agent_row))
    for i, item in enumerate(right):
        tokens += ([plus] if i else []) + [item]

    lines, line = [], []
    for token in tokens:
        if line and sum(t.width for t in line) + token.width > max_width and token is not plus:
            lines.append(line)
            line = []
        line.append(token)
    lines.append(line)
    rows = [_hstack(l) for l in lines]
    width = max(r.width for r in rows)
    out = Image.new("RGB", (width, sum(r.height for r in rows) + GAP * (len(rows) - 1)), "white")
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height + GAP
    return out


def differences(reaction, others, stereo=True):
    """Molecules of ``reaction`` absent from the same side of any reaction in ``others``."""
    reactants, _, products = split_reaction(reaction)
    own = side_counts(reactants, stereo), side_counts(products, stereo)
    shade = (set(), set())
    for other in others:
        other_reactants, _, other_products = split_reaction(other)
        theirs = side_counts(other_reactants, stereo), side_counts(other_products, stereo)
        for i in (0, 1):
            shade[i].update(own[i] - theirs[i])
    if not stereo:
        # Shading looks molecules up by their stereo-aware key.
        lookup = [{molecule_key(s, False): molecule_key(s) for s in side}
                  for side in (reactants, products)]
        shade = tuple({lookup[i][k] for k in shade[i]} for i in (0, 1))
    return shade

"""Draw reaction comparisons and failure-count charts as SVG images."""
from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D

NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)


def _element(parent,tag,**attributes):
    return ET.SubElement(parent,f'{{{NS}}}{tag}',{k.replace('_','-'):str(v) for k,v in attributes.items()})


@dataclass(frozen=True)
class ReactionView:
    svg: str

    def _repr_svg_(self):return self.svg

    def save(self,path):
        path=Path(path)
        if path.suffix.lower()!='.svg':raise ValueError('Save reaction views with an .svg extension')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as handle:handle.write(self.svg)


def _draw(reaction,width,height):
    sides=re.split(r'(?<!-)>',reaction.strip())
    if len(sides)!=3:raise ValueError('Expected reactants>agents>products')
    rxn=rdChemReactions.ChemicalReaction()
    with rdBase.BlockLogs():
        for text,add in zip(sides,(rxn.AddReactantTemplate,rxn.AddAgentTemplate,rxn.AddProductTemplate)):
            for smiles in filter(None,text.split('.')):
                molecule=Chem.MolFromSmiles(smiles)
                if molecule is None:raise ValueError('Invalid molecule')
                for atom in molecule.GetAtoms():atom.SetAtomMapNum(0)
                add(molecule)
    if not rxn.GetNumReactantTemplates() and not rxn.GetNumProductTemplates():
        raise ValueError('Empty reaction')
    drawer=rdMolDraw2D.MolDraw2DSVG(width,height)
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()
    return ET.fromstring(drawer.GetDrawingText())


def compare_reactions(rows,*,title='',width=1000,panel_height=220):
    """Stack labelled reaction panels. Show a message for undrawable predictions."""
    root=ET.Element(f'{{{NS}}}svg',dict(width=str(width),height=str(40+len(rows)*(panel_height+30)),
                                      viewBox=f'0 0 {width} {40+len(rows)*(panel_height+30)}'))
    _element(root,'rect',width='100%',height='100%',fill='white')
    _element(root,'text',x=12,y=24,font_size=16).text=title
    for i,(label,reaction) in enumerate(rows):
        y=40+i*(panel_height+30)
        _element(root,'text',x=12,y=y+20,font_size=15).text=label
        try:
            drawing=_draw(reaction,width,panel_height)
            drawing.set('y',str(y+30));root.append(drawing)
        except (ValueError,RuntimeError):
            _element(root,'text',x=12,y=y+60,font_size=13,fill='#9c2f2f').text=(
                'No completed reaction' if not reaction else 'Cannot draw this reaction: '+reaction[:180])
    return ReactionView(ET.tostring(root,encoding='unicode'))


def plot_reaction(reaction,*,width=1000,height=220):
    """Draw one completed reaction. Invalid input raises ValueError."""
    return ReactionView(ET.tostring(_draw(reaction,width,height),encoding='unicode'))


def failure_bars(counts):
    """Plot automated failure counts. The categories are not chemical diagnoses."""
    rows=sorted(counts.items(),key=lambda item:(-item[1],item[0]))
    height=65+35*max(1,len(rows));width=760
    root=ET.Element(f'{{{NS}}}svg',dict(width=str(width),height=str(height),viewBox=f'0 0 {width} {height}'))
    _element(root,'rect',width='100%',height='100%',fill='white')
    _element(root,'text',x=12,y=25,font_size=16).text='Failures by evaluation category'
    maximum=max(counts.values(),default=1)
    if not rows:_element(root,'text',x=12,y=55,font_size=14).text='No reference-negative predictions'
    for i,(label,count) in enumerate(rows):
        y=45+35*i
        _element(root,'text',x=12,y=y+16,font_size=13).text=label.replace('_',' ')
        _element(root,'rect',x=180,y=y,width=480*count/maximum,height=22,fill='#487fa7')
        _element(root,'text',x=670,y=y+16,font_size=13).text=str(count)
    return ReactionView(ET.tostring(root,encoding='unicode'))

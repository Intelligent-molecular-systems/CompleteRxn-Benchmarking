#!/usr/bin/env python3
"""Compare synthetic omissions with observed omissions and check target selection bias."""
from __future__ import annotations

import argparse, json, random, re, sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from utils.chemistry import canonicalize_smiles, proton_transfer_identity
HELPERS = ["CC(=O)O","O","[H][H]","O=O","N","N#N","[Na+]","[K+]","O=[N+]=O",
 "O=S(O)[O-]","C","Cl","Br","I","F","P","S","O=P(O)O","O=S(=O)(O)O",
 "O=N(=O)O","O=Cl(=O)O","CO","CCO","CCCO","CCCCO","CCC(=O)O",
 "CCCC(=O)O","COCC(=O)O","c1ccccc1","Cc1ccccc1","Oc1ccccc1","Clc1ccccc1"]
FORMULA = re.compile(r"(\d*)([A-Z][a-z]?)")
BALANCE = re.compile(r"\$L:?\s*(.*?)\s*\$R:?\s*(.*?)\s*\$Charge:?")

def parts(x):
    return [] if pd.isna(x) or not str(x).strip() else str(x).split(".")

def split(x):
    a,b,c = str(x).split(">")
    return parts(a),parts(b),parts(c)

def join(r,g,p): return f"{'.'.join(r)}>{'.'.join(g)}>{'.'.join(p)}"

@lru_cache(maxsize=500_000)
def prop(s):
    m=Chem.MolFromSmiles(s)
    if m is None: raise ValueError(s)
    for a in m.GetAtoms(): a.SetAtomMapNum(0)
    ident=Chem.MolToSmiles(m,canonical=True,isomericSmiles=False)
    atoms=Counter(a.GetSymbol() for a in Chem.AddHs(m).GetAtoms())
    carbon=sum(a.GetAtomicNum()==6 for a in m.GetAtoms())
    kind="helper" if ident in HELPER_IDS else ("carbon" if carbon else "noncarbon")
    return ident,atoms,sum(atoms.values()),kind

def ident(s):
    m=Chem.MolFromSmiles(s)
    if m is None: raise ValueError(s)
    for a in m.GetAtoms(): a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m,canonical=True,isomericSmiles=False)

HELPER_IDS=frozenset(ident(s) for s in HELPERS)

@lru_cache(maxsize=500_000)
def proton_key(s):
    c=canonicalize_smiles(s)
    try:
        return canonicalize_smiles(proton_transfer_identity(c))
    except Exception:
        # Some valid dataset species cannot be processed by RDKit ChargeParent
        # (for example, unusual hypervalent phosphorus representations). Keep
        # their ordinary canonical identity so they are never rewritten.
        return c

def identify_recovered_reagents(rxn,uspto_reagents):
    """Remove recovered proton variants anchored to identified USPTO reagents."""
    r,g,p=split(rxn); used_r=set(); used_p=set(); moved=[]
    reagent_keys={proton_key(x) for x in uspto_reagents}
    slots=defaultdict(list)
    for j,x in enumerate(p): slots[proton_key(x)].append(j)
    # Prefer an already identified FlowER reagent; otherwise move the matching
    # FlowER reactant into its reagent field. Each molecular occurrence is used once.
    for field,i,x in ([('reagent',i,x) for i,x in enumerate(g)]+
                      [('reactant',i,x) for i,x in enumerate(r)]):
        k=proton_key(x); candidates=slots.get(k,[])
        if not candidates or k not in reagent_keys: continue
        j=next((j for j in candidates if prop(x)[0]!=prop(p[j])[0]),None)
        if j is None: continue
        candidates.remove(j)
        if field=='reactant': used_r.add(i)
        used_p.add(j)
        moved.append({"source_field":field,"reactant":x,"product":p[j],
                      "reason":"protonation_equivalent_recovered"})
    cleaned=join([x for i,x in enumerate(r) if i not in used_r],
                 g+[r[i] for i in sorted(used_r)],
                 [x for i,x in enumerate(p) if i not in used_p])
    return cleaned,moved

def key(c): return tuple(sorted((e,int(n)) for e,n in c.items() if n))

def formula(x):
    out=Counter()
    for n,e in FORMULA.findall(str(x)): out[e]+=int(n or 1)
    return out

def balance(x):
    if pd.isna(x) or not str(x).strip(): return Counter(),Counter()
    m=BALANCE.search(str(x))
    if not m: raise ValueError(x)
    return formula(m.group(1)),formula(m.group(2))

def helper_lookup():
    d=defaultdict(set)
    for s in HELPERS:
        a=prop(s)[1]
        for n in range(1,7): d[key(Counter({e:v*n for e,v in a.items()}))].add(ident(s))
    return d

HLOOK=helper_lookup()

def helper_match(left,right):
    if left: return "left_deficit",""
    hit=sorted(HLOOK.get(key(right),set()))
    return (("unique_helper",hit[0]) if len(hit)==1 else
            ("ambiguous_helper","|".join(hit)) if hit else ("no_helper",""))

def missing(source,target,keyfn=lambda x:prop(x)[0]):
    have=Counter(keyfn(s) for s in source); out=[]
    for s in target:
        i=keyfn(s)
        if have[i]: have[i]-=1
        else: out.append(s)
    return out

def atom_sum(xs):
    out=Counter()
    for x in xs: out.update(prop(x)[1])
    return out

def net_atoms(added,removed):
    a,r=atom_sum(added),atom_sum(removed)
    return {e:a[e]-r[e] for e in set(a)|set(r) if a[e]-r[e]}

def side(nl,nr):
    return "both" if nl and nr else "left" if nl else "right" if nr else "none"

def category(xs):
    k={prop(x)[3] for x in xs}
    return "+".join(x for x in ("helper","noncarbon","carbon") if x in k) or "none"

class Stats:
    def __init__(self): self.n=0; self.count=Counter(); self.side=Counter(); self.kind=Counter(); self.cat=Counter(); self.atoms=Counter()
    def add(self,l,r):
        x=l+r; self.n+=1; self.count[len(x)]+=1; self.side[side(len(l),len(r))]+=1; self.cat[category(x)]+=1
        self.atoms[sum(prop(i)[2] for i in x)]+=1
        for i in x: self.kind[prop(i)[3]]+=1

def q(c,p):
    n=sum(c.values()); target=p*(n-1); seen=0
    for v in sorted(c):
        seen+=c[v]
        if seen>target:return v
    return np.nan

def add(rows,analysis,pool,dimension,values,denom=None):
    if isinstance(values,Counter):
        total=denom if denom is not None else sum(values.values())
        for k,v in values.items(): rows.append([analysis,pool,dimension,str(k),v,100*v/total if total else np.nan])
    else: rows.append([analysis,pool,dimension,"",values,np.nan])

def van_subset(weights,rng):
    cap=sum(weights)/2; n=len(weights)
    while True:
        mask=rng.getrandbits(n)
        chosen=[i for i in range(n) if mask>>i&1]
        if chosen and sum(weights[i] for i in chosen)<=cap:return chosen

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=42); p.add_argument("--limit",type=int)
    p.add_argument("--benchmark",default="data/Template_curated/benchmark/benchmark_final.csv")
    p.add_argument("--tpl",default="data/USPTO_1k_TPL/uspto_1k_TPL_full_enhanced.csv")
    p.add_argument("--out",default="results/additional/bias_analysis"); a=p.parse_args()
    out=ROOT/a.out; out.mkdir(parents=True,exist_ok=True); rng=random.Random(a.seed)
    stats={x:Stats() for x in ("CompleteRXN empirical","ChemMLM/ChemBalancer","van Wijngaarden")}
    groups=defaultdict(lambda:{"n":0,"helper":Counter(),"missing":Counter(),"atoms":Counter(),"helper_atoms":Counter(),"exact_atoms":0,"net_atoms":0})
    use=["Unique_identifier","1kTPLLabel","Balance","CanonicalReaction",
         "rxn_reagents_identified_flower"]
    df=pd.read_csv(ROOT/a.benchmark,usecols=use,nrows=a.limit,low_memory=False)
    df["template_group"]=np.where(
        pd.to_numeric(df["1kTPLLabel"],errors="coerce").notna(),
        "1kTPL-labelled","No 1kTPL label")
    reagent_audit=Counter(); derived=[]
    for row in tqdm(df.itertuples(index=False),total=len(df),desc="CompleteRXN"):
        sl,sg,sr=split(row.CanonicalReaction)
        cleaned,moved=identify_recovered_reagents(
            row.rxn_reagents_identified_flower,sg)
        tl,tg,tr=split(cleaned); known=sl+sg
        ml,mr=missing(known,tl,proton_key),missing(sr,tr,proton_key)
        rl,rr=missing(tl,sl,proton_key),missing(tr,sr,proton_key)
        restored=missing(known,tg,proton_key)
        proton_moved=[x for x in moved if x["reason"].startswith("protonation")]
        reagent_audit["reactions"]+=1
        reagent_audit["target_reagent_occurrences"]+=len(tg)
        reagent_audit["restored_reagent_occurrences"]+=len(restored)
        reagent_audit["reactions_with_restored_reagent"]+=bool(restored)
        reagent_audit["proton_recovered_occurrences"]+=len(proton_moved)
        reagent_audit["reactions_with_proton_recovered"]+=bool(proton_moved)
        derived.append([str(row.Unique_identifier),row.rxn_reagents_identified_flower,
                        cleaned,json.dumps(moved,separators=(",",":")),
                        len(proton_moved),len(restored),".".join(restored)])
        stats["CompleteRXN empirical"].add(ml,mr)
        target=tl+tr
        for i,x in enumerate(target):
            stats["ChemMLM/ChemBalancer"].add([x] if i<len(tl) else [],[x] if i>=len(tl) else [])
        if target:
            idx=van_subset([prop(x)[2] for x in target],rng)
            stats["van Wijngaarden"].add([target[i] for i in idx if i<len(tl)],[target[i] for i in idx if i>=len(tl)])
        left,right=balance(row.Balance); status,_=helper_match(left,right)
        for g in (row.template_group,"All CompleteRXN"):
            z=groups[g]; z["n"]+=1; z["helper"][status]+=1; z["missing"][len(ml)+len(mr)]+=1
            z["atoms"][sum(prop(x)[2] for x in ml+mr)]+=1
            if status=="unique_helper": z["helper_atoms"][sum(right.values())]+=1
            z["exact_atoms"]+=atom_sum(ml)==left and atom_sum(mr)==right
            z["net_atoms"]+=net_atoms(ml,rl)==dict(left) and net_atoms(mr,rr)==dict(right)
    rows=[]
    for name,s in stats.items():
        add(rows,"corruption",name,"tasks",s.n); add(rows,"corruption",name,"missing_molecule_count",s.count)
        add(rows,"corruption",name,"missing_side",s.side); add(rows,"corruption",name,"missing_molecule_type",s.kind)
        add(rows,"corruption",name,"missing_type_combination",s.cat)
        add(rows,"corruption",name,"median_missing_atoms",q(s.atoms,.5)); add(rows,"corruption",name,"q95_missing_atoms",q(s.atoms,.95))
    for name,z in groups.items():
        add(rows,"construction",name,"reactions",z["n"]); add(rows,"construction",name,"helper_balance_status",z["helper"],z["n"])
        add(rows,"construction",name,"missing_molecule_count",z["missing"],z["n"])
        add(rows,"construction",name,"actual_added_molecules_exactly_match_balance_pct",100*z["exact_atoms"]/z["n"])
        add(rows,"construction",name,"net_molecular_change_matches_balance_pct",100*z["net_atoms"]/z["n"])
        for pctl in (.5,.9,.95,.99):
            add(rows,"construction",name,f"q{int(pctl*100)}_actual_missing_atoms",q(z["atoms"],pctl))
            add(rows,"construction",name,f"q{int(pctl*100)}_unique_helper_atoms",q(z["helper_atoms"],pctl))
    for metric,value in reagent_audit.items():
        add(rows,"reagent_audit","All CompleteRXN",metric,value)
    label_source=df if a.limit is None else pd.read_csv(ROOT/a.benchmark,usecols=["1kTPLLabel"])
    labels=set(pd.to_numeric(label_source["1kTPLLabel"],errors="coerce").dropna().astype(int))
    tpl=pd.read_csv(ROOT/a.tpl,usecols=["labels","balance_diff"],low_memory=False)
    helper_labels=set()
    for r in tpl.itertuples(index=False):
        l,rr=balance(r.balance_diff)
        if helper_match(l,rr)[0]=="unique_helper": helper_labels.add(int(r.labels))
    overlap=labels&helper_labels
    for metric,value in {"CompleteRXN_labels":len(labels),"unique_helper_labels":len(helper_labels),
      "overlap":len(overlap),"CompleteRXN_only":len(labels-helper_labels),"helper_only":len(helper_labels-labels)}.items():
        add(rows,"construction","1kTPL universe","template_coverage",value); rows[-1][3]=metric
    result=pd.DataFrame(rows,columns=["analysis","pool","dimension","category","value","percent"])
    result.to_csv(out/"results.csv",index=False)
    pd.DataFrame(derived,columns=["Unique_identifier","original_flower_target",
      "flower_target_proton_reagents_identified","recovered_reagent_pairs",
      "n_proton_recovered","n_restored_reagents","restored_reagents"]
    ).to_csv(out/"derived_targets.csv",index=False)
    def val(an,pool,dim,cat=""):
        x=result[(result.analysis==an)&(result.pool==pool)&(result.dimension==dim)&(result.category==cat)]
        return x.iloc[0].value if len(x) else np.nan
    md=f"""# Synthetic-corruption and construction-bias audit

This is the compact analysis of synthetic corruption and target bias.

## 1. Held-target corruption comparison

All methods use the reaction core of `rxn_reagents_identified_flower` as the balanced
target. Protonation/deprotonation-equivalent pairs anchored to a reagent identified in
`CanonicalReaction` are additionally reassigned as reagents. Target reagents are then excluded from
the completion diff; known USPTO reactants and reagents are pooled when matching the
remaining target reactants. Matching is stoichiometry-aware.

| Pool | Tasks | Mean missing molecules | >1 missing | Left / right / both (%) |
|---|---:|---:|---:|---:|
"""
    for name,s in stats.items():
        mean=sum(k*v for k,v in s.count.items())/s.n; multi=sum(v for k,v in s.count.items() if k>1)
        md+=f"| {name} | {s.n:,} | {mean:.2f} | {100*multi/s.n:.2f}% | {100*s.side['left']/s.n:.1f} / {100*s.side['right']/s.n:.1f} / {100*s.side['both']/s.n:.1f} |\n"
    md+="\n| Pool | Helper | Other non-carbon | Carbon-containing |\n|---|---:|---:|---:|\n"
    for name,s in stats.items():
        total=sum(s.kind.values())
        md+=f"| {name} | {100*s.kind['helper']/total:.2f}% | {100*s.kind['noncarbon']/total:.2f}% | {100*s.kind['carbon']/total:.2f}% |\n"
    md+="\nFull missing-count, side, molecule-type, and type-combination distributions are in `results.csv`.\n\n## 2. Balanced-target construction bias\n\n"
    for name,z in groups.items():
        md+=f"- **{name}:** {z['n']:,} inputs; {100*z['helper']['unique_helper']/z['n']:.2f}% have an unambiguous single-helper atom-balance match; added molecules alone explain {100*z['exact_atoms']/z['n']:.2f}% and net additions minus replacements explain {100*z['net_atoms']/z['n']:.2f}% of recorded atom balances. Actual/helper q95 missing atoms: {q(z['atoms'],.95)}/{q(z['helper_atoms'],.95)}.\n"
    md+=f"\nTemplate reach on the independent 1kTPL universe: CompleteRXN {len(labels)}/1,000, unique-helper closure {len(helper_labels)}/1,000; overlap {len(overlap)}, CompleteRXN-only {len(labels-helper_labels)}, helper-only {len(helper_labels-labels)}.\n\n"
    md+="The atom-count quantiles and missing-count distributions in `results.csv` quantify the long tail. Helper matching uses one helper identity at 1-6 equivalents and excludes ambiguous formula matches. Charge is not used, so this is a conservative atom-balance diagnostic rather than a full ChemBalancer execution.\n"
    n=reagent_audit["reactions"]
    md+=f"""\n## 3. Recovered-reagent audit

- Additional protonation/deprotonation-equivalent recovered reagents were found in {reagent_audit['reactions_with_proton_recovered']:,}/{n:,} reactions ({100*reagent_audit['reactions_with_proton_recovered']/n:.2f}%), comprising {reagent_audit['proton_recovered_occurrences']:,} molecular occurrences.
- After reagent reassignment, {reagent_audit['reactions_with_restored_reagent']:,}/{n:,} reactions ({100*reagent_audit['reactions_with_restored_reagent']/n:.2f}%) contain at least one target reagent not already present in the USPTO pre-product record under proton-transfer equivalence ({reagent_audit['restored_reagent_occurrences']:,} occurrences).

`derived_targets.csv` stores the original target, cleaned target, paired species and
separate restored-reagent counts for every UID. The pairing uses the guarded
classification normalization from `utils/chemistry.py`; it does not use the
stoichiometry-agnostic set paths of deep-pragmatic accuracy.
"""
    (out/"summary.md").write_text(md)
    print(f"Wrote {out/'results.csv'} and {out/'summary.md'}")

if __name__=="__main__": main()

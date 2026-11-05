#!/usr/bin/env python3
"""Check how reactions move between missing-species categories."""
from __future__ import annotations

import argparse, json, sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from analysis.missing_species.classify_missing_species import classify, component_type

RDLogger.DisableLog("rdApp.*")

@lru_cache(maxsize=500_000)
def size(s):
    m=Chem.MolFromSmiles(s)
    if m is None: raise ValueError(s)
    return Chem.AddHs(m).GetNumAtoms(),sum(a.GetAtomicNum()==6 for a in m.GetAtoms())

def molecules(x):
    return [] if pd.isna(x) or not str(x) else [s for s in str(x).split(".") if s]

def q(x,p): return float(np.quantile(x,p)) if len(x) else np.nan

def main():
    p=argparse.ArgumentParser(); p.add_argument("--benchmark",default="data/Template_curated/benchmark/benchmark_final.csv")
    p.add_argument("--derived",default="results/additional/bias_analysis/derived_targets.csv")
    p.add_argument("--categories",default="results/additional/missing_species_results/reaction_categories.csv")
    p.add_argument("--out",default="results/additional/missing_species_results"); a=p.parse_args()
    out=ROOT/a.out; out.mkdir(parents=True,exist_ok=True)
    derived=pd.read_csv(ROOT/a.derived,dtype={"Unique_identifier":str})
    current=pd.read_csv(ROOT/a.categories,dtype={"Unique_identifier":str})
    benchmark=pd.read_csv(ROOT/a.benchmark,usecols=["Unique_identifier","CanonicalReaction"],dtype={"Unique_identifier":str},low_memory=False)
    base=benchmark.merge(derived[["Unique_identifier","original_flower_target","n_proton_recovered","recovered_reagent_pairs"]],on="Unique_identifier",validate="one_to_one")
    affected=base[base.n_proton_recovered.gt(0)].copy()
    old=[]
    for r in tqdm(affected.itertuples(index=False),total=len(affected),desc="Before/after categories"):
        row=pd.Series({"Unique_identifier":r.Unique_identifier,"CanonicalReaction":r.CanonicalReaction,
                       "flower_target_proton_reagents_identified":r.original_flower_target})
        z=classify(row); old.append([r.Unique_identifier,z["missing_category"],z["n_missing"]])
    old=pd.DataFrame(old,columns=["Unique_identifier","category_before","n_missing_before"])
    changes=(affected[["Unique_identifier","n_proton_recovered","recovered_reagent_pairs"]]
             .merge(old,on="Unique_identifier").merge(current[["Unique_identifier","missing_category","n_missing"]],on="Unique_identifier"))
    changes=changes.rename(columns={"missing_category":"category_after","n_missing":"n_missing_after"})
    changes["category_changed"]=changes.category_before.ne(changes.category_after)
    changes["missing_reduction"]=changes.n_missing_before-changes.n_missing_after
    trans=(changes.groupby(["category_before","category_after"],sort=False).size().rename("reactions").reset_index())
    trans["percent_of_affected"]=100*trans.reactions/len(changes); trans.to_csv(out/"reagent_category_transitions.csv",index=False)

    removed=Counter()
    for x in changes.recovered_reagent_pairs:
        for pair in json.loads(x): removed[component_type(pair["product"])]+=1

    detail=[]
    for r in tqdm(current.itertuples(index=False),total=len(current),desc="Category complexity"):
        helper=molecules(r.helper_molecules); nonhelper=molecules(r.noncarbon_molecules)+molecules(r.carbon_molecules)
        for scope,xs in (("all_missing",helper+nonhelper),("nonhelper_only",nonhelper)):
            sizes=[size(x) for x in xs]
            detail.append([r.missing_category,scope,len(xs),sum(x[0] for x in sizes),sum(x[1] for x in sizes)])
    detail=pd.DataFrame(detail,columns=["category","scope","molecules","atoms","carbons"])
    rows=[]
    for (cat,scope),g in detail.groupby(["category","scope"],sort=False):
        row={"category":cat,"scope":scope,"reactions":len(g)}
        for col in ("molecules","atoms","carbons"):
            x=g[col].to_numpy(); row.update({f"{col}_mean":x.mean(),f"{col}_median":q(x,.5),
              f"{col}_p25":q(x,.25),f"{col}_p75":q(x,.75),f"{col}_p90":q(x,.9),f"{col}_p95":q(x,.95),f"{col}_max":x.max()})
        rows.append(row)
    complexity=pd.DataFrame(rows); complexity.to_csv(out/"category_complexity.csv",index=False)

    changed=int(changes.category_changed.sum()); reduced=int(changes.missing_reduction.gt(0).sum()); total=len(current)
    md=["# Stratification change and complexity audit","",
        "The before/after comparison changes only the additional protonation-aware reagent assignment; target-reagent exclusion and proton-aware input-target matching are held fixed.","",
        f"- Additional reagent identification affected **{len(changes):,}/{total:,} reactions ({100*len(changes)/total:.2f}%)**.",
        f"- It changed the category of **{changed:,} affected reactions ({100*changed/len(changes):.2f}%; {100*changed/total:.2f}% of the benchmark)**.",
        f"- It reduced the missing-molecule count in **{reduced:,} affected reactions ({100*reduced/len(changes):.2f}%)**; mean reduction {changes.missing_reduction.mean():.2f} molecules.",
        f"- Reassigned recovered product forms: carbon {removed['carbon']:,} ({100*removed['carbon']/sum(removed.values()):.2f}%), non-carbon {removed['non-carbon']:,} ({100*removed['non-carbon']/sum(removed.values()):.2f}%), helper {removed['helper']:,} ({100*removed['helper']/sum(removed.values()):.2f}%).","",
        "## Most frequent category transitions","","| Before | After | Reactions | % affected |","|---|---|---:|---:|"]
    for r in trans.sort_values("reactions",ascending=False).head(15).itertuples(index=False):
        md.append(f"| {r.category_before} | {r.category_after} | {r.reactions:,} | {r.percent_of_affected:.2f}% |")
    md += ["","## Final category complexity","",
      "Values are per reaction and count implicit hydrogens. The table shows all remaining missing molecules; `category_complexity.csv` additionally reports non-helper-only values and p25/p75/p95/max.","",
      "| Category | n | Missing molecules median [IQR] | Missing atoms median [IQR], p90 | Missing carbons median [IQR], p90 |","|---|---:|---:|---:|---:|"]
    for r in complexity[complexity.scope.eq("all_missing")].itertuples(index=False):
        md.append(f"| {r.category} | {r.reactions:,} | {r.molecules_median:.0f} [{r.molecules_p25:.0f}-{r.molecules_p75:.0f}] | {r.atoms_median:.0f} [{r.atoms_p25:.0f}-{r.atoms_p75:.0f}], {r.atoms_p90:.0f} | {r.carbons_median:.0f} [{r.carbons_p25:.0f}-{r.carbons_p75:.0f}], {r.carbons_p90:.0f} |")
    (out/"stratification_audit.md").write_text("\n".join(md)+"\n")
    print("\n".join(md[:12])); print(f"Wrote audit outputs to {out}")

if __name__=="__main__": main()

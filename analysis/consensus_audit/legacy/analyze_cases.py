"""Inspect cases where the models agree but do not match the reference."""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/archive/metric_v1/additional/consensus_audit"
sys.path.insert(0, str(ROOT))
from utils.metrics import compare_reactions
from utils.reaction_balance import is_balanced_reaction

RDLogger.DisableLog('rdApp.*')


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def metrics(prediction, target):
    try:
        result = compare_reactions(prediction, target)
        return {key: int(bool(getattr(result, key))) for key in
                ('exact_match', 'proton_shuffle_match', 'all_normalizations_match',
                 'pragmatic_equivalence_match', 'deep_pragmatic_equivalence_match')}
    except Exception:
        return {key: 0 for key in
                ('exact_match', 'proton_shuffle_match', 'all_normalizations_match',
                 'pragmatic_equivalence_match', 'deep_pragmatic_equivalence_match')}


def product_molecules(reaction):
    items = []
    for smiles in reaction.split('>')[-1].split('.'):
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        items.append((Chem.MolToSmiles(mol, isomericSmiles=True),
                      mol.GetNumHeavyAtoms(),
                      sum(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())))
    return items


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    historical_cases = OUT / 'stereo_sensitive_both_wrong_cases.csv'
    if not historical_cases.exists():
        historical_cases = OUT / 'agreed_wrong_cases.csv'
    cases = read(historical_cases)
    by_uid = defaultdict(list)
    for row in cases:
        by_uid[row['Unique_identifier']].append(row)
    bench = {row['Unique_identifier']: row for row in
             read(ROOT / 'data/Template_curated/benchmark/benchmark_final.csv')
             if row['Unique_identifier'] in by_uid}
    assert len(by_uid) == len(bench) == 93
    rows = []
    for uid, observations in sorted(by_uid.items()):
        # Prediction strings can differ across repetitions. Keep a row for each
        # distinct pair and preserve how often that pair appears.
        by_pair = defaultdict(list)
        for row in observations:
            by_pair[(row['syn_prediction'], row['crb_prediction'])].append(row)
        target_flow = bench[uid]['rxn_flower']
        target_stereo = bench[uid]['TargetReaction_with_stereo']
        for (syn, crb), copies in by_pair.items():
            syn_flow = metrics(syn, target_flow)
            crb_flow = metrics(crb, target_flow)
            syn_stereo = metrics(syn, target_stereo)
            crb_stereo = metrics(crb, target_stereo)
            direct = metrics(syn, crb)
            assert direct['deep_pragmatic_equivalence_match'] == 1
            assert is_balanced_reaction(syn) and is_balanced_reaction(crb)
            syn_products = product_molecules(syn)
            crb_products = product_molecules(crb)
            gt_products = product_molecules(target_flow)
            main_syn = max(syn_products, key=lambda item: (item[2], item[1])) if syn_products else ('', 0, 0)
            main_crb = max(crb_products, key=lambda item: (item[2], item[1])) if crb_products else ('', 0, 0)
            gt_smiles = {item[0] for item in gt_products}
            rows.append(dict(
                Unique_identifier=uid, observations=len(copies),
                folds=';'.join(sorted(row['fold'] for row in copies)),
                syn_solved_all=int(all(row['syn_solved'] == 'True' for row in copies)),
                high_confidence_any=int(any(float(row['syn_confidence']) >= .90 and float(row['crb_confidence']) >= .95 for row in copies)),
                exact_same_prediction=int(syn == crb),
                same_main_product_syn=int(main_syn[0] in gt_smiles),
                same_main_product_crb=int(main_crb[0] in gt_smiles),
                main_product_syn=main_syn[0], main_product_crb=main_crb[0],
                main_product_gt=';'.join(item[0] for item in gt_products if item[2] == max((x[2] for x in gt_products), default=0)),
                syn_vs_flow_exact=syn_flow['exact_match'],
                syn_vs_flow_proton=syn_flow['proton_shuffle_match'],
                syn_vs_flow_equiv=syn_flow['deep_pragmatic_equivalence_match'],
                crb_vs_flow_exact=crb_flow['exact_match'],
                crb_vs_flow_proton=crb_flow['proton_shuffle_match'],
                crb_vs_flow_equiv=crb_flow['deep_pragmatic_equivalence_match'],
                syn_vs_stereo_equiv=syn_stereo['deep_pragmatic_equivalence_match'],
                crb_vs_stereo_equiv=crb_stereo['deep_pragmatic_equivalence_match'],
                syn_prediction=syn, crb_prediction=crb, target_flow=target_flow,
                target_stereo=target_stereo,
            ))
        print(uid, len(by_pair), flush=True)
    with (OUT / 'case_analysis.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('prediction pairs', len(rows), 'distinct UIDs', len(by_uid))
    for key in ('exact_same_prediction', 'syn_solved_all', 'high_confidence_any',
                'same_main_product_syn', 'same_main_product_crb',
                'syn_vs_flow_equiv', 'crb_vs_flow_equiv',
                'syn_vs_stereo_equiv', 'crb_vs_stereo_equiv'):
        print(key, sum(int(row[key]) for row in rows))


if __name__ == '__main__':
    main()

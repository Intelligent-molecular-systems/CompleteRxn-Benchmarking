"""Recalculate CRB and SynRBL agreement on uncurated USPTO, ignoring stereochemistry.

Read saved SynRBL outputs and CRB predictions without rerunning either model.
The benchmark metric checks both reactions for balance. Historical scoring
with --metric v1 keeps the original CRB balance labels
and checks SynRBL balance using the original procedure.
"""
import argparse
import csv
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/additional/uncurated_stereo_free_agreement"
sys.path.insert(0, str(ROOT))
from analysis.uncurated.legacy.analyze_uncurated_predictions import _is_reaction_balanced, dpe_match
from utils.result_paths import metric_name
from utils.scoring import METRICS, labels, results_root, provenance
from utils.reaction_balance import is_balanced_reaction
_METRIC = "v2"

RDLogger.DisableLog('rdApp.*')


def read(path):
    with open(path, newline='') as handle:
        yield from csv.DictReader(handle)


def flag(value):
    return str(value).strip().lower() in ('1', 'true', 't', 'yes')


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def strip_stereo(reaction):
    if not any(marker in reaction for marker in ('@', '/', '\\')):
        return reaction
    sides = []
    for side in reaction.split('>'):
        molecules = []
        for smiles in side.split('.'):
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            molecules.append(Chem.MolToSmiles(mol, isomericSmiles=False) if mol else smiles)
        sides.append('.'.join(molecules))
    return '>'.join(sides)


def agrees(pair):
    syn, crb = pair
    if _METRIC == "v1":
        return int(dpe_match(strip_stereo(syn), strip_stereo(crb)))
    return int(labels(syn,crb,metric=_METRIC,stereo=False)[1])


def balanced(pair):
    syn, crb = pair
    if _METRIC == "v1": return _is_reaction_balanced(syn)
    return is_balanced_reaction(syn) and is_balanced_reaction(crb)


def main():
    global OUT, _METRIC
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric",type=metric_name, choices=METRICS,default="v2")
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args();_METRIC=args.metric
    OUT=args.output_dir or results_root(args.metric)/"additional/uncurated_stereo_free_agreement"
    if (OUT/"summary.csv").exists(): parser.error("Choose a new output directory")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.metric == 'v1' and os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('Run with APPTAINERENV_PYTHONHASHSEED=0 to make halide normalization reproducible')
    syn = {}
    source = ROOT / 'results/SynRBL/uspto_uncurated_CanonicalReaction_results'
    for path in sorted(glob.glob(str(source / '*_results.csv'))):
        for row in read(path):
            uid = row['Unique_identifier']
            syn.setdefault(uid, (row['Reaction'], number(row['confidence'])))
    print('loaded SynRBL UIDs:', len(syn), flush=True)

    historical_path = ROOT / 'results/archive/metric_v1/additional/uncurated_consensus_summary.csv'
    historical = read(historical_path)[0]
    historical_inputs = [historical_path]
    old_agree = set()
    for name in ('confident_agree_uids.csv', 'agree_not_confident_uids.csv'):
        historical_file = ROOT / 'results/archive/metric_v1/additional/uncurated_consensus_uids' / name
        historical_inputs.append(historical_file)
        old_agree.update(r['Unique_identifier'] for r in read(historical_file))
    assert len(old_agree) == int(historical["agreement_dpe_n"]) == 201223, len(old_agree)

    records = []
    syntax_rows = []
    crb_balanced_syntax = 0
    cache = ROOT / 'results/main/predictions/crb/uncurated/loaded_balanced.csv'
    for row in read(cache):
        uid = row['Unique_identifier']
        sr = syn.get(uid)
        if sr is None:
            continue
        sp, sc = sr
        cp = row['prediction_reaction']
        if '>' not in sp or '>' not in cp:
            continue
        if not flag(row['is_balanced']):
            continue
        syntax_rows.append((uid, sp, cp, sc, number(row['crb_sequence_probability']), flag(row['is_balanced'])))
    print('cached CRB balanced syntax:', len(syntax_rows), flush=True)
    with Pool(processes=8) as pool:
        jobs = ((sp, cp) for _, sp, cp, _, _, _ in syntax_rows)
        for i, both_ok in enumerate(pool.imap(balanced, jobs, chunksize=64)):
            uid, sp, cp, sc, cc, cached_balanced = syntax_rows[i]
            if both_ok:
                records.append((uid, sp, cp, sc, cc))
            if (i + 1) % 50000 == 0:
                print('balance checked:', i + 1, 'both balanced:', len(records), flush=True)
    print('cached CRB balanced syntax:', len(syntax_rows),
          'both balanced:', len(records), flush=True)
    if args.metric == "v1": assert len(records) == 384579, len(records)
    missing_old = old_agree - {r[0] for r in records}
    print('historically agreeing UIDs outside reconstructed population:',
          len(missing_old), sorted(missing_old)[:10], flush=True)

    both_confident = old_confident = 0
    new_confident = new_agree = gained = lost = 0
    gained_rows, lost_rows = [], []
    jobs = ((sp, cp) for _, sp, cp, _, _ in records)
    with Pool(processes=8) as pool:
        for i, new_match in enumerate(pool.imap(agrees, jobs, chunksize=64)):
            uid, _, _, sc, cc = records[i]
            old_match = uid in old_agree
            confident = sc >= .90 and cc >= .95
            new_agree += new_match
            both_confident += confident
            old_confident += confident and old_match
            new_confident += confident and bool(new_match)
            if new_match and not old_match:
                gained += 1
                gained_rows.append((uid, sc, cc))
            elif old_match and not new_match:
                lost += 1
                lost_rows.append((uid, sc, cc))
            if (i + 1) % 10000 == 0:
                print('checked', i + 1, '/', len(records), flush=True)

    summary = dict(
        overlap_both_balanced_rows=len(records),
        old_stereo_sensitive_agreement_n=len(old_agree),
        # Keep the reported historical percentage; the current overlap can differ.
        old_stereo_sensitive_agreement_pct=float(historical["agreement_dpe_pct_overlap"]),
        stereo_free_agreement_n=new_agree,
        stereo_free_agreement_pct=100 * new_agree / len(records),
        gained_agreement_n=gained,
        lost_agreement_n=lost,
        both_confident_n=both_confident,
        old_stereo_sensitive_confident_agreement_n=old_confident,
        stereo_free_confident_agreement_n=new_confident,
        stereo_free_confident_agreement_pct_overlap=100 * new_confident / len(records),
        syn_conf_thr=.90, crb_conf_thr=.95,
    )
    with (OUT / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    for name, rows in [('gained_uids.csv', gained_rows), ('lost_uids.csv', lost_rows)]:
        with (OUT / name).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['Unique_identifier', 'synrbl_confidence', 'crb_sequence_probability'])
            writer.writerows(rows)
    provenance(OUT,args.metric,[cache,Path(__file__),*historical_inputs,*sorted(source.glob("*_results.csv"))],
               stereo=False,crb_cached_population=True,crb_balance_rechecked=args.metric != "v1",
               syn_confidence=.90,crb_confidence=.95)
    print(summary, flush=True)


if __name__ == '__main__':
    main()

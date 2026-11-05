"""List distinct prediction pairs where both models miss the reference."""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/archive/metric_v1/additional/consensus_audit"


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def confidence(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = read(OUT / 'stereo_free_both_wrong_cases.csv')
    by_pair = defaultdict(list)
    for row in cases:
        by_pair[(row['Unique_identifier'], row['syn_prediction'], row['crb_prediction'])].append(row)
    assert len(cases) == 138
    ids = {key[0] for key in by_pair}
    references = {row['Unique_identifier']: row for row in
                  read(ROOT / 'data/Template_curated/benchmark/benchmark_final.csv')
                  if row['Unique_identifier'] in ids}
    assert len(references) == len(ids) == 127
    historical_cases = OUT / 'stereo_sensitive_both_wrong_cases.csv'
    if not historical_cases.exists():
        historical_cases = OUT / 'agreed_wrong_cases.csv'
    stereo_sensitive = {(r['fold'], r['Unique_identifier']) for r in read(historical_cases)}
    rows = []
    for (uid, syn, crb), copies in by_pair.items():
        reference = references[uid]
        high = [r for r in copies if r['syn_solved'] == '1' and
                confidence(r['syn_confidence']) >= .90 and confidence(r['crb_confidence']) >= .95]
        representative = max(copies, key=lambda r:
                             (r in high, confidence(r['syn_confidence']), confidence(r['crb_confidence'])))
        rows.append(dict(
            Unique_identifier=uid,
            observations=len(copies),
            folds=';'.join(sorted(r['fold'] for r in copies)),
            high_confidence_any=int(bool(high)),
            exact_same_prediction=int(syn == crb),
            stereo_sensitive_agreement_any=int(any((r['fold'], uid) in stereo_sensitive for r in copies)),
            syn_solved_representative=representative['syn_solved'],
            syn_confidence_representative=representative['syn_confidence'],
            crb_confidence_representative=representative['crb_confidence'],
            input_reaction=reference['InputReaction'],
            synrbl_reference=reference['rxn_flower'],
            crb_reference=reference['TargetReaction_with_stereo'],
            synrbl_prediction=syn,
            crb_prediction=crb,
        ))
    rows.sort(key=lambda r: (-r['high_confidence_any'], -r['exact_same_prediction'],
                             -r['observations'], r['Unique_identifier']))
    with (OUT / 'stereo_free_both_wrong_review.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('review rows:', len(rows), 'unique UIDs:', len(ids),
          'high-confidence rows:', sum(r['high_confidence_any'] for r in rows))


if __name__ == '__main__':
    main()

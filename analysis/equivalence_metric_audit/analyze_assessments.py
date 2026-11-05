#!/usr/bin/env python3
"""Summarize the fixed expert review using the blinded sample and its answer key.

The sample was selected using v1. It is not a new sample of errors under the benchmark metric.
Optionally include the separate LLM review or saved prediction scores.
This script does not rerun a model or recalculate chemical matches.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / 'results/archive/metric_v1/additional/equivalence_metric_audit'


def clean(value):
    return (value or '').strip().strip('"').strip()


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    indexed = {}
    for row in rows:
        audit_id = int(row['audit_id'])
        if audit_id in indexed:
            raise ValueError(f'Duplicate audit_id {audit_id} in {path}')
        # Preserve notes containing unquoted commas in the original assessment.
        extra = row.pop(None, []) or []
        row['notes'] = clean(', '.join([row.get('notes', ''), *extra]))
        indexed[audit_id] = row
    if not indexed:
        raise ValueError(f'Empty audit input: {path}')
    return indexed


def preference(value):
    value = clean(value).lower()
    if not value:
        return 'none'
    if 'prefer a' in value or 'a more plausible' in value or 'a better' in value:
        return 'A'
    if any(text in value for text in ('prefer b', 'b more plausible', 'b better',
                                     'b is more realistic', 'b is the more realistic')):
        return 'B'
    if 'both plausible' in value or 'either is acceptable' in value:
        return 'both'
    raise ValueError(f'Unknown preference: {value!r}')


def assess(rows, blinded, key, assessor):
    details = []
    for audit_id, row in sorted(rows.items()):
        if audit_id not in blinded or audit_id not in key:
            raise ValueError(f'Audit {audit_id} missing from blinded sample or key')
        for field in ('input_reaction', 'completion_a', 'completion_b'):
            if clean(row[field]) != clean(blinded[audit_id][field]):
                raise ValueError(f'{assessor} audit {audit_id}: {field} differs from blinded sample')
        labels = key[audit_id]
        if {labels['completion_a_source'], labels['completion_b_source']} != {'SynRBL', 'FlowER'}:
            raise ValueError(f'Invalid source labels for audit {audit_id}')
        sources = {'SynRBL': labels['synrbl_completion'], 'FlowER': labels['flower_target']}
        for letter in ('a', 'b'):
            if clean(blinded[audit_id][f'completion_{letter}']) != clean(sources[labels[f'completion_{letter}_source']]):
                raise ValueError(f'Key does not identify completion {letter} for audit {audit_id}')
        equivalence = {'yes': 'equivalent', 'no': 'not_equivalent', 'uncertain': 'uncertain'}.get(clean(row['equivalent']).lower())
        if equivalence is None:
            raise ValueError(f'Unknown equivalence judgment for audit {audit_id}')
        pref = preference(row['assessment'])
        source = labels[f'completion_{pref.lower()}_source'] if pref in ('A', 'B') else pref
        outcome = source if equivalence == 'not_equivalent' else equivalence
        details.append(dict(assessor=assessor, audit_id=audit_id,
                            Unique_identifier=labels['Unique_identifier'],
                            completion_a_source=labels['completion_a_source'],
                            completion_b_source=labels['completion_b_source'],
                            equivalence=equivalence, preference_blinded=pref,
                            preferred_source=source, outcome=outcome,
                            confidence=clean(row.get('confidence')).title(), notes=row['notes']))
    return details


def add_scores(details, path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    scores = {row['Unique_identifier']: row for row in rows}
    if len(scores) != len(rows):
        raise ValueError('Duplicate reaction IDs in score file; provide one fold only')
    for row in details:
        score = scores.get(row['Unique_identifier'])
        if score is None:
            raise ValueError('Missing audit UID in score file; use the full-test fold')
        if score['equivalence'] not in ('0', '1'):
            raise ValueError('Expected binary equivalence labels in score file')
        row['saved_equivalence'] = int(score['equivalence'])
        row['retained'] = int(score['retained'])


def summarize(details):
    summary = []
    for assessor in sorted({row['assessor'] for row in details}):
        selected = [row for row in details if row['assessor'] == assessor]
        for field, categories in [
            ('equivalence', ('equivalent', 'not_equivalent', 'uncertain')),
            ('outcome', ('equivalent', 'FlowER', 'SynRBL', 'both', 'none', 'uncertain')),
        ]:
            counts = Counter(row[field] for row in selected)
            for category in categories:
                summary.append(dict(assessor=assessor, measure=field, category=category,
                                    count=counts[category], observations=len(selected),
                                    percentage=100 * counts[category] / len(selected)))
    return summary


def write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert', type=Path, default=HERE/'reaction_completion_assessment_expert_first20.csv')
    parser.add_argument('--blinded', type=Path, default=HERE/'audit_sample_blinded.csv')
    parser.add_argument('--key', type=Path, default=HERE/'audit_key.csv')
    parser.add_argument('--llm', type=Path, help='Optional supplementary assessment; must cover the same audit IDs')
    parser.add_argument('--scores', type=Path, help='Optional saved full-test SynRBL score CSV for this fold')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args(argv)
    expert, blinded, key = map(read_rows, (args.expert, args.blinded, args.key))
    details = assess(expert, blinded, key, 'expert')
    if args.llm:
        llm = read_rows(args.llm)
        if set(llm) != set(expert):
            raise ValueError('Expert and LLM assessment IDs differ')
        details += assess(llm, blinded, key, 'llm')
    if args.scores:
        add_scores(details, args.scores)
    outputs = [args.output_dir/'expert_audit_details.csv', args.output_dir/'expert_audit_summary.csv']
    if any(path.exists() for path in outputs):
        raise FileExistsError('Audit outputs already exist; choose another --output-dir')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(outputs[0], details)
    write(outputs[1], summarize(details))
    print('Wrote ' + ', '.join(str(path) for path in outputs))


if __name__ == '__main__':
    main()

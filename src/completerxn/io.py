"""Small CSV and checksum helpers shared by the public commands."""
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path, required):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f'Empty or duplicate CSV header: {path}')
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError(f'Missing columns in {path}: {", ".join(sorted(missing))}')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Malformed CSV row in {path}, line {reader.line_num}')
            yield row


def write_csv(path, rows, fields):
    with Path(path).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')

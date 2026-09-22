#!/usr/bin/env python3
"""Predict total tokens for every candidate using a fitted DynCost model."""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

import argparse

from carp.cost import QueryOnlyDynCost
from carp.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True, help='Canonical JSONL containing question and id')
    parser.add_argument('--output', required=True)
    parser.add_argument('--methods', nargs='+', required=True)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    costs = QueryOnlyDynCost.load(args.model).predict([record['question'] for record in records])
    write_jsonl(args.output, [
        {'id': record.get('id', index), 'predicted_total_tokens': {method: float(costs[index, column]) for column, method in enumerate(args.methods)}}
        for index, record in enumerate(records)
    ])


if __name__ == '__main__':
    main()

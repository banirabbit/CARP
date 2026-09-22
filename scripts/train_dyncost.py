#!/usr/bin/env python3
"""Fit the query-only DynCost component from canonical CARP execution JSONL."""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

import argparse
import json

from carp.cost import QueryOnlyDynCost
from carp.io import read_jsonl, total_token_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='Canonical train JSONL')
    parser.add_argument('--output', required=True, help='Output joblib path')
    parser.add_argument('--methods', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = read_jsonl(args.train)
    model = QueryOnlyDynCost().fit(
        [record['question'] for record in train],
        total_token_matrix(train, args.methods),
        random_state=args.seed,
    )
    model.save(args.output)
    print(json.dumps({'records': len(train), 'methods': args.methods, 'output': args.output}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Convert per-service execution logs into CARP's canonical split JSONL files.

The converter consumes the original repository layout:
  <dataset-dir>/<method>/<benchmark>/results.score.json  (JSONL despite suffix)
  <dataset-dir>/data_splits.json
"""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

import argparse
import json
from pathlib import Path
from typing import Any

from carp.io import write_jsonl


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if 'id' not in item:
                raise ValueError(f'{path}:{line_number} has no id')
            rows[int(item['id'])] = item
    return rows


def total_tokens(item: dict[str, Any], source: str) -> float:
    tokens = float((item.get('token_cost') or {}).get('total_tokens', 0) or 0)
    if tokens <= 0:
        raise ValueError(f'{source}: id {item.get("id")} has non-positive total_tokens')
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=Path, default=Path('legacy_original/dataset'))
    parser.add_argument('--benchmark', default='hotpot')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['qagn', 'dalk', 'gr', 'hippo', 'lgraph', 'light'])
    args = parser.parse_args()

    split_payload = json.loads((args.dataset_dir / 'data_splits.json').read_text(encoding='utf-8'))
    source = {
        method: load_jsonl(args.dataset_dir / method / args.benchmark / 'results.score.json')
        for method in args.methods
    }
    common_ids = set.intersection(*(set(rows) for rows in source.values()))
    if not common_ids:
        raise ValueError('no query id is shared by all requested methods')

    split_keys = {'train': 'train_qids', 'validation': 'val_qids', 'test': 'test_qids'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for output_name, split_key in split_keys.items():
        if split_key not in split_payload:
            raise KeyError(f'{args.dataset_dir / "data_splits.json"} lacks {split_key}')
        ids = sorted(int(qid) for qid in split_payload[split_key] if int(qid) in common_ids)
        records = []
        for qid in ids:
            samples = {method: source[method][qid] for method in args.methods}
            questions = {str(sample.get('question', '')) for sample in samples.values()}
            if len(questions) != 1 or not next(iter(questions)).strip():
                raise ValueError(f'id {qid} has inconsistent or missing question text across services')
            records.append({
                'id': qid,
                'question': next(iter(questions)),
                'methods': {
                    method: {
                        'f1': float(samples[method].get('f1', 0.0)),
                        'total_tokens': total_tokens(samples[method], method),
                    }
                    for method in args.methods
                },
            })
        output = args.output_dir / f'{output_name}.jsonl'
        write_jsonl(output, records)
        print(json.dumps({'split': output_name, 'queries': len(records), 'output': str(output)}))


if __name__ == '__main__':
    main()

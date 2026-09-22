#!/usr/bin/env python3
"""Calibrate P3 on validation predictions and route aligned test predictions."""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

import argparse
import json

from carp.io import read_jsonl, write_jsonl
from carp.routing import P3Router


def pair_predictions(uplift_path: str, cost_path: str) -> list[tuple[dict, dict]]:
    uplifts = read_jsonl(uplift_path)
    costs = read_jsonl(cost_path)
    cost_by_id = {str(record.get('id', index)): record['predicted_total_tokens'] for index, record in enumerate(costs)}
    pairs = []
    for index, record in enumerate(uplifts):
        identifier = str(record.get('id', index))
        if identifier not in cost_by_id:
            raise ValueError(f'no cost prediction for id {identifier}')
        pairs.append((record, {'predicted_total_tokens': cost_by_id[identifier]}))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-uplifts', required=True)
    parser.add_argument('--validation-costs', required=True)
    parser.add_argument('--test-uplifts', required=True)
    parser.add_argument('--test-costs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    validation = pair_predictions(args.validation_uplifts, args.validation_costs)
    router = P3Router().fit(
        [record['uplifts'] for record, _ in validation],
        [record['predicted_total_tokens'] for _, record in validation],
    )
    test = pair_predictions(args.test_uplifts, args.test_costs)
    routed = []
    for index, (uplift, cost) in enumerate(test):
        decision = router.route(uplift['uplifts'], cost['predicted_total_tokens'])
        routed.append({
            'id': uplift.get('id', index),
            'selected_method': decision.chosen_method,
            'pareto_front': list(decision.pareto_front),
        })
    write_jsonl(args.output, routed)
    scale_path = Path(args.output).with_suffix(Path(args.output).suffix + '.scales.json')
    scale_path.write_text(json.dumps(router.scales.__dict__, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

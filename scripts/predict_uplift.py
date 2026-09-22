#!/usr/bin/env python3
"""Run a trained CARP uplift scorer on queries in canonical JSONL."""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

import argparse

import torch

from carp.io import read_jsonl, write_jsonl
from carp.scoring import FrozenQueryEncoder, PriorResidualUpliftHead, build_full_uplifts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    encoder = FrozenQueryEncoder(checkpoint['encoder_name'], device=args.device, max_length=checkpoint['max_length'])
    head = PriorResidualUpliftHead(encoder.hidden_size, checkpoint['prior'], checkpoint['hidden_dim']).to(encoder.device)
    head.load_state_dict(checkpoint['state_dict'])
    head.eval()
    records = read_jsonl(args.input)
    features = encoder.encode([record['question'] for record in records]).to(encoder.device)
    reference = encoder.encode([checkpoint['reference_text']]).to(encoder.device)
    with torch.no_grad():
        full = build_full_uplifts(checkpoint['anchor'], checkpoint['methods'], head(features, reference)).cpu().numpy()
    write_jsonl(args.output, [
        {'id': record.get('id', index), 'uplifts': {method: float(full[index, column]) for column, method in enumerate(checkpoint['methods'])}}
        for index, record in enumerate(records)
    ])


if __name__ == '__main__':
    main()

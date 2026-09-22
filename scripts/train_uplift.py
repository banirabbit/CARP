#!/usr/bin/env python3
"""Train CARP's frozen-encoder anchor-relative continuous uplift scorer."""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from carp.io import read_jsonl, uplift_targets
from carp.scoring import FrozenQueryEncoder, PriorResidualUpliftHead


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', required=True, help='Checkpoint .pt path')
    parser.add_argument('--anchor', default='qagn')
    parser.add_argument('--methods', nargs='+', required=True)
    parser.add_argument('--encoder', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accumulation', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--reference-text', default='Question:')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device')
    return parser.parse_args()


def mean_mse(head: PriorResidualUpliftHead, features: torch.Tensor, targets: torch.Tensor, reference: torch.Tensor) -> float:
    head.eval()
    with torch.no_grad():
        return float(torch.nn.functional.mse_loss(head(features, reference), targets).item())


def main() -> None:
    args = arguments()
    if args.anchor not in args.methods:
        raise ValueError('anchor must be one of methods')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train = read_jsonl(args.train)
    validation = read_jsonl(args.validation)
    train_targets = torch.tensor(uplift_targets(train, args.anchor, args.methods))
    validation_targets = torch.tensor(uplift_targets(validation, args.anchor, args.methods))

    encoder = FrozenQueryEncoder(args.encoder, device=args.device, max_length=args.max_length)
    train_features = encoder.encode([record['question'] for record in train])
    validation_features = encoder.encode([record['question'] for record in validation])
    reference = encoder.encode([args.reference_text])
    device = encoder.device
    head = PriorResidualUpliftHead(encoder.hidden_size, train_targets.mean(0), args.hidden_dim).to(device)
    train_features, validation_features, reference = train_features.to(device), validation_features.to(device), reference.to(device)
    train_targets, validation_targets = train_targets.to(device), validation_targets.to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=args.batch_size, shuffle=True)
    best_state, best_validation, stale = None, float('inf'), 0
    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (features, targets) in enumerate(loader, 1):
            loss = torch.nn.functional.mse_loss(head(features, reference), targets) / args.grad_accumulation
            loss.backward()
            if step % args.grad_accumulation == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation_mse = mean_mse(head, validation_features, validation_targets, reference)
        print(json.dumps({'epoch': epoch + 1, 'validation_mse': validation_mse}))
        if validation_mse < best_validation:
            best_validation, stale = validation_mse, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    assert best_state is not None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': best_state,
        'prior': train_targets.mean(0).detach().cpu(),
        'anchor': args.anchor,
        'methods': args.methods,
        'encoder_name': args.encoder,
        'max_length': args.max_length,
        'hidden_dim': args.hidden_dim,
        'reference_text': args.reference_text,
        'validation_mse': best_validation,
    }, output)


if __name__ == '__main__':
    main()

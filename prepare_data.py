#!/usr/bin/env python3
"""
Prepare Hotpot train/val/test splits from the repository-local full dataset.

The current repository already ships a canonical `dataset/full_dataset.csv`
containing the enriched per-(qid, method) rows used by training. This script
rebuilds `train.csv`, `val.csv`, `test.csv` and their JSONL mirrors according
to `dataset/data_splits.json`, instead of relying on stale absolute paths from
an older workspace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "dataset"
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild train/val/test CSV + JSONL files from dataset/full_dataset.csv."
    )
    parser.add_argument(
        "--full-dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR / "full_dataset.csv",
        help="Path to the canonical full_dataset CSV.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_DATASET_DIR / "data_splits.json",
        help="Path to the qid split definition JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory where train/val/test CSV and JSONL files will be written.",
    )
    return parser.parse_args()


def load_split_map(split_path: Path) -> Dict[str, set[int]]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    split_map: Dict[str, set[int]] = {}
    for split_name in SPLIT_NAMES:
        key = f"{split_name}_qids"
        if key not in payload:
            raise KeyError(f"Missing split key: {key} in {split_path}")
        split_map[split_name] = {int(qid) for qid in payload[key]}
    return split_map


def write_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_stats(splits: Dict[str, pd.DataFrame]) -> Dict:
    dataset_stats = {}
    for split_name, df in splits.items():
        labels = df["label"].value_counts().sort_index().to_dict() if "label" in df.columns else {}
        dataset_stats[split_name] = {
            "num_samples": int(len(df)),
            "num_questions": int(df["qid"].nunique()),
            "label_distribution": {str(k): int(v) for k, v in labels.items()},
        }

    return {
        "total_questions": int(splits["full"]["qid"].nunique()),
        "methods": sorted(splits["full"]["method_id"].unique().tolist())
        if "method_id" in splits["full"].columns
        else [],
        "split_ratios": {
            "train": round(dataset_stats["train"]["num_questions"] / dataset_stats["full"]["num_questions"], 4),
            "val": round(dataset_stats["val"]["num_questions"] / dataset_stats["full"]["num_questions"], 4),
            "test": round(dataset_stats["test"]["num_questions"] / dataset_stats["full"]["num_questions"], 4),
        },
        "dataset_stats": dataset_stats,
    }


def main() -> None:
    args = parse_args()

    if not args.full_dataset.exists():
        raise FileNotFoundError(f"Full dataset not found: {args.full_dataset}")
    if not args.split_path.exists():
        raise FileNotFoundError(f"Split file not found: {args.split_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Rebuilding Hotpot train/val/test splits")
    print("=" * 80)
    print(f"Full dataset: {args.full_dataset}")
    print(f"Split file:    {args.split_path}")
    print(f"Output dir:    {args.output_dir}")

    full_df = pd.read_csv(args.full_dataset)
    split_map = load_split_map(args.split_path)

    splits: Dict[str, pd.DataFrame] = {"full": full_df}
    for split_name in SPLIT_NAMES:
        qids = split_map[split_name]
        df = full_df[full_df["qid"].isin(qids)].copy()
        df.sort_values(["qid", "method_id"], inplace=True, ignore_index=True)
        splits[split_name] = df

        csv_path = args.output_dir / f"{split_name}.csv"
        jsonl_path = args.output_dir / f"{split_name}.jsonl"
        df.to_csv(csv_path, index=False)
        write_jsonl(df, jsonl_path)

        print(
            f"{split_name:>5}: questions={df['qid'].nunique():>4} "
            f"rows={len(df):>6} -> {csv_path.name}, {jsonl_path.name}"
        )

    stats = build_stats(splits)
    stats_path = args.output_dir / "statistics.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()

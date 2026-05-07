#!/usr/bin/env python3
"""
Prepare Hotpot train/val/test splits from per-method Hotpot result files.

Source of truth:
- `dataset/<method>/hotpot/results.score.json` for method-level Hotpot results
- `dataset/data_splits.json` for train/val/test qid splits

This script rebuilds:
- `dataset/train.csv`, `dataset/val.csv`, `dataset/test.csv`
- `dataset/train.jsonl`, `dataset/val.jsonl`, `dataset/test.jsonl`
- `dataset/statistics.json`
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "dataset"
SPLIT_NAMES = ("train", "val", "test")
METHODS = ["dalk", "gr", "hippo", "lgraph", "light", "qagn"]
METHOD_PROFILES = {
    "dalk": "retrieval-based QA method",
    "gr": "graph reasoning oriented method",
    "hippo": "multi-step retrieval and reasoning method",
    "lgraph": "graph-centric reasoning and linking method",
    "light": "lightweight retrieval and reasoning method",
    "qagn": "question answering with graph/navigation signals",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild train/val/test CSV + JSONL files from dataset/<method>/hotpot/results.score.json."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset root directory containing per-method result files.",
    )
    parser.add_argument(
        "--dataset-name",
        default="hotpot",
        help="Dataset name under each method directory.",
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
    parser.add_argument(
        "--methods",
        nargs="*",
        default=METHODS,
        help="Methods to aggregate from dataset/<method>/<dataset-name>/results.score.json.",
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


def load_method_results(dataset_dir: Path, method: str, dataset_name: str) -> Dict[int, Dict]:
    result_path = dataset_dir / method / dataset_name / "results.score.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    rows: Dict[int, Dict] = {}
    with result_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "id" not in item:
                raise KeyError(f"Missing `id` at {result_path}:{line_no}")
            qid = int(item["id"])
            rows[qid] = item
    return rows


def build_input_text(question_text: str, method_id: str, sample: Dict) -> str:
    question_type = sample.get("type", "unknown")
    difficulty = sample.get("level", "unknown")
    supporting_facts = sample.get("supporting_facts") or []
    return (
        f"Question: {question_text}\n"
        f"Method: {method_id}\n"
        f"Method profile: {METHOD_PROFILES.get(method_id, method_id)}\n"
        f"Question type: {question_type}\n"
        f"Difficulty: {difficulty}\n"
        f"Supporting facts count: {len(supporting_facts)}"
    )


def is_close_to_best(score: float, best_score: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(score), float(best_score), rel_tol=0.0, abs_tol=tol)


def build_rows_for_qid(qid: int, methods: Iterable[str], per_method: Dict[str, Dict[int, Dict]]) -> List[Dict]:
    method_samples = {method: per_method[method][qid] for method in methods}
    first_sample = next(iter(method_samples.values()))
    question_text = first_sample.get("question", "")
    gold_answer = first_sample.get("label", first_sample.get("answer", ""))

    best_f1 = max(float(sample.get("f1", 0.0)) for sample in method_samples.values())
    best_methods = [
        method
        for method in methods
        if is_close_to_best(float(method_samples[method].get("f1", 0.0)), best_f1)
    ]

    rows: List[Dict] = []
    for method in methods:
        sample = method_samples[method]
        token_cost = sample.get("token_cost") or {}
        rows.append(
            {
                "qid": qid,
                "question_text": question_text,
                "method_id": method,
                "method_profile": METHOD_PROFILES.get(method, method),
                "input_text": build_input_text(question_text, method, sample),
                "label": 1 if method in best_methods else 0,
                "best_methods_json": json.dumps(best_methods, ensure_ascii=False),
                "num_best_methods": len(best_methods),
                "is_tied_best": len(best_methods) > 1,
                "f1_score": float(sample.get("f1", 0.0)),
                "accuracy": float(sample.get("accuracy", 0.0)),
                "precision": float(sample.get("precision", 0.0)),
                "recall": float(sample.get("recall", 0.0)),
                "em": bool(sample.get("em", False)),
                "question_type": sample.get("type", ""),
                "difficulty": sample.get("level", ""),
                "gold_answer": gold_answer,
                "parsed_answer": sample.get("parsed_answer", ""),
                "raw_answer": sample.get("answer", ""),
                "method_output": sample.get("output", ""),
                "supporting_facts_json": json.dumps(sample.get("supporting_facts", []), ensure_ascii=False),
                "sample_uid": sample.get("_id", ""),
                "prompt_tokens": int(token_cost.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(token_cost.get("completion_tokens", 0) or 0),
                "total_tokens": int(token_cost.get("total_tokens", 0) or 0),
                "token_cost": float(token_cost.get("cost", 0.0) or 0.0),
                "source_dataset": "hotpot",
            }
        )
    return rows


def build_stats(
    splits: Dict[str, pd.DataFrame],
    methods: List[str],
    common_qids: set[int],
    split_map: Dict[str, set[int]],
) -> Dict:
    dataset_stats = {}
    for split_name, df in splits.items():
        if split_name == "full":
            continue
        labels = df["label"].value_counts().sort_index().to_dict() if "label" in df.columns else {}
        dataset_stats[split_name] = {
            "num_samples": int(len(df)),
            "num_questions": int(df["qid"].nunique()),
            "label_distribution": {str(k): int(v) for k, v in labels.items()},
        }

    total_questions = max(len(common_qids), 1)
    return {
        "total_questions_all_splits": int(sum(len(split_map[name] & common_qids) for name in SPLIT_NAMES)),
        "total_questions_common_across_methods": int(len(common_qids)),
        "methods": methods,
        "missing_from_common_qids": {
            split_name: int(len(split_map[split_name] - common_qids))
            for split_name in SPLIT_NAMES
        },
        "split_ratios_within_common_qids": {
            split_name: round(dataset_stats[split_name]["num_questions"] / total_questions, 4)
            for split_name in SPLIT_NAMES
        },
        "dataset_stats": dataset_stats,
    }


def main() -> None:
    args = parse_args()

    if not args.split_path.exists():
        raise FileNotFoundError(f"Split file not found: {args.split_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Rebuilding Hotpot train/val/test splits from per-method result files")
    print("=" * 80)
    print(f"Dataset dir:  {args.dataset_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Split file:   {args.split_path}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Methods:      {args.methods}")

    split_map = load_split_map(args.split_path)
    per_method = {
        method: load_method_results(args.dataset_dir, method, args.dataset_name)
        for method in args.methods
    }

    common_qids = set.intersection(*(set(rows.keys()) for rows in per_method.values())) if per_method else set()
    if not common_qids:
        raise RuntimeError("No common qids found across methods.")

    print(f"Common qids across all methods: {len(common_qids)}")

    splits: Dict[str, pd.DataFrame] = {}
    full_rows: List[Dict] = []

    for split_name in SPLIT_NAMES:
        split_qids = sorted(split_map[split_name] & common_qids)
        missing_qids = len(split_map[split_name] - common_qids)
        split_rows: List[Dict] = []
        for qid in split_qids:
            split_rows.extend(build_rows_for_qid(qid=qid, methods=args.methods, per_method=per_method))

        df = pd.DataFrame(split_rows)
        if not df.empty:
            df.sort_values(["qid", "method_id"], inplace=True, ignore_index=True)
        splits[split_name] = df
        full_rows.extend(split_rows)

        csv_path = args.output_dir / f"{split_name}.csv"
        jsonl_path = args.output_dir / f"{split_name}.jsonl"
        df.to_csv(csv_path, index=False)
        write_jsonl(df, jsonl_path)

        print(
            f"{split_name:>5}: questions={df['qid'].nunique():>4} "
            f"rows={len(df):>6} missing_from_common={missing_qids:>3} "
            f"-> {csv_path.name}, {jsonl_path.name}"
        )

    full_df = pd.DataFrame(full_rows)
    if not full_df.empty:
        full_df.sort_values(["qid", "method_id"], inplace=True, ignore_index=True)
    splits["full"] = full_df

    stats = build_stats(splits, methods=list(args.methods), common_qids=common_qids, split_map=split_map)
    stats_path = args.output_dir / "statistics.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()

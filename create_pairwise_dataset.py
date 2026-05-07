#!/usr/bin/env python3
"""
Build pairwise Hotpot datasets and transfer-test question lists.

Outputs:
1. `dataset/pairwise/{train,val,test}_pairwise.csv`
2. `dataset/eval_questions/{multihop,2wiki,...}.csv`
3. Legacy compatibility copies: `dataset/pairwise/{dataset}_pairwise.csv`
   for transfer test sets, containing only `qid,question_text`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from tqdm import tqdm


METHODS = ["dalk", "gr", "hippo", "lgraph", "light", "qagn"]
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pairwise datasets from repository-local CSV/JSONL files.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/pairwise"))
    parser.add_argument("--eval-questions-dir", type=Path, default=Path("dataset/eval_questions"))
    parser.add_argument("--train-csv", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("dataset/val.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--hotpot-dataset-name", default="hotpot")
    parser.add_argument("--text-column", default="input_text")
    parser.add_argument(
        "--transfer-datasets",
        nargs="*",
        default=["multihop", "2wiki"],
        help="Additional datasets to export as evaluation question lists.",
    )
    return parser.parse_args()


def load_scores(dataset_dir: Path, method: str, dataset_name: str) -> Dict[int, float]:
    score_file = dataset_dir / method / dataset_name / "results.score.json"
    scores: Dict[int, float] = {}
    with score_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "id" in item:
                scores[int(item["id"])] = float(item.get("f1", 0.0))
    return scores


def parse_best_methods(row_or_group) -> List[str]:
    payload = row_or_group.get("best_methods_json", "[]")
    if isinstance(payload, list):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return []
    return []


def fallback_input_text(row: pd.Series) -> str:
    return f"Question: {row['question_text']}\nMethod: {row['method_id']}"


def resolve_input_text(row: pd.Series, text_column: str) -> str:
    value = row.get(text_column, "")
    if isinstance(value, str) and value.strip():
        return value
    return fallback_input_text(row)


def create_pair(
    qid: int,
    question_text: str,
    pos_row: pd.Series,
    neg_row: pd.Series,
    all_scores: Dict[str, Dict[int, float]],
    best_methods: List[str],
    text_column: str,
) -> Dict:
    pos_method_id = pos_row["method_id"]
    neg_method_id = neg_row["method_id"]
    score_pos = all_scores.get(pos_method_id, {}).get(qid, float(pos_row.get("f1_score", 0.0)))
    score_neg = all_scores.get(neg_method_id, {}).get(qid, float(neg_row.get("f1_score", 0.0)))

    if random.random() < 0.5:
        method_a_id, method_b_id = pos_method_id, neg_method_id
        method_a_text = resolve_input_text(pos_row, text_column)
        method_b_text = resolve_input_text(neg_row, text_column)
        score_a, score_b = score_pos, score_neg
        pair_label = 1
    else:
        method_a_id, method_b_id = neg_method_id, pos_method_id
        method_a_text = resolve_input_text(neg_row, text_column)
        method_b_text = resolve_input_text(pos_row, text_column)
        score_a, score_b = score_neg, score_pos
        pair_label = 0

    return {
        "qid": qid,
        "question_text": question_text,
        "methodA_id": method_a_id,
        "methodB_id": method_b_id,
        "methodA_text": method_a_text,
        "methodB_text": method_b_text,
        "scoreA": float(score_a),
        "scoreB": float(score_b),
        "pair_label": pair_label,
        "best_methods_json": json.dumps(best_methods, ensure_ascii=False),
        "num_best_methods": len(best_methods),
        "is_tied_best": len(best_methods) > 1,
    }


def create_pairwise_data(
    df: pd.DataFrame,
    all_scores: Dict[str, Dict[int, float]],
    text_column: str,
) -> List[Dict]:
    pairwise_data: List[Dict] = []
    grouped = df.groupby("qid")

    for qid, group in tqdm(grouped, desc="Processing questions"):
        best_methods = parse_best_methods(group.iloc[0])
        positive_methods = group[group["label"] == 1]
        negative_methods = group[group["label"] == 0]
        if positive_methods.empty or negative_methods.empty:
            continue

        question_text = group.iloc[0]["question_text"]
        for _, pos_row in positive_methods.iterrows():
            for _, neg_row in negative_methods.iterrows():
                pairwise_data.append(
                    create_pair(
                        qid=qid,
                        question_text=question_text,
                        pos_row=pos_row,
                        neg_row=neg_row,
                        all_scores=all_scores,
                        best_methods=best_methods,
                        text_column=text_column,
                    )
                )
    return pairwise_data


def save_pairwise_split(
    split_name: str,
    df: pd.DataFrame,
    output_dir: Path,
    all_scores: Dict[str, Dict[int, float]],
    text_column: str,
) -> None:
    print(f"\n{'=' * 80}\nProcessing {split_name}\n{'=' * 80}")
    pairwise_df = pd.DataFrame(create_pairwise_data(df, all_scores, text_column=text_column))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}_pairwise.csv"
    pairwise_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"  pairs={len(pairwise_df)} questions={pairwise_df['qid'].nunique()}")
    if len(pairwise_df) > 0:
        label_dist = pairwise_df["pair_label"].value_counts(normalize=True).sort_index().to_dict()
        print(f"  pair_label distribution={label_dist}")


def load_question_map(dataset_dir: Path, dataset_name: str, methods: Iterable[str]) -> Dict[int, str]:
    per_method_qids: List[set[int]] = []
    question_map: Dict[int, str] = {}

    for method in methods:
        score_file = dataset_dir / method / dataset_name / "results.score.json"
        qids_for_method: set[int] = set()
        with score_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                qid = int(item["id"])
                qids_for_method.add(qid)
                question_map.setdefault(qid, item.get("question", ""))
        per_method_qids.append(qids_for_method)

    common_qids = sorted(set.intersection(*per_method_qids)) if per_method_qids else []
    return {qid: question_map[qid] for qid in common_qids}


def save_eval_questions(
    dataset_name: str,
    dataset_dir: Path,
    eval_questions_dir: Path,
    pairwise_dir: Path,
    methods: Iterable[str],
) -> None:
    question_map = load_question_map(dataset_dir, dataset_name, methods)
    eval_questions_dir.mkdir(parents=True, exist_ok=True)
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [{"qid": qid, "question_text": question_text} for qid, question_text in question_map.items()]
    )
    eval_path = eval_questions_dir / f"{dataset_name}.csv"
    legacy_path = pairwise_dir / f"{dataset_name}_pairwise.csv"
    df.to_csv(eval_path, index=False)
    df.to_csv(legacy_path, index=False)
    print(f"Saved eval questions: {eval_path} ({len(df)} questions)")
    print(f"Saved legacy compatibility copy: {legacy_path}")


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)

    print("=" * 80)
    print("Creating pairwise datasets and transfer evaluation question lists")
    print("=" * 80)

    all_scores = {
        method: load_scores(args.dataset_dir, method, args.hotpot_dataset_name)
        for method in METHODS
    }

    for split_name, csv_path in (
        ("train", args.train_csv),
        ("val", args.val_csv),
        ("test", args.test_csv),
    ):
        df = pd.read_csv(csv_path)
        save_pairwise_split(
            split_name=split_name,
            df=df,
            output_dir=args.output_dir,
            all_scores=all_scores,
            text_column=args.text_column,
        )

    for dataset_name in args.transfer_datasets:
        save_eval_questions(
            dataset_name=dataset_name,
            dataset_dir=args.dataset_dir,
            eval_questions_dir=args.eval_questions_dir,
            pairwise_dir=args.output_dir,
            methods=METHODS,
        )


if __name__ == "__main__":
    main()

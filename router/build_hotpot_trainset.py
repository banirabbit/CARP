import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


METHOD_ORDER = ["qagn", "dalk", "gr", "hippo", "light", "lgraph"]


def load_jsonl(path: Path) -> Dict[int, Dict]:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data[int(item["id"])] = item
    return data


def extract_token_cost(item: Dict) -> float:
    token_cost = item.get("token_cost", {})
    if isinstance(token_cost, dict):
        return float(token_cost.get("prompt_tokens", token_cost.get("total_tokens", 0)))
    if isinstance(token_cost, (int, float)):
        return float(token_cost)
    return 0.0


def choose_best_actual(methods: Dict[str, Dict]) -> str:
    if not methods:
        return ""
    ranked: List[Tuple[str, float, float]] = []
    for method, result in methods.items():
        f1 = float(result.get("f1", 0.0))
        cost = extract_token_cost(result)
        ranked.append((method, f1, cost))
    ranked.sort(key=lambda x: (-x[1], x[2], x[0]))
    return ranked[0][0]


def compute_tau(train_items: List[Dict]) -> float:
    f1_values: List[float] = []
    for item in train_items:
        for result in item.get("methods", {}).values():
            f1 = result.get("f1", None)
            if isinstance(f1, (int, float)):
                f1_values.append(float(f1))
    if not f1_values:
        return 0.0
    f1_values.sort()
    n = len(f1_values)
    mid = n // 2
    if n % 2 == 1:
        return f1_values[mid]
    return (f1_values[mid - 1] + f1_values[mid]) / 2.0


def build_trainset(dataset_dir: Path, split_path: Path, output_path: Path, stats_path: Path):
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    train_qids = set(split_data["train_qids"])

    per_method = {}
    for method in METHOD_ORDER:
        method_path = dataset_dir / method / "hotpot" / "results.score.json"
        per_method[method] = load_jsonl(method_path)

    train_items: List[Dict] = []
    missing_counts = {m: 0 for m in METHOD_ORDER}

    for qid in sorted(train_qids):
        methods = {}
        base_question = ""
        base_answer = ""

        for method in METHOD_ORDER:
            result = per_method[method].get(qid)
            if result is None:
                missing_counts[method] += 1
                continue
            if not base_question:
                base_question = result.get("question", "")
            if not base_answer:
                base_answer = result.get("answer", "")
            methods[method] = result

        if not methods:
            continue

        item = {
            "id": qid,
            "question": base_question,
            "answer": base_answer,
            "methods": methods,
            "p2l_scores": {},
            "best_method_by_p2l": "",
            "best_method_actual": choose_best_actual(methods),
            "prediction_correct": False,
        }
        train_items.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    tau = compute_tau(train_items)
    stats = {
        "dataset": "hotpot",
        "split": "train",
        "train_questions": len(train_items),
        "methods": METHOD_ORDER,
        "cpp_plus_tau": tau,
        "epsilon": 1e-8,
        "missing_counts": missing_counts,
        "output_path": str(output_path),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build merged Hotpot train set for router evaluation.")
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Directory containing per-method dataset subdirectories.",
    )
    parser.add_argument(
        "--split-path",
        default="dataset/data_splits.json",
        help="Path to data_splits.json.",
    )
    parser.add_argument(
        "--output-path",
        default="router/data/hotpot_train.jsonl",
        help="Output merged train JSONL path.",
    )
    parser.add_argument(
        "--stats-path",
        default="router/data/hotpot_train_cpp_plus_stats.json",
        help="Output JSON path for computed CPP+ stats.",
    )
    args = parser.parse_args()

    build_trainset(
        dataset_dir=Path(args.dataset_dir),
        split_path=Path(args.split_path),
        output_path=Path(args.output_path),
        stats_path=Path(args.stats_path),
    )


if __name__ == "__main__":
    main()

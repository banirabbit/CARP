from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from router.adaptive_router import RouterConfig, RouterMode
    from router.evaluate_router_method import METHOD_ORDER, RouterMethodEvaluator
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from router.adaptive_router import RouterConfig, RouterMode
    from router.evaluate_router_method import METHOD_ORDER, RouterMethodEvaluator


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
COSTS_PATH = ROOT / "router" / "method_costs.json"

DATASETS = {
    "hotpot": {
        "scored": ROOT / "router" / "results" / "scorer" / "hotpot.jsonl",
        "train_jsonl": ROOT / "router" / "data" / "hotpot_train.jsonl",
        "raw": ROOT / "router" / "data" / "hotpot414.jsonl",
    },
    "multi": {
        "scored": ROOT / "router" / "results" / "scorer" / "multihop.jsonl",
        "train_jsonl": ROOT / "router" / "data" / "hotpot_train.jsonl",
        "raw": ROOT / "router" / "data" / "multihop414.jsonl",
    },
    "2wiki": {
        "scored": ROOT / "router" / "results" / "scorer" / "2wiki.jsonl",
        "train_jsonl": ROOT / "router" / "data" / "hotpot_train.jsonl",
        "raw": ROOT / "router" / "data" / "2wiki414.jsonl",
    },
}


@dataclass(frozen=True)
class DynamicCostConfig:
    top_k: int
    sim_floor: float
    blend: float
    p: float
    stat: str
    analyzer: str
    sim_power: float

    def asdict(self) -> Dict:
        return asdict(self)

    def name(self) -> str:
        return (
            f"k{self.top_k}_sf{self.sim_floor:g}_b{self.blend:g}_p{self.p:g}_"
            f"{self.stat}_{self.analyzer}_sp{self.sim_power:g}"
        )


BEST_DYNAMIC_COST_CONFIG = DynamicCostConfig(
    top_k=5,
    sim_floor=0.0,
    blend=0.5,
    p=6.0,
    stat="median",
    analyzer="word",
    sim_power=1.0,
)


def load_static_costs(costs_path: Path = COSTS_PATH) -> Dict[str, float]:
    data = json.load(open(costs_path, "r", encoding="utf-8"))
    return {m: float(stats["mean"]) for m, stats in data.items()}


def load_train_cost_matrix(dataset_dir: Path = DATA_DIR) -> Tuple[List[str], np.ndarray]:
    train_df = pd.read_csv(dataset_dir / "train.csv", usecols=["qid", "question_text"])
    train_questions = (
        train_df.drop_duplicates(subset=["qid"])
        .sort_values("qid")
        .reset_index(drop=True)
    )
    qids = train_questions["qid"].tolist()
    qid_to_idx = {int(qid): i for i, qid in enumerate(qids)}
    questions = train_questions["question_text"].tolist()

    cost_matrix = np.zeros((len(qids), len(METHOD_ORDER)), dtype=np.float32)
    for m_idx, method in enumerate(METHOD_ORDER):
        path = dataset_dir / method / "hotpot" / "results.score.json"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = row.get("id")
                if qid is None or int(qid) not in qid_to_idx:
                    continue
                token_cost = row.get("token_cost", {})
                cost = float(token_cost.get("prompt_tokens", token_cost.get("total_tokens", 0.0)))
                cost_matrix[qid_to_idx[int(qid)], m_idx] = cost
    return questions, cost_matrix


def fit_vectorizer(train_questions: List[str], analyzer: str):
    if analyzer == "char":
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    else:
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)
    train_matrix = vectorizer.fit_transform(train_questions)
    return vectorizer, train_matrix


def precompute_neighbors(vectorizer, train_matrix, test_questions: List[str], max_k: int) -> Tuple[np.ndarray, np.ndarray]:
    test_matrix = vectorizer.transform(test_questions)
    sim = test_matrix @ train_matrix.T
    sim = sim.toarray()
    top_idx = np.argsort(-sim, axis=1)[:, :max_k]
    top_sim = np.take_along_axis(sim, top_idx, axis=1)
    return top_idx, top_sim


def weighted_stat(values: np.ndarray, weights: np.ndarray, stat: str) -> float:
    if len(values) == 0:
        return float("nan")
    if stat == "median":
        order = np.argsort(values)
        vals = values[order]
        wts = weights[order]
        cdf = np.cumsum(wts) / max(np.sum(wts), 1e-12)
        return float(vals[np.searchsorted(cdf, 0.5, side="left")])
    if stat == "p25":
        order = np.argsort(values)
        vals = values[order]
        wts = weights[order]
        cdf = np.cumsum(wts) / max(np.sum(wts), 1e-12)
        return float(vals[np.searchsorted(cdf, 0.25, side="left")])
    return float(np.average(values, weights=weights))


def build_dynamic_costs_for_question(
    static_costs: Dict[str, float],
    top_idx_row: np.ndarray,
    top_sim_row: np.ndarray,
    train_cost_matrix: np.ndarray,
    cfg: DynamicCostConfig,
) -> Dict[str, float]:
    dynamic_costs = {}
    sims = np.maximum(top_sim_row[: cfg.top_k], 0.0)
    if cfg.sim_power != 1.0:
        sims = sims ** cfg.sim_power

    valid = sims >= cfg.sim_floor
    max_sim = float(np.max(sims)) if len(sims) else 0.0
    alpha = cfg.blend * max(0.0, min(1.0, max_sim))

    for m_idx, method in enumerate(METHOD_ORDER):
        static_cost = static_costs[method]
        if np.any(valid):
            idx = top_idx_row[: cfg.top_k][valid]
            wts = sims[valid]
            vals = train_cost_matrix[idx, m_idx]
            local_cost = weighted_stat(vals, wts, cfg.stat)
            if math.isnan(local_cost) or local_cost <= 0:
                local_cost = static_cost
        else:
            local_cost = static_cost
        dynamic_costs[method] = (1.0 - alpha) * static_cost + alpha * local_cost
    return dynamic_costs


def build_dynamic_cost_router_config(cfg: DynamicCostConfig) -> RouterConfig:
    return RouterConfig(
        methods=METHOD_ORDER,
        routing_mode=RouterMode.PARETO_COMPROMISE,
        confidence_threshold=0.15,
        quality_threshold=0.35,
        compromise_p=cfg.p,
        score_band_delta=0.6,
        fallback_method="qagn",
    )


def extract_method_costs_from_item(item: Dict) -> Dict[str, float]:
    costs: Dict[str, float] = {}
    for method in METHOD_ORDER:
        method_result = item.get("methods", {}).get(method, {})
        token_cost = method_result.get("token_cost", {})
        costs[method] = float(token_cost.get("prompt_tokens", token_cost.get("total_tokens", 0.0)) or 0.0)
    return costs


def build_routed_record(
    evaluator: RouterMethodEvaluator,
    item: Dict,
    source_index: int,
    chosen_method: str,
    cost_details: Dict,
) -> Dict:
    method_result = item.get("methods", {}).get(chosen_method, {})
    return {
        "id": item.get("id", source_index),
        "source_index": source_index,
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "chosen_method": chosen_method,
        "router_decision": cost_details,
        "output": method_result.get("output", ""),
        "parsed_answer": method_result.get("parsed_answer", ""),
        "accuracy": method_result.get("accuracy", 0),
        "f1": method_result.get("f1", 0.0),
        "em": method_result.get("em", False),
        "token_cost": method_result.get("token_cost", {}),
        "time_cost": evaluator._extract_time_cost(chosen_method, method_result),
        "p2l_scores": item.get("p2l_scores", {}),
        "best_method_by_p2l": item.get("best_method_by_p2l", ""),
        "best_method_actual": item.get("best_method_actual", ""),
    }


def route_scored_file_with_cost_resolver(
    scored_path: str | Path,
    train_jsonl_path: str | Path,
    router_config: RouterConfig,
    cost_resolver: Callable[[int, Dict], Dict[str, float]],
    cost_mode: str = "custom",
) -> Tuple[RouterMethodEvaluator, List[Dict]]:
    evaluator = RouterMethodEvaluator(
        test_data_path=str(scored_path),
        costs_path=str(COSTS_PATH),
        train_data_path=str(train_jsonl_path),
        router_config=router_config,
    )

    routed_results: List[Dict] = []
    for i, item in enumerate(evaluator.test_data):
        p2l_scores = item.get("p2l_scores", {})
        if not p2l_scores:
            continue
        costs = cost_resolver(i, item)
        decision = evaluator.router.route(scores=p2l_scores, costs=costs)
        routed_results.append(
            build_routed_record(
                evaluator,
                item,
                i,
                decision.chosen_method,
                {
                    "scores": decision.scores,
                    "probs": decision.probs,
                    "gap": decision.gap,
                    "uncertainty": decision.uncertainty,
                    "reason": decision.reason,
                    f"{cost_mode}_costs": costs,
                },
            )
        )
    return evaluator, routed_results


def route_dataset_with_dynamic_cost(
    dataset_name: str,
    cfg: DynamicCostConfig,
    train_questions: Optional[List[str]] = None,
    train_cost_matrix: Optional[np.ndarray] = None,
    static_costs: Optional[Dict[str, float]] = None,
) -> Tuple[RouterMethodEvaluator, List[Dict]]:
    ds_cfg = DATASETS[dataset_name]
    static_costs = static_costs or load_static_costs()
    if train_questions is None or train_cost_matrix is None:
        train_questions, train_cost_matrix = load_train_cost_matrix()

    vectorizer, train_matrix = fit_vectorizer(train_questions, cfg.analyzer)
    evaluator = RouterMethodEvaluator(
        test_data_path=str(ds_cfg["scored"]),
        costs_path=str(COSTS_PATH),
        train_data_path=str(ds_cfg["train_jsonl"]),
        router_config=build_dynamic_cost_router_config(cfg),
    )
    test_questions = [item.get("question", "") for item in evaluator.test_data]
    top_idx, top_sim = precompute_neighbors(vectorizer, train_matrix, test_questions, cfg.top_k)

    routed_results: List[Dict] = []
    for i, item in enumerate(evaluator.test_data):
        p2l_scores = item.get("p2l_scores", {})
        if not p2l_scores:
            continue
        dynamic_costs = build_dynamic_costs_for_question(
            static_costs, top_idx[i], top_sim[i], train_cost_matrix, cfg
        )
        decision = evaluator.router.route(scores=p2l_scores, costs=dynamic_costs)
        chosen_method = decision.chosen_method
        method_result = item.get("methods", {}).get(chosen_method, {})
        routed_results.append(
            {
                "id": item.get("id", i),
                "source_index": i,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "chosen_method": chosen_method,
                "router_decision": {
                    "scores": decision.scores,
                    "probs": decision.probs,
                    "gap": decision.gap,
                    "uncertainty": decision.uncertainty,
                    "reason": decision.reason,
                    "dynamic_costs": dynamic_costs,
                },
                "output": method_result.get("output", ""),
                "parsed_answer": method_result.get("parsed_answer", ""),
                "accuracy": method_result.get("accuracy", 0),
                "f1": method_result.get("f1", 0.0),
                "em": method_result.get("em", False),
                "token_cost": method_result.get("token_cost", {}),
                "time_cost": evaluator._extract_time_cost(chosen_method, method_result),
                "p2l_scores": p2l_scores,
                "best_method_by_p2l": item.get("best_method_by_p2l", ""),
                "best_method_actual": item.get("best_method_actual", ""),
            }
        )
    return evaluator, routed_results


def route_scored_file_with_dynamic_cost(
    scored_path: str | Path,
    train_jsonl_path: str | Path,
    cfg: DynamicCostConfig,
    train_questions: Optional[List[str]] = None,
    train_cost_matrix: Optional[np.ndarray] = None,
    static_costs: Optional[Dict[str, float]] = None,
) -> Tuple[RouterMethodEvaluator, List[Dict]]:
    static_costs = static_costs or load_static_costs()
    if train_questions is None or train_cost_matrix is None:
        train_questions, train_cost_matrix = load_train_cost_matrix()

    vectorizer, train_matrix = fit_vectorizer(train_questions, cfg.analyzer)
    router_config = build_dynamic_cost_router_config(cfg)
    evaluator = RouterMethodEvaluator(
        test_data_path=str(scored_path),
        costs_path=str(COSTS_PATH),
        train_data_path=str(train_jsonl_path),
        router_config=router_config,
    )
    test_questions = [item.get("question", "") for item in evaluator.test_data]
    top_idx, top_sim = precompute_neighbors(vectorizer, train_matrix, test_questions, cfg.top_k)

    def resolver(i: int, item: Dict) -> Dict[str, float]:
        return build_dynamic_costs_for_question(
            static_costs, top_idx[i], top_sim[i], train_cost_matrix, cfg
        )

    return route_scored_file_with_cost_resolver(
        scored_path=scored_path,
        train_jsonl_path=train_jsonl_path,
        router_config=router_config,
        cost_resolver=resolver,
        cost_mode="dynamic",
    )


def evaluate_dynamic_cost_dataset(
    dataset_name: str,
    cfg: DynamicCostConfig,
    train_questions: Optional[List[str]] = None,
    train_cost_matrix: Optional[np.ndarray] = None,
    static_costs: Optional[Dict[str, float]] = None,
) -> Dict:
    evaluator, routed_results = route_dataset_with_dynamic_cost(
        dataset_name,
        cfg,
        train_questions=train_questions,
        train_cost_matrix=train_cost_matrix,
        static_costs=static_costs,
    )
    return evaluator.calculate_metrics(routed_results)


def attach_icer_against_qagn(
    evaluator: RouterMethodEvaluator,
    router_metrics: Dict,
    individual_metrics: Dict[str, Dict],
) -> None:
    baseline_metrics = individual_metrics.get("qagn")
    if not baseline_metrics:
        return

    router_icer = evaluator._calculate_icer_against_baseline(
        target_f1=router_metrics.get("f1", 0.0),
        target_cost=router_metrics.get("avg_token_cost", 0.0),
        baseline_f1=baseline_metrics.get("f1", 0.0),
        baseline_cost=baseline_metrics.get("avg_token_cost", 0.0),
    )
    router_metrics["icer_qagn"] = router_icer
    router_metrics["icer_qagn_valid_rate"] = 0.0 if router_icer is None else 100.0
    router_metrics["icer_qagn_zero_effect_rate"] = 100.0 if router_icer is None else 0.0

    for method_name, method_metrics in individual_metrics.items():
        avg_icer = evaluator._calculate_icer_against_baseline(
            target_f1=method_metrics.get("f1", 0.0),
            target_cost=method_metrics.get("avg_token_cost", 0.0),
            baseline_f1=baseline_metrics.get("f1", 0.0),
            baseline_cost=baseline_metrics.get("avg_token_cost", 0.0),
        )
        method_metrics["icer_qagn"] = avg_icer
        method_metrics["icer_qagn_valid_rate"] = 0.0 if avg_icer is None else 100.0
        method_metrics["icer_qagn_zero_effect_rate"] = 100.0 if avg_icer is None else 0.0


def save_jsonl(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic-cost + pareto-compromise routing.")
    parser.add_argument(
        "--scored-path",
        type=Path,
        default=ROOT / "router" / "results" / "scorer" / "hotpot.jsonl",
        help="Scored JSONL produced by generate_test_scores.py",
    )
    parser.add_argument(
        "--train-jsonl-path",
        type=Path,
        default=ROOT / "router" / "data" / "hotpot_train.jsonl",
        help="Merged Hotpot training JSONL for CPP+ tau and router-side metadata.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATA_DIR,
        help="Dataset directory containing train.csv and per-method result files.",
    )
    parser.add_argument(
        "--costs-path",
        type=Path,
        default=COSTS_PATH,
        help="Static method cost summary JSON.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=ROOT / "router" / "results" / "dynamic_cost" / "hotpot.routed.jsonl",
        help="Path to write routed per-question records.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "router" / "results" / "dynamic_cost" / "hotpot.summary.json",
        help="Path to write routing metrics summary.",
    )
    parser.add_argument("--top-k", type=int, default=BEST_DYNAMIC_COST_CONFIG.top_k)
    parser.add_argument("--sim-floor", type=float, default=BEST_DYNAMIC_COST_CONFIG.sim_floor)
    parser.add_argument("--blend", type=float, default=BEST_DYNAMIC_COST_CONFIG.blend)
    parser.add_argument("--compromise-p", type=float, default=BEST_DYNAMIC_COST_CONFIG.p)
    parser.add_argument("--stat", choices=["mean", "median", "p25"], default=BEST_DYNAMIC_COST_CONFIG.stat)
    parser.add_argument("--analyzer", choices=["word", "char"], default=BEST_DYNAMIC_COST_CONFIG.analyzer)
    parser.add_argument("--sim-power", type=float, default=BEST_DYNAMIC_COST_CONFIG.sim_power)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DynamicCostConfig(
        top_k=args.top_k,
        sim_floor=args.sim_floor,
        blend=args.blend,
        p=args.compromise_p,
        stat=args.stat,
        analyzer=args.analyzer,
        sim_power=args.sim_power,
    )

    train_questions, train_cost_matrix = load_train_cost_matrix(args.dataset_dir)
    static_costs = load_static_costs(args.costs_path)
    evaluator, routed_results = route_scored_file_with_dynamic_cost(
        scored_path=args.scored_path,
        train_jsonl_path=args.train_jsonl_path,
        cfg=cfg,
        train_questions=train_questions,
        train_cost_matrix=train_cost_matrix,
        static_costs=static_costs,
    )

    router_metrics = evaluator.calculate_metrics(routed_results)
    individual_metrics = evaluator.calculate_individual_method_metrics()
    attach_icer_against_qagn(evaluator, router_metrics, individual_metrics)
    evaluator.print_metrics(router_metrics, individual_metrics)

    save_jsonl(routed_results, args.output_jsonl)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": cfg.asdict(),
        "scored_path": str(args.scored_path),
        "train_jsonl_path": str(args.train_jsonl_path),
        "costs_path": str(args.costs_path),
        "router_metrics": router_metrics,
        "individual_metrics": individual_metrics,
        "num_routed_questions": len(routed_results),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved routed records to: {args.output_jsonl}")
    print(f"Saved summary to: {args.summary_json}")


if __name__ == "__main__":
    main()

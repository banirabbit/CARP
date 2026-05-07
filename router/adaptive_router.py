"""
GraphRAG 路由器：
1) score_only：只按预测质量 Top1
2) cheapest_first：先跑便宜方法，置信度/质量不达标再升级
3) confidence_cost：置信度门槛 + fallback + 成本效用
4) pareto_compromise：基于 query 级质量预测 + 方法级静态成本先验，选择质量-成本折中解
5) score_band_cheapest：只在接近 Top1 的候选带内选择最低静态成本方法
"""

import logging
from typing import Dict, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class RouterMode:
    SCORE_ONLY = "score_only"
    CHEAPEST_FIRST = "cheapest_first"
    CONFIDENCE_COST = "confidence_cost"
    PARETO_COMPROMISE = "pareto_compromise"
    SCORE_BAND_CHEAPEST = "score_band_cheapest"


@dataclass
class RouterConfig:
    methods: list = None
    routing_mode: str = RouterMode.SCORE_BAND_CHEAPEST

    # 成本权重 (仅 confidence_cost 模式使用)
    lambda_min: float = 0.4
    lambda_max: float = 1.0

    # 置信度/质量阈值
    confidence_threshold: float = 0.15
    quality_threshold: Optional[float] = None  # 仅 cheapest_first 模式用
    compromise_p: float = 4.0
    score_band_delta: float = 0.6

    fallback_method: Optional[str] = None  # 默认 methods[0]

    def __post_init__(self):
        if self.methods is None:
            self.methods = ["qagn", "gr", "dalk", "hippo", "lgraph", "light"]
        if self.fallback_method is None:
            self.fallback_method = self.methods[0]
        if self.routing_mode not in {
            RouterMode.SCORE_ONLY,
            RouterMode.CHEAPEST_FIRST,
            RouterMode.CONFIDENCE_COST,
            RouterMode.PARETO_COMPROMISE,
            RouterMode.SCORE_BAND_CHEAPEST,
        }:
            raise ValueError(f"未知的 routing_mode: {self.routing_mode}")


@dataclass
class RouterDecision:
    chosen_method: str
    scores: Dict[str, float]
    probs: Dict[str, float]
    costs: Dict[str, float]
    norm_costs: Dict[str, float]
    utilities: Dict[str, float]
    gap: float
    uncertainty: float
    lambda_dyn: float
    reason: str


class AdaptiveGraphRAGRouter:
    def __init__(self, config: RouterConfig = None):
        self.config = config or RouterConfig()
        logger.info(f"初始化路由器: {self.config}")

    # 核心入口 -----------------------------------------------------
    def route(self, scores: Dict[str, float], costs: Dict[str, float]) -> RouterDecision:
        methods, raw_scores, cost_arr = self._align_data(scores, costs)
        if len(methods) == 0:
            return self._create_fallback_decision(scores, costs, "所有方法缺失数据")

        probs = self._softmax(raw_scores)
        probs_dict = dict(zip(methods, probs))

        mode = self.config.routing_mode
        if mode == RouterMode.SCORE_ONLY:
            return self._route_score_only(methods, raw_scores, probs, cost_arr, probs_dict)
        if mode == RouterMode.CHEAPEST_FIRST:
            return self._route_cheapest_first(methods, raw_scores, probs, cost_arr, probs_dict)
        if mode == RouterMode.CONFIDENCE_COST:
            return self._route_confidence_cost(methods, raw_scores, probs, cost_arr, probs_dict)
        if mode == RouterMode.PARETO_COMPROMISE:
            return self._route_pareto_compromise(methods, raw_scores, probs, cost_arr, probs_dict)
        if mode == RouterMode.SCORE_BAND_CHEAPEST:
            return self._route_score_band_cheapest(methods, raw_scores, probs, cost_arr, probs_dict)

        raise ValueError(f"未知的 routing_mode: {mode}")

    # 模式 1：只看分数 --------------------------------------------
    def _route_score_only(
        self,
        methods: list,
        scores: np.ndarray,
        probs: np.ndarray,
        costs: np.ndarray,
        probs_dict: Dict[str, float],
    ) -> RouterDecision:
        best_idx = int(np.argmax(scores))
        best_prob = float(probs[best_idx])
        sorted_probs = np.sort(probs)[::-1]
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

        return RouterDecision(
            chosen_method=methods[best_idx],
            scores=dict(zip(methods, scores)),
            probs=probs_dict,
            costs=dict(zip(methods, costs)),
            norm_costs={},
            utilities={},
            gap=gap,
            uncertainty=1.0 - best_prob,
            lambda_dyn=0.0,
            reason=f"score_only: 选择 Top1 预测质量 {methods[best_idx]} (prob={best_prob:.2f})",
        )

    # 模式 2：最便宜优先级联 --------------------------------------
    def _route_cheapest_first(
        self,
        methods: list,
        scores: np.ndarray,
        probs: np.ndarray,
        costs: np.ndarray,
        probs_dict: Dict[str, float],
    ) -> RouterDecision:
        order = list(np.argsort(costs))  # 从便宜到昂贵
        reason_trace: List[str] = []
        chosen_idx = order[-1]
        trigger_reason = ""

        for idx in order:
            method = methods[idx]
            prob = float(probs[idx])
            score = float(scores[idx])
            cost = float(costs[idx])
            prob_ok = prob >= self.config.confidence_threshold
            quality_ok = (
                True
                if self.config.quality_threshold is None
                else score >= self.config.quality_threshold
            )
            reason_trace.append(
                f"{method}(cost={cost:.0f}, prob={prob:.2f}, score={score:.2f})"
            )
            if prob_ok and quality_ok:
                chosen_idx = idx
                trigger_reason = "置信度达标"
                if self.config.quality_threshold is not None:
                    trigger_reason += f" 且分数≥{self.config.quality_threshold:.2f}"
                break
        else:
            chosen_idx = int(np.argmax(probs))
            trigger_reason = "无方法达标，回退概率最高"

        sorted_probs = np.sort(probs)[::-1]
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        reason = (
            f"cheapest_first: {trigger_reason} -> {methods[chosen_idx]} | "
            f"路径: {' -> '.join(reason_trace)}"
        )

        return RouterDecision(
            chosen_method=methods[chosen_idx],
            scores=dict(zip(methods, scores)),
            probs=probs_dict,
            costs=dict(zip(methods, costs)),
            norm_costs={},
            utilities={},
            gap=gap,
            uncertainty=1.0 - float(probs[chosen_idx]),
            lambda_dyn=0.0,
            reason=reason,
        )

    # 模式 3：置信度门槛 + fallback + 成本 -----------------------
    def _route_confidence_cost(
        self,
        methods: list,
        scores: np.ndarray,
        probs: np.ndarray,
        costs: np.ndarray,
        probs_dict: Dict[str, float],
    ) -> RouterDecision:
        score_dict = dict(zip(methods, scores))
        cost_dict = dict(zip(methods, costs))
        norm_costs = self._normalize_costs(costs)
        norm_cost_dict = dict(zip(methods, norm_costs))
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        sorted_probs = np.sort(probs)[::-1]
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        uncertainty = 1.0 - top_prob

        if top_prob < self.config.confidence_threshold:
            return RouterDecision(
                chosen_method=self.config.fallback_method,
                scores=score_dict,
                probs=probs_dict,
                costs=cost_dict,
                norm_costs=norm_cost_dict,
                utilities={},
                gap=gap,
                uncertainty=uncertainty,
                lambda_dyn=0.0,
                reason=(
                    f"confidence_cost: 置信度不足 ({top_prob:.2f} < {self.config.confidence_threshold}), "
                    f"fallback -> {self.config.fallback_method}"
                ),
            )

        lambda_dyn = self._compute_dynamic_lambda(uncertainty)
        utilities = probs - lambda_dyn * norm_costs
        utility_dict = dict(zip(methods, utilities))
        chosen_idx = int(np.argmax(utilities))

        return RouterDecision(
            chosen_method=methods[chosen_idx],
            scores=score_dict,
            probs=probs_dict,
            costs=cost_dict,
            norm_costs=norm_cost_dict,
            utilities=utility_dict,
            gap=gap,
            uncertainty=uncertainty,
            lambda_dyn=lambda_dyn,
            reason=(
                "confidence_cost: 置信度达标后按动态成本效用选路, "
                f"chosen={methods[chosen_idx]}, "
                f"top_prob={top_prob:.2f}, lambda={lambda_dyn:.2f}, "
                f"utility={utilities[chosen_idx]:.4f}"
            ),
        )

    def _route_pareto_compromise(
        self,
        methods: list,
        scores: np.ndarray,
        probs: np.ndarray,
        costs: np.ndarray,
        probs_dict: Dict[str, float],
    ) -> RouterDecision:
        score_dict = dict(zip(methods, scores))
        cost_dict = dict(zip(methods, costs))

        quality_scores = self._normalize_scores(scores)
        cost_benefits = self._cost_benefits(costs)
        pareto_idx = self._pareto_frontier_indices(quality_scores, cost_benefits)
        p = max(float(self.config.compromise_p), 1.0)

        distances = np.full(len(methods), np.inf, dtype=float)
        for idx in pareto_idx:
            distances[idx] = ((1.0 - quality_scores[idx]) ** p + (1.0 - cost_benefits[idx]) ** p) ** (1.0 / p)

        chosen_idx = min(
            pareto_idx,
            key=lambda idx: (distances[idx], float(costs[idx]), -float(quality_scores[idx])),
        )
        shortlist_methods = [methods[idx] for idx in pareto_idx]

        return RouterDecision(
            chosen_method=methods[chosen_idx],
            scores=score_dict,
            probs=probs_dict,
            costs=cost_dict,
            norm_costs=dict(zip(methods, 1.0 - cost_benefits)),
            utilities=dict(zip(methods, -distances)),
            gap=0.0,
            uncertainty=1.0 - float(np.max(probs)),
            lambda_dyn=0.0,
            reason=(
                "pareto_compromise: 基于 query 级分数与方法级静态成本先验选折中解, "
                f"p={p:.2f}, shortlist={shortlist_methods}, "
                f"chosen={methods[chosen_idx]}, distance={distances[chosen_idx]:.4f}"
            ),
        )

    def _route_score_band_cheapest(
        self,
        methods: list,
        scores: np.ndarray,
        probs: np.ndarray,
        costs: np.ndarray,
        probs_dict: Dict[str, float],
    ) -> RouterDecision:
        score_dict = dict(zip(methods, scores))
        cost_dict = dict(zip(methods, costs))
        quality_scores = self._normalize_scores(scores)
        delta = max(float(self.config.score_band_delta), 0.0)
        quality_floor = max(float(np.max(quality_scores)) - delta, 0.0)
        candidate_idx = [
            idx for idx, quality in enumerate(quality_scores)
            if float(quality) >= quality_floor - 1e-12
        ]
        chosen_idx = min(
            candidate_idx,
            key=lambda idx: (float(costs[idx]), -float(quality_scores[idx]), -float(probs[idx])),
        )

        sorted_probs = np.sort(probs)[::-1]
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        shortlist_methods = [methods[idx] for idx in candidate_idx]

        return RouterDecision(
            chosen_method=methods[chosen_idx],
            scores=score_dict,
            probs=probs_dict,
            costs=cost_dict,
            norm_costs=dict(zip(methods, self._normalize_costs(costs))),
            utilities=dict(zip(methods, quality_scores)),
            gap=gap,
            uncertainty=1.0 - float(np.max(probs)),
            lambda_dyn=0.0,
            reason=(
                "score_band_cheapest: 只在 query 级高分候选带内按最低静态成本选路, "
                f"delta={delta:.2f}, quality_floor={quality_floor:.4f}, "
                f"shortlist={shortlist_methods}, chosen={methods[chosen_idx]}"
            ),
        )

    # 辅助函数 ----------------------------------------------------
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _align_data(self, scores: Dict, costs: Dict) -> Tuple[list, np.ndarray, np.ndarray]:
        methods = []
        score_list = []
        cost_list = []
        for m in self.config.methods:
            if m not in scores or m not in costs:
                continue
            if costs[m] <= 0:
                continue
            methods.append(m)
            score_list.append(scores[m])
            cost_list.append(costs[m])
        return methods, np.array(score_list), np.array(cost_list)

    def _normalize_costs(self, costs: np.ndarray) -> np.ndarray:
        if len(costs) == 0:
            return np.array([])
        log_costs = np.log1p(costs)
        median_log = np.median(log_costs)
        if median_log < 1e-6:
            median_log = 1.0
        return log_costs / median_log

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return np.array([])
        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        if score_max - score_min < 1e-9:
            return np.ones_like(scores, dtype=float)
        return (scores - score_min) / (score_max - score_min)

    def _cost_benefits(self, costs: np.ndarray) -> np.ndarray:
        if len(costs) == 0:
            return np.array([])
        normalized_cost = np.log1p(costs) / (1.0 + np.log1p(costs))
        return 1.0 - normalized_cost

    def _pareto_frontier_indices(self, quality_scores: np.ndarray, cost_benefits: np.ndarray) -> List[int]:
        frontier = []
        for idx in range(len(quality_scores)):
            dominated = False
            for other_idx in range(len(quality_scores)):
                if idx == other_idx:
                    continue
                if (
                    quality_scores[other_idx] >= quality_scores[idx]
                    and cost_benefits[other_idx] >= cost_benefits[idx]
                    and (
                        quality_scores[other_idx] > quality_scores[idx]
                        or cost_benefits[other_idx] > cost_benefits[idx]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(idx)
        return frontier

    def _compute_dynamic_lambda(self, uncertainty: float) -> float:
        return self.config.lambda_min + uncertainty * (self.config.lambda_max - self.config.lambda_min)

    def _create_fallback_decision(self, scores, costs, reason):
        return RouterDecision(
            chosen_method=self.config.fallback_method,
            scores=scores,
            probs={},
            costs=costs,
            norm_costs={},
            utilities={},
            gap=0.0,
            uncertainty=1.0,
            lambda_dyn=0.0,
            reason=reason,
        )

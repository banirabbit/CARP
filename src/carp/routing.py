"""P3: validation-calibrated Pareto-compromise routing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class P3Scales:
    quality_q05: float
    quality_q95: float
    log_cost_q05: float
    log_cost_q95: float


@dataclass(frozen=True)
class P3Config:
    quality_weight: float = 0.5
    cost_weight: float = 0.5
    quantiles: tuple[float, float] = (0.05, 0.95)
    fallback_method: str | None = None


@dataclass(frozen=True)
class RoutingDecision:
    chosen_method: str
    pareto_front: tuple[str, ...]
    normalized_quality: Mapping[str, float]
    normalized_cost: Mapping[str, float]
    distances: Mapping[str, float]


class P3Router:
    """P3 router with global validation calibration and equal-weight L1 selection.

    Pareto membership is determined in raw predicted-uplift and predicted
    total-token space. Validation-calibrated coordinates are only used for
    selection within that frontier.
    """

    def __init__(self, config: P3Config | None = None) -> None:
        self.config = config or P3Config()
        self.scales: P3Scales | None = None

    def fit(
        self,
        validation_scores: Sequence[Mapping[str, float]],
        validation_costs: Sequence[Mapping[str, float]],
    ) -> "P3Router":
        if len(validation_scores) != len(validation_costs) or not validation_scores:
            raise ValueError("validation scores and costs must be non-empty and aligned")
        quality, log_cost = [], []
        for scores, costs in zip(validation_scores, validation_costs):
            methods = set(scores).intersection(costs)
            quality.extend(float(scores[m]) for m in methods)
            log_cost.extend(float(np.log1p(max(float(costs[m]), 0.0))) for m in methods)
        if not quality or not log_cost:
            raise ValueError("validation records contain no common candidate methods")
        low, high = self.config.quantiles
        self.scales = P3Scales(
            quality_q05=float(np.quantile(quality, low)),
            quality_q95=float(np.quantile(quality, high)),
            log_cost_q05=float(np.quantile(log_cost, low)),
            log_cost_q95=float(np.quantile(log_cost, high)),
        )
        return self

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip((value - low) / max(high - low, 1e-12), 0.0, 1.0))

    @staticmethod
    def _front(methods: list[str], scores: Mapping[str, float], costs: Mapping[str, float]) -> list[str]:
        front: list[str] = []
        for method in methods:
            dominated = any(
                other != method
                and scores[other] >= scores[method]
                and costs[other] <= costs[method]
                and (scores[other] > scores[method] or costs[other] < costs[method])
                for other in methods
            )
            if not dominated:
                front.append(method)
        return front

    def route(self, scores: Mapping[str, float], costs: Mapping[str, float]) -> RoutingDecision:
        if self.scales is None:
            raise RuntimeError("fit P3Router on validation records before routing test queries")
        methods = sorted(set(scores).intersection(costs))
        methods = [m for m in methods if np.isfinite(scores[m]) and np.isfinite(costs[m]) and costs[m] > 0]
        if not methods:
            if self.config.fallback_method is None:
                raise ValueError("no candidate has finite score and positive cost")
            return RoutingDecision(self.config.fallback_method, (), {}, {}, {})
        front = self._front(methods, scores, costs)
        q = {m: self._clip(float(scores[m]), self.scales.quality_q05, self.scales.quality_q95) for m in methods}
        k = {m: self._clip(float(np.log1p(costs[m])), self.scales.log_cost_q05, self.scales.log_cost_q95) for m in methods}
        distance = {m: self.config.quality_weight * (1.0 - q[m]) + self.config.cost_weight * k[m] for m in front}
        chosen = min(front, key=lambda m: (distance[m], k[m], costs[m], -q[m], m))
        return RoutingDecision(chosen, tuple(front), q, k, distance)

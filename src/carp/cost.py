"""DynCost: query-only total-token cost estimation used by CARP."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DynCostConfig:
    """Hyperparameters for the query-only total-token regressor."""

    svd_dimensions: int = 128
    max_features: int = 30_000
    huber_epsilon: float = 1.35
    alpha: float = 1e-4
    max_iter: int = 200
    tolerance: float = 1e-3
    initial_learning_rate: float = 1e-3


class QueryOnlyDynCost:
    """One shared query representation and one Huber residual head per service.

    Each service target is (tokens - median) / max(MAD, 1). Predictions are
    transformed back to total-token space and clipped to at least one token.
    """

    def __init__(self, config: DynCostConfig | None = None) -> None:
        self.config = config or DynCostConfig()
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=self.config.max_features, sublinear_tf=True,
        )
        self.svd: TruncatedSVD | None = None
        self.scaler = StandardScaler()
        self.medians: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.estimators: list[SGDRegressor] = []

    def fit(
        self,
        questions: Sequence[str],
        total_tokens: np.ndarray,
        *,
        random_state: int = 42,
    ) -> "QueryOnlyDynCost":
        costs = np.asarray(total_tokens, dtype=np.float64)
        if costs.ndim != 2 or costs.shape[0] != len(questions) or np.any(costs <= 0):
            raise ValueError("total_tokens must be a positive [queries, services] matrix")
        sparse = self.vectorizer.fit_transform([str(question) for question in questions])
        dimensions = min(self.config.svd_dimensions, sparse.shape[0] - 1, sparse.shape[1] - 1)
        if dimensions < 1:
            raise ValueError("DynCost requires at least two non-empty training questions")
        self.svd = TruncatedSVD(n_components=dimensions, random_state=42)
        features = self.scaler.fit_transform(self.svd.fit_transform(sparse)).astype(np.float32)
        self.medians = np.median(costs, axis=0)
        self.scales = np.maximum(np.median(np.abs(costs - self.medians), axis=0), 1.0)
        self.estimators = []
        for service_index in range(costs.shape[1]):
            head = SGDRegressor(
                loss="huber", epsilon=self.config.huber_epsilon, alpha=self.config.alpha,
                max_iter=self.config.max_iter, tol=self.config.tolerance, average=True,
                random_state=random_state, learning_rate="adaptive",
                eta0=self.config.initial_learning_rate,
            )
            target = (costs[:, service_index] - self.medians[service_index]) / self.scales[service_index]
            head.fit(features, target)
            self.estimators.append(head)
        return self

    def predict(self, questions: Sequence[str]) -> np.ndarray:
        if self.svd is None or self.medians is None or self.scales is None or not self.estimators:
            raise RuntimeError("DynCost has not been fitted")
        sparse = self.vectorizer.transform([str(question) for question in questions])
        features = self.scaler.transform(self.svd.transform(sparse)).astype(np.float32)
        residuals = np.column_stack([head.predict(features) for head in self.estimators])
        return np.maximum(self.medians[None, :] + self.scales[None, :] * residuals, 1.0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "QueryOnlyDynCost":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected QueryOnlyDynCost, got {type(model)!r}")
        return model

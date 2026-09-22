import numpy as np
import torch

from carp.cost import QueryOnlyDynCost
from carp.routing import P3Router
from carp.scoring import PriorResidualUpliftHead, build_full_uplifts


def test_anchor_coordinate_is_zero():
    head = PriorResidualUpliftHead(3, torch.tensor([0.1, -0.2]))
    features = torch.ones((2, 3))
    predicted = head(features, torch.zeros(1, 3))
    full = build_full_uplifts('qagn', ['qagn', 'dalk', 'light'], predicted)
    assert torch.allclose(full[:, 0], torch.zeros(2))


def test_dyncost_produces_positive_service_costs():
    questions = ['alpha alpha', 'alpha beta', 'alpha gamma', 'alpha delta']
    costs = np.asarray([[10, 20], [11, 19], [12, 22], [10, 21]], dtype=float)
    prediction = QueryOnlyDynCost().fit(questions, costs).predict(['alpha epsilon'])
    assert prediction.shape == (1, 2)
    assert np.all(prediction > 0)


def test_p3_uses_raw_cost_for_pareto_filtering():
    router = P3Router().fit(
        [{'qagn': 0.0, 'dalk': 0.2}, {'qagn': 0.0, 'dalk': 0.3}],
        [{'qagn': 10.0, 'dalk': 8.0}, {'qagn': 11.0, 'dalk': 9.0}],
    )
    decision = router.route({'qagn': 0.0, 'dalk': 0.2}, {'qagn': 10.0, 'dalk': 8.0})
    assert decision.chosen_method == 'dalk'
    assert decision.pareto_front == ('dalk',)

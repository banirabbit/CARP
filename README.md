# CARP

CARP is a query-aware router for a fixed candidate set of LLM-orchestrated services. Its three production components are:

- an anchor-relative continuous uplift scorer, using a frozen query encoder and a small MLP head;
- DynCost, a query-only total-token predictor with a robust per-service baseline and residual model;
- P3, a validation-calibrated Pareto-compromise policy.


## Installation

~~~bash
python -m pip install -e .
~~~

The implementation requires Python 3.10 or newer. The first use of the frozen encoder downloads the configured Hugging Face model unless it is already cached.

## Repository layout

~~~text
src/carp/
  scoring.py    frozen query encoder and uplift head
  cost.py       query-only DynCost estimator
  routing.py    P3 Pareto-compromise router
  io.py         canonical JSONL validation and I/O
scripts/
  train_uplift.py, predict_uplift.py
  train_dyncost.py, predict_cost.py
  route_p3.py
configs/carp.yaml  paper-default component settings
~~~

## Input format

Training inputs are JSONL files. Each row records one complete query and observed executions for every fixed candidate service:

~~~json
{"id":"q-001","question":"...","methods":{"qagn":{"f1":0.42,"total_tokens":8500},"dalk":{"f1":0.46,"total_tokens":2600},"gr":{"f1":0.44,"total_tokens":4200},"hippo":{"f1":0.47,"total_tokens":6400},"lgraph":{"f1":0.45,"total_tokens":5100},"light":{"f1":0.41,"total_tokens":2300}}}
~~~

The default candidate order is `qagn dalk gr hippo lgraph light`, with QAGN as the zero-uplift anchor. All training rows must contain F1 and total-token observations for every candidate. This fixed-candidate assumption is deliberate: a newly added service needs sufficient observations to train both its quality coordinate and DynCost head before it can be routed.


## Main workflow

1. Fit the uplift scorer using only training labels and select its checkpoint on validation MSE.

~~~bash
python scripts/train_uplift.py \
  --train data/hotpot/train.jsonl --validation data/hotpot/validation.jsonl \
  --output checkpoints/uplift.pt \
  --methods qagn dalk gr hippo lgraph light
~~~

2. Fit DynCost on the same training queries and produce validation/test predictions.

~~~bash
python scripts/train_dyncost.py \
  --train data/hotpot/train.jsonl --output checkpoints/dyncost.joblib \
  --methods qagn dalk gr hippo lgraph light
python scripts/predict_cost.py --model checkpoints/dyncost.joblib \
  --input data/hotpot/validation.jsonl --output artifacts/validation_cost.jsonl \
  --methods qagn dalk gr hippo lgraph light
~~~

3. Predict uplift coordinates for validation and test queries, then calibrate and apply P3. The router estimates global coordinate quantiles from validation predictions only; Pareto membership itself uses raw uplift and predicted token cost.

~~~bash
python scripts/predict_uplift.py --checkpoint checkpoints/uplift.pt \
  --input data/hotpot/validation.jsonl --output artifacts/validation_uplift.jsonl
python scripts/route_p3.py \
  --validation-uplifts artifacts/validation_uplift.jsonl \
  --validation-costs artifacts/validation_cost.jsonl \
  --test-uplifts artifacts/test_uplift.jsonl \
  --test-costs artifacts/test_cost.jsonl \
  --output artifacts/test_routes.jsonl
~~~

Run `python scripts/<name>.py --help` for all options. Generated data, checkpoints, artifacts, and the old repository snapshot are ignored by Git.

## Validation

~~~bash
python -m pytest -q
~~~

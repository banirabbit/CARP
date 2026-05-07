# CARP

Language:
English | [中文](README.zh.md)

# CARP: Query-Aware QoS Routing for LLM-Orchestrated Services

Service computing has long focused on selecting or composing services under Quality of Service (QoS) constraints. With the emergence of large language models (LLMs), a new class of LLM-orchestrated services has arisen, where complex multi-stage pipelines make service effectiveness and execution cost strongly dependent on the input query and internal orchestration process. This challenges conventional QoS-aware service selection methods that rely primarily on stable service-level characteristics. In this paper, we study a practical pre-execution routing setting for LLM-orchestrated services, where the router must select a service strategy without observing the realized quality or exact execution cost of all candidates. To address this problem, we propose CARP, a query-aware and cost-profile-guided QoS routing framework. CARP formulates service routing as a query-level decision problem, uses a preference-based scoring model to estimate fine-grained service suitability, and applies a Pareto-compromise routing policy that combines suitability scores with historical service-level cost priors. This enables lightweight and interpretable service selection under partial QoS observability. Extensive experiments on multiple multi-hop question answering benchmarks demonstrate that CARP achieves favorable empirical quality-cost trade-offs compared with fixed pipelines and representative routing baselines. In particular, CARP achieves the highest efficiency scores of 16.49, 15.60, and 15.93 on HotpotQA, MultiHop-RAG, and 2WikiMultiHopQA, respectively, while introducing only millisecond-level routing overhead.

## Goal

This repository implements the CARP pipeline described in `method.tex`:

1. Scoring model: `pairwise + pointwise + dynamic margin`
2. Router: `dynamic_cost + pareto_compromise`
3. Main config: `configs/qwen_pairwise_pairweight_soft_pointwise_margin.yaml`

This project is designed for query-level pre-execution routing across multiple GraphRAG/QA methods. Instead of invoking every candidate first, it predicts which method is most suitable for the current question and combines that prediction with historical cost signals to achieve a better practical trade-off between answer quality and token/time cost.

Chinese version: [README.zh.md](README.zh.md)

## Environment setup

### Option A: `venv`

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: `conda`

```bash
conda create -n carp python=3.10 -y
conda activate carp
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch note

`requirements.txt` includes a generic `torch` dependency. If you need GPU PyTorch, install the proper CUDA build from the official PyTorch instructions first, then run:

```bash
pip install -r requirements.txt
```

### Optional dependency

If you want to use `train_pairwise.py --deepspeed ...`, install:

```bash
pip install deepspeed
```

## Run pipeline

### 1. Rebuild Hotpot train/val/test

```bash
python prepare_data.py
```

### 2. Build pairwise data

```bash
python create_pairwise_dataset.py
```

This also exports transfer-test question lists:

- `dataset/eval_questions/multihop.csv`
- `dataset/eval_questions/2wiki.csv`

Current shared-question counts:

- `multihop`: 2556
- `2wiki`: 1495

### 3. Build router training archive

```bash
python router/build_hotpot_trainset.py
```

### 4. Train the scorer

```bash
python train_pairwise.py \
  --config configs/qwen_pairwise_pairweight_soft_pointwise_margin.yaml
```

### 5. Generate test scores

Hotpot:

```bash
CONFIG_PATH=configs/qwen_pairwise_pairweight_soft_pointwise_margin.yaml \
MODEL_PATH=outputs/pairwise_pairweight_soft_pointwise_margin/last_model.pt \
DATASET_NAME=hotpot \
P2L_OUTPUT_FILE=router/results/scorer/hotpot.jsonl \
python generate_test_scores.py
```

Use the same pattern for `multihop` and `2wiki`.

### 6. Run dynamic-cost routing

```bash
python router/dynamic_cost_router.py \
  --scored-path router/results/scorer/hotpot.jsonl \
  --train-jsonl-path router/data/hotpot_train.jsonl \
  --output-jsonl router/results/dynamic_cost/hotpot.routed.jsonl \
  --summary-json router/results/dynamic_cost/hotpot.summary.json
```

Use the same pattern for `multihop` and `2wiki`.

## Metrics

`router/results/dynamic_cost/*.summary.json` contains:

- `router_metrics.f1`
- `router_metrics.avg_token_cost`
- `router_metrics.avg_time_cost`
- `router_metrics.efficiency_balance`
- `router_metrics.cpp`
- `router_metrics.icer_qagn`

These map to:

- `F1`
- `Tokens`
- `Time`
- `Effic.`
- `CPP`
- `ICER-Q`

## Current limitations

1. `prepare_data.py` rebuilds splits from the shipped `dataset/full_dataset.csv`.
2. You may hit OOM if the selected GPU is busy.
3. The smoke checkpoint is only for pipeline verification.

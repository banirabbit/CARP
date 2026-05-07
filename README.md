# CARP

Language:
[中文](#中文) | [English](#english)

<details open>
<summary><strong id="中文">中文</strong></summary>

# CARP：面向 LLM 编排服务的查询感知 QoS 路由方法
服务计算长期以来关注在服务质量（QoS）约束下进行服务选择或服务组合。随着大语言模型（LLM）的兴起，一类由 LLM 编排的新型服务逐渐出现。这类服务通常包含复杂的多阶段流水线，使得服务效果和执行成本高度依赖于输入查询及其内部编排过程。这对传统主要依赖稳定服务级特征的 QoS 感知服务选择方法提出了挑战。
本文研究了一种实用的 LLM 编排服务预执行路由场景，其中路由器必须在无法观测所有候选服务的实际质量和精确执行成本的情况下选择服务策略。为解决这一问题，我们提出了 CARP，一个查询感知且由成本画像引导的 QoS 路由框架。CARP 将服务路由建模为查询级决策问题，使用基于偏好的评分模型估计细粒度服务适配性，并采用 Pareto-compromise 路由策略，将适配性得分与历史服务级成本先验相结合。这使得在部分 QoS 可观测条件下实现轻量、可解释的服务选择成为可能。
在多个多跳问答基准上的大量实验表明，与固定流水线和代表性路由基线相比，CARP 实现了更优的经验质量—成本权衡。具体而言，CARP 在 HotpotQA、MultiHop-RAG 和 2WikiMultiHopQA 上分别取得了 16.49、15.60 和 15.93 的最高效率得分，同时仅引入毫秒级路由开销，体现了其在 LLM 编排系统中进行 QoS 感知服务计算的实际有效性。
## 项目目标


本仓库实现 `method.tex` 中描述的 CARP 框架：

1. 打分模型：`pairwise + pointwise + dynamic margin`
2. 路由策略：`dynamic_cost + pareto_compromise`
3. 主配置：`configs/qwen_pairwise_pairweight_soft_pointwise_margin.yaml`

本项目用于在多种 GraphRAG/QA 方法之间做查询级预执行路由，在不提前实际调用所有候选方法的前提下，预测当前问题更适合哪种方法，并结合历史成本信号，在答案质量与 token/时间开销之间取得更好的折中。

## 环境配置

### 方案 A：`venv`

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 方案 B：`conda`

```bash
conda create -n carp python=3.10 -y
conda activate carp
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch 说明

`requirements.txt` 里包含通用 `torch` 依赖，但如果你需要 GPU 版 PyTorch，建议按你的 CUDA 版本从 PyTorch 官方命令安装，再执行：

```bash
pip install -r requirements.txt
```

如果你只想先验证流程，CPU 版也可以安装，但训练和打分会明显更慢。

### 可选依赖

如果你要用 `train_pairwise.py --deepspeed ...`，需要额外安装：

```bash
pip install deepspeed
```

## 运行流程

### 1. 重建 Hotpot train/val/test

```bash
python prepare_data.py
```

输出：

- `dataset/train.csv`
- `dataset/val.csv`
- `dataset/test.csv`
- `dataset/train.jsonl`
- `dataset/val.jsonl`
- `dataset/test.jsonl`

### 2. 构造 pairwise 数据

```bash
python create_pairwise_dataset.py
```

输出：

- `dataset/pairwise/train_pairwise.csv`
- `dataset/pairwise/val_pairwise.csv`
- `dataset/pairwise/test_pairwise.csv`
- `dataset/eval_questions/multihop.csv`
- `dataset/eval_questions/2wiki.csv`

说明：

- `multihop` 和 `2wiki` 这里生成的是测试问题清单，不是监督式 pairwise 标注。
- 当前仓库实际构造出的共同问题数是：
  - `multihop`: 2556
  - `2wiki`: 1495

### 3. 构造路由训练归档

```bash
python router/build_hotpot_trainset.py
```

输出：

- `router/data/hotpot_train.jsonl`

### 4. 训练打分模型

```bash
CUDA_VISIBLE_DEVICES=0 \
python train_pairwise.py \
  --config configs/qwen_pairwise_pairweight_soft_pointwise_margin.yaml
```

默认输出目录：

- `outputs/pairwise_pairweight_soft_pointwise_margin/`

### 5. 生成测试集打分

#### Hotpot

```bash
CONFIG_PATH=configs/qwen_pairwise_pairweight_soft_pointwise_margin.yaml \
MODEL_PATH=outputs/pairwise_pairweight_soft_pointwise_margin/last_model.pt \
DATASET_NAME=hotpot \
P2L_OUTPUT_FILE=router/results/scorer/hotpot.jsonl \
python generate_test_scores.py
```

#### Multihop

```bash
CONFIG_PATH=configs/qwen_pairwise_pairweight_soft_pointwise_margin.yaml \
MODEL_PATH=outputs/pairwise_pairweight_soft_pointwise_margin/last_model.pt \
DATASET_NAME=multihop \
P2L_OUTPUT_FILE=router/results/scorer/multihop.jsonl \
python generate_test_scores.py
```

#### 2Wiki

```bash
CONFIG_PATH=configs/qwen_pairwise_pairweight_soft_pointwise_margin.yaml \
MODEL_PATH=outputs/pairwise_pairweight_soft_pointwise_margin/last_model.pt \
DATASET_NAME=2wiki \
P2L_OUTPUT_FILE=router/results/scorer/2wiki.jsonl \
python generate_test_scores.py
```

### 6. 动态成本路由评估

#### Hotpot

```bash
python router/dynamic_cost_router.py \
  --scored-path router/results/scorer/hotpot.jsonl \
  --train-jsonl-path router/data/hotpot_train.jsonl \
  --output-jsonl router/results/dynamic_cost/hotpot.routed.jsonl \
  --summary-json router/results/dynamic_cost/hotpot.summary.json
```

#### Multihop

```bash
python router/dynamic_cost_router.py \
  --scored-path router/results/scorer/multihop.jsonl \
  --train-jsonl-path router/data/hotpot_train.jsonl \
  --output-jsonl router/results/dynamic_cost/multihop.routed.jsonl \
  --summary-json router/results/dynamic_cost/multihop.summary.json
```

#### 2Wiki

```bash
python router/dynamic_cost_router.py \
  --scored-path router/results/scorer/2wiki.jsonl \
  --train-jsonl-path router/data/hotpot_train.jsonl \
  --output-jsonl router/results/dynamic_cost/2wiki.routed.jsonl \
  --summary-json router/results/dynamic_cost/2wiki.summary.json
```

## 输出指标

`router/results/dynamic_cost/*.summary.json` 中包含论文对应指标：

- `router_metrics.f1` -> `F1`
- `router_metrics.avg_token_cost` -> `Tokens`
- `router_metrics.avg_time_cost` -> `Time`
- `router_metrics.efficiency_balance` -> `Effic.`
- `router_metrics.cpp` -> `CPP`
- `router_metrics.icer_qagn` -> `ICER-Q`

## 当前限制

1. `prepare_data.py` 当前是从仓库自带的 `dataset/full_dataset.csv` 重建 split。
2. 如果目标 GPU 很忙，训练或前向会 OOM。

</details>

<details>
<summary><strong id="english">English</strong></summary>

# CARP: Query-Aware QoS Routing for LLM-Orchestrated Services

Service computing has long focused on selecting or composing services under Quality of Service (QoS) constraints. With the emergence of large language models (LLMs), a new class of LLM-orchestrated services has arisen, where complex multi-stage pipelines make service effectiveness and execution cost strongly dependent on the input query and internal orchestration process. This challenges conventional QoS-aware service selection methods that rely primarily on stable service-level characteristics. In this paper, we study a practical pre-execution routing setting for LLM-orchestrated services, where the router must select a service strategy without observing the realized quality or exact execution cost of all candidates. To address this problem, we propose CARP, a query-aware and cost-profile-guided QoS routing framework. CARP formulates service routing as a query-level decision problem, uses a preference-based scoring model to estimate fine-grained service suitability, and applies a Pareto-compromise routing policy that combines suitability scores with historical service-level cost priors. This enables lightweight and interpretable service selection under partial QoS observability. Extensive experiments on multiple multi-hop question answering benchmarks demonstrate that CARP achieves favorable empirical quality--cost trade-offs compared with fixed pipelines and representative routing baselines. In particular, CARP achieves the highest efficiency scores of 16.49, 15.60, and 15.93 on HotpotQA, MultiHop-RAG, and 2WikiMultiHopQA, respectively, while introducing only millisecond-level routing overhead, highlighting its practical effectiveness for QoS-aware service computing in LLM-orchestrated systems.


## Goal

This repository implements the CARP pipeline described in `method.tex`:

1. Scoring model: `pairwise + pointwise + dynamic margin`
2. Router: `dynamic_cost + pareto_compromise`
3. Main config: `configs/qwen_pairwise_pairweight_soft_pointwise_margin.yaml`

This project is designed for query-level pre-execution routing across multiple GraphRAG/QA methods. Instead of invoking every candidate first, it predicts which method is most suitable for the current question and combines that prediction with historical cost signals to achieve a better practical trade-off between answer quality and token/time cost.


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

</details>

"""
评估工具函数
从原始方法评分文件加载数据并进行对比评估
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path


def load_method_scores(
    dataset_dir: str = 'dataset',
    dataset_name: str = 'hotpot',
    split: str = 'val',
    methods: List[str] = None
) -> Dict:
    """
    从方法评分文件加载数据，并按 split 过滤 qids
    """
    if methods is None:
        methods = ['dalk', 'gr', 'hippo', 'lgraph', 'light', 'qagn']

    dataset_path = Path(dataset_dir)

    # 读取 split qids
    split_qids = None
    splits_file = dataset_path / 'data_splits.json'
    if splits_file.exists():
        with open(splits_file, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        key = f'{split}_qids'
        if key in splits:
            split_qids = set(splits[key])

    all_data = {}
    for method in methods:
        score_file = dataset_path / method / dataset_name / 'results.score.json'
        if not score_file.exists():
            print(f"Warning: {score_file} not found, skipping {method}")
            continue

        data_list = []
        with open(score_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    data_list.append(data)
                except json.JSONDecodeError:
                    continue

        all_data[method] = {item['id']: item for item in data_list}

    if not all_data:
        raise ValueError(f"No method data loaded from {dataset_dir}")

    first_method = list(all_data.keys())[0]
    qids = sorted(all_data[first_method].keys())

    if split_qids is not None:
        qids = [qid for qid in qids if qid in split_qids]

    questions = []
    labels = []
    scores_list = []

    for qid in qids:
        sample = all_data[first_method][qid]
        questions.append(sample['question'])
        labels.append(sample.get('label', sample.get('answer', '')))

        method_scores = []
        for method in methods:
            if method in all_data and qid in all_data[method]:
                f1_score = all_data[method][qid].get('f1', 0.0)
                method_scores.append(f1_score)
            else:
                method_scores.append(0.0)

        scores_list.append(method_scores)

    return {
        'questions': questions,
        'qids': qids,
        'methods': methods,
        'scores': np.array(scores_list),
        'labels': labels
    }


def evaluate_model_vs_methods(
    model_predictions: np.ndarray,
    method_scores: np.ndarray,
    method_names: List[str],
    opportunity_threshold: float = 0.05
) -> Dict:
    """
    评估模型预测与真实方法 F1 分数的对比

    说明：
    - 保留原有 overlap / correlation 指标
    - 新增 tie-aware 指标
    """
    num_questions = model_predictions.shape[0]
    num_methods = len(method_names)

    metrics = {}

    # 动态 top-k
    topk_list = [k for k in [1, 3, 5, num_methods] if k <= num_methods]
    topk_list = sorted(set(topk_list))

    # 1. Top-k overlap (proxy)
    for k in topk_list:
        top_k_overlap = 0.0
        for i in range(num_questions):
            pred_topk = set(np.argsort(model_predictions[i])[-k:])
            true_topk = set(np.argsort(method_scores[i])[-k:])
            overlap = len(pred_topk & true_topk) / k
            top_k_overlap += overlap

        metrics[f'top{k}_overlap'] = top_k_overlap / num_questions

    # 2. Top-k hit against true-best-set (tie-aware)
    for k in topk_list:
        hit = 0
        for i in range(num_questions):
            pred_topk = set(np.argsort(model_predictions[i])[-k:])
            max_true = np.max(method_scores[i])
            true_best_set = set(np.where(np.isclose(method_scores[i], max_true))[0].tolist())
            if len(pred_topk & true_best_set) > 0:
                hit += 1
        metrics[f'top{k}_hit_bestset'] = hit / num_questions

    # 3. Spearman
    from scipy.stats import spearmanr
    correlations = []
    for i in range(num_questions):
        corr, _ = spearmanr(model_predictions[i], method_scores[i])
        if not np.isnan(corr):
            correlations.append(corr)
    metrics['spearman_corr'] = np.mean(correlations) if correlations else 0.0

    # 4. Kendall tau
    from scipy.stats import kendalltau
    tau_values = []
    for i in range(num_questions):
        tau, _ = kendalltau(model_predictions[i], method_scores[i])
        if not np.isnan(tau):
            tau_values.append(tau)
    metrics['kendall_tau'] = np.mean(tau_values) if tau_values else 0.0

    # 5. NDCG
    from sklearn.metrics import ndcg_score
    for k in topk_list:
        try:
            metrics[f'ndcg@{k}'] = ndcg_score(method_scores, model_predictions, k=k)
        except Exception:
            metrics[f'ndcg@{k}'] = 0.0

    # 6. 单 argmax best_method_acc (proxy)
    pred_best = np.argmax(model_predictions, axis=1)
    true_best = np.argmax(method_scores, axis=1)
    metrics['best_method_acc'] = np.mean(pred_best == true_best)

    # 7. tie-aware best_method_acc
    tie_correct = 0
    for i in range(num_questions):
        max_true = np.max(method_scores[i])
        true_best_set = set(np.where(np.isclose(method_scores[i], max_true))[0].tolist())
        if pred_best[i] in true_best_set:
            tie_correct += 1
    metrics['best_method_acc_tie'] = tie_correct / num_questions

    # 8. 每个方法相关性
    for i, method in enumerate(method_names):
        pred_scores = model_predictions[:, i]
        true_scores = method_scores[:, i]
        corr, _ = spearmanr(pred_scores, true_scores)
        metrics[f'{method}_corr'] = corr if not np.isnan(corr) else 0.0

    # 9. Predicted best F1 / Oracle best F1
    pred_best_f1 = method_scores[np.arange(num_questions), pred_best]
    true_best_f1 = np.max(method_scores, axis=1)
    metrics['predicted_best_f1'] = np.mean(pred_best_f1)
    metrics['oracle_best_f1'] = np.mean(true_best_f1)
    metrics['f1_gap'] = metrics['oracle_best_f1'] - metrics['predicted_best_f1']

    # 10. baseline-aware / tie-aware router metrics
    strongest_single_idx = int(np.argmax(np.mean(method_scores, axis=0)))
    strongest_single_method = method_names[strongest_single_idx]
    strongest_single_f1 = method_scores[:, strongest_single_idx]
    opportunity = true_best_f1 - strongest_single_f1
    hard_mask = opportunity >= opportunity_threshold
    easy_mask = ~hard_mask
    pred_gain_vs_single = pred_best_f1 - strongest_single_f1
    switched = pred_best != strongest_single_idx

    metrics['strongest_single_method'] = strongest_single_method
    metrics['strongest_single_avg_f1'] = float(np.mean(strongest_single_f1))
    metrics['opportunity_threshold'] = float(opportunity_threshold)
    metrics['opportunity_ratio'] = float(np.mean(hard_mask))
    metrics['no_opportunity_ratio'] = float(np.mean(easy_mask))
    metrics['overall_gain'] = float(np.mean(pred_gain_vs_single))
    metrics['switch_rate'] = float(np.mean(switched))

    if np.any(hard_mask):
        hard_denominator = opportunity[hard_mask]
        metrics['hard_gain'] = float(np.mean(pred_gain_vs_single[hard_mask]))
        metrics['hard_oracle_gap'] = float(
            np.mean(true_best_f1[hard_mask] - pred_best_f1[hard_mask])
        )
        metrics['hard_regret_reduction'] = float(
            np.mean(pred_gain_vs_single[hard_mask] / hard_denominator)
        )
        metrics['hard_bestset_hit_rate'] = float(
            np.mean(np.isclose(pred_best_f1[hard_mask], true_best_f1[hard_mask]))
        )

        hard_switched = hard_mask & switched
        if np.any(hard_switched):
            metrics['hard_switch_precision'] = float(
                np.mean(pred_best_f1[hard_switched] > strongest_single_f1[hard_switched])
            )
        else:
            metrics['hard_switch_precision'] = 0.0
    else:
        metrics['hard_gain'] = 0.0
        metrics['hard_oracle_gap'] = 0.0
        metrics['hard_regret_reduction'] = 0.0
        metrics['hard_bestset_hit_rate'] = 0.0
        metrics['hard_switch_precision'] = 0.0

    if np.any(easy_mask):
        easy_losses = np.maximum(0.0, strongest_single_f1[easy_mask] - pred_best_f1[easy_mask])
        metrics['easy_regret'] = float(np.mean(easy_losses))
        metrics['easy_not_worse_rate'] = float(
            np.mean(pred_best_f1[easy_mask] >= strongest_single_f1[easy_mask])
        )
    else:
        metrics['easy_regret'] = 0.0
        metrics['easy_not_worse_rate'] = 1.0

    # 11. 完全排名匹配率
    exact_matches = 0
    for i in range(num_questions):
        pred_ranking = np.argsort(model_predictions[i])
        true_ranking = np.argsort(method_scores[i])
        if np.array_equal(pred_ranking, true_ranking):
            exact_matches += 1
    metrics['exact_ranking_match'] = exact_matches / num_questions

    # 12. 位置准确率
    position_correct = np.zeros(num_methods)
    for k in range(1, num_methods + 1):
        correct_at_k = 0
        for i in range(num_questions):
            pred_topk = set(np.argsort(model_predictions[i])[-k:])
            true_topk = set(np.argsort(method_scores[i])[-k:])
            if pred_topk == true_topk:
                correct_at_k += 1
        position_correct[k - 1] = correct_at_k / num_questions

    metrics['avg_position_accuracy'] = np.mean(position_correct)
    for k in range(1, num_methods + 1):
        metrics[f'position_acc_top{k}'] = position_correct[k - 1]

    return metrics


def create_comparison_report(metrics: Dict, output_file: str = None) -> str:
    """
    创建对比评估报告
    
    Args:
        metrics: 评估指标字典
        output_file: 输出文件路径（可选）
    
    Returns:
        格式化的报告字符串
    """
    report_lines = [
        "=" * 80,
        "模型预测 vs 方法真实评分对比报告",
        "=" * 80,
        "",
        "📊 排序准确性指标:",
        f"  • Top-1 重叠率: {metrics['top1_overlap']:.3f}",
        f"  • Top-3 重叠率: {metrics['top3_overlap']:.3f}",
        f"  • Top-5 重叠率: {metrics['top5_overlap']:.3f}",
        f"  • Spearman相关系数: {metrics['spearman_corr']:.3f}",
        f"  • Kendall's Tau: {metrics['kendall_tau']:.3f}",
        "",
        "🎯 排名对应率指标:",
        f"  • 完全排名匹配率: {metrics['exact_ranking_match']:.3f}",
        f"  • 平均位置准确率: {metrics['avg_position_accuracy']:.3f}",
        f"  • Top-1位置准确率: {metrics.get('position_acc_top1', 0):.3f}",
        f"  • Top-2位置准确率: {metrics.get('position_acc_top2', 0):.3f}",
        f"  • Top-3位置准确率: {metrics.get('position_acc_top3', 0):.3f}",
        f"  • Top-4位置准确率: {metrics.get('position_acc_top4', 0):.3f}",
        f"  • Top-5位置准确率: {metrics.get('position_acc_top5', 0):.3f}",
        "",
        "📈 NDCG指标（排序质量）:",
        f"  • NDCG@1: {metrics['ndcg@1']:.3f}",
        f"  • NDCG@3: {metrics['ndcg@3']:.3f}",
        f"  • NDCG@5: {metrics['ndcg@5']:.3f}",
        "",
        "🏆 最优方法选择:",
        f"  • 最优方法准确率: {metrics['best_method_acc']:.3f}",
        f"  • 预测最优方法平均F1: {metrics['predicted_best_f1']:.3f}",
        f"  • Oracle最优方法平均F1: {metrics['oracle_best_f1']:.3f}",
        f"  • F1差距: {metrics['f1_gap']:.3f}",
        "",
        "🧭 Router Checkpoint 指标:",
        f"  • Strongest single: {metrics.get('strongest_single_method', 'N/A')}",
        f"  • Strongest single平均F1: {metrics.get('strongest_single_avg_f1', 0.0):.3f}",
        f"  • Opportunity ratio: {metrics.get('opportunity_ratio', 0.0):.3f}",
        f"  • Overall gain vs strongest single: {metrics.get('overall_gain', 0.0):.3f}",
        f"  • Hard gain: {metrics.get('hard_gain', 0.0):.3f}",
        f"  • Hard regret reduction: {metrics.get('hard_regret_reduction', 0.0):.3f}",
        f"  • Easy regret: {metrics.get('easy_regret', 0.0):.3f}",
        "",
        "🔍 各方法预测相关性:",
    ]

    method_names = ['dalk', 'gr', 'hippo', 'lgraph', 'light', 'qagn']
    for method in method_names:
        key = f'{method}_corr'
        if key in metrics:
            report_lines.append(f"  • {method}: {metrics[key]:.3f}")
    
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Report saved to {output_file}")
    
    return report

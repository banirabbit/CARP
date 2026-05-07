"""
成对学习模型评估器

推理策略：
- 对同一问题的所有方法分别打分
- 选择得分最高的方法作为预测
- 不做两两比较

评估指标：
- Top-k 准确率 (k=1,2,3,5)
- 每类 Precision/Recall/F1
- 方法分布对比
- 混淆矩阵
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
def parse_best_methods(best_methods_json):
    if isinstance(best_methods_json, str):
        try:
            return json.loads(best_methods_json)
        except Exception:
            return []
    elif isinstance(best_methods_json, list):
        return best_methods_json
    return []

class PairwiseEvaluator:
    """
    成对学习模型评估器
    
    推理：对每个问题的所有方法单独打分，选最高分
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Args:
            model: 训练好的模型
            device: 设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def encode_and_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        method_ids=None
    ) -> torch.Tensor:
        """
        编码并打分
        """
        with torch.no_grad():
            return self.model.score_single(input_ids, attention_mask, method_ids=method_ids)
    
    def evaluate_on_original_dataset(
        self,
        original_csv: str,
        tokenizer,
        max_length: int = 256
    ) -> Dict:
        """
        在原始数据集上评估

        支持 tie-best：
        - 每个问题可能有多个 label=1
        - 或从 best_methods_json 中读取最佳方法集合
        """
        print(f"\nEvaluating on {original_csv}...")

        df = pd.read_csv(original_csv)
        print(f"  Loaded {len(df)} samples")
        print(f"  Unique questions: {df['qid'].nunique()}")

        grouped = df.groupby('qid')

        all_predictions = []
        all_target_sets = []
        all_primary_targets = []
        all_scores_per_question = []
        all_method_names = []

        skipped_questions = 0

        print("  Running inference...")
        for qid, group in tqdm(grouped, desc="Questions"):
            method_scores = []
            method_ids = []

            # 优先从 best_methods_json 读取
            first_row = group.iloc[0]
            target_methods = parse_best_methods(first_row.get('best_methods_json', '[]'))

            # 如果没有 best_methods_json，则退回到 label=1
            if not target_methods:
                target_methods = group[group['label'] == 1]['method_id'].tolist()

            target_methods = sorted(set(target_methods))
            if not target_methods:
                skipped_questions += 1
                continue

            for _, row in group.iterrows():
                encoding = tokenizer(
                    row['input_text'],
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                score = self.encode_and_score(input_ids, attention_mask, method_ids=[row['method_id']])
                method_scores.append(score.item())
                method_ids.append(row['method_id'])

            max_idx = int(np.argmax(method_scores))
            predicted_method = method_ids[max_idx]

            all_predictions.append(predicted_method)
            all_target_sets.append(target_methods)
            all_primary_targets.append(target_methods[0])  # 用于单标签 proxy 指标
            all_scores_per_question.append(method_scores)
            all_method_names.append(method_ids)

        if skipped_questions > 0:
            print(f"  ⚠️ Skipped questions with empty targets: {skipped_questions}")

        results = self._compute_metrics(
            predictions=all_predictions,
            target_sets=all_target_sets,
            primary_targets=all_primary_targets,
            scores=all_scores_per_question,
            method_names=all_method_names
        )

        return results
    
    def _compute_metrics(
        self,
        predictions: List[str],
        target_sets: List[List[str]],
        primary_targets: List[str],
        scores: List[List[float]],
        method_names: List[List[str]]
    ) -> Dict:
        """
        计算评估指标

        说明：
        - Top-k 按 target set（支持 tie-best）
        - 每类 Precision/Recall/F1、混淆矩阵按 primary target 计算（proxy）
        """
        results = {}

        print("\n" + "=" * 80)
        print("Top-k Accuracy (against best-method sets)")
        print("=" * 80)

        num_methods = len(method_names[0]) if method_names else 0
        topk_list = [k for k in [1, 2, 3, 5, num_methods] if k <= num_methods]
        topk_list = sorted(set(topk_list))

        for k in topk_list:
            correct = 0
            for target_set, score_list, names in zip(target_sets, scores, method_names):
                top_k_indices = np.argsort(score_list)[::-1][:k]
                top_k_methods = [names[i] for i in top_k_indices]

                if any(t in top_k_methods for t in target_set):
                    correct += 1

            accuracy = correct / len(target_sets) if target_sets else 0.0
            results[f'top{k}_accuracy'] = accuracy
            print(f"  Top-{k}: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # 单标签 proxy 指标
        print("\n" + "=" * 80)
        print("Per-Method Metrics (proxy using primary target)")
        print("=" * 80)

        all_methods = sorted(set(predictions + primary_targets))

        precision, recall, f1, support = precision_recall_fscore_support(
            primary_targets, predictions, labels=all_methods, average=None, zero_division=0
        )

        per_method_metrics = {}
        print(f"{'Method':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 80)

        for i, method in enumerate(all_methods):
            per_method_metrics[method] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
            }
            print(f"{method:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")

        results['per_method_metrics'] = per_method_metrics

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            primary_targets, predictions, average='macro', zero_division=0
        )
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            primary_targets, predictions, average='micro', zero_division=0
        )

        print("-" * 80)
        print(f"{'Macro Avg':<10} {macro_p:<12.4f} {macro_r:<12.4f} {macro_f1:<12.4f}")
        print(f"{'Micro Avg':<10} {micro_p:<12.4f} {micro_r:<12.4f} {micro_f1:<12.4f}")

        results['macro_avg'] = {
            'precision': float(macro_p),
            'recall': float(macro_r),
            'f1': float(macro_f1),
        }
        results['micro_avg'] = {
            'precision': float(micro_p),
            'recall': float(micro_r),
            'f1': float(micro_f1),
        }

        # 预测分布 vs primary target 分布
        print("\n" + "=" * 80)
        print("Method Distribution Comparison")
        print("=" * 80)

        pred_dist = Counter(predictions)
        target_dist = Counter(primary_targets)

        print(f"{'Method':<10} {'Predicted':<12} {'Actual':<12} {'Diff':<12}")
        print("-" * 80)

        distribution_comparison = {}
        for method in all_methods:
            pred_count = pred_dist.get(method, 0)
            target_count = target_dist.get(method, 0)
            pred_ratio = pred_count / len(predictions) if predictions else 0.0
            target_ratio = target_count / len(primary_targets) if primary_targets else 0.0
            diff = pred_ratio - target_ratio

            distribution_comparison[method] = {
                'predicted_count': pred_count,
                'actual_count': target_count,
                'predicted_ratio': float(pred_ratio),
                'actual_ratio': float(target_ratio),
                'difference': float(diff),
            }

            print(f"{method:<10} {pred_count:<4} ({pred_ratio:>5.1%}) "
                  f"{target_count:<4} ({target_ratio:>5.1%}) "
                  f"{diff:>+6.1%}")

        results['distribution_comparison'] = distribution_comparison

        # 混淆矩阵（proxy）
        cm = confusion_matrix(primary_targets, predictions, labels=all_methods)
        results['confusion_matrix'] = cm.tolist()
        results['method_labels'] = all_methods

        print("\n" + "=" * 80)
        print("Confusion Matrix (proxy, rows=primary target, cols=predicted)")
        print("=" * 80)

        print(f"{'':>10}", end='')
        for method in all_methods:
            print(f"{method:>10}", end='')
        print()

        for i, true_method in enumerate(all_methods):
            print(f"{true_method:>10}", end='')
            for j in range(len(all_methods)):
                print(f"{cm[i, j]:>10}", end='')
            print()

        # tie-best 统计
        tie_count = sum(1 for ts in target_sets if len(ts) > 1)
        results['tie_best_questions'] = tie_count
        results['tie_best_ratio'] = tie_count / len(target_sets) if target_sets else 0.0

        results['total_questions'] = len(predictions)
        results['num_methods'] = len(all_methods)

        return results
    
    def save_results(self, results: Dict, output_file: str):
        """保存评估结果"""
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 转换numpy类型为Python类型
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        results = convert_types(results)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")


def create_evaluation_report(results: Dict, output_md: str):
    """创建Markdown格式的评估报告"""
    with open(output_md, 'w') as f:
        f.write("# 模型评估报告\n\n")
        
        # Top-k准确率
        f.write("## Top-k 准确率\n\n")
        f.write("| 指标 | 准确率 |\n")
        f.write("|------|--------|\n")
        for k in [1, 2, 3, 5]:
            key = f'top{k}_accuracy'
            if key in results:
                f.write(f"| Top-{k} | {results[key]:.4f} ({results[key]*100:.2f}%) |\n")
        
        # 每类指标
        f.write("\n## 每类指标\n\n")
        f.write("| 方法 | Precision | Recall | F1 | Support |\n")
        f.write("|------|-----------|--------|----|---------|\n")
        
        if 'per_method_metrics' in results:
            for method, metrics in results['per_method_metrics'].items():
                f.write(f"| {method} | {metrics['precision']:.4f} | "
                       f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | "
                       f"{metrics['support']} |\n")
        
        # 平均指标
        f.write("\n### 平均指标\n\n")
        f.write("| 类型 | Precision | Recall | F1 |\n")
        f.write("|------|-----------|--------|----|\n")
        
        if 'macro_avg' in results:
            m = results['macro_avg']
            f.write(f"| Macro | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n")
        
        if 'micro_avg' in results:
            m = results['micro_avg']
            f.write(f"| Micro | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n")
        
        # 分布对比
        f.write("\n## 方法分布对比\n\n")
        f.write("| 方法 | 预测数量 | 实际数量 | 预测占比 | 实际占比 | 差异 |\n")
        f.write("|------|----------|----------|----------|----------|------|\n")
        
        if 'distribution_comparison' in results:
            for method, dist in results['distribution_comparison'].items():
                f.write(f"| {method} | {dist['predicted_count']} | "
                       f"{dist['actual_count']} | "
                       f"{dist['predicted_ratio']*100:.1f}% | "
                       f"{dist['actual_ratio']*100:.1f}% | "
                       f"{dist['difference']*100:+.1f}% |\n")
    
    print(f"✓ Report saved to {output_md}")


# 测试代码
if __name__ == '__main__':
    print("PairwiseEvaluator module loaded")
    print("Use this module to evaluate pairwise-trained models")

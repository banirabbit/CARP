"""
计算各GraphRAG方法的预估成本

功能:
1. 从训练集统计每个方法的token消耗
2. 计算多种成本指标（平均值、中位数、分位数等）
3. 生成路由器配置文件

成本计算策略:
- 平均成本: 简单平均，可能受极值影响
- 中位数成本: 更稳健，不受极值影响
- P75成本: 75%的问题成本低于此值（保守估计）
- P90成本: 90%的问题成本低于此值（非常保守）
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from typing import Dict, List
import argparse
from collections import defaultdict


def load_data_splits(splits_file: str) -> Dict[str, List[int]]:
    """加载数据划分信息"""
    with open(splits_file, 'r') as f:
        return json.load(f)


def load_method_results(method_file: str) -> Dict[int, dict]:
    """加载单个方法的结果文件"""
    results = {}
    with open(method_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            results[item['id']] = item
    return results


def extract_token_cost(item: dict) -> int:
    """从结果中提取token成本"""
    token_cost = item.get('token_cost', {})

    # 尝试不同的字段名
    if isinstance(token_cost, dict):
        return token_cost.get('total_tokens', 0)
    elif isinstance(token_cost, (int, float)):
        return int(token_cost)
    else:
        return 0


def extract_time_cost(item: dict) -> float:
    """从结果中提取时间成本（秒）"""
    # 尝试多种可能的时间字段名
    time_fields = ['time', 'elapsed_time', 'duration', 'execution_time', 'query_time']

    for field in time_fields:
        if field in item:
            time_value = item[field]
            if isinstance(time_value, (int, float)) and time_value > 0:
                return float(time_value)

    # 如果有详细的时间统计，尝试提取总时间
    time_stats = item.get('time_stats', {})
    if isinstance(time_stats, dict):
        total_time = time_stats.get('total', 0) or time_stats.get('total_time', 0)
        if total_time > 0:
            return float(total_time)

    return 0.0


def compute_cost_statistics(costs: List[int]) -> Dict:
    """
    计算成本统计信息
    
    返回多种成本指标，用于不同场景：
    - mean: 平均成本（常用）
    - median: 中位数成本（稳健）
    - p75: 75分位数（保守估计）
    - p90: 90分位数（非常保守）
    - min/max: 最小/最大值
    - std: 标准差
    """
    costs_array = np.array(costs)
    
    return {
        'count': len(costs),
        'mean': float(np.mean(costs_array)),
        'median': float(np.median(costs_array)),
        'p25': float(np.percentile(costs_array, 25)),
        'p75': float(np.percentile(costs_array, 75)),
        'p90': float(np.percentile(costs_array, 90)),
        'p95': float(np.percentile(costs_array, 95)),
        'min': float(np.min(costs_array)),
        'max': float(np.max(costs_array)),
        'std': float(np.std(costs_array))
    }


def main():
    parser = argparse.ArgumentParser(description="计算GraphRAG方法的预估成本")
    parser.add_argument("--methods", nargs='+', 
                        default=["qagn", "dalk", "gr", "light", "hippo", "lgraph"],
                        help="方法列表")
    parser.add_argument("--results-dir", type=str,
                        default=str(project_root / "dataset"),
                        help="结果文件目录")
    parser.add_argument("--splits-file", type=str,
                        default=str(project_root / "dataset/graphrag/data_splits.json"),
                        help="数据划分文件")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="使用哪个数据集统计（推荐用train）")
    parser.add_argument("--output", type=str,
                        default=str(project_root / "dataset/graphrag/method_costs.json"),
                        help="输出文件路径")
    parser.add_argument("--output-yaml", type=str,
                        default=str(project_root / "route/graphrag_costs.yaml"),
                        help="输出YAML配置文件")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GraphRAG 方法成本统计")
    print("=" * 80)
    print(f"方法列表: {args.methods}")
    print(f"数据集: {args.split}")
    print(f"输出文件: {args.output}")
    print("=" * 80)
    print()
    
    # 1. 加载数据划分
    print("📂 加载数据划分...")
    splits = load_data_splits(args.splits_file)
    question_ids = splits[args.split]
    print(f"  ✓ {args.split}集包含 {len(question_ids)} 个问题\n")
    
    # 2. 加载每个方法的结果并提取成本
    print("💰 统计各方法成本...")
    method_costs = {}
    method_times = {}  # 新增：时间统计

    for method in args.methods:
        # 尝试不同的路径格式
        possible_paths = [
            Path(args.results_dir) / method / "hotpot" / "results.score.json",
            Path(args.results_dir) / "graphrag" / "test" / f"{method}_test.jsonl"
        ]

        method_file = None
        for path in possible_paths:
            if path.exists():
                method_file = path
                break

        if method_file is None:
            print(f"  ⚠️  警告: 找不到方法 {method} 的结果文件，跳过")
            continue

        print(f"  - 处理 {method}...")
        print(f"    文件: {method_file}")

        # 加载结果
        results = load_method_results(method_file)

        # 提取该方法在训练集中的成本和时间
        costs = []
        times = []  # 新增：时间列表
        missing_count = 0
        missing_time_count = 0  # 新增：缺失时间计数

        for qid in question_ids:
            if qid in results:
                cost = extract_token_cost(results[qid])
                time_cost = extract_time_cost(results[qid])  # 新增：提取时间

                if cost > 0:
                    costs.append(cost)
                else:
                    missing_count += 1

                if time_cost > 0:
                    times.append(time_cost)
                else:
                    missing_time_count += 1
            else:
                missing_count += 1
                missing_time_count += 1

        if len(costs) == 0:
            print(f"    ✗ 没有有效的成本数据")
            continue

        # 计算统计信息
        stats = compute_cost_statistics(costs)
        method_costs[method] = stats

        # 新增：计算时间统计
        if len(times) > 0:
            time_stats = compute_cost_statistics(times)
            method_times[method] = time_stats
            print(f"    ✓ 成功统计 {stats['count']} 个问题的成本 | {len(times)} 个问题的时间")
        else:
            print(f"    ✓ 成功统计 {stats['count']} 个问题的成本")

        if missing_count > 0:
            print(f"    ⚠️  缺失 {missing_count} 个问题的成本数据")
        if missing_time_count > 0:
            print(f"    ⚠️  缺失 {missing_time_count} 个问题的时间数据")
    
    # 3. 保存结果
    print(f"\n💾 保存统计结果...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 合并成本和时间统计
    combined_stats = {}
    for method in method_costs.keys():
        combined_stats[method] = {
            'token_costs': method_costs[method],
            'time_costs': method_times.get(method, None)
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_stats, f, indent=2, ensure_ascii=False)

    print(f"  ✓ JSON格式已保存到: {output_path}")
    
    # 4. 生成YAML配置文件（用于路由器）
    if args.output_yaml:
        yaml_path = Path(args.output_yaml)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("# GraphRAG 方法成本配置\n")
            f.write("# 自动生成，请勿手动编辑\n\n")
            f.write("method_costs:\n")

            for method, stats in method_costs.items():
                f.write(f"  {method}:\n")
                f.write(f"    mean: {stats['mean']:.2f}      # 平均成本（推荐用于路由）\n")
                f.write(f"    median: {stats['median']:.2f}  # 中位数成本\n")
                f.write(f"    p75: {stats['p75']:.2f}        # 75分位数（保守估计）\n")
                f.write(f"    p90: {stats['p90']:.2f}        # 90分位数（非常保守）\n")
                f.write(f"    min: {stats['min']:.2f}\n")
                f.write(f"    max: {stats['max']:.2f}\n")
                f.write(f"    std: {stats['std']:.2f}\n")
                f.write(f"    count: {stats['count']}\n")

                # 新增：时间统计
                if method in method_times:
                    time_stats = method_times[method]
                    f.write(f"    # 时间成本（秒）\n")
                    f.write(f"    time_mean: {time_stats['mean']:.2f}\n")
                    f.write(f"    time_median: {time_stats['median']:.2f}\n")
                    f.write(f"    time_p75: {time_stats['p75']:.2f}\n")
                    f.write(f"    time_p90: {time_stats['p90']:.2f}\n")

                f.write("\n")

        print(f"  ✓ YAML配置已保存到: {yaml_path}")
    
    # 5. 打印详细统计报告
    print("\n" + "=" * 80)
    print("📊 成本统计报告")
    print("=" * 80)
    print()

    # Token成本表
    print("【Token成本统计】")
    print(f"{'方法':<10} {'平均':<10} {'中位数':<10} {'P75':<10} {'P90':<10} {'样本数':<8}")
    print("-" * 68)

    # 按平均成本排序
    sorted_methods = sorted(method_costs.items(), key=lambda x: x[1]['mean'])

    for method, stats in sorted_methods:
        print(f"{method:<10} {stats['mean']:<10.0f} {stats['median']:<10.0f} "
              f"{stats['p75']:<10.0f} {stats['p90']:<10.0f} {stats['count']:<8}")

    # 新增：时间成本表
    if method_times:
        print("\n【时间成本统计（秒）】")
        print(f"{'方法':<10} {'平均':<10} {'中位数':<10} {'P75':<10} {'P90':<10} {'样本数':<8}")
        print("-" * 68)

        # 按平均时间排序
        sorted_time_methods = sorted(method_times.items(), key=lambda x: x[1]['mean'])

        for method, stats in sorted_time_methods:
            print(f"{method:<10} {stats['mean']:<10.2f} {stats['median']:<10.2f} "
                  f"{stats['p75']:<10.2f} {stats['p90']:<10.2f} {stats['count']:<8}")
    
    # 6. 推荐成本配置
    print("\n" + "=" * 80)
    print("💡 路由器成本配置推荐")
    print("=" * 80)
    print()
    print("根据不同的应用场景，推荐使用以下成本值：\n")

    print("【Token成本配置】\n")

    print("场景 1: 常规使用（推荐）")
    print("  使用 mean (平均值)")
    print("  ```python")
    print("  METHOD_COSTS = {")
    for method, stats in sorted_methods:
        print(f"      '{method}': {stats['mean']:.0f},")
    print("  }")
    print("  ```\n")

    print("场景 2: 稳健估计")
    print("  使用 median (中位数)")
    print("  ```python")
    print("  METHOD_COSTS = {")
    for method, stats in sorted_methods:
        print(f"      '{method}': {stats['median']:.0f},")
    print("  }")
    print("  ```\n")

    print("场景 3: 保守估计（避免超支）")
    print("  使用 p75 (75分位数)")
    print("  ```python")
    print("  METHOD_COSTS = {")
    for method, stats in sorted_methods:
        print(f"      '{method}': {stats['p75']:.0f},")
    print("  }")
    print("  ```\n")

    print("场景 4: 非常保守（严格预算控制）")
    print("  使用 p90 (90分位数)")
    print("  ```python")
    print("  METHOD_COSTS = {")
    for method, stats in sorted_methods:
        print(f"      '{method}': {stats['p90']:.0f},")
    print("  }")
    print("  ```\n")

    # 新增：时间成本配置推荐
    if method_times:
        sorted_time_methods = sorted(method_times.items(), key=lambda x: x[1]['mean'])
        print("\n【时间成本配置（秒）】\n")

        print("场景 1: 常规使用（推荐）")
        print("  使用 mean (平均值)")
        print("  ```python")
        print("  METHOD_TIME_COSTS = {")
        for method, stats in sorted_time_methods:
            print(f"      '{method}': {stats['mean']:.2f},")
        print("  }")
        print("  ```\n")

        print("场景 2: 稳健估计")
        print("  使用 median (中位数)")
        print("  ```python")
        print("  METHOD_TIME_COSTS = {")
        for method, stats in sorted_time_methods:
            print(f"      '{method}': {stats['median']:.2f},")
        print("  }")
        print("  ```\n")
    
    # 7. 成本分布分析
    print("=" * 80)
    print("📈 成本分布分析")
    print("=" * 80)
    print()
    
    for method, stats in sorted_methods:
        cv = (stats['std'] / stats['mean']) * 100  # 变异系数
        range_ratio = (stats['max'] - stats['min']) / stats['mean']
        
        print(f"{method.upper()}:")
        print(f"  范围: {stats['min']:.0f} - {stats['max']:.0f} tokens")
        print(f"  标准差: {stats['std']:.0f} (变异系数: {cv:.1f}%)")
        print(f"  分布特征: ", end="")
        
        if cv < 20:
            print("稳定 ✓")
        elif cv < 50:
            print("中等波动")
        else:
            print("波动较大 ⚠️")
        
        print()
    
    print("=" * 80)
    print("✅ 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()


"""
使用自适应路由器选择最佳方法并评估性能

该脚本通过分析 test_scored.jsonl 中每个问题的 p2l_score 和 method_costs.json 中的成本，
调用 adaptive_router 的三种模式进行路由，并评估整体性能指标。
"""

import argparse
import json
import logging
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    from router.adaptive_router import AdaptiveGraphRAGRouter, RouterConfig, RouterMode
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from router.adaptive_router import AdaptiveGraphRAGRouter, RouterConfig, RouterMode

# 方法列表集中配置
METHOD_ORDER = ["qagn", "dalk", "gr", "hippo", "light", "lgraph"]

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_shqfc_from_summary_points(
    summary_points: Dict[str, Dict[str, float]],
    baseline_names: List[str],
    quality_scale: float = 3.0,
    cost_power: float = 0.5,
    gap_lambda: float = 1.0,
    frontier_source_points: Dict[str, Dict[str, float]] = None,
    max_tokens_override: float = None,
    quality_floor_override: float = None,
) -> Tuple[Dict[str, float], float, List[str], Dict[str, float]]:
    """
    通用 sHQFC 计算函数。

    summary_points 格式:
        {
            "method_name": {"f1": 30.49, "tokens": 5955},
            ...
        }
    baseline_names 用于定义 quality floor，不要求它们全部在 frontier 上。
    """
    if not summary_points:
        return {}, 0.0, [], {}

    if quality_floor_override is None:
        available_baselines = [
            name for name in baseline_names
            if name in summary_points
        ]
        baseline_f1_values = [
            float(summary_points[name]['f1'])
            for name in available_baselines
        ]
        quality_floor = statistics.median(baseline_f1_values) if baseline_f1_values else 0.0
    else:
        quality_floor = float(quality_floor_override)

    frontier_points = frontier_source_points or summary_points

    sorted_points = sorted(
        frontier_points.items(),
        key=lambda kv: (kv[1]['tokens'], -kv[1]['f1'], kv[0])
    )

    frontier: List[Tuple[str, float, float]] = []
    best_f1 = float('-inf')
    for name, point in sorted_points:
        if point['f1'] > best_f1:
            frontier.append((name, point['f1'], point['tokens']))
            best_f1 = point['f1']

    if max_tokens_override is None:
        max_tokens = max(point['tokens'] for point in summary_points.values())
    else:
        max_tokens = float(max_tokens_override)
    scores = {}

    def softplus(delta: float, scale: float) -> float:
        return scale * math.log1p(math.exp(delta / scale))

    def frontier_f1_at_cost(tokens: float) -> float:
        eligible = [f1 for _, f1, t in frontier if t <= tokens]
        return max(eligible) if eligible else 0.0

    for name, point in summary_points.items():
        f1 = float(point['f1'])
        tokens = float(point['tokens'])
        quality_term = softplus(f1 - quality_floor, quality_scale)
        cost_term = (max_tokens / tokens) ** cost_power if tokens > 0 else 0.0
        frontier_gap = max(frontier_f1_at_cost(tokens) - f1, 0.0)
        penalty_term = 1.0 / (1.0 + gap_lambda * frontier_gap)
        scores[name] = quality_term * cost_term * penalty_term

    frontier_names = [name for name, _, _ in frontier]
    params = {
        'quality_scale': quality_scale,
        'cost_power': cost_power,
        'gap_lambda': gap_lambda,
    }
    return scores, quality_floor, frontier_names, params


class RouterMethodEvaluator:
    """路由器方法评估器"""
    
    def __init__(self, 
                 test_data_path: str,
                 costs_path: str,
                 train_data_path: str = None,
                 router_config: RouterConfig = None):
        self.test_data_path = Path(test_data_path)
        self.costs_path = Path(costs_path)
        self.train_data_path = Path(train_data_path) if train_data_path else None
        self.router = AdaptiveGraphRAGRouter(router_config or RouterConfig())
        
        self.test_data = self._load_test_data()
        self.method_costs = self._load_method_costs()
        self.time_dataset_key = self._infer_time_dataset_key()
        self.method_time_costs = self._load_method_time_costs()
        self.cpp_plus_tau = self._compute_cpp_plus_tau()
        self.cpp_plus_tau_percent = self.cpp_plus_tau * 100.0
        self.cpp_plus_epsilon = 1e-3
        self.cpp_plus_cost_scale = 1000.0
        
        logger.info(f"加载了 {len(self.test_data)} 个测试问题")
        logger.info(f"方法成本: {self.method_costs}")
        if self.method_time_costs:
            logger.info(f"方法平均时间成本: {self.method_time_costs}")
        logger.info(
            f"CPP+ threshold tau: raw={self.cpp_plus_tau:.6f}, percent={self.cpp_plus_tau_percent:.4f}, "
            f"epsilon={self.cpp_plus_epsilon:.1e}, cost_scale={self.cpp_plus_cost_scale:.0f}"
        )
    
    def _load_test_data(self) -> List[Dict]:
        data = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _load_method_costs(self) -> Dict[str, float]:
        with open(self.costs_path, 'r', encoding='utf-8') as f:
            costs_data = json.load(f)
        method_costs = {}
        for method, stats in costs_data.items():
            method_costs[method] = stats['mean']
        return method_costs

    def _infer_time_dataset_key(self) -> str | None:
        path_text = str(self.test_data_path).lower()
        if "2wiki" in path_text:
            return "2wiki"
        if "hotpot" in path_text:
            return "hotpot"
        if "multihop" in path_text or "multi" in path_text:
            return "multi"
        if "musique" in path_text:
            return "musique"
        if "popqa" in path_text:
            return "popqa"
        return None

    def _compute_average_time_from_jsonl(self, file_path: Path) -> float:
        first_seen_timestamps: Dict[int, datetime] = {}
        ordered_timestamps: List[datetime] = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                qid = item.get('question_idx')
                ts = item.get('timestamp')
                if qid is None or not isinstance(ts, str):
                    continue
                if qid in first_seen_timestamps:
                    continue
                try:
                    dt = datetime.fromisoformat(ts)
                except ValueError:
                    continue
                first_seen_timestamps[int(qid)] = dt
                ordered_timestamps.append(dt)

        if len(ordered_timestamps) < 2:
            return 0.0

        ordered_timestamps.sort()
        deltas = []
        for prev, curr in zip(ordered_timestamps, ordered_timestamps[1:]):
            seconds = (curr - prev).total_seconds()
            if seconds > 0:
                deltas.append(seconds)

        return sum(deltas) / len(deltas) if deltas else 0.0

    def _compute_average_time_from_hotpot_gr_log(self, file_path: Path) -> float:
        start_pattern = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*正在处理问题 (?P<qid>\d+)/")
        save_pattern = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*问题 (?P<qid>\d+)/\d+ 已保存")

        start_times: Dict[int, datetime] = {}
        durations: List[float] = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                start_match = start_pattern.search(line)
                if start_match:
                    qid = int(start_match.group('qid'))
                    start_times[qid] = datetime.strptime(start_match.group('ts'), "%Y-%m-%d %H:%M:%S.%f")
                    continue

                save_match = save_pattern.search(line)
                if not save_match:
                    continue
                qid = int(save_match.group('qid'))
                start_dt = start_times.get(qid)
                if start_dt is None:
                    continue
                end_dt = datetime.strptime(save_match.group('ts'), "%Y-%m-%d %H:%M:%S.%f")
                seconds = (end_dt - start_dt).total_seconds()
                if seconds > 0:
                    durations.append(seconds)

        return sum(durations) / len(durations) if durations else 0.0

    def _load_method_time_costs(self) -> Dict[str, float]:
        base_dir = Path(__file__).parent / "results" / "time"
        dataset_key = self.time_dataset_key
        if not dataset_key:
            return {}

        dataset_aliases = {
            "2wiki": ["2wiki"],
            "hotpot": ["hotpot"],
            "multi": ["multi"],
            "musique": ["musique", "multi"],
            "popqa": ["popqa", "multi"],
        }
        candidate_dirs = dataset_aliases.get(dataset_key, [dataset_key])

        method_times: Dict[str, float] = {}
        for method in METHOD_ORDER:
            avg_time = 0.0
            for candidate_dir in candidate_dirs:
                method_dir = base_dir / candidate_dir / method
                if method == "gr" and candidate_dir == "hotpot":
                    log_files = sorted(method_dir.glob("*.log"))
                    if log_files:
                        avg_time = self._compute_average_time_from_hotpot_gr_log(log_files[0])
                        if avg_time > 0:
                            break
                jsonl_path = method_dir / "temp_results.jsonl"
                if jsonl_path.exists():
                    avg_time = self._compute_average_time_from_jsonl(jsonl_path)
                    if avg_time > 0:
                        break
            if avg_time > 0:
                method_times[method] = avg_time

        return method_times

    def _extract_time_cost(self, method_name: str, result: Dict | None = None) -> float:
        if method_name in self.method_time_costs:
            return float(self.method_time_costs[method_name])
        if result is not None:
            time_cost = result.get('time', result.get('time_cost', 0.0))
            if isinstance(time_cost, (int, float)):
                return float(time_cost)
        return 0.0

    def _compute_cpp_plus_tau(self) -> float:
        """
        tau is defined as the median F1 over all query-method pairs in the training set.
        If no training data is provided, fall back to the current evaluation set.
        """
        if self.train_data_path is None:
            source_path = self.test_data_path
            logger.warning("未提供训练集路径，CPP+ 的 tau 将使用当前评估数据估计。")
        elif not self.train_data_path.exists():
            source_path = self.test_data_path
            logger.warning("训练集路径不存在 (%s)，CPP+ 的 tau 将使用当前评估数据估计。", self.train_data_path)
        else:
            source_path = self.train_data_path

        f1_values: List[float] = []
        with open(source_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                for method_result in item.get('methods', {}).values():
                    f1 = method_result.get('f1', None)
                    if isinstance(f1, (int, float)) and not math.isnan(float(f1)):
                        f1_values.append(float(f1))

        if not f1_values:
            logger.warning("训练集未找到有效 F1，CPP+ tau 回退为 0.0")
            return 0.0

        f1_values.sort()
        n = len(f1_values)
        mid = n // 2
        if n % 2 == 1:
            return f1_values[mid]
        return (f1_values[mid - 1] + f1_values[mid]) / 2.0

    def _extract_token_cost(self, result: Dict) -> float:
        tc = result.get('token_cost', {})
        if isinstance(tc, dict):
            return float(tc.get('prompt_tokens', tc.get('total_tokens', 0)))
        if isinstance(tc, (int, float)):
            return float(tc)
        return 0.0

    def _calculate_cpp_plus(self, token_costs: List[float], f1_scores: List[float]) -> Tuple[float, float]:
        """
        Dataset-level CPP+ is the average over routed queries:
            Cost(q,m) / max(F1(q,m) - tau, epsilon)
        We compute it on the percentage scale to keep values interpretable:
            (Cost(q,m) / cost_scale) / max(F1_percent(q,m) - tau_percent, epsilon)
        where cost_scale=1000 converts raw tokens into K-tokens.
        """
        if not token_costs or not f1_scores or len(token_costs) != len(f1_scores):
            return float('inf'), 0.0

        cpp_plus_values = []
        below_tau_count = 0
        for cost, f1 in zip(token_costs, f1_scores):
            f1_percent = float(f1) * 100.0
            if f1_percent <= self.cpp_plus_tau_percent:
                below_tau_count += 1
            effective_perf = max(f1_percent - self.cpp_plus_tau_percent, self.cpp_plus_epsilon)
            scaled_cost = float(cost) / self.cpp_plus_cost_scale
            cpp_plus_values.append(scaled_cost / effective_perf)

        avg_cpp_plus = sum(cpp_plus_values) / len(cpp_plus_values) if cpp_plus_values else float('inf')
        below_tau_rate = below_tau_count / len(cpp_plus_values) * 100 if cpp_plus_values else 0.0
        return avg_cpp_plus, below_tau_rate
    
    def route_all_questions(self) -> List[Dict]:
        """为所有问题进行路由选择"""
        routed_results = []
        
        for idx, item in enumerate(self.test_data):
            p2l_scores = item.get('p2l_scores', {})
            if not p2l_scores:
                continue
            
            decision = self.router.route(
                scores=p2l_scores,
                costs=self.method_costs
            )
            
            chosen_method = decision.chosen_method
            method_result = item.get('methods', {}).get(chosen_method, {})
            
            result = {
                'id': item.get('id', idx),
                'source_index': idx,
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'chosen_method': chosen_method,
                'router_decision': {
                    'scores': decision.scores,
                    'probs': decision.probs, # 记录一下概率
                    'gap': decision.gap,
                    'uncertainty': decision.uncertainty,
                    'reason': decision.reason
                },
                'output': method_result.get('output', ''),
                'parsed_answer': method_result.get('parsed_answer', ''),
                'accuracy': method_result.get('accuracy', 0),
                'f1': method_result.get('f1', 0.0),
                'em': method_result.get('em', False),
                'token_cost': method_result.get('token_cost', {}),
                'time_cost': self._extract_time_cost(chosen_method, method_result),
                'p2l_scores': p2l_scores,
                'best_method_by_p2l': item.get('best_method_by_p2l', ''),
                'best_method_actual': item.get('best_method_actual', ''),
                'routing_mode': self.router.config.routing_mode,
            }
            routed_results.append(result)
        
        logger.info(f"完成 {len(routed_results)} 个问题的路由选择")
        return routed_results
    
    def calculate_metrics(self, routed_results: List[Dict]) -> Dict:
        """计算Router的评估指标"""
        if not routed_results:
            return {}

        accuracy_list = [r['accuracy'] for r in routed_results]
        f1_list = [r['f1'] for r in routed_results]
        em_list = [r['em'] for r in routed_results]
        token_costs = []
        time_costs = []  # 新增：时间成本列表

        for r in routed_results:
            token_costs.append(self._extract_token_cost(r))

            # 新增：提取时间成本
            time_cost = r.get('time_cost', 0.0)
            time_costs.append(time_cost if isinstance(time_cost, (int, float)) else 0.0)

        accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        f1 = sum(f1_list) * 100 / len(f1_list)
        em = sum(em_list) * 100 / len(em_list)
        avg_token_cost = sum(token_costs) / len(token_costs)
        avg_time_cost = sum(time_costs) / len(time_costs)  # 新增：平均时间成本

        efficiency_balance = self._calculate_efficiency_balance(token_costs, f1)

        # 计算CPP (Cost per unit performance): 平均成本 / 平均性能
        cpp = (avg_token_cost / f1) if f1 > 0 else float('inf')
        cpp_plus, below_tau_rate = self._calculate_cpp_plus(token_costs, f1_list)

        method_distribution = defaultdict(int)
        fallback_count = 0
        for r in routed_results:
            method_distribution[r['chosen_method']] += 1
            # 判断是否使用了fallback策略
            reason = r.get('router_decision', {}).get('reason', '')
            if 'fallback' in reason.lower() or '回退' in reason:
                fallback_count += 1

        matches_p2l_best = sum(1 for r in routed_results if r['chosen_method'] == r.get('best_method_by_p2l', ''))
        matches_actual_best = sum(1 for r in routed_results if r['chosen_method'] == r.get('best_method_actual', ''))

        return {
            'total_questions': len(routed_results),
            'accuracy': accuracy,
            'f1': f1,
            'em': em,
            'avg_token_cost': avg_token_cost,
            'avg_time_cost': avg_time_cost,  # 新增：平均时间成本
            'efficiency_balance': efficiency_balance,
            'cpp': cpp,
            'cpp_plus': cpp_plus,
            'cpp_plus_tau': self.cpp_plus_tau,
            'cpp_plus_tau_percent': self.cpp_plus_tau_percent,
            'cpp_plus_epsilon': self.cpp_plus_epsilon,
            'cpp_plus_cost_scale': self.cpp_plus_cost_scale,
            'cpp_plus_below_tau_rate': below_tau_rate,
            'method_distribution': dict(method_distribution),
            'fallback_count': fallback_count,
            'fallback_rate': fallback_count / len(routed_results) * 100,
            'p2l_best_match_rate': matches_p2l_best / len(routed_results) * 100,
            'actual_best_match_rate': matches_actual_best / len(routed_results) * 100,
        }

    def calculate_individual_method_metrics(self) -> Dict[str, Dict]:
        """计算每个单独方法的平均性能指标"""
        methods = METHOD_ORDER
        method_metrics = {}

        for method in methods:
            accuracy_list = []
            f1_list = []
            em_list = []
            token_costs = []
            time_costs = []  # 新增：时间成本列表
            raw_f1_scores = []
            valid_count = 0

            for item in self.test_data:
                method_result = item.get('methods', {}).get(method, {})
                if method_result:
                    accuracy_list.append(method_result.get('accuracy', 0))
                    f1_list.append(method_result.get('f1', 0.0))
                    raw_f1_scores.append(method_result.get('f1', 0.0))
                    em_list.append(method_result.get('em', False))
                    token_costs.append(self._extract_token_cost(method_result))

                    # 新增：提取时间成本
                    time_costs.append(self._extract_time_cost(method, method_result))

                    valid_count += 1

            if valid_count > 0:
                accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
                f1 = sum(f1_list) * 100 / len(f1_list)
                em = sum(em_list) * 100 / len(em_list)
                avg_token_cost = sum(token_costs) / len(token_costs)
                avg_time_cost = sum(time_costs) / len(time_costs)  # 新增：平均时间成本
                efficiency_balance = self._calculate_efficiency_balance(token_costs, f1)
                cpp = (avg_token_cost / f1) if f1 > 0 else float('inf')
                cpp_plus, below_tau_rate = self._calculate_cpp_plus(token_costs, raw_f1_scores)

                method_metrics[method] = {
                    'valid_questions': valid_count,
                    'accuracy': accuracy,
                    'f1': f1,
                    'em': em,
                    'avg_token_cost': avg_token_cost,
                    'avg_time_cost': avg_time_cost,  # 新增：平均时间成本
                    'efficiency_balance': efficiency_balance,
                    'cpp': cpp,
                    'cpp_plus': cpp_plus,
                    'cpp_plus_below_tau_rate': below_tau_rate
                }
            else:
                method_metrics[method] = {
                    'valid_questions': 0, 'accuracy': 0.0, 'f1': 0.0, 'em': 0.0,
                    'avg_token_cost': 0.0, 'avg_time_cost': 0.0, 'efficiency_balance': 0.0,
                    'cpp': float('inf'), 'cpp_plus': float('inf'), 'cpp_plus_below_tau_rate': 0.0
                }
        return method_metrics

    # ✅ 新增：计算 Oracle（理论上限）的方法
    def calculate_oracle_metrics(self) -> Dict:
        """
        计算 Oracle (理论上限) 指标
        假设对于每个 Query，我们都能选中该 Query 下 F1 最高的方法。
        如果多个方法 F1 相同，则选择 Cost 最低的那个（模拟"完美且高效"的路由器）。
        """
        accuracy_list = []
        f1_list = []
        em_list = []
        token_costs = []
        time_costs = []  # 新增：时间成本列表
        raw_f1_scores = []

        valid_count = 0

        for item in self.test_data:
            methods_data = item.get('methods', {})
            if not methods_data:
                continue

            # 1. 找出当前 Query 下所有方法的表现
            candidates = []
            for m_name, m_res in methods_data.items():
                # 排除 gr=0 等无效数据
                cost_val = self._extract_token_cost(m_res)

                # 新增：提取时间成本
                time_val = self._extract_time_cost(m_name, m_res)

                candidates.append({
                    'name': m_name,
                    'accuracy': m_res.get('accuracy', 0),
                    'f1': m_res.get('f1', 0.0),
                    'em': m_res.get('em', False),
                    'cost': cost_val,
                    'time': time_val  # 新增：时间成本
                })

            if not candidates:
                continue

            # 2. Oracle 逻辑：选 F1 最高的；如果 F1 一样，选 Cost 最低的
            # 排序优先级：F1 (降序) -> Cost (升序)
            best_candidate = sorted(candidates, key=lambda x: (-x['f1'], x['cost']))[0]

            # 或者：如果更看重 Accuracy，可以改为：
            # best_candidate = sorted(candidates, key=lambda x: (-x['accuracy'], -x['f1'], x['cost']))[0]

            accuracy_list.append(best_candidate['accuracy'])
            f1_list.append(best_candidate['f1'])
            em_list.append(best_candidate['em'])
            token_costs.append(best_candidate['cost'])
            time_costs.append(best_candidate['time'])  # 新增：收集时间成本
            raw_f1_scores.append(best_candidate['f1'])
            valid_count += 1

        if valid_count == 0:
            return {}

        # 计算平均值
        accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        f1 = sum(f1_list) * 100 / len(f1_list)
        em = sum(em_list) * 100 / len(em_list)
        avg_token_cost = sum(token_costs) / len(token_costs)
        avg_time_cost = sum(time_costs) / len(time_costs)  # 新增：平均时间成本

        # Oracle 的 efficiency 其实不公平（因为它总是变来变去），但可以作为参考
        efficiency_balance = self._calculate_efficiency_balance(token_costs, f1)
        cpp = avg_token_cost / f1 if f1 > 0 else float('inf')
        cpp_plus, below_tau_rate = self._calculate_cpp_plus(token_costs, raw_f1_scores)

        return {
            'valid_questions': valid_count,
            'accuracy': accuracy,
            'f1': f1,
            'em': em,
            'avg_token_cost': avg_token_cost,
            'avg_time_cost': avg_time_cost,  # 新增：平均时间成本
            'efficiency_balance': efficiency_balance,
            'cpp': cpp,
            'cpp_plus': cpp_plus,
            'cpp_plus_below_tau_rate': below_tau_rate
        }

    def _calculate_efficiency_balance(self, token_costs: List[float], f1_score: float) -> float:
        if not token_costs:
            return 0.0

        # 对每个 query 的原始 token cost 做对数归一化：
        #   Cost(q,m) = log(1 + c(q,m)) / (1 + log(1 + c(q,m)))
        # 再在数据集层面取平均，避免被单个极端大值主导。
        normalized_costs = []
        for raw_cost in token_costs:
            safe_cost = max(float(raw_cost), 0.0)
            log_cost = math.log1p(safe_cost)
            normalized_costs.append(log_cost / (1.0 + log_cost))

        normalized_cost = sum(normalized_costs) / len(normalized_costs)
        f1_decimal = f1_score / 100.0
        denominator = f1_decimal + (1 - normalized_cost)
        if denominator == 0:
            return 0.0
        efficiency_balance = (2 * f1_decimal * (1 - normalized_cost)) / denominator
        return efficiency_balance * 100

    def _calculate_shqfc_scores(
        self,
        router_metrics: Dict,
        individual_metrics: Dict[str, Dict],
    ) -> Tuple[Dict[str, float], float, List[str], Dict[str, float]]:
        """
        sHQFC (smoothed High-Quality Frontier Contribution)

        为避免硬阈值导致大量方法得分为 0，这里使用平滑版：
        1. 用静态 baseline 方法的 F1 中位数作为 quality floor；
        2. 对质量项使用 softplus(F1 - floor)，保留“高于 floor 更值钱”的性质；
        3. 对成本项使用 mild cost reward = (T_max / T)^beta；
        4. 对被前沿支配的方法施加 frontier gap penalty：
           1 / (1 + lambda_gap * (frontier_f1_at_cost - f1)).

        这个指标与 Efficiency Balance 的功能不同：
        - Efficiency Balance 是单点平衡分数；
        - sHQFC 看的是“高质量 + 成本 + frontier 相对位置”的联合效用。
        """
        summary_points: Dict[str, Dict[str, float]] = {}

        for method in METHOD_ORDER:
            method_metrics = individual_metrics.get(method, {})
            if method_metrics.get('valid_questions', 0) <= 0:
                continue
            summary_points[method] = {
                'f1': float(method_metrics.get('f1', 0.0)),
                'tokens': float(method_metrics.get('avg_token_cost', 0.0)),
            }

        if router_metrics:
            summary_points['Router'] = {
                'f1': float(router_metrics.get('f1', 0.0)),
                'tokens': float(router_metrics.get('avg_token_cost', 0.0)),
            }

        return calculate_shqfc_from_summary_points(
            summary_points=summary_points,
            baseline_names=METHOD_ORDER,
            quality_scale=3.0,
            cost_power=0.5,
            gap_lambda=1.0,
        )

    def _calculate_icer_against_baseline(
        self,
        target_f1: float,
        target_cost: float,
        baseline_f1: float,
        baseline_cost: float,
    ) -> float | None:
        """
        ICER-Q: incremental cost-effectiveness ratio against qagn.

        ICER(r | qagn) = (c_r - c_qagn) / (e_r - e_qagn)
        where c is avg token cost and e is avg F1 (% scale).
        """
        delta_effect = float(target_f1) - float(baseline_f1)
        delta_cost = float(target_cost) - float(baseline_cost)

        if abs(delta_effect) < 1e-12:
            if abs(delta_cost) < 1e-12:
                return None
            return float('inf')

        return delta_cost / delta_effect

    def _calculate_average_query_icer(
        self,
        query_pairs: List[Tuple[float, float, float, float]],
    ) -> Tuple[float | None, float, float]:
        """
        逐题计算 ICER，再对有限值取平均。

        query_pairs 元组为:
            (target_f1_raw, target_cost, baseline_f1_raw, baseline_cost)

        F1 使用原始 [0,1] 标度输入，这里统一转换到 percentage-point 再计算。
        当某题 delta_F1 == 0 时，该题 ICER 视为未定义，不纳入均值；
        同时返回 zero-effect rate 便于判断该均值覆盖率。
        """
        if not query_pairs:
            return None, 0.0, 0.0

        icer_values: List[float] = []
        zero_effect_count = 0

        for target_f1_raw, target_cost, baseline_f1_raw, baseline_cost in query_pairs:
            delta_effect = (float(target_f1_raw) - float(baseline_f1_raw)) * 100.0
            delta_cost = float(target_cost) - float(baseline_cost)

            if abs(delta_effect) < 1e-12:
                zero_effect_count += 1
                continue

            icer_values.append(delta_cost / delta_effect)

        avg_icer = (sum(icer_values) / len(icer_values)) if icer_values else None
        valid_rate = len(icer_values) * 100.0 / len(query_pairs)
        zero_effect_rate = zero_effect_count * 100.0 / len(query_pairs)
        return avg_icer, valid_rate, zero_effect_rate

    def _build_method_query_icer_pairs(self, method_name: str) -> List[Tuple[float, float, float, float]]:
        pairs: List[Tuple[float, float, float, float]] = []
        for item in self.test_data:
            methods = item.get('methods', {})
            target_result = methods.get(method_name, {})
            baseline_result = methods.get('qagn', {})
            if not target_result or not baseline_result:
                continue
            pairs.append((
                float(target_result.get('f1', 0.0)),
                self._extract_token_cost(target_result),
                float(baseline_result.get('f1', 0.0)),
                self._extract_token_cost(baseline_result),
            ))
        return pairs

    def _build_router_query_icer_pairs(self, routed_results: List[Dict]) -> List[Tuple[float, float, float, float]]:
        pairs: List[Tuple[float, float, float, float]] = []
        for result in routed_results:
            source_index = result.get('source_index')
            if not isinstance(source_index, int) or source_index < 0 or source_index >= len(self.test_data):
                continue
            item = self.test_data[source_index]
            baseline_result = item.get('methods', {}).get('qagn', {})
            if not baseline_result:
                continue
            pairs.append((
                float(result.get('f1', 0.0)),
                self._extract_token_cost(result),
                float(baseline_result.get('f1', 0.0)),
                self._extract_token_cost(baseline_result),
            ))
        return pairs

    def _build_oracle_query_icer_pairs(self) -> List[Tuple[float, float, float, float]]:
        pairs: List[Tuple[float, float, float, float]] = []
        for item in self.test_data:
            methods_data = item.get('methods', {})
            baseline_result = methods_data.get('qagn', {})
            if not methods_data or not baseline_result:
                continue

            candidates = []
            for m_name, m_res in methods_data.items():
                candidates.append({
                    'name': m_name,
                    'f1': float(m_res.get('f1', 0.0)),
                    'cost': self._extract_token_cost(m_res),
                })

            if not candidates:
                continue

            best_candidate = sorted(candidates, key=lambda x: (-x['f1'], x['cost']))[0]
            pairs.append((
                best_candidate['f1'],
                best_candidate['cost'],
                float(baseline_result.get('f1', 0.0)),
                self._extract_token_cost(baseline_result),
            ))
        return pairs
    
    def print_metrics(self, metrics: Dict, individual_metrics: Dict[str, Dict] = None, oracle_metrics: Dict = None):
        """打印评估指标 (包含 Oracle)"""
        print("\n" + "=" * 90)
        print("📊 路由器方法评估结果")
        print("=" * 90)
        print(f"\n总问题数: {metrics['total_questions']}")
        print(f"\n性能指标:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  F1 Score: {metrics['f1']:.2f}%")
        print(f"  EM Score: {metrics['em']:.2f}%")
        print(f"\n成本指标:")
        print(f"  平均Token消耗: {metrics['avg_token_cost']:.0f}")
        print(f"  平均时间消耗: {metrics.get('avg_time_cost', 0.0):.2f}秒")  # 新增：时间成本
        print(f"  Efficiency Balance: {metrics['efficiency_balance']:.2f}%")
        print(f"  CPP (每单位F1的Token消耗): {metrics['cpp']:.2f}")
        icer_q = metrics.get('icer_qagn')
        if icer_q is None:
            print("  ICER-Q (vs qagn): base")
        else:
            icer_display = f"{icer_q:.2f}" if math.isfinite(icer_q) else "inf"
            print(f"  ICER-Q (vs qagn, token/F1-pt): {icer_display}")
            print(f"  ICER-Q Valid Rate: {metrics.get('icer_qagn_valid_rate', 0.0):.2f}%")

        # 打印方法分派比例统计
        if 'method_distribution' in metrics:
            print(f"\n方法分派统计:")
            method_dist = metrics['method_distribution']
            total = metrics['total_questions']

            # 按照分派数量降序排序
            sorted_methods = sorted(method_dist.items(), key=lambda x: x[1], reverse=True)

            for method, count in sorted_methods:
                percentage = (count / total * 100) if total > 0 else 0
                print(f"  {method:<12} {count:>5} 次  ({percentage:>5.2f}%)")

            # 打印fallback统计
            if 'fallback_count' in metrics:
                fallback_count = metrics['fallback_count']
                fallback_rate = metrics['fallback_rate']
                print(f"\n  Fallback回退: {fallback_count:>5} 次  ({fallback_rate:>5.2f}%)")

        print("=" * 90)

        if individual_metrics:
            print("\n" + "=" * 100)
            print("📈 各方法独立性能对比 (含 Oracle 上限)")
            print("=" * 100)
            # 表头 - 新增时间列
            headers = ['Method', 'Count', 'Acc (%)', 'F1 (%)', 'EM (%)', 'Tokens', 'Time(s)', 'CPP', 'Effic.', 'ICER-Q']
            print(f"{headers[0]:<12} {headers[1]:<8} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} "
                  f"{headers[5]:<10} {headers[6]:<10} {headers[7]:<10} {headers[8]:<10} {headers[9]:<10}")
            print("-" * 112)

            # 1. 打印各单体方法
            for method in METHOD_ORDER:
                if method in individual_metrics:
                    m = individual_metrics[method]
                    cpp_display = f"{m['cpp']:.2f}" if m['cpp'] != float('inf') else "inf"
                    time_display = f"{m.get('avg_time_cost', 0.0):.2f}"  # 新增：时间显示
                    icer_q = m.get('icer_qagn')
                    icer_display = "base" if icer_q is None else (
                        f"{icer_q:.2f}" if math.isfinite(icer_q) else "inf"
                    )
                    print(f"{method:<12} {m['valid_questions']:<8} "
                          f"{m['accuracy']:<10.2f} {m['f1']:<10.2f} {m['em']:<10.2f} "
                          f"{m['avg_token_cost']:<10.0f} {time_display:<10} {cpp_display:<10} {m['efficiency_balance']:<10.2f} {icer_display:<10}")

            print("-" * 112)

            # 2. 打印 Router 结果
            cpp_display = f"{metrics['cpp']:.2f}" if metrics['cpp'] != float('inf') else "inf"
            time_display = f"{metrics.get('avg_time_cost', 0.0):.2f}"  # 新增：时间显示
            icer_q = metrics.get('icer_qagn')
            icer_display = "base" if icer_q is None else (
                f"{icer_q:.2f}" if math.isfinite(icer_q) else "inf"
            )
            print(f"{'Router':<12} {metrics['total_questions']:<8} "
                  f"{metrics['accuracy']:<10.2f} {metrics['f1']:<10.2f} {metrics['em']:<10.2f} "
                  f"{metrics['avg_token_cost']:<10.0f} {time_display:<10} {cpp_display:<10} {metrics['efficiency_balance']:<10.2f} {icer_display:<10}")

            # 3. 打印 Oracle 结果 (如果有)
            if oracle_metrics:
                print("-" * 112)
                cpp_display = f"{oracle_metrics['cpp']:.2f}" if oracle_metrics['cpp'] != float('inf') else "inf"
                time_display = f"{oracle_metrics.get('avg_time_cost', 0.0):.2f}"  # 新增：时间显示
                icer_q = oracle_metrics.get('icer_qagn')
                icer_display = "base" if icer_q is None else (
                    f"{icer_q:.2f}" if math.isfinite(icer_q) else "inf"
                )
                print(f"{'Oracle':<12} {oracle_metrics['valid_questions']:<8} "
                      f"{oracle_metrics['accuracy']:<10.2f} {oracle_metrics['f1']:<10.2f} {oracle_metrics['em']:<10.2f} "
                      f"{oracle_metrics['avg_token_cost']:<10.0f} {time_display:<10} {cpp_display:<10} {oracle_metrics['efficiency_balance']:<10.2f} {icer_display:<10}")

            print("=" * 112)

    def run_evaluation(self, output_path: str = None) -> Tuple[List[Dict], Dict, Dict]:
        # 1. 路由
        logger.info("开始路由选择...")
        routed_results = self.route_all_questions()

        # 2. 计算路由器指标
        metrics = self.calculate_metrics(routed_results)

        # 3. 计算各方法独立指标
        individual_metrics = self.calculate_individual_method_metrics()
        
        # 4. ✅ 计算 Oracle 指标
        oracle_metrics = self.calculate_oracle_metrics()

        # 5. 计算 ICER-Q（相对 qagn 的总体增量 token/F1 成本）
        baseline_metrics = individual_metrics.get('qagn')
        if baseline_metrics:
            router_icer = self._calculate_icer_against_baseline(
                target_f1=metrics.get('f1', 0.0),
                target_cost=metrics.get('avg_token_cost', 0.0),
                baseline_f1=baseline_metrics.get('f1', 0.0),
                baseline_cost=baseline_metrics.get('avg_token_cost', 0.0),
            )
            metrics['icer_qagn'] = router_icer
            metrics['icer_qagn_valid_rate'] = 0.0 if router_icer is None else 100.0
            metrics['icer_qagn_zero_effect_rate'] = 100.0 if router_icer is None else 0.0

            for method in METHOD_ORDER:
                if method in individual_metrics:
                    m = individual_metrics[method]
                    avg_icer = self._calculate_icer_against_baseline(
                        target_f1=m.get('f1', 0.0),
                        target_cost=m.get('avg_token_cost', 0.0),
                        baseline_f1=baseline_metrics.get('f1', 0.0),
                        baseline_cost=baseline_metrics.get('avg_token_cost', 0.0),
                    )
                    m['icer_qagn'] = avg_icer
                    m['icer_qagn_valid_rate'] = 0.0 if avg_icer is None else 100.0
                    m['icer_qagn_zero_effect_rate'] = 100.0 if avg_icer is None else 0.0
            if oracle_metrics:
                avg_icer = self._calculate_icer_against_baseline(
                    target_f1=oracle_metrics.get('f1', 0.0),
                    target_cost=oracle_metrics.get('avg_token_cost', 0.0),
                    baseline_f1=baseline_metrics.get('f1', 0.0),
                    baseline_cost=baseline_metrics.get('avg_token_cost', 0.0),
                )
                oracle_metrics['icer_qagn'] = avg_icer
                oracle_metrics['icer_qagn_valid_rate'] = 0.0 if avg_icer is None else 100.0
                oracle_metrics['icer_qagn_zero_effect_rate'] = 100.0 if avg_icer is None else 0.0

        # 6. 打印结果 (传入 oracle_metrics)
        self.print_metrics(metrics, individual_metrics, oracle_metrics)


        return routed_results, metrics, individual_metrics


def main():
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Evaluate GraphRAG routing modes")
    parser.add_argument(
        "--mode",
        choices=[
            RouterMode.SCORE_ONLY,
            RouterMode.CHEAPEST_FIRST,
            RouterMode.CONFIDENCE_COST,
            RouterMode.PARETO_COMPROMISE,
            RouterMode.SCORE_BAND_CHEAPEST,
        ],
        default=RouterMode.PARETO_COMPROMISE,
        help="路由模式：score_only / cheapest_first / confidence_cost / pareto_compromise / score_band_cheapest",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.15,
        help="Softmax 置信度阈值 (cheapest_first & confidence_cost)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.35,
        help="cheapest_first 模式下的分数阈值；留空则不启用",
    )
    parser.add_argument(
        "--compromise-p",
        type=float,
        default=6.0,
        help="pareto_compromise 模式的 compromise programming 范数 p (默认 4.0)",
    )
    parser.add_argument(
        "--score-band-delta",
        type=float,
        default=0.3,
        help="score_band_cheapest 模式的候选带宽；只在归一化分数距 top1 不超过该值的候选中选最便宜方法",
    )
    parser.add_argument(
        "--test-data",
        default=base_dir / "results/scorer/hotpot.jsonl",
        help="带 p2l_scores 的测试数据路径",
    )
    parser.add_argument(
        "--costs-path",
        default=base_dir / "method_costs.json",
        help="方法成本统计路径",
    )
    parser.add_argument(
        "--train-data",
        default=base_dir / "data/hotpot_train.jsonl",
        help="用于计算 CPP+ tau 的训练集路径；若文件不存在则回退为当前 test-data",
    )
    args = parser.parse_args()

    cfg = RouterConfig(
        methods=METHOD_ORDER,
        routing_mode=args.mode,
        confidence_threshold=args.confidence_threshold,
        quality_threshold=args.quality_threshold,
        compromise_p=args.compromise_p,
        score_band_delta=args.score_band_delta,
        fallback_method=METHOD_ORDER[0],
    )

    logger.info(
        "Router Mode: %s | confidence_threshold=%.2f | quality_threshold=%s | compromise_p=%.2f | score_band_delta=%.2f",
        cfg.routing_mode,
        cfg.confidence_threshold,
        cfg.quality_threshold,
        cfg.compromise_p,
        cfg.score_band_delta,
    )

    evaluator = RouterMethodEvaluator(
        test_data_path=str(args.test_data),
        costs_path=str(args.costs_path),
        train_data_path=str(args.train_data) if args.train_data else None,
        router_config=cfg,
    )

    evaluator.run_evaluation()

if __name__ == "__main__":
    main()

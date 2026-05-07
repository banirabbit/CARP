"""
成对学习的批采样器 (Pairwise Batch Sampler)

功能：
1. 以 qid 为采样单位（不是单个 pair）
2. 同一 qid 的所有 pair 在同一 batch
3. 跨 qid 分层均衡最佳方法类别
4. 可配置 batch_size_q（每批问题数）
"""

import torch
import numpy as np
import pandas as pd
from typing import Iterator, List, Optional
from collections import defaultdict
import random
import json

def parse_best_methods(best_methods_json):
    """解析 best_methods_json 字段"""
    if isinstance(best_methods_json, str):
        try:
            methods = json.loads(best_methods_json)
        except Exception:
            methods = []
    elif isinstance(best_methods_json, list):
        methods = best_methods_json
    else:
        methods = []
    return methods


class PairwiseBatchSampler:
    """
    成对学习的批采样器
    
    确保：
    - 以 qid 为单位采样
    - 同一 qid 的所有 pairs 在同一 batch
    - 跨 qid 分层均衡最佳方法（缓解类不平衡）
    """
    
    def __init__(
        self,
        dataset_file: str,
        batch_size_q: int = 4,
        shuffle: bool = True,
        min_method_ratio: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Args:
            dataset_file: 成对数据集CSV文件路径
            batch_size_q: 每个batch包含的问题数量
            shuffle: 是否打乱
            min_method_ratio: 最长尾方法的最小占比（默认≥10%）
            seed: 随机种子
        """
        self.batch_size_q = batch_size_q
        self.shuffle = shuffle
        self.min_method_ratio = min_method_ratio
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 加载数据集
        print(f"Loading pairwise dataset from {dataset_file}...")
        self.df = pd.read_csv(dataset_file)
        
        # 分析数据集结构
        self._analyze_dataset()

        # 构建 qid 到 indices 的映射
        self._build_qid_index()

        # 构建 qid 元信息（best_methods / primary_method）
        self._build_qid_metadata()

        # 按最佳方法分组 qid（用于分层采样）
        self._group_qids_by_optimal_method()
        
        # 生成批次
        self._generate_batches()
        
        print(f"✓ Sampler initialized:")
        print(f"  Total questions: {self.num_questions}")
        print(f"  Total pairs: {len(self.df)}")
        print(f"  Batch size (questions): {self.batch_size_q}")
        print(f"  Total batches: {len(self.batches)}")
        print(f"  Method distribution in batches:")
        self._validate_batch_distribution()
        
    def _build_qid_metadata(self):
        """
        为每个 qid 构建问题级元信息：
        - best_methods: 该问题的最佳方法集合
        - primary_method: 用于采样分桶的主方法（稳定选一个）
        - is_tied_best: 是否存在并列最佳
        """
        self.qid_to_best_methods = {}
        self.qid_to_primary_method = {}
        self.qid_to_is_tied = {}

        for qid in self.unique_qids:
            qid_rows = self.df[self.df['qid'] == qid]
            first_row = qid_rows.iloc[0]

            best_methods = parse_best_methods(first_row.get('best_methods_json', '[]'))
            if not best_methods:
                # 回退策略：如果字段缺失，尽量别让程序崩
                best_methods = []

            best_methods = sorted(best_methods)
            is_tied_best = len(best_methods) > 1

            # 采样分桶需要一个稳定主标签
            if len(best_methods) > 0:
                primary_method = best_methods[0]
            else:
                # 极端兜底
                primary_method = 'UNKNOWN'

            self.qid_to_best_methods[qid] = best_methods
            self.qid_to_primary_method[qid] = primary_method
            self.qid_to_is_tied[qid] = is_tied_best

        # 统计 tie 问题比例
        tied_count = sum(self.qid_to_is_tied.values())
        print(f"  Tie-best questions: {tied_count} / {self.num_questions} ({tied_count / self.num_questions:.2%})")
    
    def _analyze_dataset(self):
        """分析数据集结构"""
        self.num_questions = self.df['qid'].nunique()
        self.pairs_per_q = self.df.groupby('qid').size()
        
        print(f"  Questions: {self.num_questions}")
        print(f"  Total pairs: {len(self.df)}")
        print(f"  Avg pairs/question: {self.pairs_per_q.mean():.2f}")
        print(f"  Min pairs/question: {self.pairs_per_q.min()}")
        print(f"  Max pairs/question: {self.pairs_per_q.max()}")
    
    def _build_qid_index(self):
        """构建 qid 到样本索引的映射"""
        self.qid_to_indices = defaultdict(list)
        
        for idx, row in self.df.iterrows():
            qid = row['qid']
            self.qid_to_indices[qid].append(idx)
            
        self.unique_qids = list(self.qid_to_indices.keys())
        print(f"  Built index for {len(self.unique_qids)} unique questions")
    
    def _group_qids_by_optimal_method(self):
        """
        按 primary_method 对 qid 分组，用于分层采样。

        注意：
        - 这里不再根据 pair_label 反推赢家
        - 直接使用 best_methods_json 中解析出的问题级最佳方法集合
        - 若一个问题有多个最佳方法，则取排序后的第一个作为 primary_method
        """
        self.method_to_qids = defaultdict(list)

        for qid in self.unique_qids:
            primary_method = self.qid_to_primary_method[qid]
            self.method_to_qids[primary_method].append(qid)

        # 统计每个方法的问题数
        self.method_counts = {
            method: len(qids)
            for method, qids in self.method_to_qids.items()
        }

        print(f"  Primary method distribution (questions):")
        for method, count in sorted(self.method_counts.items(), key=lambda x: x[1], reverse=True):
            ratio = count / self.num_questions
            print(f"    {method}: {count} ({ratio:.2%})")
    
    def _generate_batches(self):
        """
        生成批次
        
        策略：
        1. 分层采样：从每个方法组中按比例采样qid
        2. 确保最长尾方法的占比 ≥ min_method_ratio
        3. 每个batch包含 batch_size_q 个问题
        """
        self.batches = []
        
        methods = list(self.method_to_qids.keys())
        num_methods = len(methods)
        
        # 理论上每个 batch 最多只能覆盖 batch_size_q 个方法
        max_distinct_methods_in_batch = min(num_methods, self.batch_size_q)
        
        min_count_per_batch = max(1, int(self.batch_size_q * self.min_method_ratio))
        
        # 如果 batch 太小，不可能让所有方法都至少出现一次
        # 后面只做“尽量平衡”，不强求每种方法都进每个 batch
        
        # 为每个方法创建qid池（可重复采样）
        method_qid_pools = {
            method: list(qids) for method, qids in self.method_to_qids.items()
        }
        
        # 如果需要打乱，打乱每个方法的qid池
        if self.shuffle:
            for method in methods:
                random.shuffle(method_qid_pools[method])
        
        # 创建方法轮转指针
        method_pointers = {method: 0 for method in methods}
        
        # 生成批次直到所有qid都被采样至少一次
        max_iters = (self.num_questions + self.batch_size_q - 1) // self.batch_size_q
        used_qids = set()
        
        for batch_idx in range(max_iters):
            batch_qids = []
            
            # 策略1：分层采样，确保每个方法至少有 min_count_per_batch 个
            for method in methods:
                qids_to_add = min(min_count_per_batch, 
                                 self.batch_size_q - len(batch_qids))
                
                for _ in range(qids_to_add):
                    if len(batch_qids) >= self.batch_size_q:
                        break
                    
                    # 从该方法的qid池中取出下一个
                    pool = method_qid_pools[method]
                    pointer = method_pointers[method]
                    
                    # 如果池子用完了，重新打乱并重置指针
                    if pointer >= len(pool):
                        if self.shuffle:
                            random.shuffle(pool)
                        pointer = 0
                        method_pointers[method] = 0
                    
                    qid = pool[pointer]
                    method_pointers[method] += 1
                    
                    batch_qids.append(qid)
                    used_qids.add(qid)
            
            # 如果batch还没满，从随机方法中补充
            while len(batch_qids) < self.batch_size_q:
                method = random.choice(methods)
                pool = method_qid_pools[method]
                pointer = method_pointers[method]
                
                if pointer >= len(pool):
                    if self.shuffle:
                        random.shuffle(pool)
                    pointer = 0
                    method_pointers[method] = 0
                
                qid = pool[pointer]
                method_pointers[method] += 1
                
                batch_qids.append(qid)
                used_qids.add(qid)
            
            # 将qid转换为样本索引
            batch_indices = []
            for qid in batch_qids:
                batch_indices.extend(self.qid_to_indices[qid])
            
            self.batches.append(batch_indices)
            
            # 如果所有qid都被采样过至少一次，可以提前结束
            if len(used_qids) >= self.num_questions and batch_idx > 0:
                break
        
        print(f"  Generated {len(self.batches)} batches")
    
    def _validate_batch_distribution(self):
        """验证每个 batch 的 primary_method 分布，并统计 tie-best 占比"""
        all_min_ratios = []
        tie_ratios = []

        for batch_idx, batch_indices in enumerate(self.batches):
            batch_qids = self.df.iloc[batch_indices]['qid'].unique()

            method_counts_in_batch = defaultdict(int)
            tied_count = 0

            for qid in batch_qids:
                primary_method = self.qid_to_primary_method[qid]
                method_counts_in_batch[primary_method] += 1

                if self.qid_to_is_tied[qid]:
                    tied_count += 1

            total_qids = len(batch_qids)
            if total_qids > 0 and len(method_counts_in_batch) > 0:
                min_ratio = min(method_counts_in_batch.values()) / total_qids
                all_min_ratios.append(min_ratio)
                tie_ratios.append(tied_count / total_qids)

        avg_min_ratio = np.mean(all_min_ratios) if all_min_ratios else 0
        min_min_ratio = min(all_min_ratios) if all_min_ratios else 0
        avg_tie_ratio = np.mean(tie_ratios) if tie_ratios else 0

        print(f"    Average min primary-method ratio: {avg_min_ratio:.2%}")
        print(f"    Worst batch min primary-method ratio: {min_min_ratio:.2%}")
        print(f"    Average tie-best ratio in batch: {avg_tie_ratio:.2%}")

        if min_min_ratio < self.min_method_ratio:
            print(f"    ⚠ Warning: Some batches have primary-method ratio < {self.min_method_ratio:.0%}")
        else:
            print(f"    ✓ All batches meet min ratio requirement")
    
    def __iter__(self) -> Iterator[List[int]]:
        """迭代返回每个batch的样本索引"""
        if self.shuffle:
            # 打乱batch顺序，但batch内的indices顺序保持不变
            indices = list(range(len(self.batches)))
            random.shuffle(indices)
            for idx in indices:
                yield self.batches[idx]
        else:
            for batch in self.batches:
                yield batch
    
    def __len__(self) -> int:
        """返回batch数量"""
        return len(self.batches)


def test_sampler():
    """测试采样器"""
    print("="*80)
    print("Testing PairwiseBatchSampler")
    print("="*80)
    
    # 测试训练集
    sampler = PairwiseBatchSampler(
        dataset_file='dataset/pairwise/train_pairwise.csv',
        batch_size_q=4,
        shuffle=True,
        min_method_ratio=0.1,
        seed=42
    )
    
    print(f"\n{'='*80}")
    print("Sample batches:")
    print(f"{'='*80}")
    
    # 检查前3个batch
    for i, batch_indices in enumerate(sampler):
        if i >= 3:
            break
        
        print(f"\nBatch {i}:")
        print(f"  Total pairs: {len(batch_indices)}")
        
        # 加载这个batch的数据
        df = pd.read_csv('dataset/pairwise/train_pairwise.csv')
        batch_df = df.iloc[batch_indices]
        
        # 统计qid
        unique_qids = batch_df['qid'].unique()
        print(f"  Unique qids: {len(unique_qids)}")
        
        # 统计方法分布
        method_counts = defaultdict(int)
        for qid in unique_qids:
            qid_pairs = batch_df[batch_df['qid'] == qid]
            # ✅ 正确识别赢家方法（根据 pair_label）
            winners = []
            for _, row in qid_pairs.iterrows():
                if row['pair_label'] == 1:
                    winners.append(row['methodA_id'])
                else:
                    winners.append(row['methodB_id'])
            optimal_method = pd.Series(winners).mode()[0]
            method_counts[optimal_method] += 1
        
        print(f"  Method distribution:")
        for method, count in sorted(method_counts.items()):
            ratio = count / len(unique_qids)
            print(f"    {method}: {count} ({ratio:.2%})")
        
        min_ratio = min(method_counts.values()) / len(unique_qids)
        print(f"  Min method ratio: {min_ratio:.2%}")


class DistributedPairwiseBatchSampler(PairwiseBatchSampler):
    """
    分布式成对学习的批采样器
    
    在多GPU训练时，每个GPU只处理一部分批次
    """
    
    def __init__(
        self,
        dataset_file: str,
        batch_size_q: int = 4,
        shuffle: bool = True,
        min_method_ratio: float = 0.1,
        seed: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None
    ):
        """
        Args:
            dataset_file: 成对数据集CSV文件路径
            batch_size_q: 每个batch包含的问题数量
            shuffle: 是否打乱
            min_method_ratio: 最长尾方法的最小占比
            seed: 随机种子
            num_replicas: 总GPU数量
            rank: 当前GPU的rank
        """
        # 先调用父类初始化（生成所有批次）
        super().__init__(
            dataset_file=dataset_file,
            batch_size_q=batch_size_q,
            shuffle=shuffle,
            min_method_ratio=min_method_ratio,
            seed=seed
        )
        
        # 分布式参数
        if num_replicas is None:
            import torch.distributed as dist
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        self.num_replicas = num_replicas
        
        if rank is None:
            import torch.distributed as dist
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.rank = rank
        
        # 将总批次数分配到各个GPU
        self.total_batches = len(self.batches)
        self.num_batches_per_replica = int(np.ceil(self.total_batches / self.num_replicas))
        
        # 当前GPU的批次范围
        self.start_idx = self.rank * self.num_batches_per_replica
        self.end_idx = min(self.start_idx + self.num_batches_per_replica, self.total_batches)
        
        # 当前GPU的批次
        self.local_batches = self.batches[self.start_idx:self.end_idx]
        
        print(f"\n✓ Distributed sampler for rank {self.rank}/{self.num_replicas}:")
        print(f"  Total batches: {self.total_batches}")
        print(f"  Local batches: {len(self.local_batches)} (indices {self.start_idx}-{self.end_idx-1})")
    
    def __iter__(self) -> Iterator[List[int]]:
        """迭代返回当前GPU的batch"""
        if self.shuffle:
            # 打乱batch顺序，但batch内的indices顺序保持不变
            # 注意：使用相同的种子确保所有GPU打乱顺序一致
            indices = list(range(len(self.local_batches)))
            random.shuffle(indices)
            for idx in indices:
                yield self.local_batches[idx]
        else:
            for batch in self.local_batches:
                yield batch
    
    def __len__(self) -> int:
        """返回当前GPU的batch数量"""
        return len(self.local_batches)


if __name__ == '__main__':
    test_sampler()


"""
成对学习数据集 (Pairwise Dataset)

加载成对数据用于排序学习。

说明：
1. 支持一个问题存在多个并列最佳方法（best_methods）
2. pair_label 使用 0/1：
   - 1: methodA 优于 methodB
   - 0: methodB 优于 methodA
3. 不再通过 pair winner 反推问题级最佳方法
4. 问题级最佳方法集合直接从 best_methods_json 读取

"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict, Optional, List
from collections import defaultdict
import json
import math

def parse_best_methods(best_methods_json) -> List[str]:
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


class PairwiseDataset(Dataset):
    """
    成对学习数据集
    
    每个样本包含：
    - 问题文本
    - 方法A的输入
    - 方法B的输入
    - pair_label: +1 表示A优于B
    """
    
    def __init__(
        self,
        csv_file: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        """
        Args:
            csv_file: 成对数据CSV文件
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loaded pairwise dataset from {csv_file}")
        print(f"  Total pairs: {len(self.df)}")
        print(f"  Unique questions: {self.df['qid'].nunique()}")
        
        # 统计最优方法分布（用于计算类权重）
        self._compute_method_distribution()
    
    def _compute_method_distribution(self):
        """
        计算问题级最佳方法分布（用于类权重）

        现在不再根据 pair_label 反推赢家，而是直接从 best_methods_json 读取。
        对于 tie-best 问题，计数会在多个最佳方法之间平分。
        """
        self.method_counts = defaultdict(float)

        # 以 qid 为单位去重，只取每个问题的一条记录读取 best_methods_json
        qid_meta = self.df[['qid', 'best_methods_json']].drop_duplicates(subset=['qid'])

        for _, row in qid_meta.iterrows():
            best_methods = parse_best_methods(row['best_methods_json'])
            if not best_methods:
                continue

            share = 1.0 / len(best_methods)
            for method in best_methods:
                self.method_counts[method] += share

        self.total_questions = len(qid_meta)

        print(f"  Best-method distribution (question-level, tie-shared):")
        for method, count in sorted(self.method_counts.items(), key=lambda x: x[1], reverse=True):
            ratio = count / self.total_questions
            print(f"    {method}: {count:.2f} ({ratio:.2%})")
    
    def get_class_weights(self, weight_type='inverse') -> Dict[str, float]:
        """
        计算类权重（用于处理类别不平衡）

        Args:
            weight_type: 权重类型
                - 'inverse': 频率反比
                - 'sqrt_inverse': 频率平方根反比
                - 'log_inverse': 频率对数反比

        Returns:
            Dict[method_id, weight]
        """
        weights = {}

        for method, count in self.method_counts.items():
            freq = count / self.total_questions if self.total_questions > 0 else 0.0

            if freq <= 0:
                weights[method] = 1.0
            elif weight_type == 'inverse':
                weights[method] = 1.0 / freq
            elif weight_type == 'sqrt_inverse':
                weights[method] = 1.0 / math.sqrt(freq)
            elif weight_type == 'log_inverse':
                weights[method] = 1.0 / math.log(1 + freq)
            else:
                weights[method] = 1.0

        # 归一化权重，使平均权重为 1
        if len(weights) > 0:
            avg_weight = sum(weights.values()) / len(weights)
            weights = {k: v / avg_weight for k, v in weights.items()}

        print(f"\nClass weights (type={weight_type}):")
        for method, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {weight:.4f}")

        return weights
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个成对样本

        Returns:
            {
                'qid': 问题ID,
                'input_ids_A': 方法A的input_ids,
                'attention_mask_A': 方法A的attention_mask,
                'input_ids_B': 方法B的input_ids,
                'attention_mask_B': 方法B的attention_mask,
                'pair_label': 0或1，1表示A优于B
                'methodA_id': 方法A名称,
                'methodB_id': 方法B名称,
                'best_methods': 问题级最佳方法集合,
                'is_tied_best': 是否并列最佳,
                'num_best_methods': 最佳方法数量
            }
        """
        row = self.df.iloc[idx]

        # 编码方法A
        encoding_A = self.tokenizer(
            row['methodA_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 编码方法B
        encoding_B = self.tokenizer(
            row['methodB_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 直接读取问题级最佳方法集合
        best_methods = parse_best_methods(row.get('best_methods_json', '[]'))

        return {
            'qid': torch.tensor(row['qid'], dtype=torch.long),
            'input_ids_A': encoding_A['input_ids'].squeeze(0),
            'attention_mask_A': encoding_A['attention_mask'].squeeze(0),
            'input_ids_B': encoding_B['input_ids'].squeeze(0),
            'attention_mask_B': encoding_B['attention_mask'].squeeze(0),
            'score_a': torch.tensor(float(row.get('scoreA', 0.0)), dtype=torch.float),
            'score_b': torch.tensor(float(row.get('scoreB', 0.0)), dtype=torch.float),
            'pair_label': torch.tensor(row['pair_label'], dtype=torch.long),
            'methodA_id': row['methodA_id'],
            'methodB_id': row['methodB_id'],
            'best_methods': best_methods,
            'is_tied_best': bool(row.get('is_tied_best', False)),
            'num_best_methods': int(row.get('num_best_methods', len(best_methods)))
        }


def pairwise_collate_fn(batch):
    """
    自定义 collate 函数用于成对数据

    统一字段命名（pairwise 架构标准）:
    - input_ids_a, attention_mask_a: 方法A
    - input_ids_b, attention_mask_b: 方法B
    - label_ab: 0或1，1表示A优于B
    """
    return {
        # 标准 pairwise 字段（必需）
        'input_ids_a': torch.stack([item['input_ids_A'] for item in batch]),
        'attention_mask_a': torch.stack([item['attention_mask_A'] for item in batch]),
        'input_ids_b': torch.stack([item['input_ids_B'] for item in batch]),
        'attention_mask_b': torch.stack([item['attention_mask_B'] for item in batch]),
        'score_a': torch.stack([item['score_a'] for item in batch]),
        'score_b': torch.stack([item['score_b'] for item in batch]),
        'label_ab': torch.stack([item['pair_label'] for item in batch]),

        # 辅助字段（用于采样/统计，不直接喂给模型）
        'qids': torch.stack([item['qid'] for item in batch]),
        'method_id_a': [item['methodA_id'] for item in batch],
        'method_id_b': [item['methodB_id'] for item in batch],
        'best_methods': [item['best_methods'] for item in batch],
        'is_tied_best': [item['is_tied_best'] for item in batch],
        'num_best_methods': [item['num_best_methods'] for item in batch]
    }


def create_pairwise_dataloaders(
    train_file: str,
    val_file: str,
    test_file: str,
    tokenizer: PreTrainedTokenizer,
    batch_size_q: int = 4,
    batch_size_pairs: Optional[int] = None,
    max_length: int = 512,
    num_workers: int = 0,
    use_sampler: bool = False,
    weight_type: str = 'sqrt_inverse',
    is_distributed: bool = False
):
    """
    创建成对学习的数据加载器

    Args:
        train_file: 训练集CSV文件
        val_file: 验证集CSV文件
        test_file: 测试集CSV文件
        tokenizer: 分词器
        batch_size_q: 若使用 sampler，则表示每批问题数
        batch_size_pairs: 若不用 sampler，则表示每批 pair 数
        max_length: 最大序列长度
        num_workers: DataLoader工作进程数
        use_sampler: 是否使用分层采样器
        weight_type: 类权重类型
        is_distributed: 是否分布式训练

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    from .pairwise_sampler import PairwiseBatchSampler, DistributedPairwiseBatchSampler

    if batch_size_pairs is None:
        batch_size_pairs = batch_size_q * 4

    # 创建数据集
    train_dataset = PairwiseDataset(train_file, tokenizer, max_length)
    val_dataset = PairwiseDataset(val_file, tokenizer, max_length)
    test_dataset = PairwiseDataset(test_file, tokenizer, max_length)

    # 获取类权重
    class_weights = train_dataset.get_class_weights(weight_type=weight_type)

    # 创建训练数据加载器
    if use_sampler:
        if is_distributed:
            import torch.distributed as dist
            train_sampler = DistributedPairwiseBatchSampler(
                dataset_file=train_file,
                batch_size_q=batch_size_q,
                shuffle=True,
                min_method_ratio=0.1,
                seed=42,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank()
            )
            print(f"\n✓ Using DistributedPairwiseBatchSampler:")
            print(f"  World size: {dist.get_world_size()}")
            print(f"  Rank: {dist.get_rank()}")
        else:
            train_sampler = PairwiseBatchSampler(
                dataset_file=train_file,
                batch_size_q=batch_size_q,
                shuffle=True,
                min_method_ratio=0.1,
                seed=42
            )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=pairwise_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_pairs,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=pairwise_collate_fn
        )

    # 验证和测试集
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_pairs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pairwise_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_pairs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pairwise_collate_fn
    )

    print(f"\n✓ Dataloaders created:")
    print(f"  Train batches: {len(train_loader)} (per GPU)" if is_distributed and use_sampler else f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_weights


# 测试代码
if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    print("Testing PairwiseDataset...")
    print("="*80)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    # 创建数据集
    dataset = PairwiseDataset(
        csv_file='dataset/pairwise/train_pairwise.csv',
        tokenizer=tokenizer,
        max_length=256
    )
    
    # 测试获取样本
    print(f"\nTesting __getitem__...")
    sample = dataset[0]
    print(f"  qid: {sample['qid']}")
    print(f"  input_ids_A shape: {sample['input_ids_A'].shape}")
    print(f"  input_ids_B shape: {sample['input_ids_B'].shape}")
    print(f"  pair_label: {sample['pair_label']}")
    print(f"  methodA_id: {sample['methodA_id']}")
    print(f"  methodB_id: {sample['methodB_id']}")
    print(f"  best_methods: {sample['best_methods']}")
    print(f"  is_tied_best: {sample['is_tied_best']}")
    print(f"  num_best_methods: {sample['num_best_methods']}")
    
    # 获取类权重
    print(f"\nClass weights:")
    weights = dataset.get_class_weights('sqrt_inverse')
    
    print("\n✓ Test completed!")

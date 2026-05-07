"""
Method Efficiency Model Package - Pairwise Learning

包含用于方法选择模型的Pairwise学习组件
"""

# 核心模型
from .cross_encoder import MethodSelectionCrossEncoder, create_tokenizer
from .config import Config, ModelConfig, TrainingConfig

# Pairwise学习组件
from .pairwise_dataset import PairwiseDataset, create_pairwise_dataloaders
from .pairwise_sampler import PairwiseBatchSampler, DistributedPairwiseBatchSampler
from .pairwise_trainer import PairwiseTrainer

# 评估工具
from .evaluation_utils import load_method_scores, evaluate_model_vs_methods

__all__ = [
    # 配置
    'Config',
    'ModelConfig',
    'TrainingConfig',
    # 模型
    'MethodSelectionCrossEncoder',
    'create_tokenizer',
    # Pairwise数据集
    'PairwiseDataset',
    'create_pairwise_dataloaders',
    # Pairwise采样器
    'PairwiseBatchSampler',
    'DistributedPairwiseBatchSampler',
    # Pairwise训练器
    'PairwiseTrainer',
    # 评估工具
    'load_method_scores',
    'evaluate_model_vs_methods',
]


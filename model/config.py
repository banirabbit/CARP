"""
训练配置
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class ModelConfig:
    """模型配置"""
    model_name_or_path: str = "Qwen/Qwen2-0.5B"  # 或 "meta-llama/Llama-2-7b-hf"
    num_methods: int = 5
    method_vocab: Optional[List[str]] = None
    use_method_embedding: bool = False
    method_embedding_scale: float = 1.0
    method_embedding_init_std: float = 0.02
    pooling_mode: str = 'mean'  # 'mean', 'cls', 'max'
    dropout: float = 0.1
    use_gradient_checkpointing: bool = False
    max_length: int = 512
    
    # 防止塌缩策略（统一 Pairwise 架构）
    temperature: float = 1.5  # 温度参数 T（1.5-3.0），防止塌缩
    learnable_temperature: bool = False  # 是否可学习（建议False）
    margin: float = 0.2  # 边际参数 γ（0.1-0.3），要求"赢得足够多"
    
    # 损失函数配置（保留，兼容旧代码）
    loss_type: str = 'pairwise'  # 'pairwise'（推荐），'listwise', 'mixed'
    pairwise_weight: float = 0.5  # 混合损失中pairwise的权重（0-1）
    pairwise_margin_mode: str = 'legacy'  # 'legacy', 'symmetric'
    dynamic_margin_enabled: bool = False
    dynamic_margin_scale: float = 0.0
    dynamic_margin_power: float = 1.0
    dynamic_margin_max: Optional[float] = None
    pointwise_weight: float = 0.0  # pointwise回归权重；0表示关闭
    pointwise_loss_type: str = 'huber'  # 'huber', 'mse', 'l1'
    pointwise_huber_beta: float = 0.1
    pointwise_apply_sigmoid: bool = True  # 用sigmoid(logit)回归[0,1] F1
    listwise_weight: float = 0.0  # listwise分布监督权重；0表示关闭
    listwise_loss_type: str = 'kl'  # 'kl', 'ce'
    listwise_target_temperature: float = 1.0  # 真实F1分布softmax温度
    listwise_min_methods: int = 3  # 至少需要多少个唯一方法才计算listwise


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据
    train_file: str = "dataset/train.csv"
    val_file: str = "dataset/val.csv"
    test_file: str = "dataset/test.csv"
    
    # 训练超参数
    batch_size_q: int = 8  # 每个 batch 的问题数量
    num_methods: int = 5   # 每个问题的方法数量
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    num_epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # 优化器
    optimizer: str = "adamw"  # 'adamw', 'sgd'
    optimizer_type: str = "adamw"  # 别名，兼容新配置
    scheduler: str = "linear"  # 'linear', 'cosine', 'constant'
    scheduler_type: str = "linear"  # 别名，兼容新配置
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # 采样器
    use_stratified_sampler: bool = True
    sampler_version: str = "v1"  # 'v1', 'v2'
    method_weights: Optional[dict] = None
    shuffle_train: bool = True
    shuffle_val: bool = False
    
    # 分布式训练
    use_ddp: bool = False
    local_rank: int = -1
    
    # 保存和日志
    output_dir: str = "outputs"
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    save_epoch_checkpoints: bool = True
    
    # 评估
    eval_strategy: str = "steps"  # 'steps', 'epoch'
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True

    # 平均 F1 先验权重（用于评估和推理）
    # final_score = model_score + f1_prior_alpha * avg_f1(method)
    f1_prior_alpha: float = 0.5

    # baseline-aware / tie-aware checkpoint 选择
    checkpoint_opportunity_threshold: float = 0.05
    checkpoint_overall_gain_floor: float = -0.005
    checkpoint_easy_regret_ceiling: float = 0.01

    # 防止塌缩策略
    freeze_backbone_epochs: int = 0  # 前N个epoch冻结backbone
    temperature_schedule: Optional[str] = None  # 温度退火: 'linear', 'cosine', None
    
    # 类权重（成对学习）
    use_class_weights: bool = False
    weight_type: str = "sqrt_inverse"  # 'inverse', 'sqrt_inverse', 'log_inverse'
    use_pair_score_weights: bool = False
    pair_score_weight_floor: float = 0.25
    pair_score_weight_power: float = 1.0
    tied_best_weight: float = 1.0
    num_best_methods_weight_power: float = 0.0
    
    # 其他
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    fp16: bool = False
    bf16: bool = False
    
    def __post_init__(self):
        """验证和调整配置"""
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 验证文件存在
        if not os.path.exists(self.train_file):
            raise FileNotFoundError(f"Training file not found: {self.train_file}")
        
        # 实际 batch size = batch_size_q * num_methods
        self.effective_batch_size = self.batch_size_q * self.num_methods


@dataclass
class Config:
    """完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典创建配置"""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        return cls(model=model_config, training=training_config)
    
    @classmethod
    def from_yaml(cls, yaml_file: str):
        """从 YAML 文件加载配置"""
        import yaml
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        from dataclasses import asdict
        return {
            'model': asdict(self.model),
            'training': asdict(self.training)
        }
    
    def save(self, save_path: str):
        """保存配置到 YAML"""
        import yaml
        with open(save_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        print(f"Config saved to {save_path}")


# 预定义配置
def get_qwen_config(size: str = "0.5B") -> Config:
    """获取 Qwen 配置"""
    config = Config()
    config.model.model_name_or_path = f"Qwen/Qwen2-{size}"
    return config


def get_llama_config(size: str = "7b") -> Config:
    """获取 LLaMA 配置"""
    config = Config()
    config.model.model_name_or_path = f"meta-llama/Llama-2-{size}-hf"
    return config


def get_debug_config() -> Config:
    """获取调试配置（小模型+小数据）"""
    config = Config()
    config.model.model_name_or_path = "bert-base-uncased"
    config.training.batch_size_q = 2
    config.training.num_epochs = 2
    config.training.logging_steps = 10
    config.training.eval_steps = 50
    return config

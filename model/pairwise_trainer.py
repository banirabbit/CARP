"""
成对学习训练器 (Pairwise Trainer)

使用 Logistic Ranking Loss 进行训练
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Dict, Optional, Any, Tuple
from tqdm import tqdm
import numpy as np


class LogisticRankingLoss(nn.Module):
    """
    Logistic Ranking Loss
    
    L = log(1 + exp(-y * (scoreA - scoreB)))
    
    其中:
    - y ∈ {+1, -1}: pair_label
    - scoreA, scoreB: 方法A和B的分数
    
    当 y=+1 时，希望 scoreA > scoreB
    当 y=-1 时，希望 scoreB > scoreA
    """
    
    def __init__(self, use_class_weights: bool = True):
        super().__init__()
        self.use_class_weights = use_class_weights
    
    def forward(
        self,
        score_diff: torch.Tensor,  # scoreA - scoreB
        pair_labels: torch.Tensor,  # +1 or -1
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            score_diff: [batch_size] scoreA - scoreB
            pair_labels: [batch_size] +1 or -1
            sample_weights: [batch_size] 每个样本的权重（可选）
        
        Returns:
            loss: 标量
        """
        # Logistic Ranking Loss
        # L = log(1 + exp(-y * (scoreA - scoreB)))
        loss_per_sample = torch.log(1 + torch.exp(-pair_labels * score_diff))
        
        # 应用样本权重
        if sample_weights is not None and self.use_class_weights:
            loss_per_sample = loss_per_sample * sample_weights
            loss = loss_per_sample.sum() / sample_weights.sum()
        else:
            loss = loss_per_sample.mean()
        
        return loss


class PairwiseTrainer:
    """
    成对学习训练器
    
    特点:
    1. 使用 Logistic Ranking Loss
    2. 支持类权重处理不平衡
    3. 分别编码方法A和B
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader: Optional = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        config: Optional[Any] = None,
        class_weights: Optional[Dict[str, float]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_amp: bool = False
    ):
        """
        Args:
            model: Cross-Encoder 模型
            train_loader: 训练数据加载器（成对数据）
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器（可选）
            optimizer: 优化器
            scheduler: 学习率调度器
            config: 训练配置
            class_weights: 类权重字典 {method_id: weight}
            device: 设备
            use_amp: 是否使用自动混合精度
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.use_amp = use_amp
        self.class_weights = class_weights or {}
        
        # 优化器和调度器
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # 损失函数
        self.criterion = LogisticRankingLoss(use_class_weights=len(self.class_weights) > 0)
        
        # 检测是否是 DeepSpeed engine
        try:
            import deepspeed
            self.use_deepspeed = isinstance(model, deepspeed.DeepSpeedEngine)
        except:
            self.use_deepspeed = hasattr(model, 'module') and hasattr(model, 'optimizer')
        
        if self.use_deepspeed:
            print("  ✓ Detected DeepSpeed engine")
            print(f"    World size: {model.world_size if hasattr(model, 'world_size') else 'N/A'}")
            print(f"    Local rank: {model.local_rank if hasattr(model, 'local_rank') else 'N/A'}")
        
        # 混合精度 - 只在 FP16 时使用 GradScaler
        from torch.cuda.amp import GradScaler
        self.use_fp16 = use_amp and (config and config.training.fp16) and not self.use_deepspeed
        self.scaler = GradScaler() if self.use_fp16 else None
        
        if self.scaler:
            print("  Using GradScaler for FP16 training")
        elif use_amp:
            print("  Using BF16 (no GradScaler needed)")
        if self.use_deepspeed:
            print("  DeepSpeed manages mixed precision internally")
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_metric = -float('inf') if config and config.training.greater_is_better else float('inf')
        self.best_top1_overlap = -float('inf')  # 追踪最佳 top-1 overlap
        self.best_method_acc_tie = -float('inf')
        self.best_router_key = None
        self.best_model_path = None

        self.checkpoint_opportunity_threshold = (
            config.training.checkpoint_opportunity_threshold
            if config and hasattr(config.training, 'checkpoint_opportunity_threshold')
            else 0.05
        )
        self.checkpoint_overall_gain_floor = (
            config.training.checkpoint_overall_gain_floor
            if config and hasattr(config.training, 'checkpoint_overall_gain_floor')
            else -0.005
        )
        self.checkpoint_easy_regret_ceiling = (
            config.training.checkpoint_easy_regret_ceiling
            if config and hasattr(config.training, 'checkpoint_easy_regret_ceiling')
            else 0.01
        )
           
        # 日志
        self.train_loss_history = []
        self.val_metrics_history = []
        
        # 分阶段训练和温度退火配置
        self.freeze_epochs = config.training.freeze_backbone_epochs if config and hasattr(config.training, 'freeze_backbone_epochs') else 0
        self.temperature_schedule = config.training.temperature_schedule if config and hasattr(config.training, 'temperature_schedule') else None
        self.initial_temperature = float(self.model.temperature) if hasattr(self.model, 'temperature') else 1.0
        
        print(f"PairwiseTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Use AMP: {use_amp}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        if class_weights:
            print(f"  Using class weights: {len(class_weights)} classes")
        print("  Router checkpoint selection:")
        print(f"    opportunity_threshold={self.checkpoint_opportunity_threshold:.3f}")
        print(f"    overall_gain_floor={self.checkpoint_overall_gain_floor:.3f}")
        print(f"    easy_regret_ceiling={self.checkpoint_easy_regret_ceiling:.3f}")
        self.save_epoch_checkpoints = (
            config.training.save_epoch_checkpoints
            if config and hasattr(config.training, 'save_epoch_checkpoints')
            else True
        )
        print(f"  Save epoch checkpoints: {self.save_epoch_checkpoints}")
        if config and hasattr(config.training, 'use_pair_score_weights'):
            print("  Pair weighting:")
            print(f"    use_pair_score_weights={config.training.use_pair_score_weights}")
            print(f"    pair_score_weight_floor={getattr(config.training, 'pair_score_weight_floor', 0.25):.3f}")
            print(f"    pair_score_weight_power={getattr(config.training, 'pair_score_weight_power', 1.0):.3f}")
            print(f"    tied_best_weight={getattr(config.training, 'tied_best_weight', 1.0):.3f}")
            print(f"    num_best_methods_weight_power={getattr(config.training, 'num_best_methods_weight_power', 0.0):.3f}")

    def _router_checkpoint_eligible(self, metrics: Dict[str, float]) -> bool:
        return (
            metrics.get('overall_gain', -float('inf')) >= self.checkpoint_overall_gain_floor
            and metrics.get('easy_regret', float('inf')) <= self.checkpoint_easy_regret_ceiling
        )

    def _build_router_checkpoint_key(self, metrics: Dict[str, float]) -> Tuple[float, ...]:
        eligible = 1.0 if self._router_checkpoint_eligible(metrics) else 0.0
        return (
            eligible,
            metrics.get('hard_regret_reduction', -float('inf')),
            metrics.get('hard_gain', -float('inf')),
            metrics.get('overall_gain', -float('inf')),
            -metrics.get('easy_regret', float('inf')),
            metrics.get('predicted_best_f1', -float('inf')),
        )
    
    def _encode_and_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        method_ids: Optional[Any] = None
    ) -> torch.Tensor:
        """
        对单个样本编码并打分（推断接口）
        
        统一使用 model.score_single() 方法
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size] 原始标量分数（logit 空间）
        
        **注意**：
        返回原始 logits（未除温度）。温度只在训练的 pairwise 损失中应用一次。
        """
        # 获取实际模型（处理DeepSpeed包装）
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 使用模型的 score_single 接口（返回原始 logits）
        logits = actual_model.score_single(input_ids, attention_mask, method_ids=method_ids)
        
        return logits
    
    def _get_sample_weights(
        self,
        best_methods_batch: list,
        score_a: Optional[torch.Tensor] = None,
        score_b: Optional[torch.Tensor] = None,
        is_tied_best: Optional[list] = None,
        num_best_methods: Optional[list] = None
    ) -> Optional[torch.Tensor]:
        """
        根据问题级最佳方法集合获取样本权重

        Args:
            best_methods_batch: List[List[str]]
                每个样本对应一个最佳方法集合，例如：
                [
                    ['hippo'],
                    ['gr', 'hippo'],
                    ...
                ]

        Returns:
            weights: [batch_size] 权重张量
        """
        use_pair_score_weights = (
            self.config is not None
            and hasattr(self.config.training, 'use_pair_score_weights')
            and self.config.training.use_pair_score_weights
        )

        if not self.class_weights and not use_pair_score_weights:
            return None

        pair_score_weight_floor = (
            self.config.training.pair_score_weight_floor
            if self.config is not None and hasattr(self.config.training, 'pair_score_weight_floor')
            else 0.25
        )
        pair_score_weight_power = (
            self.config.training.pair_score_weight_power
            if self.config is not None and hasattr(self.config.training, 'pair_score_weight_power')
            else 1.0
        )
        tied_best_weight = (
            self.config.training.tied_best_weight
            if self.config is not None and hasattr(self.config.training, 'tied_best_weight')
            else 1.0
        )
        num_best_methods_weight_power = (
            self.config.training.num_best_methods_weight_power
            if self.config is not None and hasattr(self.config.training, 'num_best_methods_weight_power')
            else 0.0
        )

        weights = []
        for idx, best_methods in enumerate(best_methods_batch):
            weight = 1.0

            if self.class_weights and best_methods:
                method_weights = [self.class_weights.get(m, 1.0) for m in best_methods]
                weight *= sum(method_weights) / len(method_weights)

            if use_pair_score_weights and score_a is not None and score_b is not None:
                gap = abs(float(score_a[idx].item()) - float(score_b[idx].item()))
                gap = max(0.0, min(1.0, gap))
                gap_weight = pair_score_weight_floor + (1.0 - pair_score_weight_floor) * (gap ** pair_score_weight_power)
                weight *= gap_weight

                if is_tied_best is not None and bool(is_tied_best[idx]):
                    weight *= tied_best_weight

                if num_best_methods is not None:
                    num_best = max(1, int(num_best_methods[idx]))
                    weight /= float(num_best ** num_best_methods_weight_power)

            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float, device=self.device)
    
    def freeze_backbone(self):
        """冻结 backbone，只训练 scorer"""
        print("\n🔒 Freezing backbone (只训练 scorer 头)")
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model

        for param in actual_model.backbone.parameters():
            param.requires_grad = False
        for param in actual_model.scorer.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in actual_model.parameters())
        print(f"  Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def unfreeze_backbone(self):
        """解冻 backbone"""
        print("\n🔓 Unfreezing backbone (全模型训练)")
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model

        for param in actual_model.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in actual_model.parameters())
        print(f"  Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def update_temperature(self, epoch: int, num_epochs: int):
        """
        更新温度参数（退火）
        
        策略：
        - 初期保持高温度（如1.5-2.0），平滑logits
        - 后期逐渐降温到1.0，恢复正常尺度
        """
        if self.temperature_schedule is None:
            return
        
        if self.temperature_schedule == 'linear':
            # 线性退火：从 initial_temperature 到 1.0
            # 最后20%的训练中完成退火
            anneal_start = int(num_epochs * 0.8)
            if epoch >= anneal_start:
                progress = (epoch - anneal_start) / (num_epochs - anneal_start)
                new_temp = self.initial_temperature + (1.0 - self.initial_temperature) * progress
                self.model.temperature.data = torch.tensor(new_temp, device=self.device)
                print(f"  🌡️  Temperature annealed to {new_temp:.3f}")
        
        elif self.temperature_schedule == 'cosine':
            # 余弦退火
            anneal_start = int(num_epochs * 0.8)
            if epoch >= anneal_start:
                progress = (epoch - anneal_start) / (num_epochs - anneal_start)
                new_temp = 1.0 + (self.initial_temperature - 1.0) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
                self.model.temperature.data = torch.tensor(new_temp, device=self.device)
                print(f"  🌡️  Temperature annealed to {new_temp:.3f}")

    def _has_non_finite_gradients(self) -> bool:
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        for param in actual_model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return True
        return False
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        skipped_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch+1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动标准 pairwise 字段到设备
            input_ids_a = batch['input_ids_a'].to(self.device)
            attention_mask_a = batch['attention_mask_a'].to(self.device)
            input_ids_b = batch['input_ids_b'].to(self.device)
            attention_mask_b = batch['attention_mask_b'].to(self.device)
            label_ab = batch['label_ab'].to(self.device)
            
            best_methods = batch.get('best_methods')
            sample_weights = self._get_sample_weights(
                best_methods,
                score_a=batch.get('score_a'),
                score_b=batch.get('score_b'),
                is_tied_best=batch.get('is_tied_best'),
                num_best_methods=batch.get('num_best_methods')
            ) if best_methods is not None else None
            
            # 前向传播（模型内部计算损失）
            if self.use_amp:
                with autocast():
                    outputs = self.model.forward_pairwise(
                        input_ids_a=input_ids_a,
                        attention_mask_a=attention_mask_a,
                        input_ids_b=input_ids_b,
                        attention_mask_b=attention_mask_b,
                        label_ab=label_ab,
                        sample_weights=sample_weights,
                        target_score_a=batch.get('score_a'),
                        target_score_b=batch.get('score_b'),
                        qids=batch.get('qids'),
                        method_id_a=batch.get('method_id_a'),
                        method_id_b=batch.get('method_id_b')
                    )
                    loss = outputs['loss']
            else:
                outputs = self.model.forward_pairwise(
                    input_ids_a=input_ids_a,
                    attention_mask_a=attention_mask_a,
                    input_ids_b=input_ids_b,
                    attention_mask_b=attention_mask_b,
                    label_ab=label_ab,
                    qids=batch.get('qids'),
                    method_id_a=batch.get('method_id_a'),
                    method_id_b=batch.get('method_id_b'),
                    sample_weights=sample_weights,
                    target_score_a=batch.get('score_a'),
                    target_score_b=batch.get('score_b'),
                )
                loss = outputs['loss']
            
            # 🔍 在这里加 NaN / Inf 检查
            if not torch.isfinite(loss):
                print(f"❌ Non-finite loss detected at epoch={self.epoch}, batch={batch_idx}, step={self.global_step}")
                with torch.no_grad():
                    score_a = outputs.get('score_a')
                    score_b = outputs.get('score_b')
                    if score_a is not None and score_b is not None:
                        print(
                            f"    score_a=[{score_a.min().item():.2f}, {score_a.max().item():.2f}], "
                            f"score_b=[{score_b.min().item():.2f}, {score_b.max().item():.2f}]"
                        )
                    print("    label_ab sample:", label_ab[:8].detach().cpu().tolist())
                # 防止这个 batch 污染优化器和统计，直接跳过
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                continue
            
            # 反向传播和优化器步骤
            if self.use_deepspeed:
                # DeepSpeed engine 自动处理混合精度、梯度累积、梯度裁剪等
                self.model.backward(loss)
                if self._has_non_finite_gradients():
                    print(f"❌ Non-finite gradients detected at epoch={self.epoch}, batch={batch_idx}, step={self.global_step}; skipping step")
                    self.model.zero_grad()
                    skipped_batches += 1
                    continue
                self.model.step()
            else:
                # 标准训练流程
                if self.scaler:  # FP16 with GradScaler
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                else:  # BF16 or FP32
                    loss.backward()

                if self._has_non_finite_gradients():
                    print(f"❌ Non-finite gradients detected at epoch={self.epoch}, batch={batch_idx}, step={self.global_step}; skipping step")
                    self.optimizer.zero_grad(set_to_none=True)
                    skipped_batches += 1
                    continue
                
                # 更新参数
                if self.scaler:  # FP16 with GradScaler
                    if self.config and self.config.training.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:  # BF16 or FP32
                    if self.config and self.config.training.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm
                        )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
            
            self.global_step += 1

            # Debug 打印 logit 范围
            if batch_idx % 200 == 0:
                with torch.no_grad():
                    score_a = outputs.get('score_a')
                    score_b = outputs.get('score_b')
                    if score_a is not None and score_b is not None:
                        print(
                            f"[debug] step={self.global_step}, "
                            f"score_a=[{score_a.min().item():.2f}, {score_a.max().item():.2f}], "
                            f"score_b=[{score_b.min().item():.2f}, {score_b.max().item():.2f}]"
                        )

            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{total_loss / num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 记录日志
            if self.config and self.global_step % self.config.training.logging_steps == 0:
                avg_loss = total_loss / max(1, num_batches)
                self.train_loss_history.append({
                    'step': self.global_step,
                    'epoch': self.epoch,
                    'loss': avg_loss
                })
                print(f"\nStep {self.global_step}: loss={avg_loss:.4f}")
            

        if num_batches == 0:
            print(f"⚠️  Epoch {self.epoch + 1} produced no valid optimization steps. skipped_batches={skipped_batches}")
            return {'loss': float('inf'), 'valid_batches': 0, 'skipped_batches': skipped_batches}

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss, 'valid_batches': num_batches, 'skipped_batches': skipped_batches}
    
    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估
        
        注意：这里只计算损失，不进行完整的Top-k评估
        完整评估需要使用原始格式数据，见 PairwiseEvaluator
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                # 移动标准 pairwise 字段到设备
                input_ids_a = batch['input_ids_a'].to(self.device)
                attention_mask_a = batch['attention_mask_a'].to(self.device)
                input_ids_b = batch['input_ids_b'].to(self.device)
                attention_mask_b = batch['attention_mask_b'].to(self.device)
                label_ab = batch['label_ab'].to(self.device)
                best_methods = batch.get('best_methods')
                sample_weights = self._get_sample_weights(
                    best_methods,
                    score_a=batch.get('score_a'),
                    score_b=batch.get('score_b'),
                    is_tied_best=batch.get('is_tied_best'),
                    num_best_methods=batch.get('num_best_methods')
                ) if best_methods is not None else None
                
                # 前向传播（模型内部计算损失）
                if self.use_amp or self.use_deepspeed:
                    with autocast():
                        outputs = self.model.forward_pairwise(
                            input_ids_a=input_ids_a,
                            attention_mask_a=attention_mask_a,
                            input_ids_b=input_ids_b,
                            attention_mask_b=attention_mask_b,
                            label_ab=label_ab,
                            sample_weights=sample_weights,
                            target_score_a=batch.get('score_a'),
                            target_score_b=batch.get('score_b'),
                            method_id_a=batch.get('method_id_a'),
                            method_id_b=batch.get('method_id_b')
                        )
                        loss = outputs['loss']
                else:
                    outputs = self.model.forward_pairwise(
                        input_ids_a=input_ids_a,
                        attention_mask_a=attention_mask_a,
                        input_ids_b=input_ids_b,
                        attention_mask_b=attention_mask_b,
                        label_ab=label_ab,
                        sample_weights=sample_weights,
                        target_score_a=batch.get('score_a'),
                        target_score_b=batch.get('score_b'),
                        method_id_a=batch.get('method_id_a'),
                        method_id_b=batch.get('method_id_b')
                    )
                    loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def evaluate_top3_coverage(self, tokenizer, max_length: int = 256,
                             dataset_dir: str = 'dataset', dataset_name: str = 'hotpot',
                             split: str = 'val', alpha: float = 0.5) -> Dict[str, float]:
        """
        从原始方法评分文件评估模型性能，与真实方法F1分数对比

        Args:
            tokenizer: 分词器
            max_length: 最大序列长度
            dataset_dir: 数据集根目录
            dataset_name: 数据集名称（hotpot等）
            split: 数据集划分（val/test）
            alpha: 平均 F1 先验权重（默认 0.5）

        Returns:
            评估指标字典
        """
        # 实验版：先完全关闭 F1 先验，只看模型自己的打分
        # alpha = 0.0
        print(f"  🔍 F1 Prior Alpha (EXP_NO_PRIOR): {alpha}")

        from model.evaluation_utils import load_method_scores, evaluate_model_vs_methods

        self.model.eval()

        # 加载方法评分数据
        print(f"\n  📂 Loading {split} data from method score files...")
        method_data = load_method_scores(dataset_dir, dataset_name, split)

        questions = method_data['questions']
        method_names = method_data['methods']
        true_scores = method_data['scores']  # shape: (num_questions, num_methods)

        num_questions = len(questions)
        num_methods = len(method_names)

        print(f"  ✓ Loaded {num_questions} questions, {num_methods} methods")

        # 加载方法效率信息（包含 avg_f1）
        import json
        import os
        efficiency_path = os.path.join(dataset_dir, 'method_efficiency.json')

        method_avg_f1 = {}
        if os.path.exists(efficiency_path):
            with open(efficiency_path, 'r') as f:
                eff_dict = json.load(f)
            method_avg_f1 = {m: v['avg_f1'] for m, v in eff_dict.items()}
            print(f"  ✓ Loaded method avg_f1 from {efficiency_path}")
            print(f"    Using alpha={alpha} for F1 prior")
        else:
            print(f"  ⚠️  Method efficiency file not found: {efficiency_path}")
            print(f"    Will use alpha=0 (no F1 prior)")
            alpha = 0.0

        # 生成方法描述（用于模型输入）
        method_descriptions = {
            'dalk': 'DALK (Dense Automatic Knowledge Linking)',
            'gr': 'GR (Graph Reasoning)',
            'hippo': 'HIPPO (Hierarchical Passage Processing)',
            'lgraph': 'LGraph (Logic Graph)',
            'light': 'LIGHT (Lightweight Inference)',
            'qagn': 'QAGN ( semantic–structural GraphRAG method )'
        }
        
        # 只在主进程（rank 0）进行评估，避免重复计算
        import torch.distributed as dist
        if self.use_deepspeed and dist.is_initialized() and dist.get_rank() != 0:
            print(f"  ⏭️  Skipping evaluation on rank {dist.get_rank()}")
            # 返回空指标，不参与评估
            return {
                'top1_overlap': 0.0,
                'top3_overlap': 0.0,
                'spearman_corr': 0.0,
                'best_method_acc': 0.0,
                'strongest_single_method': '',
                'strongest_single_avg_f1': 0.0,
                'opportunity_ratio': 0.0,
                'overall_gain': 0.0,
                'hard_gain': 0.0,
                'hard_regret_reduction': 0.0,
                'easy_regret': 0.0,
                'predicted_best_f1': 0.0,
                'oracle_best_f1': 0.0,
                'f1_gap': 0.0
            }
        
        # 批量评估：一次处理多个(问题, 方法)对
        model_predictions = np.zeros((num_questions, num_methods))
        batch_size = 16  # 批处理大小
        
        # 准备所有文本对
        all_texts = []
        text_indices = []  # (question_idx, method_idx)
        for i, question in enumerate(questions):
            for j, method in enumerate(method_names):
                method_desc = method_descriptions.get(method, method)
                text = f"Question: {question}\nMethod: {method_desc}"
                all_texts.append(text)
                text_indices.append((i, j))
        
        print(f"  🚀 Evaluating {len(all_texts)} samples in batches of {batch_size}")
        
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(all_texts), batch_size), 
                                   desc=f"Evaluating on {split}", leave=False):
                batch_end = min(batch_start + batch_size, len(all_texts))
                batch_texts = all_texts[batch_start:batch_end]
                
                # 批量tokenize
                encodings = tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                batch_method_ids = [method for _, method in text_indices[batch_start:batch_end]]
                
                # 批量推理
                if self.use_amp or self.use_deepspeed:
                    with autocast():
                        scores = self._encode_and_score(input_ids, attention_mask, method_ids=batch_method_ids)
                else:
                    scores = self._encode_and_score(input_ids, attention_mask, method_ids=batch_method_ids)

                # 将分数填入结果矩阵，加入 F1 先验
                for k, (i, j) in enumerate(text_indices[batch_start:batch_end]):
                    raw_score = scores[k].item() if scores.dim() > 0 else scores.item()
                    method_name = method_names[j]  # j 对应的方法名
                    prior_f1 = method_avg_f1.get(method_name, 0.0)

                    # 融合模型分数与 F1 先验
                    final_score = raw_score + alpha * prior_f1
                    model_predictions[i, j] = final_score
        
        self.model.train()
        
        # 计算评估指标（模型预测vs真实方法F1分数）
        metrics = evaluate_model_vs_methods(
            model_predictions,
            true_scores,
            method_names,
            opportunity_threshold=self.checkpoint_opportunity_threshold
        )
        
        return metrics
    
    def save_checkpoint(self, save_path: str, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'best_top1_overlap': self.best_top1_overlap,
            'best_method_acc_tie': self.best_method_acc_tie,
            'best_router_key': self.best_router_key,
            'global_step': self.global_step,
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)
        tag = "best checkpoint" if is_best else "checkpoint"
        print(f"  ✓ Saved {tag} to {save_path}")
    
    def _plot_training_curves(self, output_dir: str):
        """
        绘制训练曲线

        Args:
            output_dir: 输出目录
        """
        import json
        import os
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt

        if not self.val_metrics_history:
            return

        # 提取数据
        epochs = [m['epoch'] for m in self.val_metrics_history]
        train_losses = [m['train_loss'] for m in self.val_metrics_history]
        val_losses = [m['val_loss'] for m in self.val_metrics_history]

        # 检查是否有评估指标
        has_eval_metrics = 'top1_overlap' in self.val_metrics_history[0]
        if has_eval_metrics:
            top1_overlaps = [m.get('top1_overlap', 0) for m in self.val_metrics_history]
            top3_overlaps = [m.get('top3_overlap', 0) for m in self.val_metrics_history]
            spearman_corrs = [m.get('spearman_corr', 0) for m in self.val_metrics_history]
            best_method_accs = [m.get('best_method_acc', 0) for m in self.val_metrics_history]
            f1_gaps = [m.get('f1_gap', 0) for m in self.val_metrics_history]
            overall_gains = [m.get('overall_gain', 0) for m in self.val_metrics_history]
            hard_regret_reductions = [m.get('hard_regret_reduction', 0) for m in self.val_metrics_history]

        # 创建图表
        if has_eval_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes = [[axes]]

        fig.suptitle('Training History', fontsize=16, fontweight='bold')

        # 1. Loss 曲线
        ax1 = axes[0][0] if has_eval_metrics else axes[0][0]
        ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 找出最佳 epoch
        best_epoch = epochs[val_losses.index(min(val_losses))]
        best_val_loss = min(val_losses)
        ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch+1}')
        ax1.text(best_epoch, best_val_loss, f' Best: {best_val_loss:.4f}',
                fontsize=9, verticalalignment='bottom')

        if has_eval_metrics:
            # 2. Top-k Overlap
            ax2 = axes[0][1]
            ax2.plot(epochs, top1_overlaps, 'g-o', label='Top-1 Overlap', linewidth=2)
            ax2.plot(epochs, top3_overlaps, 'b-s', label='Top-3 Overlap', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Overlap', fontsize=12)
            ax2.set_title('Top-k Overlap with Best Methods', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1.1])

            # 3. Spearman Correlation & Best Method Accuracy
            ax3 = axes[1][0]
            ax3_twin = ax3.twinx()

            line1 = ax3.plot(epochs, spearman_corrs, 'purple', marker='o',
                            label='Spearman Corr.', linewidth=2)
            line2 = ax3_twin.plot(epochs, best_method_accs, 'orange', marker='s',
                                label='Best Method Acc', linewidth=2)

            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Spearman Correlation', fontsize=12, color='purple')
            ax3_twin.set_ylabel('Best Method Accuracy', fontsize=12, color='orange')
            ax3.set_title('Correlation & Accuracy', fontsize=14, fontweight='bold')
            ax3.tick_params(axis='y', labelcolor='purple')
            ax3_twin.tick_params(axis='y', labelcolor='orange')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([-1.1, 1.1])
            ax3_twin.set_ylim([0, 1.1])

            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='best', fontsize=10)

            # 4. Router checkpoint metrics
            ax4 = axes[1][1]
            ax4.plot(epochs, f1_gaps, 'r-o', label='F1 Gap', linewidth=2)
            ax4.plot(epochs, overall_gains, 'g-s', label='Overall Gain', linewidth=2)
            ax4.plot(epochs, hard_regret_reductions, 'b-^', label='Hard Regret Reduction', linewidth=2)
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero')
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Metric Value', fontsize=12)
            ax4.set_title('Router Checkpoint Metrics', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        output_file = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  📊 Training curves saved to {output_file}")

    def train(self, start_epoch: int = 0):
        """训练循环"""
        import os
        import json

        print("\n" + "="*80)
        print("Starting pairwise training...")
        print("="*80)

        num_epochs = self.config.training.num_epochs if self.config else 10
        output_dir = self.config.training.output_dir if self.config else 'outputs/pairwise'
        
        # 初始化：如果需要分阶段训练，先冻结 backbone
        if self.freeze_epochs > 0:
            print(f"\n📌 Training strategy: Freeze backbone for {self.freeze_epochs} epochs, then unfreeze")
            self.freeze_backbone()
        
        # 温度调度信息
        if self.temperature_schedule:
            print(f"📌 Temperature schedule: {self.temperature_schedule}")
            print(f"   Initial temperature: {self.initial_temperature:.2f} → 1.0")
        
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*80}")
            
            # 分阶段训练：解冻 backbone
            if self.freeze_epochs > 0 and epoch == self.freeze_epochs:
                self.unfreeze_backbone()
                # 重新创建优化器（包含新解冻的参数）
                if not self.use_deepspeed:
                    print("  🔄 Recreating optimizer with unfrozen parameters...")
                    # 直接创建优化器
                    optimizer_name = self.config.training.optimizer.lower()
                    lr = self.config.training.learning_rate
                    weight_decay = self.config.training.weight_decay if hasattr(self.config.training, 'weight_decay') else 0.01
                    
                    if optimizer_name == 'adamw':
                        self.optimizer = torch.optim.AdamW(
                            self.model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay,
                            betas=(0.9, 0.999),
                            eps=1e-8
                        )
                    elif optimizer_name == 'adam':
                        self.optimizer = torch.optim.Adam(
                            self.model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay
                        )
                    else:
                        self.optimizer = torch.optim.SGD(
                            self.model.parameters(),
                            lr=lr,
                            momentum=0.9,
                            weight_decay=weight_decay
                        )
            
            # 温度退火
            self.update_temperature(epoch, num_epochs)
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            print(f"  Train loss: {train_metrics['loss']:.4f}")
            
            # 验证
            val_metrics = self.evaluate()
            print(f"  Val loss: {val_metrics['loss']:.4f}")
            
            # 评估模型预测 vs 真实方法F1分数
            if hasattr(self, 'tokenizer'):
                try:
                    # 从配置中读取 alpha，如果没有则使用默认值 0.5
                    alpha = self.config.training.f1_prior_alpha if hasattr(self.config.training, 'f1_prior_alpha') else 0.5

                    eval_metrics = self.evaluate_top3_coverage(
                        tokenizer=self.tokenizer,
                        max_length=self.config.model.max_length if self.config else 256,
                        split='val',
                        alpha=alpha
                    )
                    print(f"  📊 Top-1 Overlap: {eval_metrics['top1_overlap']:.3f} | "
                          f"Top-3 Overlap: {eval_metrics['top3_overlap']:.3f} | "
                          f"Spearman: {eval_metrics['spearman_corr']:.3f}")
                    print(f"  🏆 Best Method Acc: {eval_metrics['best_method_acc']:.3f} | "
                          f"Tie-Acc: {eval_metrics.get('best_method_acc_tie', 0.0):.3f} | "
                          f"Pred F1: {eval_metrics['predicted_best_f1']:.3f} | "
                          f"Gap: {eval_metrics['f1_gap']:.3f}")
                    print(
                        f"  🧭 Router vs {eval_metrics.get('strongest_single_method', 'N/A')}: "
                        f"overall_gain={eval_metrics.get('overall_gain', 0.0):.3f} | "
                        f"hard_gain={eval_metrics.get('hard_gain', 0.0):.3f} | "
                        f"hard_rr={eval_metrics.get('hard_regret_reduction', 0.0):.3f} | "
                        f"easy_regret={eval_metrics.get('easy_regret', 0.0):.3f}"
                    )
                    print(f"  🔍 F1 Prior Alpha: {alpha}")

                    # 合并到验证指标
                    val_metrics.update(eval_metrics)
                except Exception as e:
                    import traceback
                    print(f"  ⚠️  Failed to evaluate: {e}")
                    print(f"  Traceback: {traceback.format_exc()}")
            
            # 先计算这些指标是否刷新历史最佳，但不再单独保存对应模型
            is_best_loss = False
            current_loss = val_metrics['loss']
            if self.config.training.greater_is_better:
                current_metric = -current_loss
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    is_best_loss = True
            else:
                if current_loss < self.best_metric or self.best_metric == float('inf'):
                    self.best_metric = current_loss
                    is_best_loss = True

            is_best_top1 = False
            if 'top1_overlap' in val_metrics:
                current_top1 = val_metrics['top1_overlap']
                if current_top1 > self.best_top1_overlap:
                    self.best_top1_overlap = current_top1
                    is_best_top1 = True

            is_best_acc_tie = False
            if 'best_method_acc_tie' in val_metrics:
                current_acc_tie = val_metrics['best_method_acc_tie']
                if current_acc_tie > self.best_method_acc_tie:
                    self.best_method_acc_tie = current_acc_tie
                    is_best_acc_tie = True

            is_best_router = False
            if 'hard_regret_reduction' in val_metrics:
                router_key = self._build_router_checkpoint_key(val_metrics)
                if self.best_router_key is None or router_key > self.best_router_key:
                    self.best_router_key = router_key
                    is_best_router = True

            # 记录验证指标
            metrics_to_save = {
                'epoch': epoch,
                'val_loss': val_metrics['loss'],
                'train_loss': train_metrics['loss'],
                'is_best_loss': is_best_loss,
                'is_best_top1_overlap': is_best_top1,
                'is_best_method_acc_tie': is_best_acc_tie,
                'is_best_router': is_best_router,
            }
            # 添加评估指标（如果存在）
            if 'top1_overlap' in val_metrics:
                metrics_to_save.update({
                    'top1_overlap': val_metrics['top1_overlap'],
                    'top3_overlap': val_metrics['top3_overlap'],
                    'spearman_corr': val_metrics['spearman_corr'],
                    'kendall_tau': val_metrics.get('kendall_tau', 0.0),
                    'best_method_acc': val_metrics['best_method_acc'],
                    'best_method_acc_tie': val_metrics.get('best_method_acc_tie', 0.0),
                    'strongest_single_method': val_metrics.get('strongest_single_method', ''),
                    'strongest_single_avg_f1': val_metrics.get('strongest_single_avg_f1', 0.0),
                    'opportunity_ratio': val_metrics.get('opportunity_ratio', 0.0),
                    'overall_gain': val_metrics.get('overall_gain', 0.0),
                    'hard_gain': val_metrics.get('hard_gain', 0.0),
                    'hard_regret_reduction': val_metrics.get('hard_regret_reduction', 0.0),
                    'easy_regret': val_metrics.get('easy_regret', 0.0),
                    'predicted_best_f1': val_metrics['predicted_best_f1'],
                    'f1_gap': val_metrics['f1_gap']
                })
            
            self.val_metrics_history.append(metrics_to_save)

            # 保存训练历史到 JSON 文件（每个 epoch 都保存）
            history_path = os.path.join(output_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump({
                    'train_loss': self.train_loss_history,
                    'val_metrics': self.val_metrics_history
                }, f, indent=2)

            # 绘制训练曲线（每个 epoch 都绘制，会覆盖之前的）
            try:
                self._plot_training_curves(output_dir)
            except Exception as e:
                import traceback
                print(f"  ⚠️  Failed to plot training curves: {e}")
                print(f"  Traceback: {traceback.format_exc()}")

            # 检查是否需要保存模型（只在主进程保存）
            import torch.distributed as dist
            should_save = True
            if self.use_deepspeed and dist.is_initialized():
                should_save = (dist.get_rank() == 0)

            if not should_save:
                continue

            # 1. 可选保存 epoch checkpoint
            if self.save_epoch_checkpoints:
                epoch_model_path = os.path.join(output_dir, f'epoch_{epoch+1:03d}.pt')
                self.save_checkpoint(epoch_model_path, epoch, is_best=False)
                print(f"  💾 Epoch checkpoint saved to {epoch_model_path}")

            # 2. 总是保存 last_model.pt（每个 epoch 都覆盖）
            last_model_path = os.path.join(output_dir, 'last_model.pt')
            self.save_checkpoint(last_model_path, epoch, is_best=False)
            print(f"  💾 Last model saved to {last_model_path}")

            if is_best_loss:
                print(f"  🎉 New best val loss: {current_loss:.4f}")

            # 3. 仅记录最佳 top-1 overlap
            if is_best_top1:
                print(f"  🏆 New best top-1 overlap: {val_metrics['top1_overlap']:.4f}")
                    
            # 4. 仅记录最佳 tie-aware method acc
            if is_best_acc_tie:
                print(f"  🏅 New best tie-aware method acc: {val_metrics['best_method_acc_tie']:.4f}")

            # 5. 仅记录最佳 router checkpoint 指标
            if is_best_router:
                status = "eligible" if self._router_checkpoint_eligible(val_metrics) else "fallback"
                print(
                    f"  🧭 New best router checkpoint ({status}): "
                    f"hard_rr={val_metrics.get('hard_regret_reduction', 0.0):.4f}, "
                    f"hard_gain={val_metrics.get('hard_gain', 0.0):.4f}, "
                    f"overall_gain={val_metrics.get('overall_gain', 0.0):.4f}, "
                    f"easy_regret={val_metrics.get('easy_regret', 0.0):.4f}"
                )

        # 训练结束
        print("\n" + "="*80)
        print("✨ Training completed!")
        print("="*80)
        print(f"  Best val loss: {self.best_metric:.4f}")
        print(f"  Best top-1 overlap: {self.best_top1_overlap:.4f}")
        print(f"  Best tie-aware method acc: {self.best_method_acc_tie:.4f}")
        if self.best_router_key is not None:
            print(f"  Best router key: {self.best_router_key}")
        print(f"  Total steps: {self.global_step}")
        print(f"  Output dir: {output_dir}")
        print(f"\n📁 Saved models:")
        print(f"  • Last model: {os.path.join(output_dir, 'last_model.pt')}")
        if self.save_epoch_checkpoints:
            print(f"  • Per-epoch checkpoints: {os.path.join(output_dir, 'epoch_XXX.pt')}")
        print(f"\n📊 Other files:")
        print(f"  • Training history: {os.path.join(output_dir, 'training_history.json')}")
        print(f"  • Training curves: {os.path.join(output_dir, 'training_curves.png')}")
        print(f"\n🎯 Next steps:")
        print(f"  Evaluate the model:")
        print(f"    python evaluate_pairwise.py --model {os.path.join(output_dir, 'last_model.pt')} --data dataset/test.csv")

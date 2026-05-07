#!/usr/bin/env python3
"""
成对学习训练脚本

特性：
1. 支持普通训练 / DeepSpeed
2. 支持是否启用 sampler
3. 支持从 checkpoint 恢复训练
4. 与当前 pairwise trainer / dataset / model 设计保持一致
"""

import argparse
import json
import os
import shutil
import torch

from model.cross_encoder import MethodSelectionCrossEncoder, create_tokenizer
from model.pairwise_dataset import create_pairwise_dataloaders
from model.pairwise_trainer import PairwiseTrainer
from model.config import Config

# DeepSpeed 支持
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("⚠️  DeepSpeed not available, will use standard training")


def create_optimizer(model, config):
    """创建优化器"""
    optimizer_name = config.training.optimizer.lower()
    lr = config.training.learning_rate
    weight_decay = config.training.weight_decay if hasattr(config.training, 'weight_decay') else 0.01

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, num_training_steps, config):
    """创建学习率调度器"""
    scheduler_type = config.training.scheduler.lower() if hasattr(config.training, 'scheduler') else 'linear'
    warmup_ratio = config.training.warmup_ratio if hasattr(config.training, 'warmup_ratio') else 0.1
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    if scheduler_type == 'linear':
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        scheduler = LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        t_max = max(1, num_training_steps - num_warmup_steps)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    elif scheduler_type == 'constant':
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)

    else:
        scheduler = None

    return scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Train pairwise method selection model')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/qwen_pairwise_pairweight_soft_pointwise_margin.yaml',
        help='Path to config file'
    )

    parser.add_argument(
        '--train_file',
        type=str,
        default='dataset/pairwise/train_pairwise.csv',
        help='Path to training pairwise CSV'
    )

    parser.add_argument(
        '--val_file',
        type=str,
        default='dataset/pairwise/val_pairwise.csv',
        help='Path to validation pairwise CSV'
    )

    parser.add_argument(
        '--test_file',
        type=str,
        default='dataset/pairwise/test_pairwise.csv',
        help='Path to test pairwise CSV'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )

    parser.add_argument(
        '--weight_type',
        type=str,
        default='sqrt_inverse',
        choices=['inverse', 'sqrt_inverse', 'log_inverse', 'none'],
        help='Type of class weighting'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=None,
        help='Override total training epochs. When used with --resume_from, this means target total epochs.'
    )

    parser.add_argument(
        '--additional_epochs',
        type=int,
        default=None,
        help='When used with --resume_from, continue training for this many more epochs from the checkpoint epoch.'
    )

    parser.add_argument(
        '--deepspeed',
        type=str,
        default=None,
        help='DeepSpeed config file (enables DeepSpeed training)'
    )

    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training (auto-set by DeepSpeed)'
    )

    parser.add_argument(
        '--use_sampler',
        action='store_true',
        help='Use pairwise batch sampler instead of standard shuffle DataLoader'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = None
    start_epoch = 0

    print("=" * 80)
    print("Pairwise Method Selection Training")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Train file: {args.train_file}")
    print(f"Val file: {args.val_file}")
    print(f"Test file: {args.test_file}")
    print(f"Weight type: {args.weight_type}")
    print(f"Use sampler: {args.use_sampler}")
    print(f"Device: {args.device}")

    # 加载配置
    print(f"\nLoading config...")
    config = Config.from_yaml(args.config)

    if getattr(config.model, 'listwise_weight', 0.0) > 0.0 and not args.use_sampler:
        args.use_sampler = True
        print("  ✓ Auto-enabled --use_sampler because listwise loss needs grouped question batches")

    # 如果使用 DeepSpeed，禁用 DDP
    if args.deepspeed:
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError("DeepSpeed requested but not available. Install with: pip install deepspeed")
        config.training.use_ddp = False
        print("  ✓ DeepSpeed enabled, DDP disabled")

    # 覆盖输出目录
    if args.output_dir:
        config.training.output_dir = args.output_dir

    print(f"  Output dir: {config.training.output_dir}")
    print(f"  Model: {config.model.model_name_or_path}")
    print(f"  Max length: {config.model.max_length}")
    print(f"  Batch size (questions): {config.training.batch_size_q}")

    # 创建输出目录
    os.makedirs(config.training.output_dir, exist_ok=True)

    # 创建 tokenizer
    print(f"\nCreating tokenizer...")
    tokenizer = create_tokenizer(
        config.model.model_name_or_path,
        max_length=config.model.max_length
    )

    # 检测是否使用分布式训练
    is_distributed = args.deepspeed is not None
    if is_distributed:
        import torch.distributed as dist
        if not dist.is_initialized():
            print("\n⚠️  Warning: DeepSpeed should initialize distributed environment automatically")

    # 创建数据加载器
    print(f"\nCreating dataloaders...")
    train_loader, val_loader, test_loader, class_weights = create_pairwise_dataloaders(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        tokenizer=tokenizer,
        batch_size_q=config.training.batch_size_q,
        max_length=config.model.max_length,
        num_workers=config.training.num_workers,
        use_sampler=args.use_sampler,
        weight_type=args.weight_type,
        is_distributed=is_distributed
    )

    if args.weight_type == 'none':
        class_weights = None
        print("\n⚠️  Class weighting disabled")

    # 创建模型
    print(f"\nCreating model...")
    temperature = config.model.temperature if hasattr(config.model, 'temperature') else 1.5
    learnable_temp = config.model.learnable_temperature if hasattr(config.model, 'learnable_temperature') else False

    model = MethodSelectionCrossEncoder(
        model_name_or_path=config.model.model_name_or_path,
        num_methods=config.model.num_methods,
        method_vocab=getattr(config.model, 'method_vocab', None),
        use_method_embedding=getattr(config.model, 'use_method_embedding', False),
        method_embedding_scale=getattr(config.model, 'method_embedding_scale', 1.0),
        method_embedding_init_std=getattr(config.model, 'method_embedding_init_std', 0.02),
        pooling_mode=config.model.pooling_mode,
        dropout=config.model.dropout,
        use_gradient_checkpointing=config.model.use_gradient_checkpointing,
        temperature=temperature,
        learnable_temperature=learnable_temp,
        margin=config.model.margin,
        loss_type=config.model.loss_type,
        pairwise_weight=config.model.pairwise_weight,
        pairwise_margin_mode=getattr(config.model, 'pairwise_margin_mode', 'legacy'),
        dynamic_margin_enabled=getattr(config.model, 'dynamic_margin_enabled', False),
        dynamic_margin_scale=getattr(config.model, 'dynamic_margin_scale', 0.0),
        dynamic_margin_power=getattr(config.model, 'dynamic_margin_power', 1.0),
        dynamic_margin_max=getattr(config.model, 'dynamic_margin_max', None),
        pointwise_weight=getattr(config.model, 'pointwise_weight', 0.0),
        pointwise_loss_type=getattr(config.model, 'pointwise_loss_type', 'huber'),
        pointwise_huber_beta=getattr(config.model, 'pointwise_huber_beta', 0.1),
        pointwise_apply_sigmoid=getattr(config.model, 'pointwise_apply_sigmoid', True),
        listwise_weight=getattr(config.model, 'listwise_weight', 0.0),
        listwise_loss_type=getattr(config.model, 'listwise_loss_type', 'kl'),
        listwise_target_temperature=getattr(config.model, 'listwise_target_temperature', 1.0),
        listwise_min_methods=getattr(config.model, 'listwise_min_methods', 3),
    )

    # 恢复模型参数
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f"  Resuming from epoch {start_epoch}")

    if args.additional_epochs is not None and not args.resume_from:
        raise ValueError("--additional_epochs requires --resume_from")

    if args.resume_from and args.additional_epochs is not None:
        if args.additional_epochs <= 0:
            raise ValueError("--additional_epochs must be > 0")
        config.training.num_epochs = start_epoch + args.additional_epochs
        print(
            f"  Continue training for {args.additional_epochs} more epochs "
            f"(target total epochs: {config.training.num_epochs})"
        )
    elif args.num_epochs is not None:
        if args.num_epochs <= 0:
            raise ValueError("--num_epochs must be > 0")
        config.training.num_epochs = args.num_epochs
        if args.resume_from and config.training.num_epochs <= start_epoch:
            raise ValueError(
                f"--num_epochs={config.training.num_epochs} is not greater than resumed start_epoch={start_epoch}. "
                "Use a larger --num_epochs or pass --additional_epochs."
            )
        print(f"  Override total epochs to {config.training.num_epochs}")

    print(f"  Num epochs (target total): {config.training.num_epochs}")

    # 创建优化器
    print(f"\nCreating optimizer...")
    optimizer = create_optimizer(model, config)

    # DeepSpeed 初始化
    if args.deepspeed:
        print(f"\nInitializing DeepSpeed...")
        print(f"  DeepSpeed config: {args.deepspeed}")

        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=args.deepspeed
        )

        scheduler = None  # 交给 DeepSpeed / config 管理
        print("  ✓ DeepSpeed initialized")
        print("  ✓ ZeRO optimization enabled")

        device = f'cuda:{model.local_rank}'
    else:
        model.to(args.device)
        device = args.device

        print("Creating scheduler...")
        num_training_steps = len(train_loader) * config.training.num_epochs
        scheduler = create_scheduler(optimizer, num_training_steps, config)

    # 恢复优化器 / 调度器
    if args.resume_from and checkpoint is not None and not args.deepspeed:
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  ✓ Optimizer state restored")

        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  ✓ Scheduler state restored")

    # 创建训练器
    print(f"\nCreating trainer...")
    trainer = PairwiseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        class_weights=class_weights,
        device=device,
        use_amp=config.training.bf16 or config.training.fp16
    )

    # tokenizer 给 trainer 做 question-level 评估
    trainer.tokenizer = tokenizer
    print("  ✓ Method comparison evaluation enabled")
    print("    Will compare model predictions with true method F1 scores")

    # 恢复 trainer 状态
    if args.resume_from and checkpoint is not None:
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.best_metric = checkpoint.get(
            'best_metric',
            -float('inf') if config.training.greater_is_better else float('inf')
        )
        trainer.best_top1_overlap = checkpoint.get('best_top1_overlap', trainer.best_top1_overlap)
        trainer.best_method_acc_tie = checkpoint.get('best_method_acc_tie', trainer.best_method_acc_tie)
        trainer.best_router_key = checkpoint.get('best_router_key', trainer.best_router_key)
        print("  ✓ Trainer state restored")
        print(f"    Global step: {trainer.global_step}")
        print(f"    Best metric: {trainer.best_metric}")

        history_path = os.path.join(config.training.output_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            trainer.train_loss_history = history.get('train_loss', [])
            trainer.val_metrics_history = history.get('val_metrics', [])
            print(
                f"  ✓ Training history restored"
                f" (train points: {len(trainer.train_loss_history)},"
                f" val epochs: {len(trainer.val_metrics_history)})"
            )

            backup_dir = os.path.join(config.training.output_dir, 'resume_backups')
            os.makedirs(backup_dir, exist_ok=True)
            backup_tag = f"epoch{start_epoch}"

            history_backup = os.path.join(backup_dir, f"training_history.before_resume_{backup_tag}.json")
            if not os.path.exists(history_backup):
                shutil.copy2(history_path, history_backup)
                print(f"  ✓ Backed up history to {history_backup}")

            curves_path = os.path.join(config.training.output_dir, 'training_curves.png')
            curves_backup = os.path.join(backup_dir, f"training_curves.before_resume_{backup_tag}.png")
            if os.path.exists(curves_path) and not os.path.exists(curves_backup):
                shutil.copy2(curves_path, curves_backup)
                print(f"  ✓ Backed up curves to {curves_backup}")

    # 开始训练
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    # 需要 trainer.train(start_epoch=...)
    trainer.train(start_epoch=start_epoch)

    print("\n" + "=" * 80)
    print("✨ Training completed successfully!")
    print("=" * 80)
    print(f"\n📁 Outputs saved to: {config.training.output_dir}")
    print(f"\n🎯 Next steps:")
    print(f"  1. Evaluate the model:")
    print(f"     python evaluate_pairwise.py --model {config.training.output_dir}/last_model.pt --data dataset/test.csv")
    if getattr(config.training, 'save_epoch_checkpoints', True):
        print(f"     # or choose one of {config.training.output_dir}/epoch_XXX.pt")
    print(f"  2. Compare with baseline:")
    print(f"     python compare_models.py")


if __name__ == '__main__':
    main()

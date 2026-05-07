import os
from typing import Optional, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig


class MethodSelectionCrossEncoder(nn.Module):
    """
    方法选择 Cross-Encoder（统一 Pairwise 架构）

    主路径：
    1. 训练:
        model.forward_pairwise(
            input_ids_a, attention_mask_a,
            input_ids_b, attention_mask_b,
            label_ab, sample_weights=None
        )

    2. 推断:
        model.score_single(input_ids, attention_mask)

    说明：
    - label_ab 使用 0/1
      - 1: A 优于 B
      - 0: B 优于 A
    - 返回原始 logits；温度只在 pairwise loss 中使用一次
    - 支持 sample-wise class weighting
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_methods: int = 6,
        method_vocab: Optional[List[str]] = None,
        use_method_embedding: bool = False,
        method_embedding_scale: float = 1.0,
        method_embedding_init_std: float = 0.02,
        pooling_mode: str = "mean",
        dropout: float = 0.2,
        use_gradient_checkpointing: bool = False,
        temperature: float = 1.5,
        learnable_temperature: bool = False,
        margin: float = 0.0,
        loss_type: str = "pairwise",
        pairwise_weight: float = 1.0,
        pairwise_margin_mode: str = "legacy",
        dynamic_margin_enabled: bool = False,
        dynamic_margin_scale: float = 0.0,
        dynamic_margin_power: float = 1.0,
        dynamic_margin_max: Optional[float] = None,
        pointwise_weight: float = 0.0,
        pointwise_loss_type: str = "huber",
        pointwise_huber_beta: float = 0.1,
        pointwise_apply_sigmoid: bool = True,
        listwise_weight: float = 0.0,
        listwise_loss_type: str = "kl",
        listwise_target_temperature: float = 1.0,
        listwise_min_methods: int = 3,
    ):
        super().__init__()

        self.num_methods = num_methods
        self.method_vocab = list(method_vocab) if method_vocab is not None else None
        self.use_method_embedding = bool(use_method_embedding)
        self.method_embedding_scale = float(method_embedding_scale)
        self.method_embedding_init_std = float(method_embedding_init_std)
        self.pooling_mode = pooling_mode
        self.margin = float(margin)
        self.loss_type = loss_type
        self.pairwise_weight = float(pairwise_weight)
        self.pairwise_margin_mode = str(pairwise_margin_mode)
        self.dynamic_margin_enabled = bool(dynamic_margin_enabled)
        self.dynamic_margin_scale = float(dynamic_margin_scale)
        self.dynamic_margin_power = float(dynamic_margin_power)
        self.dynamic_margin_max = (
            None if dynamic_margin_max is None else float(dynamic_margin_max)
        )
        self.pointwise_weight = float(pointwise_weight)
        self.pointwise_loss_type = str(pointwise_loss_type)
        self.pointwise_huber_beta = float(pointwise_huber_beta)
        self.pointwise_apply_sigmoid = bool(pointwise_apply_sigmoid)
        self.listwise_weight = float(listwise_weight)
        self.listwise_loss_type = str(listwise_loss_type)
        self.listwise_target_temperature = float(listwise_target_temperature)
        self.listwise_min_methods = max(2, int(listwise_min_methods))
        self.learnable_temperature = bool(learnable_temperature)

        # 温度参数
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(float(temperature), dtype=torch.float))
            print(f"✓ Learnable temperature initialized to {temperature}")
        else:
            self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float))
            print(f"✓ Fixed temperature set to {temperature}")

        print(f"✓ Loss type: {loss_type}")

        # 加载 backbone
        print(f"Loading model from {model_name_or_path}...")
        self.config = AutoConfig.from_pretrained(model_name_or_path)

        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config,
            trust_remote_code=True,
        )

        if use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")

        # 训练时不需要 cache
        self.config.use_cache = False
        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False

        # hidden size
        self.hidden_size = self.config.hidden_size

        # scorer
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(self.hidden_size, 1)

        nn.init.xavier_uniform_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)

        self.method_to_idx = None
        self.method_embedding = None
        if self.use_method_embedding:
            if not self.method_vocab:
                raise ValueError("use_method_embedding=True requires method_vocab")
            self.method_to_idx = {method: idx for idx, method in enumerate(self.method_vocab)}
            self.method_embedding = nn.Embedding(len(self.method_vocab), self.hidden_size)
            nn.init.normal_(self.method_embedding.weight, mean=0.0, std=self.method_embedding_init_std)

        print("Model initialized:")
        print(f"  Backbone: {model_name_or_path}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Pooling mode: {self.pooling_mode}")
        print(f"  Num methods: {self.num_methods}")
        print(f"  Dropout: {self.dropout.p}")
        print(f"  Margin: {self.margin}")
        print(f"  Pairwise margin mode: {self.pairwise_margin_mode}")
        print(f"  Dynamic margin enabled: {self.dynamic_margin_enabled}")
        if self.dynamic_margin_enabled:
            print(f"  Dynamic margin scale: {self.dynamic_margin_scale}")
            print(f"  Dynamic margin power: {self.dynamic_margin_power}")
            print(f"  Dynamic margin max: {self.dynamic_margin_max}")
        print(f"  Temperature: {float(self.temperature.detach().cpu().item()) if isinstance(self.temperature, torch.Tensor) else self.temperature}")
        print(f"  Use method embedding: {self.use_method_embedding}")
        print(f"  Pointwise weight: {self.pointwise_weight}")
        if self.pointwise_weight > 0.0:
            print(f"  Pointwise loss: {self.pointwise_loss_type}")
            print(f"  Pointwise huber beta: {self.pointwise_huber_beta}")
            print(f"  Pointwise apply sigmoid: {self.pointwise_apply_sigmoid}")
        print(f"  Listwise weight: {self.listwise_weight}")
        if self.listwise_weight > 0.0:
            print(f"  Listwise loss: {self.listwise_loss_type}")
            print(f"  Listwise target temperature: {self.listwise_target_temperature}")
            print(f"  Listwise min methods: {self.listwise_min_methods}")
        if self.use_method_embedding:
            print(f"  Method embedding scale: {self.method_embedding_scale}")
            print(f"  Method vocab: {self.method_vocab}")

    def _compute_pointwise_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        preds = torch.sigmoid(logits) if self.pointwise_apply_sigmoid else logits
        targets = targets.to(preds.dtype).clamp(0.0, 1.0)

        if self.pointwise_loss_type == "mse":
            return F.mse_loss(preds, targets, reduction="none")
        if self.pointwise_loss_type == "l1":
            return F.l1_loss(preds, targets, reduction="none")
        if self.pointwise_loss_type == "huber":
            return F.smooth_l1_loss(
                preds,
                targets,
                beta=self.pointwise_huber_beta,
                reduction="none",
            )
        raise ValueError(
            f"Unsupported pointwise_loss_type={self.pointwise_loss_type}. "
            "Expected one of: 'huber', 'mse', 'l1'."
        )

    def _compute_effective_margin(
        self,
        base_dtype: torch.dtype,
        base_device: torch.device,
        target_score_a: Optional[torch.Tensor] = None,
        target_score_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        margin = torch.as_tensor(self.margin, dtype=base_dtype, device=base_device)

        if not self.dynamic_margin_enabled:
            return margin

        if target_score_a is None or target_score_b is None:
            return margin

        gap = torch.abs(
            target_score_a.to(device=base_device, dtype=base_dtype)
            - target_score_b.to(device=base_device, dtype=base_dtype)
        )
        dynamic_term = self.dynamic_margin_scale * torch.pow(
            gap.clamp_min(0.0),
            self.dynamic_margin_power,
        )
        effective_margin = margin + dynamic_term

        if self.dynamic_margin_max is not None:
            effective_margin = torch.clamp(effective_margin, max=self.dynamic_margin_max)

        return effective_margin

    def _compute_listwise_loss(
        self,
        logit_a: torch.Tensor,
        logit_b: torch.Tensor,
        qids: Optional[Union[torch.Tensor, List[int], tuple]],
        method_id_a: Optional[Union[List[Union[str, int]], tuple]],
        method_id_b: Optional[Union[List[Union[str, int]], tuple]],
        target_score_a: Optional[torch.Tensor],
        target_score_b: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.listwise_weight <= 0.0:
            return None
        if qids is None or method_id_a is None or method_id_b is None:
            return None
        if target_score_a is None or target_score_b is None:
            return None

        if isinstance(qids, torch.Tensor):
            qid_list = qids.detach().cpu().tolist()
        else:
            qid_list = list(qids)

        target_score_a = target_score_a.to(device=logit_a.device, dtype=logit_a.dtype)
        target_score_b = target_score_b.to(device=logit_b.device, dtype=logit_b.dtype)
        target_temp = max(self.listwise_target_temperature, 1e-6)

        grouped = {}

        def add_entry(qid, method_id, pred_logit, target_score):
            bucket = grouped.setdefault(qid, {})
            method_bucket = bucket.setdefault(method_id, {"logits": [], "targets": []})
            method_bucket["logits"].append(pred_logit)
            method_bucket["targets"].append(target_score)

        for idx, qid in enumerate(qid_list):
            add_entry(qid, method_id_a[idx], logit_a[idx], target_score_a[idx])
            add_entry(qid, method_id_b[idx], logit_b[idx], target_score_b[idx])

        group_losses = []
        for methods in grouped.values():
            if len(methods) < self.listwise_min_methods:
                continue

            pred_logits = []
            target_scores = []
            for stats in methods.values():
                pred_logits.append(torch.stack(stats["logits"]).mean().float())
                target_scores.append(torch.stack(stats["targets"]).mean().float())

            pred_logits = torch.stack(pred_logits)
            target_scores = torch.stack(target_scores)
            target_probs = torch.softmax(target_scores / target_temp, dim=0)

            if not torch.isfinite(pred_logits).all() or not torch.isfinite(target_probs).all():
                continue

            if self.listwise_loss_type == "kl":
                log_probs = F.log_softmax(pred_logits, dim=0)
                loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
            elif self.listwise_loss_type == "ce":
                log_probs = F.log_softmax(pred_logits, dim=0)
                loss = -(target_probs * log_probs).sum()
            else:
                raise ValueError(
                    f"Unsupported listwise_loss_type={self.listwise_loss_type}. "
                    "Expected one of: 'kl', 'ce'."
                )

            group_losses.append(loss)

        if not group_losses:
            return None

        return torch.stack(group_losses).mean()

    def pooling(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        从序列表示中提取句子表示

        支持:
        - mean
        - last
        - max
        - cls
        """
        if self.pooling_mode == "cls":
            return last_hidden_state[:, 0, :]

        if self.pooling_mode == "mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, S, 1]
            summed = torch.sum(last_hidden_state * mask, dim=1)              # [B, H]
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)                   # [B, 1]
            return summed / denom

        if self.pooling_mode == "max":
            masked = last_hidden_state.masked_fill(
                attention_mask.unsqueeze(-1) == 0,
                torch.finfo(last_hidden_state.dtype).min
            )
            return torch.max(masked, dim=1).values

        if self.pooling_mode == "last":
            lengths = attention_mask.sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
            return last_hidden_state[batch_idx, lengths, :]

        raise ValueError(
            f"Unknown pooling mode: {self.pooling_mode}. "
            f"Supported modes: 'mean', 'last', 'max', 'cls'."
        )

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码输入并返回 pooled representation
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state
        sentence_embedding = self.pooling(last_hidden_state, attention_mask)
        return sentence_embedding

    def _normalize_method_ids(
        self,
        method_ids: Optional[Union[torch.Tensor, List[Union[str, int]], tuple]],
        device: torch.device
    ) -> Optional[torch.Tensor]:
        if method_ids is None:
            return None

        if isinstance(method_ids, torch.Tensor):
            return method_ids.to(device=device, dtype=torch.long)

        normalized = []
        for method_id in method_ids:
            if isinstance(method_id, str):
                if self.method_to_idx is None or method_id not in self.method_to_idx:
                    raise KeyError(f"Unknown method id for method embedding: {method_id}")
                normalized.append(self.method_to_idx[method_id])
            else:
                normalized.append(int(method_id))

        return torch.tensor(normalized, dtype=torch.long, device=device)

    def score_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        method_ids: Optional[Union[torch.Tensor, List[Union[str, int]], tuple]] = None,
    ) -> torch.Tensor:
        """
        单样本打分接口（推断/评估/内部训练）

        Returns:
            logits: [batch_size]
        """
        sentence_embedding = self._encode(input_ids, attention_mask)

        if self.use_method_embedding:
            method_idx = self._normalize_method_ids(method_ids, sentence_embedding.device)
            if method_idx is None:
                raise ValueError("method_ids must be provided when use_method_embedding=True")
            sentence_embedding = sentence_embedding + self.method_embedding_scale * self.method_embedding(method_idx)

        sentence_embedding = self.dropout(sentence_embedding)
        if self.scorer.weight.dtype != sentence_embedding.dtype:
            sentence_embedding = sentence_embedding.to(self.scorer.weight.dtype)
        logits = self.scorer(sentence_embedding).squeeze(-1)
        return logits

    def forward_pairwise(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
        label_ab: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Pairwise 训练接口
    
        关键点：
        - A/B 不再分别调用两次 backbone
        - 而是在 batch 维拼接后，一次性过 backbone
        - 这样更兼容 DeepSpeed ZeRO-2 + gradient checkpointing
        """
        batch_size = input_ids_a.size(0)
        method_idx = None
        if self.use_method_embedding:
            method_id_a = kwargs.get("method_id_a")
            method_id_b = kwargs.get("method_id_b")
            method_idx_a = self._normalize_method_ids(method_id_a, input_ids_a.device)
            method_idx_b = self._normalize_method_ids(method_id_b, input_ids_a.device)
            if method_idx_a is None or method_idx_b is None:
                raise ValueError("method_id_a/method_id_b are required when use_method_embedding=True")
            method_idx = torch.cat([method_idx_a, method_idx_b], dim=0)
    
        # [2B, L]
        input_ids = torch.cat([input_ids_a, input_ids_b], dim=0)
        attention_mask = torch.cat([attention_mask_a, attention_mask_b], dim=0)
    
        # 一次 backbone 编码
        sentence_embedding = self._encode(input_ids, attention_mask)
        if self.use_method_embedding:
            sentence_embedding = sentence_embedding + self.method_embedding_scale * self.method_embedding(method_idx)
        sentence_embedding = self.dropout(sentence_embedding)
        if self.scorer.weight.dtype != sentence_embedding.dtype:
            sentence_embedding = sentence_embedding.to(self.scorer.weight.dtype)
        logits = self.scorer(sentence_embedding).squeeze(-1)  # [2B]
    
        # 拆回 A / B
        logit_a = logits[:batch_size]
        logit_b = logits[batch_size:]
    
        result = {
            "score_a": logit_a,
            "score_b": logit_b,
        }

        pairwise_loss = None
        target_score_a = kwargs.get("target_score_a")
        target_score_b = kwargs.get("target_score_b")
        if label_ab is not None:
            label_ab = label_ab.to(logit_a.dtype)
            y = label_ab * 2.0 - 1.0  # 0 -> -1, 1 -> +1
    
            temp = torch.clamp(self.temperature, min=1e-3, max=10.0)
            margin = self._compute_effective_margin(
                base_dtype=logit_a.dtype,
                base_device=logit_a.device,
                target_score_a=target_score_a,
                target_score_b=target_score_b,
            )

            if self.pairwise_margin_mode == "legacy":
                diff = (logit_a - logit_b - margin) / temp
                diff = torch.clamp(diff, min=-20.0, max=20.0)
                loss_per_sample = torch.nn.functional.softplus(-y * diff)
            elif self.pairwise_margin_mode == "symmetric":
                signed_diff = y * (logit_a - logit_b)
                loss_input = (margin - signed_diff) / temp
                loss_input = torch.clamp(loss_input, min=-20.0, max=20.0)
                loss_per_sample = torch.nn.functional.softplus(loss_input)
            else:
                raise ValueError(
                    f"Unsupported pairwise_margin_mode={self.pairwise_margin_mode}. "
                    "Expected one of: 'legacy', 'symmetric'."
                )
    
            if sample_weights is not None:
                sample_weights = sample_weights.to(loss_per_sample.dtype)
                pairwise_loss = (loss_per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
            else:
                pairwise_loss = loss_per_sample.mean()

            result["pairwise_loss"] = pairwise_loss

        pointwise_loss = None
        if self.pointwise_weight > 0.0 and target_score_a is not None and target_score_b is not None:
            target_score_a = target_score_a.to(logit_a.device, dtype=logit_a.dtype)
            target_score_b = target_score_b.to(logit_b.device, dtype=logit_b.dtype)

            pointwise_loss_a = self._compute_pointwise_loss(logit_a, target_score_a)
            pointwise_loss_b = self._compute_pointwise_loss(logit_b, target_score_b)
            pointwise_loss_per_pair = 0.5 * (pointwise_loss_a + pointwise_loss_b)

            if sample_weights is not None:
                sample_weights = sample_weights.to(pointwise_loss_per_pair.dtype)
                pointwise_loss = (
                    (pointwise_loss_per_pair * sample_weights).sum()
                    / sample_weights.sum().clamp_min(1e-8)
                )
            else:
                pointwise_loss = pointwise_loss_per_pair.mean()

            result["pointwise_loss"] = pointwise_loss

        listwise_loss = self._compute_listwise_loss(
            logit_a=logit_a,
            logit_b=logit_b,
            qids=kwargs.get("qids"),
            method_id_a=kwargs.get("method_id_a"),
            method_id_b=kwargs.get("method_id_b"),
            target_score_a=target_score_a,
            target_score_b=target_score_b,
        )
        if listwise_loss is not None:
            result["listwise_loss"] = listwise_loss

        weighted_terms = []
        if pairwise_loss is not None:
            weighted_terms.append((self.pairwise_weight, pairwise_loss))
        if pointwise_loss is not None:
            weighted_terms.append((self.pointwise_weight, pointwise_loss))
        if listwise_loss is not None:
            weighted_terms.append((self.listwise_weight, listwise_loss))

        if len(weighted_terms) > 1:
            total_weight = sum(weight for weight, _ in weighted_terms)
            if total_weight <= 0.0:
                raise ValueError("Total enabled loss weight must be > 0")
            result["loss"] = sum(weight * term for weight, term in weighted_terms) / total_weight
        elif pairwise_loss is not None:
            result["loss"] = pairwise_loss
        elif pointwise_loss is not None:
            result["loss"] = pointwise_loss
        elif listwise_loss is not None:
            result["loss"] = listwise_loss
    
        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        method_ids: Optional[Union[torch.Tensor, List[Union[str, int]], tuple]] = None,
    ) -> torch.Tensor:
        """
        推断接口，返回原始 logits
        """
        with torch.no_grad():
            return self.score_single(input_ids, attention_mask, method_ids=method_ids)

    def save_pretrained(self, save_directory: str):
        """
        保存模型
        """
        os.makedirs(save_directory, exist_ok=True)

        # 保存 backbone
        self.backbone.save_pretrained(os.path.join(save_directory, "backbone"))

        # 保存 scorer 与配置
        scorer_state = {
            "scorer": self.scorer.state_dict(),
            "num_methods": self.num_methods,
            "method_vocab": self.method_vocab,
            "use_method_embedding": self.use_method_embedding,
            "method_embedding_scale": self.method_embedding_scale,
            "method_embedding_init_std": self.method_embedding_init_std,
            "pooling_mode": self.pooling_mode,
            "hidden_size": self.hidden_size,
            "margin": self.margin,
            "loss_type": self.loss_type,
            "pairwise_weight": self.pairwise_weight,
            "pairwise_margin_mode": self.pairwise_margin_mode,
            "dynamic_margin_enabled": self.dynamic_margin_enabled,
            "dynamic_margin_scale": self.dynamic_margin_scale,
            "dynamic_margin_power": self.dynamic_margin_power,
            "dynamic_margin_max": self.dynamic_margin_max,
            "pointwise_weight": self.pointwise_weight,
            "pointwise_loss_type": self.pointwise_loss_type,
            "pointwise_huber_beta": self.pointwise_huber_beta,
            "pointwise_apply_sigmoid": self.pointwise_apply_sigmoid,
            "listwise_weight": self.listwise_weight,
            "listwise_loss_type": self.listwise_loss_type,
            "listwise_target_temperature": self.listwise_target_temperature,
            "listwise_min_methods": self.listwise_min_methods,
            "temperature": float(self.temperature.detach().cpu().item())
            if isinstance(self.temperature, torch.Tensor)
            else float(self.temperature),
            "learnable_temperature": self.learnable_temperature,
            "dropout_p": self.dropout.p,
        }
        if self.method_embedding is not None:
            scorer_state["method_embedding"] = self.method_embedding.state_dict()
        torch.save(scorer_state, os.path.join(save_directory, "scorer.pt"))

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        model_name_or_path: Optional[str] = None,
    ):
        """
        加载模型
        """
        scorer_state = torch.load(
            os.path.join(load_directory, "scorer.pt"),
            map_location="cpu",
        )

        if model_name_or_path is None:
            model_name_or_path = os.path.join(load_directory, "backbone")

        model = cls(
            model_name_or_path=model_name_or_path,
            num_methods=scorer_state["num_methods"],
            method_vocab=scorer_state.get("method_vocab"),
            use_method_embedding=scorer_state.get("use_method_embedding", False),
            method_embedding_scale=scorer_state.get("method_embedding_scale", 1.0),
            method_embedding_init_std=scorer_state.get("method_embedding_init_std", 0.02),
            pooling_mode=scorer_state["pooling_mode"],
            dropout=scorer_state.get("dropout_p", 0.2),
            temperature=scorer_state.get("temperature", 1.5),
            learnable_temperature=scorer_state.get("learnable_temperature", False),
            margin=scorer_state.get("margin", 0.0),
            loss_type=scorer_state.get("loss_type", "pairwise"),
            pairwise_weight=scorer_state.get("pairwise_weight", 1.0),
            pairwise_margin_mode=scorer_state.get("pairwise_margin_mode", "legacy"),
            dynamic_margin_enabled=scorer_state.get("dynamic_margin_enabled", False),
            dynamic_margin_scale=scorer_state.get("dynamic_margin_scale", 0.0),
            dynamic_margin_power=scorer_state.get("dynamic_margin_power", 1.0),
            dynamic_margin_max=scorer_state.get("dynamic_margin_max"),
            pointwise_weight=scorer_state.get("pointwise_weight", 0.0),
            pointwise_loss_type=scorer_state.get("pointwise_loss_type", "huber"),
            pointwise_huber_beta=scorer_state.get("pointwise_huber_beta", 0.1),
            pointwise_apply_sigmoid=scorer_state.get("pointwise_apply_sigmoid", True),
            listwise_weight=scorer_state.get("listwise_weight", 0.0),
            listwise_loss_type=scorer_state.get("listwise_loss_type", "kl"),
            listwise_target_temperature=scorer_state.get("listwise_target_temperature", 1.0),
            listwise_min_methods=scorer_state.get("listwise_min_methods", 3),
        )

        model.scorer.load_state_dict(scorer_state["scorer"])
        if model.method_embedding is not None and "method_embedding" in scorer_state:
            model.method_embedding.load_state_dict(scorer_state["method_embedding"])

        print(f"Model loaded from {load_directory}")
        return model


def create_tokenizer(model_name_or_path: str, max_length: int = 512):
    """
    创建 tokenizer
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.model_max_length = max_length
    return tokenizer

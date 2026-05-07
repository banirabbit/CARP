"""
生成测试集的P2L模型打分文件
从原始方法结果加载测试集数据，使用训练好的模型对每个(问题,方法)对打分，
生成与 dataset/example/test_scored.jsonl 相同格式的文件
"""

import os
import json
import time
import urllib.request
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoTokenizer

from model.cross_encoder import MethodSelectionCrossEncoder
from model.config import Config

# ⚠️ 方法顺序必须与训练时一致，新增方法请在此处维护
DEFAULT_METHOD_ORDER = ['dalk', 'gr', 'hippo', 'lgraph', 'light','qagn']


def get_method_order():
    """
    获取方法顺序，优先使用环境变量 METHOD_ORDER（逗号分隔），否则用默认顺序。
    """
    env_value = os.getenv('METHOD_ORDER')
    if env_value:
        methods = [m.strip() for m in env_value.split(',') if m.strip()]
        if methods:
            print(f"📋 Using method order from env: {methods}")
            return methods
        print("⚠️  METHOD_ORDER env var is empty after parsing, fallback to default.")

    print(f"📋 Using default method order: {DEFAULT_METHOD_ORDER}")
    return DEFAULT_METHOD_ORDER


METHOD_ORDER = get_method_order()


def _qid_key(qid: Any) -> str:
    """统一qid键，便于断点恢复时匹配"""
    return str(qid)


def load_llm_progress(progress_file: str, method_order: List[str]) -> Dict[str, Dict[str, float]]:
    """
    加载LLM打分进度文件（JSONL），返回:
      {qid_key: {method: score}}
    同一qid多条记录时以最后一条为准。
    """
    path = Path(progress_file)
    if not path.exists():
        return {}

    progress: Dict[str, Dict[str, float]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = item.get('id')
            scores = item.get('p2l_scores')
            if qid is None or not isinstance(scores, dict):
                continue

            normalized_scores = {}
            valid = True
            for method in method_order:
                value = scores.get(method)
                if not isinstance(value, (int, float)):
                    valid = False
                    break
                normalized_scores[method] = float(value)

            if valid:
                progress[_qid_key(qid)] = normalized_scores

    return progress


def append_llm_progress(progress_file: str, qid: Any, scores: Dict[str, float]) -> None:
    """实时追加单题打分进度，确保崩溃后可恢复"""
    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        'id': qid,
        'p2l_scores': scores,
        'updated_at': time.time()
    }
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()


def load_prompt_template(prompt_path: str) -> str:
    """加载LLM打分prompt模板"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    从模型输出中提取JSON对象。
    支持纯JSON或包含额外文本/代码块的情况。
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("No JSON object found in model output")


def parse_method_scores_from_llm_output(content: str, method_order: List[str]) -> Dict[str, float]:
    """
    解析LLM返回的method_scores，输出 {method: score}
    """
    parsed = extract_json_from_text(content)
    if 'method_scores' not in parsed or not isinstance(parsed['method_scores'], list):
        raise ValueError("LLM output missing `method_scores` list")

    scores = {}
    for item in parsed['method_scores']:
        if not isinstance(item, dict):
            continue
        method = item.get('method')
        score = item.get('score')
        if method in method_order and isinstance(score, (int, float)):
            scores[method] = float(score)

    missing = [m for m in method_order if m not in scores]
    if missing:
        raise ValueError(f"Missing method scores from LLM output: {missing}")

    return scores


def call_openai_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int = 120
) -> str:
    """
    调用OpenAI兼容Chat Completions接口，返回message.content字符串
    """
    api_base = api_base.rstrip('/')
    url = f"{api_base}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    data = json.dumps(payload).encode('utf-8')

    req = urllib.request.Request(
        url=url,
        data=data,
        method='POST',
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp_body = resp.read().decode('utf-8')
        result = json.loads(resp_body)

    choices = result.get('choices', [])
    if not choices:
        raise ValueError("Empty `choices` in API response")

    message = choices[0].get('message', {})
    content = message.get('content')
    if not isinstance(content, str):
        raise ValueError("Missing `message.content` in API response")
    return content


def load_best_model(config_path: str = 'configs/qwen_pairwise.yaml',
                    model_path: str = 'outputs/pair_v1/best_model_top1.pt'):
    """加载训练好的最佳模型"""
    print(f"📂 Loading config from {config_path}")
    config = Config.from_yaml(config_path)
    
    print(f"📂 Loading tokenizer: {config.model.model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name_or_path,
            trust_remote_code=True
        )
    
    print(f"🤖 Initializing model")
    model = MethodSelectionCrossEncoder(
        model_name_or_path=config.model.model_name_or_path,
        num_methods=config.model.num_methods,
        method_vocab=getattr(config.model, 'method_vocab', None),
        use_method_embedding=getattr(config.model, 'use_method_embedding', False),
        method_embedding_scale=getattr(config.model, 'method_embedding_scale', 1.0),
        method_embedding_init_std=getattr(config.model, 'method_embedding_init_std', 0.02),
        pooling_mode=config.model.pooling_mode,
        dropout=config.model.dropout,
        use_gradient_checkpointing=False,  # 评估时不需要梯度检查点
        temperature=config.model.temperature,
        learnable_temperature=config.model.learnable_temperature,
        margin=getattr(config.model, 'margin', 0.0),  # 边际参数
        loss_type=getattr(config.model, 'loss_type', 'pairwise'),
        pairwise_weight=getattr(config.model, 'pairwise_weight', 1.0),
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
    
    print(f"📂 Loading model weights from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 处理不同的checkpoint格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 移除可能的 'module.' 前缀（来自DDP/DeepSpeed）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"⚠️ Missing keys: {missing}")
    if unexpected:
        print(f"⚠️ Unexpected keys: {unexpected}")
    model.eval()
    
    # 移动到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✅ Model loaded successfully on {device}")
    return model, tokenizer, config, device


def load_test_ids(test_csv: str = 'dataset/pairwise/test_pairwise.csv') -> set:
    """加载测试集的问题ID列表"""
    import pandas as pd
    df = pd.read_csv(test_csv)
    # 从pairwise数据集中提取唯一的问题ID
    test_qids = set(df['qid'].unique().tolist())
    print(f"  📋 Loaded {len(test_qids)} unique test questions from {test_csv}")
    return test_qids


def resolve_default_test_id_file(dataset_name: str) -> str:
    if dataset_name == "hotpot":
        return "dataset/test.csv"

    candidates = [
        f"dataset/eval_questions/{dataset_name}.csv",
        f"dataset/pairwise/{dataset_name}_pairwise.csv",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    return candidates[0]


def load_test_data(dataset_dir: str = 'dataset', dataset_name: str = 'hotpot', 
                  test_ids: set = None):
    """
    从方法评分文件加载测试集数据
    
    Args:
        dataset_dir: 数据集根目录
        dataset_name: 数据集名称
        test_ids: 测试集ID列表（如果提供，则只加载这些ID的数据）
    
    Returns:
        {
            qid: {
                'question': str,
                'answer': str,
                'methods': {
                    method_name: {
                        'output': str,
                        'parsed_answer': str,
                        'accuracy': int,
                        'f1': float,
                        'em': bool,
                        'token_cost': int
                    }
                }
            }
        }
    """
    # ⚠️ 方法顺序必须与训练时一致
    methods = METHOD_ORDER
    print(f"  📋 Method order: {methods}")
    dataset_path = Path(dataset_dir)
    
    # 加载所有方法的评分
    all_data = {}
    
    for method in methods:
        score_file = dataset_path / method / dataset_name / 'results.score.json'
        if not score_file.exists():
            # 兼容没有子目录的数据布局（例如 dataset/qagn/results.score.json）
            fallback_file = dataset_path / method / 'results.score.json'
            if fallback_file.exists():
                score_file = fallback_file
                print(f"⚠️  Warning: {dataset_path / method / dataset_name / 'results.score.json'} not found, using {fallback_file}")
            else:
                print(f"⚠️  Warning: {score_file} not found, skipping {method}")
                continue
        
        print(f"📂 Loading {method} data from {score_file}")
        
        # 读取JSONL文件
        with open(score_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    qid = data['id']
                    
                    # 如果指定了test_ids，只加载测试集的数据（根据qid过滤）
                    if test_ids is not None and qid not in test_ids:
                        continue
                    
                    if qid not in all_data:
                        all_data[qid] = {
                            'question': data['question'],
                            'answer': data.get('label', data.get('answer', '')),
                            'methods': {}
                        }
                    
                    # 添加该方法的结果
                    all_data[qid]['methods'][method] = {
                        'output': data.get('output', ''),
                        'parsed_answer': data.get('parsed_answer', ''),
                        'accuracy': data.get('accuracy', 0),
                        'f1': data.get('f1', 0.0),
                        'em': data.get('em', False),
                        'token_cost': data.get('token_cost', 0)
                    }
                    
                except json.JSONDecodeError:
                    continue
    
    print(f"✅ Loaded {len(all_data)} questions from {len(methods)} methods")
    return all_data


def score_with_model(model, tokenizer, device, questions: List[str],
                     method_descriptions: Dict[str, str],
                     max_length: int = 256, batch_size: int = 32) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    """
    使用模型对所有(问题, 方法)对打分

    Args:
        model: 训练好的模型
        tokenizer: 分词器
        device: 设备
        questions: 问题列表 [(qid, question_text), ...]
        method_descriptions: 方法描述字典
        max_length: 最大序列长度
        batch_size: 批处理大小

    Returns:
        scores: {qid: {method_name: p2l_score}}
        timing_stats: 时间统计信息
    """
    # ⚠️ 方法顺序必须与训练时一致
    methods = METHOD_ORDER
    print(f"  📋 Scoring method order: {methods}")
    
    # 准备所有文本对
    all_texts = []
    text_indices = []  # (qid, method_name)
    
    for qid, question in questions:
        for method in methods:
            method_desc = method_descriptions.get(method, method)
            text = f"Question: {question}\nMethod: {method_desc}"
            all_texts.append(text)
            text_indices.append((qid, method))
    
    print(f"🚀 Scoring {len(all_texts)} (question, method) pairs in batches of {batch_size}")

    # 时间统计变量
    total_inference_time = 0.0  # 总推理时间（秒）
    num_batches = 0

    # 批量推理
    all_scores = []

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(all_texts), batch_size),
                                desc="Scoring with model"):
            batch_end = min(batch_start + batch_size, len(all_texts))
            batch_texts = all_texts[batch_start:batch_end]

            # 批量tokenize
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            # 移动到设备
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            batch_method_ids = [method for _, method in text_indices[batch_start:batch_end]]

            # 记录推理时间
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            # 前向传播（使用BF16加速）
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    scores = model.score_single(input_ids, attention_mask, method_ids=batch_method_ids)  # shape: (batch_size,)
            else:
                scores = model.score_single(input_ids, attention_mask, method_ids=batch_method_ids)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            # 累加推理时间
            batch_time = end_time - start_time
            total_inference_time += batch_time
            num_batches += 1

            # BFloat16 需要先转换为 float32 再转 numpy
            all_scores.extend(scores.float().cpu().numpy().tolist())
    
    # 组织结果
    results = {}
    for (qid, method), score in zip(text_indices, all_scores):
        if qid not in results:
            results[qid] = {}
        results[qid][method] = float(score)

    # 计算时间统计
    num_questions = len(questions)
    num_methods = len(METHOD_ORDER)

    # 每个问题需要对所有方法打分
    avg_time_per_question = total_inference_time / num_questions if num_questions > 0 else 0.0
    avg_time_per_pair = total_inference_time / len(all_texts) if len(all_texts) > 0 else 0.0
    avg_time_per_batch = total_inference_time / num_batches if num_batches > 0 else 0.0

    timing_stats = {
        'total_inference_time': total_inference_time,
        'num_questions': num_questions,
        'num_methods': num_methods,
        'num_pairs': len(all_texts),
        'num_batches': num_batches,
        'avg_time_per_question': avg_time_per_question,
        'avg_time_per_pair': avg_time_per_pair,
        'avg_time_per_batch': avg_time_per_batch,
    }

    print(f"\n⏱️  Timing Statistics:")
    print(f"  Total inference time: {total_inference_time:.4f}s")
    print(f"  Average time per question: {avg_time_per_question*1000:.2f}ms")
    print(f"  Average time per (question, method) pair: {avg_time_per_pair*1000:.2f}ms")
    print(f"  Average time per batch: {avg_time_per_batch*1000:.2f}ms")

    return results, timing_stats


def score_with_llm_api(
    questions: List[Tuple[Any, str]],
    prompt_template: str,
    api_base: str,
    api_key: str,
    model: str,
    progress_file: Optional[str] = None,
    resume_from_progress: bool = True,
    request_timeout: int = 120,
    retry_times: int = 3,
    retry_wait_seconds: float = 2.0
) -> Tuple[Dict[Any, Dict[str, float]], Dict[str, float]]:
    """
    使用LLM API对每个问题进行方法打分
    """
    methods = METHOD_ORDER
    qid_lookup = {_qid_key(qid): qid for qid, _ in questions}
    results = {}
    total_inference_time = 0.0
    total_calls = 0
    total_retries = 0
    resumed_count = 0

    if progress_file and resume_from_progress:
        progress_scores = load_llm_progress(progress_file, methods)
        for key, scores in progress_scores.items():
            if key in qid_lookup:
                results[qid_lookup[key]] = scores
                resumed_count += 1

    print(f"🌐 Scoring {len(questions)} questions with LLM API")
    print(f"  Model: {model}")
    print(f"  API base: {api_base}")
    print(f"  Method order: {methods}")
    if progress_file:
        print(f"  Progress file: {progress_file}")
        print(f"  Resume enabled: {resume_from_progress}, resumed questions: {resumed_count}")

    for qid, question in tqdm(questions, desc="Scoring with LLM API"):
        if qid in results:
            continue

        prompt = prompt_template.replace("{{question}}", question)
        last_error = None

        for attempt in range(1, retry_times + 1):
            start_time = time.time()
            try:
                content = call_openai_chat_completion(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    timeout=request_timeout
                )
                scores = parse_method_scores_from_llm_output(content, methods)
                end_time = time.time()

                total_inference_time += (end_time - start_time)
                total_calls += 1
                results[qid] = scores
                if progress_file:
                    append_llm_progress(progress_file, qid, scores)
                break
            except Exception as e:
                end_time = time.time()
                total_inference_time += (end_time - start_time)
                total_calls += 1
                last_error = e
                if attempt < retry_times:
                    total_retries += 1
                    time.sleep(retry_wait_seconds)
                else:
                    raise RuntimeError(
                        f"Failed to score question qid={qid} after {retry_times} attempts: {e}"
                    ) from e

        if last_error is not None and qid not in results:
            raise RuntimeError(f"Failed to score question qid={qid}: {last_error}")

    num_questions = len(questions)
    num_methods = len(methods)
    num_pairs = num_questions * num_methods
    avg_time_per_question = total_inference_time / num_questions if num_questions > 0 else 0.0
    avg_time_per_pair = total_inference_time / num_pairs if num_pairs > 0 else 0.0

    timing_stats = {
        'total_inference_time': total_inference_time,
        'num_questions': num_questions,
        'num_questions_resumed': resumed_count,
        'num_methods': num_methods,
        'num_pairs': num_pairs,
        'num_batches': total_calls,
        'avg_time_per_question': avg_time_per_question,
        'avg_time_per_pair': avg_time_per_pair,
        'avg_time_per_batch': total_inference_time / total_calls if total_calls > 0 else 0.0,
        'total_api_calls': total_calls,
        'total_retries': total_retries
    }

    print(f"\n⏱️  Timing Statistics:")
    print(f"  Total API time: {total_inference_time:.4f}s")
    print(f"  API calls: {total_calls}, retries: {total_retries}")
    print(f"  Average time per question: {avg_time_per_question*1000:.2f}ms")
    print(f"  Average time per (question, method) pair: {avg_time_per_pair*1000:.2f}ms")

    return results, timing_stats


def generate_scored_file(test_data: Dict, p2l_scores: Dict[int, Dict[str, float]], 
                         output_file: str):
    """
    生成带有P2L打分的测试集文件
    
    Args:
        test_data: 原始测试数据
        p2l_scores: P2L模型打分 {qid: {method: score}}
        output_file: 输出文件路径
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Generating scored file: {output_file}")
    
    results = []
    
    for qid in sorted(test_data.keys()):
        question_data = test_data[qid]
        question = question_data['question']
        answer = question_data['answer']
        methods = question_data['methods']
        
        # 获取该问题的P2L打分
        qid_p2l_scores = p2l_scores.get(qid, {})
        
        # 构建methods字段（包含所有方法的详细结果，添加p2l_score）
        methods_dict = {}
        for method, method_data in methods.items():
            methods_dict[method] = {
                'output': method_data['output'],
                'parsed_answer': method_data['parsed_answer'],
                'accuracy': method_data['accuracy'],
                'f1': method_data['f1'],
                'em': method_data['em'],
                'token_cost': method_data['token_cost'],
                'p2l_score': qid_p2l_scores.get(method, 0.0)
            }
        
        # 构建p2l_scores字段（只包含打分）
        p2l_scores_dict = {method: qid_p2l_scores.get(method, 0.0) 
                          for method in methods.keys()}
        
        # 确定最佳方法
        best_method_by_p2l = max(p2l_scores_dict, key=p2l_scores_dict.get) if p2l_scores_dict else None
        best_method_actual = max(methods, key=lambda m: methods[m]['f1']) if methods else None
        prediction_correct = (best_method_by_p2l == best_method_actual)
        
        # 构建输出记录
        record = {
            'id': qid,
            'question': question,
            'answer': answer,
            'methods': methods_dict,
            'p2l_scores': p2l_scores_dict,
            'best_method_by_p2l': best_method_by_p2l,
            'best_method_actual': best_method_actual,
            'prediction_correct': prediction_correct
        }
        
        results.append(record)
    
    # 写入JSONL文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # 统计信息
    total = len(results)
    correct = sum(1 for r in results if r['prediction_correct'])
    accuracy = correct / total if total > 0 else 0.0
    
    # 统计方法分布
    from collections import Counter
    pred_distribution = Counter(r['best_method_by_p2l'] for r in results if r['best_method_by_p2l'])
    actual_distribution = Counter(r['best_method_actual'] for r in results if r['best_method_actual'])
    
    print(f"\n✅ Generated {total} scored samples")
    print(f"📊 P2L Model Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"💾 Saved to: {output_file}")
    
    # 输出分布对比
    print("\n" + "="*80)
    print("📊 Method Distribution Analysis")
    print("="*80)
    
    methods = METHOD_ORDER
    
    print(f"\n{'Method':<10} | {'Predicted (P2L)':<20} | {'Actual (F1)':<20} | {'Difference':<12}")
    print("-" * 80)
    
    for method in methods:
        pred_count = pred_distribution.get(method, 0)
        actual_count = actual_distribution.get(method, 0)
        pred_pct = pred_count / total * 100 if total > 0 else 0
        actual_pct = actual_count / total * 100 if total > 0 else 0
        diff = pred_pct - actual_pct
        
        print(f"{method:<10} | {pred_count:>6} ({pred_pct:>5.1f}%) | {actual_count:>6} ({actual_pct:>5.1f}%) | {diff:>+6.1f}%")
    
    print("-" * 80)
    print(f"{'Total':<10} | {sum(pred_distribution.values()):>6} ({100.0:>5.1f}%) | {sum(actual_distribution.values()):>6} ({100.0:>5.1f}%) | {0.0:>+6.1f}%")
    
    # 计算分布差异（KL散度或简单的绝对差异）
    total_diff = sum(abs(pred_distribution.get(m, 0) - actual_distribution.get(m, 0)) for m in methods)
    print(f"\n📈 Total distribution difference: {total_diff} samples ({total_diff/total*100:.1f}%)")
    
    # 找出预测偏好和实际偏好
    most_pred = pred_distribution.most_common(1)[0] if pred_distribution else (None, 0)
    most_actual = actual_distribution.most_common(1)[0] if actual_distribution else (None, 0)
    
    print(f"\n🎯 Most predicted method: {most_pred[0]} ({most_pred[1]} times, {most_pred[1]/total*100:.1f}%)")
    print(f"🎯 Most optimal method: {most_actual[0]} ({most_actual[1]} times, {most_actual[1]/total*100:.1f}%)")
    
    if most_pred[0] == most_actual[0]:
        print(f"✅ Model correctly identifies the most common optimal method!")
    else:
        print(f"⚠️  Model prediction bias differs from actual optimal distribution")


def main():
    """主函数"""
    # 配置（允许通过环境变量覆盖）
    score_backend = os.getenv('SCORE_BACKEND', 'local').strip().lower()
    config_path = os.getenv('CONFIG_PATH', 'configs/qwen_pairwise_pairweight_soft_pointwise_margin.yaml')
    model_path = os.getenv('MODEL_PATH', '').strip()
    dataset_dir = os.getenv('DATASET_DIR', 'dataset')
    dataset_name = os.getenv('DATASET_NAME', 'hotpot')
    output_file = os.getenv('P2L_OUTPUT_FILE', '').strip()
    batch_size = int(os.getenv('BATCH_SIZE', '32'))
    test_pairwise_file = os.getenv('PAIRWISE_TEST_FILE', '').strip()
    prompt_template_file = os.getenv('PROMPT_TEMPLATE_FILE', 'dataset/prompt/prompt.txt')
    openai_api_key = os.getenv('OPENAI_API_KEY', '')
    openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    openai_model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
    llm_progress_file = os.getenv('LLM_PROGRESS_FILE', '')
    llm_resume = os.getenv('LLM_RESUME', '1').strip().lower() not in {'0', 'false', 'no'}
    llm_timeout = int(os.getenv('LLM_TIMEOUT', '120'))
    llm_retries = int(os.getenv('LLM_RETRIES', '3'))
    llm_retry_wait = float(os.getenv('LLM_RETRY_WAIT', '2'))
    
    print("="*80)
    print("🚀 P2L Test Set Scoring")
    print("="*80)
    print(f"Score backend: {score_backend}")
    print(f"Dataset name: {dataset_name}")
    print(f"Methods: {METHOD_ORDER}")

    if score_backend not in {'local', 'llm'}:
        raise ValueError(f"Invalid SCORE_BACKEND={score_backend}, expected one of: local, llm")

    if not output_file:
        output_file = f"router/results/scorer/{dataset_name}.jsonl"
    if not test_pairwise_file:
        test_pairwise_file = resolve_default_test_id_file(dataset_name)
    print(f"Test ID file: {test_pairwise_file}")
    print(f"Output file:  {output_file}")

    # 1. 加载测试集ID
    print("\n📦 Step 1: Loading test set IDs...")
    test_ids = load_test_ids(test_pairwise_file)
    
    # 2. 加载测试数据（只加载测试集）
    print("\n📦 Step 2: Loading test data...")
    test_data = load_test_data(dataset_dir, dataset_name, test_ids)
    questions = [(qid, data['question']) for qid, data in test_data.items()]
    
    # 3. 打分
    if score_backend == 'local':
        print("\n📦 Step 3: Loading local model...")
        if not model_path:
            config_for_path = Config.from_yaml(config_path)
            model_path = str(Path(config_for_path.training.output_dir) / 'last_model.pt')
        print(f"Resolved model path: {model_path}")
        model, tokenizer, config, device = load_best_model(config_path, model_path)

        method_descriptions = {
            'dalk': 'DALK (Dense Automatic Knowledge Linking)',
            'gr': 'GR (Graph Reasoning)',
            'hippo': 'HIPPO (Hierarchical Passage Processing)',
            'lgraph': 'LGraph (Logic Graph)',
            'light': 'LIGHT (Lightweight Inference)',
            'qagn': 'QAGN (Question Answering Graph Network)'
        }
        method_descriptions = {
            k: v for k, v in method_descriptions.items()
            if k in METHOD_ORDER
        }

        print("\n📦 Step 4: Scoring with local P2L model...")
        p2l_scores, timing_stats = score_with_model(
            model, tokenizer, device, questions,
            method_descriptions,
            max_length=config.model.max_length,
            batch_size=batch_size
        )
    else:
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when SCORE_BACKEND=llm")
        prompt_template = load_prompt_template(prompt_template_file)
        if not llm_progress_file:
            llm_progress_file = f"{output_file}.llm_progress.jsonl"
        print("\n📦 Step 3: Scoring with LLM API...")
        p2l_scores, timing_stats = score_with_llm_api(
            questions=questions,
            prompt_template=prompt_template,
            api_base=openai_base_url,
            api_key=openai_api_key,
            model=openai_model,
            progress_file=llm_progress_file,
            resume_from_progress=llm_resume,
            request_timeout=llm_timeout,
            retry_times=llm_retries,
            retry_wait_seconds=llm_retry_wait
        )
    
    # 4. 生成打分文件
    print("\n📦 Step 4: Generating scored file...")
    generate_scored_file(test_data, p2l_scores, output_file)

    # 5. 输出时间统计摘要
    print("\n" + "="*80)
    print("⏱️  Final Timing Summary")
    print("="*80)
    print(f"Total questions processed: {timing_stats['num_questions']}")
    print(f"Methods per question: {timing_stats['num_methods']}")
    print(f"Total (question, method) pairs: {timing_stats['num_pairs']}")
    print(f"Total batches: {timing_stats['num_batches']}")
    print(f"\nTotal inference time: {timing_stats['total_inference_time']:.4f}s ({timing_stats['total_inference_time']/60:.2f} min)")
    print(f"Average time per question: {timing_stats['avg_time_per_question']*1000:.2f}ms")
    print(f"Average time per (question, method) pair: {timing_stats['avg_time_per_pair']*1000:.2f}ms")
    print(f"Average time per batch: {timing_stats['avg_time_per_batch']*1000:.2f}ms")
    print(f"\nThroughput: {timing_stats['num_questions']/timing_stats['total_inference_time']:.2f} questions/sec")

    print("\n" + "="*80)
    print("✅ All done!")
    print("="*80)


if __name__ == '__main__':
    main()

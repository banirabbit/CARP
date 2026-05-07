# Pairwise Learning Dataset

这个文件夹包含用于成对学习（Pairwise Learning）的数据集。

## 📁 文件列表

- `train_pairwise.csv` - 训练集（14,224 配对）
- `val_pairwise.csv` - 验证集（762 配对）
- `test_pairwise.csv` - 测试集（763 配对）

## 📊 数据格式

每一行代表一个方法配对，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `qid` | int | 问题ID |
| `question_text` | str | 问题文本 |
| `methodA_id` | str | 方法A的名称（dalk/gr/hippo/lgraph/light） |
| `methodB_id` | str | 方法B的名称 |
| `methodA_text` | str | 方法A的完整输入文本 |
| `methodB_text` | str | 方法B的完整输入文本 |
| `scoreA` | float | 方法A的实际得分（F1分数） |
| `scoreB` | float | 方法B的实际得分 |
| `pair_label` | int | 配对标签（+1表示A优于B） |

## 🔄 数据构造策略

### 训练集（all_pairs）
- 对于每个问题，选择所有正样本（label=1）与所有负样本（label=0）两两配对
- 确保正样本放在A位置（pair_label=+1）
- 总配对数 = Σ(每题正样本数 × 负样本数)

### 验证/测试集（balanced）
- 对于每个问题，随机选择1个正样本和1个负样本配对
- 保持数据集大小接近原始大小
- 避免验证时的计算开销过大

## 📈 数据统计

### 训练集
- 总配对数：14,224
- 唯一问题：3,556
- 平均每题配对：4.00
- 平均得分差（A-B）：0.3046 ± 0.3715

### 验证集
- 总配对数：762
- 唯一问题：762
- 平均每题配对：1.00
- 平均得分差（A-B）：0.3039 ± 0.3635

### 测试集
- 总配对数：763
- 唯一问题：763
- 平均每题配对：1.00
- 平均得分差（A-B）：0.3036 ± 0.3772

## 🎯 使用方法

### 重新生成数据集

```bash
python create_pairwise_dataset.py
```

### 加载数据

```python
import pandas as pd

train_df = pd.read_csv('dataset/pairwise/train_pairwise.csv')
val_df = pd.read_csv('dataset/pairwise/val_pairwise.csv')
test_df = pd.read_csv('dataset/pairwise/test_pairwise.csv')
```

## 📝 注意事项

1. **得分来源**：实际得分来自 `dataset/{method}/hotpot/results.score.json` 文件
2. **配对原则**：始终确保 scoreA ≥ scoreB（如果不满足会自动交换）
3. **标签含义**：pair_label=+1 表示A优于B，适用于 MarginRankingLoss
4. **数据平衡**：通过成对学习自然解决了类别不平衡问题

## 🔗 相关文档

- 详细设计文档：`../../PAIRWISE_LEARNING.md`
- 数据构造脚本：`../../create_pairwise_dataset.py`


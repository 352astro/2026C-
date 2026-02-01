# 投票数预测模型使用说明

## 模型概述

这个模型使用选手的特征（fame、年龄、行业类别、score_sum）来预测投票数，并通过模拟每周比赛计算交叉熵损失进行优化。

## 数据要求

1. **withWeek/2026_MCM_Problem_Fame_Data.csv**: 包含特征信息
   - `fame_1`: 知名度特征
   - `celebrity_age_during_season`: 年龄
   - `industry_*`: 行业类别（one-hot编码）
   - `score_sum`: 评委评分总和
   - `elimination_order`: 淘汰顺序

2. **combined_season_stats.csv**: 包含season信息
   - `season`: 季节编号

## 模型原理

### 1. 特征到投票数的映射
使用线性模型：
```
votes = X @ weights + bias
```
其中X是特征矩阵，包含：
- fame_1
- celebrity_age_during_season
- industry_* (多个行业类别)
- score_sum

### 2. 淘汰概率计算
使用Plackett-Luce模型的softmax：
```
P(淘汰选手i) = exp(-log(votes_i)) / sum(exp(-log(votes_j))) for all j in 存活选手
```

### 3. 损失函数
交叉熵损失：
```
loss = -sum(真实标签 * log(预测概率))
```
其中真实标签是one-hot向量（被淘汰的选手=1，其他=0）

### 4. 优化目标
最小化总损失：
```
总损失 = 交叉熵损失 + L2正则化项
```

## 使用方法

### 基本使用

```python
from train_vote_model import *

# 1. 加载数据
data = load_and_prepare_data()

# 2. 提取特征（可以指定season）
X, elimination_order, player_indices, season_data = extract_features(data, season=1)

# 3. 训练模型
result, best_params = train_model(X, elimination_order, reg_lambda=0.01)

# 4. 评估模型
metrics = evaluate_model(X, elimination_order, 
                        best_params['weights'], 
                        best_params['bias'])
```

### 运行完整训练

```bash
cd bayes
python train_vote_model.py
```

## 输出说明

### 训练过程输出
- 特征数量、选手数量、淘汰轮数
- 优化过程（如果verbose=True）
- 最终损失和优化状态

### 评估结果
- 总损失和平均每轮损失
- 淘汰顺序准确率
- 预测的淘汰顺序 vs 真实淘汰顺序

### 保存的文件
- `bayes/vote_model_params.json`: 模型参数（权重和偏置）

## 模型参数

- `weights`: 特征权重向量，形状为(n_features,)
- `bias`: 偏置项，标量
- `reg_lambda`: 正则化系数（默认0.01）

## 关键函数说明

### `predict_votes_from_features(X, weights, bias)`
从特征预测投票数

### `calculate_elimination_probabilities(votes)`
计算淘汰概率（softmax）

### `weekly_elimination_loss(X_all, weights, bias, elimination_order)`
计算多轮淘汰的交叉熵损失

### `train_model(X, elimination_order, reg_lambda, max_iter)`
训练模型，返回最佳参数

## 注意事项

1. **数据关联**: 确保fame_data和season_data能够正确关联（通过index或行号）
2. **特征标准化**: 特征会自动标准化（均值为0，标准差为1）
3. **投票数约束**: 预测的投票数会被约束为≥1
4. **正则化**: 使用L2正则化防止过拟合

## 扩展建议

1. **非线性模型**: 可以尝试使用神经网络替代线性模型
2. **时间特征**: 可以添加周数作为特征
3. **集成学习**: 可以训练多个模型并集成
4. **交叉验证**: 可以添加交叉验证来评估模型泛化能力


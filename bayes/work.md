combined_season_stats.csv这里面存储了我需要的season信息，withWeek\2026_MCM_Problem_Fame_Data.csv这里存储了我需要的其他特征信息。

我利用一个选手的fame，年龄，行业类别，score_sum（评委评分总和）来推断选手的得票数量；在训练时，模拟每周的比赛，预测出投票数后计算每个人的淘汰概率和真实的淘汰情形做对比，计算出总的交叉熵损失，进行优化。

## 已完成的工作

✅ **模型文件**: `bayes/train_vote_model.py`
- 数据加载和预处理
- 特征提取（fame, 年龄, 行业类别, score_sum）
- 线性模型：从特征预测投票数
- 淘汰概率计算：使用softmax（Plackett-Luce模型）
- 交叉熵损失：每轮计算预测概率与真实标签的交叉熵
- 模型优化：使用L-BFGS-B算法最小化总损失（交叉熵 + L2正则化）

✅ **使用说明**: `bayes/README_vote_model.md`

## 使用方法

```bash
cd bayes
python train_vote_model.py
```

## 模型特点

1. **清晰的损失函数**: 显式计算每轮淘汰的交叉熵损失
2. **逐轮模拟**: 模拟每周比赛，逐步淘汰选手
3. **特征标准化**: 自动标准化特征
4. **正则化**: L2正则化防止过拟合
5. **完整评估**: 输出淘汰顺序准确率等指标

"""
投票数预测模型训练脚本（神经网络版本）
使用sklearn的MLPClassifier，将每轮淘汰作为多分类问题
让模型直接学习index与淘汰结果的关系
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载和预处理 ====================

def load_and_prepare_data():
    """
    加载并准备数据
    
    返回:
    - data: 合并后的数据DataFrame
    """
    print("=" * 60)
    print("步骤1: 加载数据")
    print("=" * 60)
    
    # 加载数据
    fame_data = pd.read_csv('withWeek/2026_MCM_Problem_Fame_Data.csv')
    season_data = pd.read_csv('combined_season_stats.csv')
    
    # 合并数据
    data = fame_data.copy()
    
    # 从season_data添加season信息
    if 'index' in data.columns:
        season_mapping = dict(zip(range(len(season_data)), season_data['season']))
        data['season'] = data['index'].map(season_mapping)
    else:
        if len(data) == len(season_data):
            data['season'] = season_data['season'].values
        else:
            print("警告: 数据长度不匹配，无法自动关联season")
            data['season'] = None
    
    print(f"加载完成: {len(data)}条记录")
    print(f"特征列: {list(data.columns[:10])}...")
    if 'season' in data.columns:
        print(f"Season范围: {data['season'].min()} - {data['season'].max()}")
    
    return data

def extract_features_for_classification(data, season=None):
    """
    提取特征用于分类（每轮预测哪个选手被淘汰）
    
    参数:
    - data: DataFrame（已经按season筛选过的）
    
    返回:
    - X_all: 所有轮次的特征矩阵列表
    - y_all: 所有轮次的标签列表（被淘汰选手的索引）
    - elimination_order: 淘汰顺序
    """
    season_data = data.copy()
    
    # 提取基础特征：fame, 年龄, 行业类别, score_sum
    feature_cols = []
    
    if 'fame_1' in season_data.columns:
        feature_cols.append('fame_1')
    if 'celebrity_age_during_season' in season_data.columns:
        feature_cols.append('celebrity_age_during_season')
    industry_cols = [col for col in season_data.columns if col.startswith('industry_')]
    feature_cols.extend(industry_cols)
    if 'score_sum' in season_data.columns:
        feature_cols.append('score_sum')
    
    # 提取特征矩阵
    available_cols = [col for col in feature_cols if col in season_data.columns]
    X_base = season_data[available_cols].values.astype(float)
    
    # 标准化基础特征
    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base)
    
    # 提取淘汰顺序
    if 'elimination_order' in season_data.columns:
        elimination_ranks = season_data['elimination_order'].values
        max_rank = elimination_ranks.max()
        eliminated_mask = elimination_ranks < max_rank
        eliminated_indices = np.where(eliminated_mask)[0].tolist()
        sorted_indices = np.argsort(elimination_ranks[eliminated_mask])
        elimination_order = [eliminated_indices[i] for i in sorted_indices]
    else:
        if 'placement' in season_data.columns:
            placement = season_data['placement'].values
            sorted_indices = np.argsort(placement)[::-1]
            elimination_order = [idx for idx in sorted_indices if placement[idx] > 1]
        else:
            elimination_order = []
    
    n_players = len(season_data)
    
    # 为每轮淘汰准备数据
    X_all_rounds = []
    y_all_rounds = []
    
    alive_indices = list(range(n_players))
    
    for round_num, true_eliminated in enumerate(elimination_order):
        # 当前存活选手的特征
        X_alive_base = X_base_scaled[alive_indices]
        
        # 添加index相关特征（不标准化，保持原始值）
        n_alive = len(alive_indices)
        season_internal_index = np.arange(n_alive).reshape(-1, 1).astype(float)
        
        # 如果有season信息，也添加season编号
        if 'season' in season_data.columns and season_data['season'].notna().any():
            season_num = season_data['season'].values[0]
            season_feature = np.full((n_alive, 1), float(season_num))
            X_index_features = np.hstack([season_feature, season_internal_index])
        else:
            X_index_features = season_internal_index
        
        # 合并特征
        X_alive = np.hstack([X_alive_base, X_index_features])
        
        # 标签：被淘汰选手在存活列表中的位置
        eliminated_idx_in_alive = alive_indices.index(true_eliminated)
        
        X_all_rounds.append(X_alive)
        y_all_rounds.append(eliminated_idx_in_alive)
        
        # 移除被淘汰的选手
        alive_indices.remove(true_eliminated)
    
    return X_all_rounds, y_all_rounds, elimination_order, scaler

# ==================== 准备训练数据 ====================

def prepare_training_data(data):
    """
    为所有season准备训练数据
    
    参数:
    - data: 完整数据DataFrame
    
    返回:
    - X_train: 所有轮次的特征矩阵（合并）
    - y_train: 所有轮次的标签（合并）
    - season_info: 每个样本属于哪个season和哪一轮
    """
    X_all = []
    y_all = []
    season_info = []
    
    if 'season' not in data.columns:
        print("警告: 数据中没有season列")
        X_rounds, y_rounds, elim_order, scaler = extract_features_for_classification(data)
        for round_num, (X_round, y_round) in enumerate(zip(X_rounds, y_rounds)):
            X_all.append(X_round)
            y_all.extend([y_round] * len(X_round))  # 每个样本的标签
            season_info.extend([(None, round_num, i) for i in range(len(X_round))])
        return np.vstack(X_all), np.array(y_all), season_info
    
    # 按season分组
    unique_seasons = sorted(data['season'].dropna().unique())
    print(f"\n找到 {len(unique_seasons)} 个season: {unique_seasons}")
    
    for season in unique_seasons:
        season_df = data[data['season'] == season].copy()
        if len(season_df) == 0:
            continue
        
        X_rounds, y_rounds, elim_order, scaler = extract_features_for_classification(season_df)
        
        if len(X_rounds) == 0:
            print(f"  警告: Season {season} 无法提取淘汰顺序，跳过")
            continue
        
        # 为每个样本添加标签
        for round_num, (X_round, y_round) in enumerate(zip(X_rounds, y_rounds)):
            # 每个样本的标签是：在这一轮中，哪个选手被淘汰（在存活列表中的位置）
            # 同一轮所有样本的标签相同
            X_all.append(X_round)
            y_all.extend([y_round] * len(X_round))
            season_info.extend([(season, round_num, i) for i in range(len(X_round))])
        
        print(f"  Season {season}: {len(season_df)}个选手, {len(X_rounds)}轮淘汰, "
              f"共{sum(len(x) for x in X_rounds)}个训练样本")
    
    X_train = np.vstack(X_all)
    y_train = np.array(y_all)
    
    print(f"\n总训练样本数: {len(X_train)}")
    print(f"特征维度: {X_train.shape[1]}")
    print(f"标签类别数: {len(np.unique(y_train))}")
    
    return X_train, y_train, season_info

# ==================== 模型训练 ====================

def train_neural_network(X_train, y_train, hidden_layer_sizes=(100, 50), max_iter=1000, verbose=True):
    """
    训练神经网络分类器
    
    参数:
    - X_train: 训练特征矩阵
    - y_train: 训练标签
    - hidden_layer_sizes: 隐藏层大小，例如(100, 50)表示两层，每层100和50个神经元
    - max_iter: 最大迭代次数
    - verbose: 是否输出训练过程
    
    返回:
    - model: 训练好的MLPClassifier
    """
    print("=" * 60)
    print("步骤2: 训练神经网络")
    print("=" * 60)
    
    print(f"训练样本数: {len(X_train)}")
    print(f"特征维度: {X_train.shape[1]}")
    print(f"标签类别数: {len(np.unique(y_train))}")
    print(f"隐藏层结构: {hidden_layer_sizes}")
    
    # 创建MLPClassifier
    # 使用交叉熵损失（log_loss）和adam优化器
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.01,  # L2正则化系数
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=42,
        tol=1e-4,
        verbose=verbose,
        early_stopping=True,  # 早停
        validation_fraction=0.1,  # 10%数据用于验证
        n_iter_no_change=10  # 10次迭代无改善则停止
    )
    
    print("开始训练...")
    model.fit(X_train, y_train)
    
    print(f"\n训练完成!")
    print(f"最终损失: {model.loss_:.4f}")
    print(f"迭代次数: {model.n_iter_}")
    print(f"训练准确率: {model.score(X_train, y_train):.4f}")
    
    return model

# ==================== 评估函数 ====================

def evaluate_model_per_season(data, model, verbose=True):
    """
    按season评估模型性能
    
    参数:
    - data: 完整数据DataFrame
    - model: 训练好的MLPClassifier
    - verbose: 是否输出详细信息
    
    返回:
    - all_metrics: 所有season的评估结果
    """
    print("=" * 60)
    print("步骤3: 评估模型（所有season）")
    print("=" * 60)
    
    all_metrics = []
    total_correct = 0
    total_rounds = 0
    
    if 'season' not in data.columns:
        X_rounds, y_rounds, elim_order, _ = extract_features_for_classification(data)
        # 评估单个season
        correct_rounds = 0
        for X_round, y_true in zip(X_rounds, y_rounds):
            y_pred = model.predict(X_round)
            # 预测概率最高的那个选手被淘汰
            y_pred_proba = model.predict_proba(X_round)
            y_pred_max = np.argmax(y_pred_proba, axis=1)
            # 取第一个样本的预测（因为同一轮所有样本的标签相同）
            if y_pred_max[0] == y_true:
                correct_rounds += 1
        accuracy = correct_rounds / len(y_rounds) if len(y_rounds) > 0 else 0
        all_metrics.append({
            'season': None,
            'accuracy': accuracy,
            'correct_rounds': correct_rounds,
            'total_rounds': len(y_rounds)
        })
        return all_metrics
    
    unique_seasons = sorted(data['season'].dropna().unique())
    
    for season in unique_seasons:
        season_df = data[data['season'] == season].copy()
        if len(season_df) == 0:
            continue
        
        X_rounds, y_rounds, elim_order, _ = extract_features_for_classification(season_df)
        
        if len(X_rounds) == 0:
            continue
        
        # 评估每一轮
        correct_rounds = 0
        for round_num, (X_round, y_true) in enumerate(zip(X_rounds, y_rounds)):
            # 预测每个存活选手被淘汰的概率
            y_pred_proba = model.predict_proba(X_round)
            # 找到概率最高的（最可能被淘汰的）
            y_pred = np.argmax(y_pred_proba, axis=1)
            # 同一轮所有样本的预测应该相同，取第一个
            if y_pred[0] == y_true:
                correct_rounds += 1
        
        accuracy = correct_rounds / len(y_rounds) if len(y_rounds) > 0 else 0
        total_correct += correct_rounds
        total_rounds += len(y_rounds)
        
        metrics = {
            'season': season,
            'accuracy': accuracy,
            'correct_rounds': correct_rounds,
            'total_rounds': len(y_rounds)
        }
        all_metrics.append(metrics)
        
        if verbose:
            print(f"Season {season}: 准确率={accuracy:.2%} "
                  f"({correct_rounds}/{len(y_rounds)})")
    
    # 总体统计
    overall_accuracy = total_correct / total_rounds if total_rounds > 0 else 0
    
    if verbose:
        print(f"\n总体统计:")
        print(f"  总体准确率: {overall_accuracy:.2%} ({total_correct}/{total_rounds})")
        print(f"  评估的season数: {len(all_metrics)}")
    
    return all_metrics, {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_rounds': total_rounds,
        'n_seasons': len(all_metrics)
    }

# ==================== 主函数 ====================

def main():
    """
    主训练流程
    """
    # 1. 加载数据
    data = load_and_prepare_data()
    
    # 2. 准备训练数据
    X_train, y_train, season_info = prepare_training_data(data)
    
    if len(X_train) == 0:
        print("错误: 没有可用的训练数据")
        return
    
    # 3. 训练神经网络
    model = train_neural_network(
        X_train, 
        y_train,
        hidden_layer_sizes=(128, 64, 32),  # 三层隐藏层
        max_iter=1000,
        verbose=True
    )
    
    # 4. 评估模型
    all_metrics, summary = evaluate_model_per_season(data, model, verbose=True)
    
    # 5. 保存模型
    print("\n" + "=" * 60)
    print("步骤4: 保存模型")
    print("=" * 60)
    
    import pickle
    import json
    
    # 保存sklearn模型
    with open('bayes/vote_model_nn.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("神经网络模型已保存到: bayes/vote_model_nn.pkl")
    
    # 保存模型信息
    model_info = {
        'model_type': 'MLPClassifier',
        'n_features': X_train.shape[1],
        'n_classes': len(np.unique(y_train)),
        'hidden_layer_sizes': model.hidden_layer_sizes,
        'final_loss': float(model.loss_),
        'n_iter': int(model.n_iter_),
        'training_accuracy': float(model.score(X_train, y_train)),
        'summary': {
            'overall_accuracy': float(summary['overall_accuracy']),
            'total_correct': int(summary['total_correct']),
            'total_rounds': int(summary['total_rounds']),
            'n_seasons': int(summary['n_seasons'])
        },
        'per_season_metrics': [
            {
                'season': int(m['season']) if m['season'] is not None else None,
                'accuracy': float(m['accuracy']),
                'correct_rounds': int(m['correct_rounds']),
                'total_rounds': int(m['total_rounds'])
            }
            for m in all_metrics
        ]
    }
    
    with open('bayes/vote_model_nn_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("模型信息已保存到: bayes/vote_model_nn_info.json")
    
    return model, all_metrics, summary

if __name__ == '__main__':
    model, all_metrics, summary = main()


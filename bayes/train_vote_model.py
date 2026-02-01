"""
投票数预测模型训练脚本
使用fame、年龄、行业类别、score_sum来预测投票数
通过模拟每周比赛，计算交叉熵损失进行优化
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
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
    
    # 合并数据：通过行索引关联（假设fame_data的index对应season_data的行）
    # 或者通过其他方式关联，这里简化处理
    data = fame_data.copy()
    
    # 从season_data添加season信息
    # 假设fame_data的index列对应season_data的行索引
    if 'index' in data.columns:
        # 如果fame_data有index列，尝试匹配
        season_mapping = dict(zip(range(len(season_data)), season_data['season']))
        data['season'] = data['index'].map(season_mapping)
    else:
        # 如果没有index列，直接按行号匹配
        if len(data) == len(season_data):
            data['season'] = season_data['season'].values
        else:
            print("警告: 数据长度不匹配，无法自动关联season")
            data['season'] = None
    
    print(f"加载完成: {len(data)}条记录")
    print(f"特征列: {list(data.columns[:10])}...")  # 只显示前10列
    if 'season' in data.columns:
        print(f"Season范围: {data['season'].min()} - {data['season'].max()}")
    
    return data

def extract_features(data, season=None):
    """
    提取特征和淘汰顺序
    
    参数:
    - data: DataFrame（已经按season筛选过的）
    - season: 季节编号（用于显示，实际数据已经筛选）
    
    返回:
    - X: 特征矩阵 (n_players, n_features)
    - elimination_order: 淘汰顺序（索引列表）
    - player_indices: 选手索引列表
    """
    # 数据已经在prepare_seasons_data中按season筛选过了
    season_data = data.copy()
    
    # 提取特征：fame, 年龄, 行业类别, score_sum, index
    feature_cols = []
    
    # 1. Fame特征（使用fame_1作为基础）
    if 'fame_1' in season_data.columns:
        feature_cols.append('fame_1')
    
    # 2. 年龄
    if 'celebrity_age_during_season' in season_data.columns:
        feature_cols.append('celebrity_age_during_season')
    
    # 3. 行业类别（one-hot编码）
    industry_cols = [col for col in season_data.columns if col.startswith('industry_')]
    feature_cols.extend(industry_cols)
    
    # 4. Score_sum（评委评分总和）
    if 'score_sum' in season_data.columns:
        feature_cols.append('score_sum')
    
    # 提取特征矩阵
    available_cols = [col for col in feature_cols if col in season_data.columns]
    X = season_data[available_cols].values.astype(float)
    
    # 5. 添加index相关特征（重要：不标准化，保持原始值）
    n_players = len(season_data)
    
    # 方式1: 添加season内相对index（0, 1, 2, ...）- 不标准化
    season_internal_index = np.arange(n_players).reshape(-1, 1).astype(float)
    
    # 方式2: 如果有season信息，也添加season编号 - 不标准化
    if 'season' in season_data.columns and season_data['season'].notna().any():
        season_num = season_data['season'].values[0]  # 同一个season内season值相同
        season_feature = np.full((n_players, 1), float(season_num))
        # 将season和season内index都加入特征（不标准化）
        X_index_features = np.hstack([season_feature, season_internal_index])
    else:
        # 如果没有season信息，只添加season内index
        X_index_features = season_internal_index
    
    # 标准化基础特征（fame, 年龄, 行业, score_sum）
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_normalized = (X - X_mean) / X_std
    
    # 将标准化后的基础特征和未标准化的index特征合并
    X = np.hstack([X_normalized, X_index_features])
    
    # 提取淘汰顺序
    if 'elimination_order' in season_data.columns:
        elimination_ranks = season_data['elimination_order'].values
        # 找到冠军（elimination_order最大的）
        max_rank = elimination_ranks.max()
        # 被淘汰的选手（排除冠军）
        eliminated_mask = elimination_ranks < max_rank
        eliminated_indices = np.where(eliminated_mask)[0].tolist()
        
        # 按elimination_order排序
        sorted_indices = np.argsort(elimination_ranks[eliminated_mask])
        elimination_order = [eliminated_indices[i] for i in sorted_indices]
    else:
        # 如果没有elimination_order，尝试从placement推导
        if 'placement' in season_data.columns:
            placement = season_data['placement'].values
            # placement=1是冠军，placement越大越早被淘汰
            sorted_indices = np.argsort(placement)[::-1]  # 从大到小
            # 排除冠军（placement=1）
            elimination_order = [idx for idx in sorted_indices if placement[idx] > 1]
        else:
            elimination_order = []
    
    player_indices = list(range(len(season_data)))
    
    return X, elimination_order, player_indices, season_data

# ==================== 模型定义 ====================

def predict_votes_from_features(X, weights, bias=0):
    """
    从特征预测投票数（线性模型）
    
    参数:
    - X: (n_players, n_features) 特征矩阵
    - weights: (n_features,) 权重向量
    - bias: 偏置项
    
    返回:
    - votes: (n_players,) 预测的投票数
    """
    votes = X @ weights + bias
    # 确保投票数为正
    votes = np.maximum(votes, 1.0)
    return votes

def calculate_elimination_probabilities(votes):
    """
    计算每个选手被淘汰的概率（softmax）
    
    参数:
    - votes: (n_alive,) 当前存活选手的投票数
    
    返回:
    - probs: (n_alive,) 每个选手被淘汰的概率
    """
    # 将投票数转换为对数空间
    mu = np.log(votes + 1e-10)
    
    # 计算 exp(-mu)
    exp_neg_mu = np.exp(-mu)
    
    # Softmax归一化
    probs = exp_neg_mu / exp_neg_mu.sum()
    
    return probs

# ==================== 损失函数 ====================

def weekly_elimination_loss(X_all, weights, bias, elimination_order, verbose=False):
    """
    计算多轮淘汰的交叉熵损失
    
    参数:
    - X_all: (n_players, n_features) 所有选手的特征矩阵
    - weights: (n_features,) 模型权重
    - bias: 偏置项
    - elimination_order: 真实淘汰顺序
    - verbose: 是否输出详细信息
    
    返回:
    - total_loss: 总交叉熵损失
    - loss_per_round: 每轮损失列表
    """
    n_players = len(X_all)
    alive_indices = list(range(n_players))
    total_loss = 0.0
    loss_per_round = []
    
    for round_num, true_eliminated in enumerate(elimination_order):
        # 获取当前存活选手的特征
        X_alive = X_all[alive_indices]
        
        # 预测投票数
        pred_votes = predict_votes_from_features(X_alive, weights, bias)
        
        # 计算淘汰概率
        pred_probs = calculate_elimination_probabilities(pred_votes)
        
        # 找到被淘汰选手在存活列表中的位置
        eliminated_idx_in_alive = alive_indices.index(true_eliminated)
        
        # 创建真实标签（one-hot）
        true_labels = np.zeros(len(pred_probs))
        true_labels[eliminated_idx_in_alive] = 1.0
        
        # 计算交叉熵损失
        cross_entropy = -np.sum(true_labels * np.log(pred_probs + 1e-10))
        total_loss += cross_entropy
        loss_per_round.append(cross_entropy)
        
        if verbose and round_num < 3:  # 只显示前3轮
            print(f"  第{round_num+1}轮: 淘汰选手{alive_indices[eliminated_idx_in_alive]}, "
                  f"预测概率={pred_probs[eliminated_idx_in_alive]:.4f}, "
                  f"损失={cross_entropy:.4f}")
        
        # 移除被淘汰的选手
        alive_indices.remove(true_eliminated)
    
    return total_loss, loss_per_round

def objective_function(params, seasons_data, reg_lambda=0.01):
    """
    目标函数（用于优化）- 处理多个season
    
    参数:
    - params: 模型参数 [weights..., bias]
    - seasons_data: list of dict，每个dict包含 {'X': ..., 'elimination_order': ...}
    - reg_lambda: 正则化系数
    
    返回:
    - loss: 总损失（所有season的交叉熵损失之和 + 正则化）
    """
    n_features = seasons_data[0]['X'].shape[1]
    weights = params[:n_features]
    bias = params[n_features]
    
    # 累加所有season的交叉熵损失
    total_cross_entropy = 0.0
    for season_data in seasons_data:
        X = season_data['X']
        elimination_order = season_data['elimination_order']
        season_loss, _ = weekly_elimination_loss(X, weights, bias, elimination_order, verbose=False)
        total_cross_entropy += season_loss
    
    # L2正则化
    reg_term = reg_lambda * np.sum(weights ** 2)
    
    total_loss = total_cross_entropy + reg_term
    
    return total_loss

# ==================== 训练函数 ====================

def train_model(seasons_data, reg_lambda=0.01, max_iter=1000, verbose=True):
    """
    训练投票数预测模型（处理多个season）
    
    参数:
    - seasons_data: list of dict，每个dict包含 {'X': ..., 'elimination_order': ..., 'season': ...}
    - reg_lambda: 正则化系数
    - max_iter: 最大迭代次数
    - verbose: 是否输出训练过程
    
    返回:
    - result: 优化结果
    - best_params: 最佳参数 {'weights': ..., 'bias': ...}
    """
    print("=" * 60)
    print("步骤2: 训练模型")
    print("=" * 60)
    
    if len(seasons_data) == 0:
        raise ValueError("没有可用的season数据")
    
    n_features = seasons_data[0]['X'].shape[1]
    n_seasons = len(seasons_data)
    
    # 初始化参数
    weights0 = np.random.randn(n_features) * 0.1
    bias0 = 1000.0  # 假设平均投票数在1000左右
    
    params0 = np.concatenate([weights0, [bias0]])
    
    # 计算初始损失
    initial_loss = objective_function(params0, seasons_data, reg_lambda)
    
    if verbose:
        print(f"Season数量: {n_seasons}")
        print(f"特征数量: {n_features}")
        total_players = sum(len(sd['X']) for sd in seasons_data)
        total_rounds = sum(len(sd['elimination_order']) for sd in seasons_data)
        print(f"总选手数: {total_players}")
        print(f"总淘汰轮数: {total_rounds}")
        print(f"初始参数: weights形状={weights0.shape}, bias={bias0}")
        print(f"初始损失: {initial_loss:.4f}")
        print("开始优化...")
    
    # 添加回调函数来监控训练过程
    loss_history = [initial_loss]
    iteration_count = [0]
    
    def callback(xk):
        """优化过程的回调函数"""
        current_loss = objective_function(xk, seasons_data, reg_lambda)
        loss_history.append(current_loss)
        iteration_count.append(len(loss_history) - 1)
        if verbose and len(loss_history) % 10 == 0:
            print(f"  迭代 {len(loss_history)-1}: 损失 = {current_loss:.4f}")
    
    # 优化
    result = minimize(
        objective_function,
        params0,
        args=(seasons_data, reg_lambda),
        method='L-BFGS-B',
        callback=callback,
        options={'maxiter': max_iter, 'disp': verbose}
    )
    
    # 提取最佳参数
    best_weights = result.x[:n_features]
    best_bias = result.x[n_features]
    
    best_params = {
        'weights': best_weights,
        'bias': best_bias
    }
    
    if verbose:
        print(f"\n优化完成!")
        print(f"初始损失: {initial_loss:.4f}")
        print(f"最终损失: {result.fun:.4f}")
        print(f"损失降低: {initial_loss - result.fun:.4f} ({((initial_loss - result.fun) / initial_loss * 100):.2f}%)")
        print(f"优化状态: {result.success}")
        print(f"迭代次数: {result.nit}")
        if not result.success:
            print(f"警告: {result.message}")
    
    return result, best_params

# ==================== 评估函数 ====================

def evaluate_model(X, elimination_order, weights, bias, verbose=True):
    """
    评估模型性能
    
    参数:
    - X: 特征矩阵
    - elimination_order: 真实淘汰顺序
    - weights: 模型权重
    - bias: 偏置项
    - verbose: 是否输出详细信息
    
    返回:
    - metrics: 评估指标字典
    """
    print("=" * 60)
    print("步骤3: 评估模型")
    print("=" * 60)
    
    # 计算损失
    total_loss, loss_per_round = weekly_elimination_loss(
        X, weights, bias, elimination_order, verbose=verbose
    )
    
    # 模拟预测的淘汰顺序
    n_players = len(X)
    alive_indices = list(range(n_players))
    predicted_elim_order = []
    
    for _ in range(len(elimination_order)):
        if len(alive_indices) <= 1:
            break
        
        X_alive = X[alive_indices]
        pred_votes = predict_votes_from_features(X_alive, weights, bias)
        
        # 找到投票数最少的（被淘汰）
        min_vote_idx = np.argmin(pred_votes)
        eliminated_idx = alive_indices[min_vote_idx]
        predicted_elim_order.append(eliminated_idx)
        alive_indices.remove(eliminated_idx)
    
    # 计算准确率
    correct_rounds = sum(1 for pred, true in zip(predicted_elim_order, elimination_order) 
                         if pred == true)
    accuracy = correct_rounds / len(elimination_order) if len(elimination_order) > 0 else 0
    
    metrics = {
        'total_loss': total_loss,
        'avg_loss_per_round': total_loss / len(elimination_order) if len(elimination_order) > 0 else 0,
        'loss_per_round': loss_per_round,
        'predicted_elim_order': predicted_elim_order,
        'true_elim_order': elimination_order,
        'round_accuracy': accuracy,
        'correct_rounds': correct_rounds,
        'total_rounds': len(elimination_order)
    }
    
    if verbose:
        print(f"总损失: {total_loss:.4f}")
        print(f"平均每轮损失: {metrics['avg_loss_per_round']:.4f}")
        print(f"淘汰顺序准确率: {accuracy:.2%} ({correct_rounds}/{len(elimination_order)})")
        print(f"预测顺序: {predicted_elim_order}")
        print(f"真实顺序: {elimination_order}")
    
    return metrics

# ==================== 主函数 ====================

def prepare_seasons_data(data):
    """
    按season准备数据
    
    参数:
    - data: 完整数据DataFrame
    
    返回:
    - seasons_data: list of dict，每个dict包含一个season的数据
    """
    seasons_data = []
    
    if 'season' not in data.columns:
        print("警告: 数据中没有season列，尝试使用所有数据作为一个season")
        X, elimination_order, player_indices, season_data = extract_features(data)
        if len(elimination_order) > 0:
            seasons_data.append({
                'X': X,
                'elimination_order': elimination_order,
                'season': None,
                'player_indices': player_indices
            })
        return seasons_data
    
    # 按season分组
    unique_seasons = sorted(data['season'].dropna().unique())
    print(f"\n找到 {len(unique_seasons)} 个season: {unique_seasons}")
    
    for season in unique_seasons:
        season_df = data[data['season'] == season].copy()
        if len(season_df) == 0:
            continue
        
        X, elimination_order, player_indices, _ = extract_features(season_df)
        
        if len(elimination_order) == 0:
            print(f"  警告: Season {season} 无法提取淘汰顺序，跳过")
            continue
        
        seasons_data.append({
            'X': X,
            'elimination_order': elimination_order,
            'season': season,
            'player_indices': player_indices,
            'n_players': len(X),
            'n_rounds': len(elimination_order)
        })
        
        print(f"  Season {season}: {len(X)}个选手, {len(elimination_order)}轮淘汰")
    
    return seasons_data

def evaluate_all_seasons(seasons_data, weights, bias, verbose=True):
    """
    评估所有season的模型性能
    
    参数:
    - seasons_data: list of dict，每个season的数据
    - weights: 模型权重
    - bias: 偏置项
    - verbose: 是否输出详细信息
    
    返回:
    - all_metrics: 所有season的评估结果
    """
    print("=" * 60)
    print("步骤3: 评估模型（所有season）")
    print("=" * 60)
    
    all_metrics = []
    total_loss_all = 0.0
    total_correct = 0
    total_rounds = 0
    
    for season_info in seasons_data:
        X = season_info['X']
        elimination_order = season_info['elimination_order']
        season = season_info['season']
        
        metrics = evaluate_model(X, elimination_order, weights, bias, verbose=False)
        metrics['season'] = season
        all_metrics.append(metrics)
        
        total_loss_all += metrics['total_loss']
        total_correct += metrics['correct_rounds']
        total_rounds += metrics['total_rounds']
        
        if verbose:
            print(f"Season {season}: 损失={metrics['total_loss']:.4f}, "
                  f"准确率={metrics['round_accuracy']:.2%} "
                  f"({metrics['correct_rounds']}/{metrics['total_rounds']})")
    
    # 总体统计
    avg_loss = total_loss_all / len(seasons_data) if len(seasons_data) > 0 else 0
    overall_accuracy = total_correct / total_rounds if total_rounds > 0 else 0
    
    if verbose:
        print(f"\n总体统计:")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  总体准确率: {overall_accuracy:.2%} ({total_correct}/{total_rounds})")
        print(f"  评估的season数: {len(seasons_data)}")
    
    return all_metrics, {
        'avg_loss': avg_loss,
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_rounds': total_rounds,
        'n_seasons': len(seasons_data)
    }

def main():
    """
    主训练流程
    """
    # 1. 加载数据
    data = load_and_prepare_data()
    
    # 2. 按season准备数据
    seasons_data = prepare_seasons_data(data)
    
    if len(seasons_data) == 0:
        print("错误: 没有可用的season数据")
        return
    
    print(f"\n数据准备完成:")
    print(f"  共 {len(seasons_data)} 个season")
    
    # 3. 训练模型（使用所有season）
    result, best_params = train_model(
        seasons_data,
        reg_lambda=0.01,
        max_iter=1000,
        verbose=True
    )
    
    # 4. 评估所有season
    all_metrics, summary = evaluate_all_seasons(
        seasons_data,
        best_params['weights'],
        best_params['bias'],
        verbose=True
    )
    
    # 5. 保存结果
    print("\n" + "=" * 60)
    print("步骤4: 保存结果")
    print("=" * 60)
    
    import json
    
    # 确保所有数值都是Python原生类型（不是numpy类型）
    def convert_to_native(obj):
        """将numpy类型转换为Python原生类型"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    model_info = {
        'weights': best_params['weights'].tolist(),
        'bias': float(best_params['bias']),
        'n_features': int(len(best_params['weights'])),
        'final_loss': float(result.fun),
        'n_seasons': int(len(seasons_data)),
        'summary': {
            'avg_loss': float(summary['avg_loss']),
            'overall_accuracy': float(summary['overall_accuracy']),
            'total_correct': int(summary['total_correct']),
            'total_rounds': int(summary['total_rounds'])
        },
        'per_season_metrics': [
            {
                'season': int(m['season']) if m['season'] is not None else None,
                'loss': float(m['total_loss']),
                'accuracy': float(m['round_accuracy']),
                'correct_rounds': int(m['correct_rounds']),
                'total_rounds': int(m['total_rounds'])
            }
            for m in all_metrics
        ]
    }
    
    # 转换所有numpy类型
    model_info = convert_to_native(model_info)
    
    with open('bayes/vote_model_params.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("模型参数已保存到: bayes/vote_model_params.json")
    
    return best_params, all_metrics, summary

if __name__ == '__main__':
    best_params, all_metrics, summary = main()


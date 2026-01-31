"""
集成投票数量预测系统
结合Plackett-Luce模型和投票数量预测
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import kendalltau

def compute_mu(X, w):
    """计算实力分数"""
    return X @ w

def elimination_loss(mu, elimination_order):
    """Plackett-Luce淘汰损失"""
    alive = list(range(len(mu)))
    loss = 0.0
    for eliminated in elimination_order:
        idx = alive.index(eliminated)
        mu_alive = mu[alive]
        log_prob = -mu_alive[idx] - logsumexp(-mu_alive)
        loss -= log_prob
        alive.remove(eliminated)
    return loss

def objective(w, X, elimination_order, reg_lambda=0.01):
    """优化目标函数"""
    mu = compute_mu(X, w)
    loss = elimination_loss(mu, elimination_order)
    reg = reg_lambda * np.sum(w ** 2)
    return loss + reg

def prepare_season_data(fame_data, wiki_data, season):
    """准备季节数据"""
    if len(fame_data) != len(wiki_data):
        min_len = min(len(fame_data), len(wiki_data))
        fame_data = fame_data.iloc[:min_len]
        wiki_data = wiki_data.iloc[:min_len]
    
    season_mask = wiki_data['season'] == season
    season_indices = season_mask[season_mask].index
    
    if len(season_indices) == 0:
        return None, None, None, None
    
    season_fame = fame_data.iloc[season_indices].copy()
    season_wiki = wiki_data.iloc[season_indices].copy()
    season_data = season_fame.copy()
    season_data['season'] = season_wiki['season'].values
    
    if len(season_data) == 0:
        return None, None, None, None
    
    feature_cols = [
        'celebrity_age_during_season',
        'industry_Actor_Performer', 'industry_Influencer', 
        'industry_Model_Fashion', 'industry_Music', 
        'industry_Others', 'industry_Sports', 'industry_TV_Media',
        'fame_1'
    ]
    
    available_cols = [col for col in feature_cols if col in season_data.columns]
    X = season_data[available_cols].values.astype(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    elimination_ranks = season_data['elimination_order'].values
    sorted_indices = np.argsort(elimination_ranks)
    max_rank = elimination_ranks.max()
    eliminated_indices = sorted_indices[elimination_ranks[sorted_indices] < max_rank]
    
    if len(eliminated_indices) == 0:
        eliminated_indices = sorted_indices[:-1]
    
    elimination_order_indices = eliminated_indices.tolist()
    
    player_names = None
    if 'celebrity_name' in season_wiki.columns:
        player_names = season_wiki['celebrity_name'].values
    
    return X, elimination_order_indices, player_names, elimination_ranks

def train_model(fame_data, wiki_data, season):
    """
    训练Plackett-Luce模型，只返回模型参数
    
    返回:
    - 模型参数字典，包含 weights, X, elimination_order, player_names 等
    """
    X, elimination_order, player_names, elimination_ranks = prepare_season_data(
        fame_data, wiki_data, season
    )
    
    if X is None:
        return None
    
    n_players, n_features = X.shape
    
    # 训练Plackett-Luce模型（估计mu）
    print(f"\n{'='*60}")
    print(f"Season {season}: 训练模型")
    print(f"{'='*60}")
    print(f"  选手数量: {n_players}, 特征数量: {n_features}")
    
    w0 = np.random.randn(n_features) * 0.01
    result = minimize(objective, w0, args=(X, elimination_order, 0.01),
                     method='BFGS', options={'maxiter': 500, 'disp': False})
    
    if not result.success:
        print(f"  ⚠ 模型训练未完全收敛: {result.message}")
    
    w_optimized = result.x
    mu_estimated = compute_mu(X, w_optimized)
    
    # 验证mu的顺序
    first_eliminated = elimination_order[0]
    winner_idx = [i for i in range(n_players) if i not in elimination_order][0]
    
    if mu_estimated[first_eliminated] > mu_estimated[winner_idx]:
        print(f"  ⚠ 警告: mu顺序反了，正在修正...")
        mu_estimated = -mu_estimated
        w_optimized = -w_optimized
    
    print(f"  模型训练完成: mu范围=[{mu_estimated.min():.4f}, {mu_estimated.max():.4f}]")
    
    return {
        'season': season,
        'n_players': n_players,
        'weights': w_optimized,
        'mu': mu_estimated,
        'X': X,
        'elimination_order': elimination_order,
        'player_names': player_names,
        'elimination_ranks': elimination_ranks
    }

def predict_votes_for_round(model_result, alive_indices, vote_params=None):
    """
    根据模型参数预测当前轮次选手的投票数
    
    参数:
    - model_result: train_model返回的模型结果
    - alive_indices: 当前轮次还存活的选手索引列表
    - vote_params: 投票数映射参数 (a, b)，如果为None则使用默认值
    
    返回:
    - 预测的投票数数组（对应alive_indices的顺序）
    """
    if model_result is None:
        return None
    
    mu = model_result['mu']
    mu_alive = mu[alive_indices]
    
    # 使用默认参数或提供的参数
    if vote_params is not None:
        a, b = vote_params
    else:
        a, b = 1000.0, 100.0
    
    # 根据mu预测投票数：votes = a * exp(mu) + b
    votes_alive = a * np.exp(mu_alive) + b
    votes_alive = np.clip(votes_alive, 1, None)
    
    return votes_alive

def simulate_round_by_round_elimination(model_result, vote_params=None):
    """
    模拟逐轮淘汰过程（综艺场景还原）
    
    按照work.md的要求：
    1. 一个季度会进行多次比赛
    2. 每次比赛：模型预测当前存活选手的投票数据
    3. 淘汰投票数最低的选手
    4. 重复直到只剩一人
    
    参数:
    - model_result: train_model返回的模型结果
    - vote_params: 投票数映射参数 (a, b)，如果为None则使用默认值
    
    返回:
    - 预测的淘汰顺序（列表）
    """
    if model_result is None:
        return None
    
    n_players = model_result['n_players']
    alive_indices = list(range(n_players))
    predicted_elim_order = []
    
    # 进行多次比赛，直到只剩一人
    while len(alive_indices) > 1:
        # 预测当前轮次存活选手的投票数
        votes_alive = predict_votes_for_round(model_result, alive_indices, vote_params)
        
        if votes_alive is None:
            break
        
        # 找到投票数最低的选手（本轮被淘汰）
        min_vote_idx = np.argmin(votes_alive)
        eliminated_idx = alive_indices[min_vote_idx]
        
        # 记录淘汰顺序
        predicted_elim_order.append(eliminated_idx)
        
        # 从存活列表中移除被淘汰的选手
        alive_indices.remove(eliminated_idx)
    
    return predicted_elim_order

def evaluate_model_consistency(model_result, vote_params=None):
    """
    评估模型的一致性：使用逐轮淘汰模拟，计算Kendall's tau
    
    参数:
    - model_result: train_model返回的模型结果
    - vote_params: 投票数映射参数 (a, b)，如果为None则使用默认值
    
    返回:
    - 字典，包含：
      - 'predicted_elim_order': 预测的淘汰顺序
      - 'elimination_order_tau': Kendall's tau 一致性系数
      - 'elimination_order_p': p值
      - 'elimination_order_match': 是否完全匹配
    """
    if model_result is None:
        return None
    
    n_players = model_result['n_players']
    elimination_order = model_result['elimination_order']
    
    # 模拟逐轮淘汰过程
    predicted_elim_order = simulate_round_by_round_elimination(model_result, vote_params)
    
    if predicted_elim_order is None or len(predicted_elim_order) != len(elimination_order):
        return {
            'predicted_elim_order': predicted_elim_order,
            'elimination_order_tau': 0.0,
            'elimination_order_p': 1.0,
            'elimination_order_match': False
        }
    
    # 计算排名
    pred_ranks = np.zeros(n_players)
    actual_ranks = np.zeros(n_players)
    
    # 为预测的淘汰顺序分配排名（第1个被淘汰的是第1名，最后被淘汰的是倒数第2名）
    for i, idx in enumerate(predicted_elim_order):
        pred_ranks[idx] = i + 1
    
    # 为实际的淘汰顺序分配排名
    for i, idx in enumerate(elimination_order):
        actual_ranks[idx] = i + 1
    
    # 冠军的排名（最后一名，最高排名）
    pred_winner = [i for i in range(n_players) if i not in predicted_elim_order][0]
    actual_winner = [i for i in range(n_players) if i not in elimination_order][0]
    pred_ranks[pred_winner] = len(elimination_order) + 1
    actual_ranks[actual_winner] = len(elimination_order) + 1
    
    # 计算Kendall's tau
    tau, p_value = kendalltau(pred_ranks, actual_ranks)
    
    return {
        'predicted_elim_order': predicted_elim_order,
        'elimination_order_tau': tau,
        'elimination_order_p': p_value,
        'elimination_order_match': (predicted_elim_order == elimination_order)
    }


def main():
    """主函数：对所有季节进行训练和预测"""
    print("="*60)
    print("加载数据...")
    print("="*60)
    
    fame_data = pd.read_csv('../2026_MCM_Problem_Fame_Data.csv')
    wiki_data = pd.read_csv('../celebrity_wikipedia_stats.csv')
    
    print(f"特征数据: {len(fame_data)} 行")
    print(f"Wiki数据: {len(wiki_data)} 行")
    
    seasons = sorted(wiki_data['season'].unique())
    print(f"\n找到 {len(seasons)} 个季节: {seasons}")
    
    all_results = []
    
    for season in seasons:
        try:
            # 只训练模型，获取模型参数（不输出结果，不进行评估）
            model_result = train_model(fame_data, wiki_data, season)
            if model_result is not None:
                all_results.append(model_result)
        except Exception as e:
            print(f"\nSeason {season} 处理失败: {str(e)}")
            continue
    
    # 评估所有模型的一致性
    if len(all_results) > 0:
        print(f"\n{'='*60}")
        print("评估模型一致性")
        print(f"{'='*60}")
        
        consistency_metrics = []
        
        for model_result in all_results:
            season = model_result['season']
            # 评估模型一致性（使用逐轮淘汰模拟）
            eval_result = evaluate_model_consistency(model_result)
            
            if eval_result and 'elimination_order_tau' in eval_result:
                tau = eval_result['elimination_order_tau']
                consistency_metrics.append({
                    'season': season,
                    'kendall_tau': tau,
                    'kendall_p': eval_result.get('elimination_order_p'),
                    'elimination_match': eval_result.get('elimination_order_match', False)
                })
        
        # 计算并显示总体一致性统计
        if consistency_metrics:
            metrics_df = pd.DataFrame(consistency_metrics)
            tau_values = metrics_df['kendall_tau'].values
            
            tau_mean = np.mean(tau_values)
            tau_std = np.std(tau_values)
            tau_median = np.median(tau_values)
            tau_min = np.min(tau_values)
            tau_max = np.max(tau_values)
            positive_tau_count = np.sum(tau_values > 0)
            negative_tau_count = np.sum(tau_values < 0)
            perfect_match_count = metrics_df['elimination_match'].sum()
            
            print(f"\nKendall's tau (排序一致性):")
            print(f"  平均值: {tau_mean:.4f} ± {tau_std:.4f}")
            print(f"  中位数: {tau_median:.4f}")
            print(f"  范围: [{tau_min:.4f}, {tau_max:.4f}]")
            print(f"  正相关季节数: {positive_tau_count}/{len(tau_values)} ({100*positive_tau_count/len(tau_values):.1f}%)")
            print(f"  负相关季节数: {negative_tau_count}/{len(tau_values)} ({100*negative_tau_count/len(tau_values):.1f}%)")
            print(f"  完全匹配季节数: {perfect_match_count}/{len(tau_values)} ({100*perfect_match_count/len(tau_values):.1f}%)")
    
    print(f"\n{'='*60}")
    print("完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

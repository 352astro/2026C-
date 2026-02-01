"""
单独训练贝叶斯模型并评估准确率和一致性
基于vote_prediction.py中的bayesian_vote_prediction方法
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import kendalltau

def get_mu_sigma(X, params, n_feats):
    """
    根据ARIMA/optimize.py的方式计算均值和方差
    mu = w1*x1 + w2*x2 ... + b
    sigma = log(1 + exp(w'*x + b'))
    """
    w_mu = params[0:n_feats]
    b_mu = params[n_feats]
    w_sig = params[n_feats + 1: 2 * n_feats + 1]
    b_sig = params[2 * n_feats + 1]

    mu = np.dot(X, w_mu) + b_mu
    y2 = np.dot(X, w_sig) + b_sig
    # 数值稳定的 log1p(exp)
    sigma = np.where(y2 > 20, y2, np.log1p(np.exp(np.clip(y2, -20, 20))))
    return mu, sigma

def pl_elimination_loss(votes, elimination_order):
    """
    Plackett-Luce模型的淘汰损失
    votes: (n_players,) 投票数量
    elimination_order: list, 淘汰顺序（索引列表）
    """
    mu = np.log(votes + 1e-10)
    alive = list(range(len(votes)))
    loss = 0.0
    
    for eliminated in elimination_order:
        idx = alive.index(eliminated)
        mu_alive = mu[alive]
        log_prob = -mu_alive[idx] - logsumexp(-mu_alive)
        loss -= log_prob
        alive.remove(eliminated)
    
    return loss

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

def bayesian_vote_prediction_with_learned_prior(X, elimination_order, params, n_feats, n_samples=100):
    """
    贝叶斯方法预测投票数量（使用从特征学习到的先验参数）
    
    参数:
    - X: 特征矩阵 (n_players, n_feats)
    - elimination_order: 淘汰顺序（索引列表）
    - params: 学习到的参数 [w_mu, b_mu, w_sig, b_sig]
    - n_feats: 特征数量
    - n_samples: MCMC采样数量（用于不确定性估计，默认100）
    
    返回:
    - 字典，包含投票数预测和不确定性估计
    """
    n_players = len(elimination_order) + 1
    
    # 从特征计算每个选手的先验均值和方差
    mu_prior, sigma_prior = get_mu_sigma(X, params, n_feats)
    
    # 将mu和sigma转换为投票数的先验参数
    # 假设投票数服从对数正态分布：log(votes) ~ N(mu_prior, sigma_prior^2)
    # 所以 votes 的先验均值 = exp(mu_prior + sigma_prior^2/2)
    # 先验标准差 = exp(mu_prior + sigma_prior^2/2) * sqrt(exp(sigma_prior^2) - 1)
    prior_means = np.exp(mu_prior + 0.5 * sigma_prior ** 2)
    prior_stds = prior_means * np.sqrt(np.exp(sigma_prior ** 2) - 1)
    # 防止过小或过大
    prior_means = np.clip(prior_means, 100, 100000)
    prior_stds = np.clip(prior_stds, 50, 50000)
    
    def objective(votes):
        likelihood = pl_elimination_loss(votes, elimination_order)
        log_votes = np.log(votes + 1e-10)
        # 每个选手使用自己的先验参数
        prior = np.sum((log_votes - mu_prior) ** 2 / (2 * (sigma_prior ** 2 + 1e-10)))
        return likelihood + prior
    
    # 使用学习到的先验均值作为初始值
    votes0 = prior_means.copy()
    votes0 = np.clip(votes0, 1, None)
    
    result = minimize(objective, votes0, method='L-BFGS-B', 
                     bounds=[(1, None)] * n_players, options={'maxiter': 1000})
    votes_map = result.x
    
    # 估计不确定性
    votes_samples = []
    for _ in range(min(50, n_samples)):
        # 从先验分布采样初始值
        votes_init = np.random.lognormal(mu_prior, sigma_prior)
        votes_init = np.clip(votes_init, 1, None)
        res = minimize(objective, votes_init, method='L-BFGS-B', 
                      bounds=[(1, None)] * n_players, options={'maxiter': 200})
        if res.success:
            votes_samples.append(res.x)
    
    if len(votes_samples) > 0:
        votes_samples = np.array(votes_samples)
        votes_mean = votes_samples.mean(axis=0)
        votes_std = votes_samples.std(axis=0)
        votes_ci_lower = np.percentile(votes_samples, 2.5, axis=0)
        votes_ci_upper = np.percentile(votes_samples, 97.5, axis=0)
    else:
        votes_mean = votes_map
        votes_std = prior_stds  # 使用先验标准差
        votes_ci_lower = np.clip(votes_mean - 1.96 * votes_std, 1, None)
        votes_ci_upper = votes_mean + 1.96 * votes_std
    
    return {
        'votes_map': votes_map,
        'votes_mean': votes_mean,
        'votes_std': votes_std,
        'votes_ci_lower': votes_ci_lower,
        'votes_ci_upper': votes_ci_upper,
        'mu_prior': mu_prior,
        'sigma_prior': sigma_prior,
        'prior_means': prior_means,
        'prior_stds': prior_stds,
        'success': result.success
    }

def bayesian_vote_prediction(elimination_order, prior_mean=1000, prior_std=500, n_samples=100):
    """
    贝叶斯方法预测投票数量（提供不确定性估计）- 使用固定先验参数（向后兼容）
    
    参数:
    - elimination_order: 淘汰顺序（索引列表）
    - prior_mean: 投票数先验均值（默认1000）
    - prior_std: 投票数先验标准差（默认500）
    - n_samples: MCMC采样数量（用于不确定性估计，默认100）
    
    返回:
    - 字典，包含投票数预测和不确定性估计
    """
    n_players = len(elimination_order) + 1
    
    def objective(votes):
        likelihood = pl_elimination_loss(votes, elimination_order)
        log_votes = np.log(votes + 1e-10)
        prior = np.sum((log_votes - np.log(prior_mean)) ** 2) / (2 * prior_std ** 2)
        return likelihood + prior
    
    votes0 = np.ones(n_players) * prior_mean
    result = minimize(objective, votes0, method='L-BFGS-B', 
                     bounds=[(1, None)] * n_players, options={'maxiter': 1000})
    votes_map = result.x
    
    # 估计不确定性
    votes_samples = []
    for _ in range(min(50, n_samples)):
        votes_init = votes_map * (1 + np.random.normal(0, 0.1, n_players))
        votes_init = np.clip(votes_init, 1, None)
        res = minimize(objective, votes_init, method='L-BFGS-B', 
                      bounds=[(1, None)] * n_players, options={'maxiter': 200})
        if res.success:
            votes_samples.append(res.x)
    
    if len(votes_samples) > 0:
        votes_samples = np.array(votes_samples)
        votes_mean = votes_samples.mean(axis=0)
        votes_std = votes_samples.std(axis=0)
        votes_ci_lower = np.percentile(votes_samples, 2.5, axis=0)
        votes_ci_upper = np.percentile(votes_samples, 97.5, axis=0)
    else:
        votes_mean = votes_map
        votes_std = votes_map * 0.1
        votes_ci_lower = votes_map * 0.9
        votes_ci_upper = votes_map * 1.1
    
    return {
        'votes_map': votes_map,
        'votes_mean': votes_mean,
        'votes_std': votes_std,
        'votes_ci_lower': votes_ci_lower,
        'votes_ci_upper': votes_ci_upper,
        'success': result.success
    }

def load_season_data_from_c_data(season=1):
    """从2026_MCM_Problem_C_Data.csv加载指定季节的数据"""
    data = pd.read_csv('../2026_MCM_Problem_C_Data.csv')
    season_data = data[data['season'] == season].copy()
    return season_data

def calculate_judge_score_percentage(season_data, week, alive_indices):
    """计算指定周存活选手的评委分数百分比"""
    judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
    
    scores = []
    for idx in alive_indices:
        row = season_data.iloc[idx]
        week_scores = []
        for col in judge_cols:
            if col in season_data.columns:
                score = row[col]
                if pd.notna(score) and score != 0 and score != 'N/A':
                    try:
                        week_scores.append(float(score))
                    except:
                        pass
        if len(week_scores) > 0:
            avg_score = np.mean(week_scores)
        else:
            avg_score = 0.0
        scores.append(avg_score)
    
    scores = np.array(scores)
    total_score = scores.sum()
    if total_score > 0:
        percentages = scores / total_score
    else:
        percentages = np.ones(len(scores)) / len(scores)
    
    return percentages

def simulate_round_by_round_elimination_bayesian(bayesian_result, season_data=None, current_week=1, return_round_details=False):
    """
    使用贝叶斯模型预测的投票数进行逐轮淘汰模拟
    
    参数:
    - bayesian_result: bayesian_vote_prediction返回的结果
    - season_data: 季节数据DataFrame（包含评委分数）
    - current_week: 当前周数（从1开始）
    - return_round_details: 是否返回每轮详细信息
    
    返回:
    - 如果return_round_details=False: 预测的淘汰顺序（列表）
    - 如果return_round_details=True: 字典，包含'elim_order'和'round_details'
    """
    if bayesian_result is None:
        return None
    
    # 使用votes_mean作为预测值
    votes_all = bayesian_result.get('votes_mean', bayesian_result.get('votes_map'))
    if votes_all is None:
        return None
    
    n_players = len(votes_all)
    alive_indices = list(range(n_players))
    predicted_elim_order = []
    round_details = []
    week = current_week
    
    # 进行多次比赛，直到只剩一人
    while len(alive_indices) > 1:
        # 1. 计算评委分数百分比
        if season_data is not None:
            judge_percentages = calculate_judge_score_percentage(season_data, week, alive_indices)
        else:
            judge_percentages = np.ones(len(alive_indices)) / len(alive_indices)
        
        # 2. 获取当前存活选手的投票数
        votes_alive = votes_all[alive_indices]
        
        # 3. 计算票数百分比
        total_votes = votes_alive.sum()
        if total_votes > 0:
            vote_percentages = votes_alive / total_votes
        else:
            vote_percentages = np.ones(len(votes_alive)) / len(votes_alive)
        
        # 4. 两个百分比相加作为排名依据
        combined_scores = judge_percentages + vote_percentages
        
        # 5. 找到排名最低的选手（本轮被淘汰）
        min_score_idx = np.argmin(combined_scores)
        eliminated_idx = alive_indices[min_score_idx]
        
        # 记录淘汰顺序
        predicted_elim_order.append(eliminated_idx)
        
        # 记录每轮详细信息
        if return_round_details:
            round_details.append({
                'round': len(predicted_elim_order),
                'week': week,
                'eliminated_idx': eliminated_idx,
                'alive_indices': alive_indices.copy(),
                'judge_percentages': judge_percentages.copy(),
                'vote_percentages': vote_percentages.copy(),
                'combined_scores': combined_scores.copy(),
                'votes_alive': votes_alive.copy()
            })
        
        # 从存活列表中移除被淘汰的选手
        alive_indices.remove(eliminated_idx)
        
        # 进入下一周
        week += 1
    
    if return_round_details:
        return {
            'elim_order': predicted_elim_order,
            'round_details': round_details
        }
    else:
        return predicted_elim_order

def evaluate_bayesian_model_consistency(bayesian_result, elimination_order, season_data=None):
    """
    评估贝叶斯模型的一致性：使用逐轮淘汰模拟，计算Kendall's tau和每轮淘汰准确性
    
    参数:
    - bayesian_result: bayesian_vote_prediction返回的结果
    - elimination_order: 实际淘汰顺序
    - season_data: 季节数据DataFrame（包含评委分数）
    
    返回:
    - 字典，包含一致性评估结果
    """
    if bayesian_result is None:
        return None
    
    n_players = len(elimination_order) + 1
    
    # 模拟逐轮淘汰过程（获取详细信息）
    sim_result = simulate_round_by_round_elimination_bayesian(
        bayesian_result, season_data, current_week=1, return_round_details=True
    )
    
    if sim_result is None:
        return {
            'predicted_elim_order': None,
            'elimination_order_tau': 0.0,
            'elimination_order_p': 1.0,
            'elimination_order_match': False,
            'round_accuracy': [],
            'overall_round_accuracy': 0.0,
            'round_details': []
        }
    
    predicted_elim_order = sim_result['elim_order']
    round_details = sim_result['round_details']
    
    if predicted_elim_order is None or len(predicted_elim_order) != len(elimination_order):
        return {
            'predicted_elim_order': predicted_elim_order,
            'elimination_order_tau': 0.0,
            'elimination_order_p': 1.0,
            'elimination_order_match': False,
            'round_accuracy': [],
            'overall_round_accuracy': 0.0,
            'round_details': round_details
        }
    
    # 计算每轮淘汰准确性
    round_accuracy = []
    correct_rounds = 0
    
    for round_num in range(len(predicted_elim_order)):
        pred_eliminated = predicted_elim_order[round_num]
        actual_eliminated = elimination_order[round_num]
        is_correct = (pred_eliminated == actual_eliminated)
        round_accuracy.append({
            'round': round_num + 1,
            'predicted_eliminated': pred_eliminated,
            'actual_eliminated': actual_eliminated,
            'is_correct': is_correct
        })
        if is_correct:
            correct_rounds += 1
    
    # 计算总体每轮淘汰准确率
    overall_round_accuracy = correct_rounds / len(predicted_elim_order) if len(predicted_elim_order) > 0 else 0.0
    
    # 计算排名
    pred_ranks = np.zeros(n_players)
    actual_ranks = np.zeros(n_players)
    
    # 为预测的淘汰顺序分配排名
    for i, idx in enumerate(predicted_elim_order):
        pred_ranks[idx] = i + 1
    
    # 为实际的淘汰顺序分配排名
    for i, idx in enumerate(elimination_order):
        actual_ranks[idx] = i + 1
    
    # 冠军的排名
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
        'elimination_order_match': (predicted_elim_order == elimination_order),
        'round_accuracy': round_accuracy,
        'overall_round_accuracy': overall_round_accuracy,
        'round_details': round_details
    }

def optimize_mu_sigma_parameters(X, elimination_order, n_feats, season_data=None, verbose=False):
    """
    优化mu和sigma的参数（w_mu, b_mu, w_sig, b_sig）
    按照ARIMA/optimize.py的方式，通过优化Plackett-Luce损失来学习参数
    
    参数:
    - X: 特征矩阵 (n_players, n_feats)
    - elimination_order: 淘汰顺序
    - n_feats: 特征数量
    - season_data: 季节数据（用于评估，可选）
    - verbose: 是否打印详细信息
    
    返回:
    - 优化后的参数数组 [w_mu, b_mu, w_sig, b_sig]
    """
    n_players = len(elimination_order) + 1
    
    def objective(params):
        """
        目标函数：Plackett-Luce损失
        """
        mu, sigma = get_mu_sigma(X, params, n_feats)
        
        # 将mu转换为投票数的对数空间
        # 使用mu作为log(votes)的估计，sigma作为不确定性
        # 为了优化，我们使用mu的指数作为投票数的估计
        votes_est = np.exp(mu)
        votes_est = np.clip(votes_est, 1, None)
        
        # Plackett-Luce损失
        loss = pl_elimination_loss(votes_est, elimination_order)
        
        # 添加正则化项防止过拟合
        reg = 0.01 * (np.sum(params ** 2))
        
        return loss + reg
    
    # 初始化参数: [w_mu(n_feats), b_mu, w_sig(n_feats), b_sig]
    init_params = np.zeros(2 * n_feats + 2)
    # 给mu的权重一个小的初始值
    init_params[0:n_feats] = np.random.randn(n_feats) * 0.01
    # 给sigma的权重一个小的初始值
    init_params[n_feats + 1: 2 * n_feats + 1] = np.random.randn(n_feats) * 0.01
    
    if verbose:
        print(f"  开始优化mu和sigma参数...")
        print(f"  参数维度: {len(init_params)} (w_mu: {n_feats}, b_mu: 1, w_sig: {n_feats}, b_sig: 1)")
    
    result = minimize(
        objective,
        init_params,
        method='L-BFGS-B',
        options={'maxiter': 500, 'disp': verbose}
    )
    
    if verbose:
        if result.success:
            print(f"  参数优化成功")
        else:
            print(f"  参数优化未完全收敛: {result.message}")
    
    return result.x

def optimize_prior_parameters(elimination_order, season_data=None, 
                              prior_mean_range=(500, 2000, 500), 
                              prior_std_range=(200, 1000, 200),
                              n_samples=50, verbose=False):
    """
    优化先验参数（prior_mean和prior_std）
    通过网格搜索找到使一致性最好的先验参数
    
    参数:
    - elimination_order: 淘汰顺序
    - season_data: 季节数据（用于一致性评估）
    - prior_mean_range: (min, max, step) 先验均值搜索范围
    - prior_std_range: (min, max, step) 先验标准差搜索范围
    - n_samples: 每次训练的采样数量（减少以加快搜索）
    - verbose: 是否打印详细信息
    
    返回:
    - 最佳先验参数字典 {'prior_mean': ..., 'prior_std': ..., 'best_score': ...}
    """
    if verbose:
        print(f"  开始优化先验参数...")
        print(f"  prior_mean范围: {prior_mean_range[0]} - {prior_mean_range[1]} (步长: {prior_mean_range[2]})")
        print(f"  prior_std范围: {prior_std_range[0]} - {prior_std_range[1]} (步长: {prior_std_range[2]})")
    
    best_score = -np.inf
    best_params = {'prior_mean': 1000, 'prior_std': 500}
    
    prior_mean_values = np.arange(prior_mean_range[0], prior_mean_range[1] + prior_mean_range[2], prior_mean_range[2])
    prior_std_values = np.arange(prior_std_range[0], prior_std_range[1] + prior_std_range[2], prior_std_range[2])
    
    total_combinations = len(prior_mean_values) * len(prior_std_values)
    current = 0
    
    for prior_mean in prior_mean_values:
        for prior_std in prior_std_values:
            current += 1
            if verbose and current % 10 == 0:
                print(f"  进度: {current}/{total_combinations} (当前: mean={prior_mean}, std={prior_std})")
            
            try:
                # 训练模型
                bayesian_result = bayesian_vote_prediction(
                    elimination_order,
                    prior_mean=prior_mean,
                    prior_std=prior_std,
                    n_samples=n_samples
                )
                
                # 评估一致性
                eval_result = evaluate_bayesian_model_consistency(
                    bayesian_result, elimination_order, season_data
                )
                
                if eval_result:
                    # 使用Kendall's tau作为评分标准
                    score = eval_result.get('elimination_order_tau', -1.0)
                    # 也可以结合每轮准确率: score = tau * 0.7 + round_accuracy * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'prior_mean': prior_mean,
                            'prior_std': prior_std,
                            'best_score': best_score,
                            'round_accuracy': eval_result.get('overall_round_accuracy', 0.0)
                        }
            except Exception as e:
                if verbose:
                    print(f"    参数组合 (mean={prior_mean}, std={prior_std}) 失败: {str(e)}")
                continue
    
    if verbose:
        print(f"  优化完成: 最佳参数 prior_mean={best_params['prior_mean']}, prior_std={best_params['prior_std']}")
        print(f"  最佳Kendall's tau: {best_params['best_score']:.4f}")
        print(f"  对应每轮准确率: {best_params.get('round_accuracy', 0.0):.4f}")
    
    return best_params

def train_bayesian_model_for_season(season, prior_mean=1000, prior_std=500, n_samples=100, 
                                    use_learned_prior=True, optimize_prior=True, verbose=True):
    """
    为指定季节训练贝叶斯模型并评估
    
    参数:
    - season: 季节编号
    - prior_mean: 投票数先验均值（当use_learned_prior=False时使用）
    - prior_std: 投票数先验标准差（当use_learned_prior=False时使用）
    - n_samples: MCMC采样数量
    - use_learned_prior: 是否使用从特征学习到的先验参数（默认True，推荐）
    - optimize_prior: 是否优化mu和sigma的参数（默认True）
    - verbose: 是否打印详细信息
    
    返回:
    - 字典，包含模型结果和评估结果
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Season {season}: 训练贝叶斯模型")
        print(f"{'='*60}")
    
    # 加载数据
    fame_data = pd.read_csv('../2026_MCM_Problem_Fame_Data.csv')
    wiki_data = pd.read_csv('../celebrity_wikipedia_stats.csv')
    c_data = load_season_data_from_c_data(season)
    
    # 准备数据
    X, elimination_order, player_names, elimination_ranks = prepare_season_data(
        fame_data, wiki_data, season
    )
    
    if X is None:
        if verbose:
            print("数据准备失败！")
        return None
    
    n_players, n_feats = X.shape
    learned_params = None
    
    # 优化mu和sigma的参数（如果需要）
    if use_learned_prior:
        if optimize_prior:
            if verbose:
                print(f"  选手数量: {n_players}, 特征数量: {n_feats}")
                print(f"  开始优化mu和sigma参数（按照ARIMA/optimize.py方式）...")
            
            learned_params = optimize_mu_sigma_parameters(
                X, elimination_order, n_feats, 
                season_data=c_data, verbose=verbose
            )
            
            if verbose:
                mu_test, sigma_test = get_mu_sigma(X, learned_params, n_feats)
                print(f"  参数优化完成")
                print(f"  mu范围: [{mu_test.min():.4f}, {mu_test.max():.4f}]")
                print(f"  sigma范围: [{sigma_test.min():.4f}, {sigma_test.max():.4f}]")
        else:
            # 即使不优化，也使用默认初始化的参数来计算mu和sigma
            if verbose:
                print(f"  选手数量: {n_players}, 特征数量: {n_feats}")
                print(f"  使用默认初始化的mu和sigma参数（不进行优化）...")
            
            # 初始化参数: [w_mu(n_feats), b_mu, w_sig(n_feats), b_sig]
            learned_params = np.zeros(2 * n_feats + 2)
            # 给mu的权重一个小的初始值
            learned_params[0:n_feats] = np.random.randn(n_feats) * 0.01
            # 给sigma的权重一个小的初始值
            learned_params[n_feats + 1: 2 * n_feats + 1] = np.random.randn(n_feats) * 0.01
            
            if verbose:
                mu_test, sigma_test = get_mu_sigma(X, learned_params, n_feats)
                print(f"  mu范围: [{mu_test.min():.4f}, {mu_test.max():.4f}]")
                print(f"  sigma范围: [{sigma_test.min():.4f}, {sigma_test.max():.4f}]")
    
    if verbose:
        print(f"  选手数量: {n_players}")
        if use_learned_prior:
            print(f"  使用从特征学习到的先验参数（mu和sigma）")
        else:
            print(f"  使用固定先验参数: mean={prior_mean}, std={prior_std}")
        print(f"  采样数量: {n_samples}")
        print(f"  开始训练...")
    
    # 训练贝叶斯模型
    if use_learned_prior and learned_params is not None:
        # 使用学习到的先验参数（从特征计算mu和sigma）
        bayesian_result = bayesian_vote_prediction_with_learned_prior(
            X, elimination_order, learned_params, n_feats, n_samples=n_samples
        )
    else:
        # 使用固定先验参数（向后兼容，不推荐）
        bayesian_result = bayesian_vote_prediction(
            elimination_order, 
            prior_mean=prior_mean, 
            prior_std=prior_std, 
            n_samples=n_samples
        )
    
    if verbose:
        votes_mean = bayesian_result.get('votes_mean', bayesian_result.get('votes_map'))
        if votes_mean is not None:
            print(f"  训练完成: 投票数范围=[{votes_mean.min():.2f}, {votes_mean.max():.2f}]")
            print(f"  优化状态: {'成功' if bayesian_result.get('success', False) else '未完全收敛'}")
    
    # 评估一致性
    if verbose:
        print(f"\n  开始评估一致性...")
    
    eval_result = evaluate_bayesian_model_consistency(
        bayesian_result, elimination_order, season_data=c_data
    )
    
    if eval_result:
        tau = eval_result['elimination_order_tau']
        round_accuracy = eval_result.get('overall_round_accuracy', 0.0)
        
        if verbose:
            print(f"\n  {'='*60}")
            print(f"  一致性评估结果")
            print(f"  {'='*60}")
            print(f"  Kendall's tau: {tau:.4f}")
            print(f"  p值: {eval_result.get('elimination_order_p', 0):.4f}")
            print(f"  完全匹配: {eval_result.get('elimination_order_match', False)}")
            correct_rounds = sum(1 for r in eval_result.get('round_accuracy', []) if r['is_correct'])
            print(f"  每轮淘汰准确率: {round_accuracy:.4f} ({correct_rounds}/{len(elimination_order)})")
            
            # 显示每轮淘汰详情
            round_accuracy_list = eval_result.get('round_accuracy', [])
            if round_accuracy_list:
                print(f"\n  每轮淘汰详情:")
                for round_info in round_accuracy_list:
                    status = "✓" if round_info['is_correct'] else "✗"
                    pred_name = player_names[round_info['predicted_eliminated']] if player_names is not None else f"Player {round_info['predicted_eliminated']}"
                    actual_name = player_names[round_info['actual_eliminated']] if player_names is not None else f"Player {round_info['actual_eliminated']}"
                    print(f"    第{round_info['round']}轮: {status} 预测={pred_name}, 实际={actual_name}")
    
    result = {
        'season': season,
        'bayesian_result': bayesian_result,
        'evaluation': eval_result,
        'player_names': player_names,
        'elimination_order': elimination_order,
        'X': X,
        'n_feats': n_feats
    }
    
    if use_learned_prior:
        result['learned_params'] = learned_params
        if learned_params is not None:
            mu_prior, sigma_prior = get_mu_sigma(X, learned_params, n_feats)
            result['mu_prior'] = mu_prior
            result['sigma_prior'] = sigma_prior
    else:
        result['prior_mean'] = prior_mean
        result['prior_std'] = prior_std
    
    return result

def main():
    """主函数：对所有季节进行贝叶斯模型训练和评估"""
    print("="*60)
    print("贝叶斯模型训练和评估")
    print("="*60)
    
    # 加载数据
    fame_data = pd.read_csv('../2026_MCM_Problem_Fame_Data.csv')
    wiki_data = pd.read_csv('../celebrity_wikipedia_stats.csv')
    
    print(f"特征数据: {len(fame_data)} 行")
    print(f"Wiki数据: {len(wiki_data)} 行")
    
    seasons = sorted(wiki_data['season'].unique())
    print(f"\n找到 {len(seasons)} 个季节: {seasons}")
    
    # 模型参数
    use_learned_prior = True   # 是否使用从特征学习到的先验参数（推荐True）
    optimize_prior = True     # 是否优化mu和sigma的参数（当use_learned_prior=True时）
    prior_mean = 1000          # 固定先验均值（当use_learned_prior=False时使用）
    prior_std = 500            # 固定先验标准差（当use_learned_prior=False时使用）
    n_samples = 100
    
    print(f"\n模型参数:")
    if use_learned_prior:
        print(f"  先验参数: 从特征学习（按照ARIMA/optimize.py方式）")
        print(f"  优化mu/sigma参数: {'是' if optimize_prior else '否（使用默认初始化）'}")
    else:
        print(f"  先验参数: 固定值")
        print(f"  先验均值: {prior_mean}")
        print(f"  先验标准差: {prior_std}")
    print(f"  采样数量: {n_samples}")
    
    all_results = []
    consistency_metrics = []
    
    for season in seasons:
        try:
            result = train_bayesian_model_for_season(
                season, 
                prior_mean=prior_mean, 
                prior_std=prior_std, 
                n_samples=n_samples,
                use_learned_prior=use_learned_prior,
                optimize_prior=optimize_prior,
                verbose=True
            )
            
            if result and result.get('evaluation'):
                eval_result = result['evaluation']
                tau = eval_result.get('elimination_order_tau', 0.0)
                round_accuracy = eval_result.get('overall_round_accuracy', 0.0)
                
                consistency_metrics.append({
                    'season': season,
                    'kendall_tau': tau,
                    'kendall_p': eval_result.get('elimination_order_p'),
                    'elimination_match': eval_result.get('elimination_order_match', False),
                    'round_accuracy': round_accuracy
                })
            
            all_results.append(result)
            
        except Exception as e:
            print(f"\nSeason {season} 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总所有季节的统计
    if consistency_metrics:
        print(f"\n{'='*60}")
        print("总体一致性统计")
        print(f"{'='*60}")
        
        metrics_df = pd.DataFrame(consistency_metrics)
        tau_values = metrics_df['kendall_tau'].values
        round_accuracy_values = metrics_df['round_accuracy'].values
        
        tau_mean = np.mean(tau_values)
        tau_std = np.std(tau_values)
        tau_median = np.median(tau_values)
        tau_min = np.min(tau_values)
        tau_max = np.max(tau_values)
        positive_tau_count = np.sum(tau_values > 0)
        negative_tau_count = np.sum(tau_values < 0)
        perfect_match_count = metrics_df['elimination_match'].sum()
        
        round_accuracy_mean = np.mean(round_accuracy_values)
        round_accuracy_std = np.std(round_accuracy_values)
        round_accuracy_median = np.median(round_accuracy_values)
        round_accuracy_min = np.min(round_accuracy_values)
        round_accuracy_max = np.max(round_accuracy_values)
        
        print(f"\nKendall's tau (排序一致性):")
        print(f"  平均值: {tau_mean:.4f} ± {tau_std:.4f}")
        print(f"  中位数: {tau_median:.4f}")
        print(f"  范围: [{tau_min:.4f}, {tau_max:.4f}]")
        print(f"  正相关季节数: {positive_tau_count}/{len(tau_values)} ({100*positive_tau_count/len(tau_values):.1f}%)")
        print(f"  负相关季节数: {negative_tau_count}/{len(tau_values)} ({100*negative_tau_count/len(tau_values):.1f}%)")
        print(f"  完全匹配季节数: {perfect_match_count}/{len(tau_values)} ({100*perfect_match_count/len(tau_values):.1f}%)")
        
        print(f"\n每轮淘汰准确率:")
        print(f"  平均值: {round_accuracy_mean:.4f} ± {round_accuracy_std:.4f}")
        print(f"  中位数: {round_accuracy_median:.4f}")
        print(f"  范围: [{round_accuracy_min:.4f}, {round_accuracy_max:.4f}]")
        
        # 保存结果
        metrics_df.to_csv('bayesian_model_consistency_metrics.csv', index=False)
        print(f"\n一致性指标已保存到: bayesian_model_consistency_metrics.csv")
    
    print(f"\n{'='*60}")
    print("完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    # 运行所有季节的训练和评估
    main()
    
    # 或者只训练单个季节
    # result = train_bayesian_model_for_season(season=1, verbose=True)


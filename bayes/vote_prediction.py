"""
从淘汰顺序反推投票数量的模型
提供多种方法：贝叶斯方法、最大似然估计、两阶段回归方法
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import kendalltau

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

def bayesian_vote_prediction(X, elimination_order, prior_mean=1000, prior_std=500, n_samples=100):
    """
    贝叶斯方法预测投票数量（提供不确定性估计）
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

def mle_vote_prediction(elimination_order, votes_bounds=(100, 100000)):
    """
    最大似然估计投票数量
    """
    n_players = len(elimination_order) + 1
    
    def objective(votes):
        return pl_elimination_loss(votes, elimination_order)
    
    votes0 = np.random.uniform(votes_bounds[0], votes_bounds[1], n_players)
    result = minimize(objective, votes0, method='L-BFGS-B',
                     bounds=[votes_bounds] * n_players, options={'maxiter': 2000})
    
    return {
        'votes': result.x,
        'loss': result.fun,
        'success': result.success
    }

def two_stage_vote_prediction(X, elimination_order, mu_from_features=None):
    """
    两阶段方法（推荐）：
    1. 估计mu（实力分数）
    2. 将mu映射到投票数量
    """
    n_players = len(elimination_order) + 1
    
    # 阶段1: 估计mu
    if mu_from_features is None:
        # 从elimination_order估计mu
        mu = np.zeros(n_players)
        for i, idx in enumerate(elimination_order):
            mu[idx] = -i
        winner_idx = [i for i in range(n_players) if i not in elimination_order][0]
        mu[winner_idx] = len(elimination_order) + 1
        
        def objective_mu(mu_vec):
            alive = list(range(n_players))
            loss = 0.0
            for eliminated in elimination_order:
                idx = alive.index(eliminated)
                mu_alive = mu_vec[alive]
                log_prob = -mu_alive[idx] - logsumexp(-mu_alive)
                loss -= log_prob
                alive.remove(eliminated)
            return loss
        
        result_mu = minimize(objective_mu, mu, method='BFGS', options={'maxiter': 500})
        mu_estimated = result_mu.x
        
        # 验证mu顺序
        mu_sorted = np.argsort(mu_estimated)
        expected_order = elimination_order + [winner_idx]
        if np.array_equal(mu_sorted, expected_order[::-1]):
            mu_estimated = -mu_estimated
    else:
        mu_estimated = mu_from_features
        first_eliminated = elimination_order[0]
        winner_idx = [i for i in range(n_players) if i not in elimination_order][0]
        
        if mu_estimated[first_eliminated] > mu_estimated[winner_idx]:
            print("  ⚠ 警告: mu顺序反了，正在修正...")
            mu_estimated = -mu_estimated
        else:
            print(f"  ✓ mu顺序正确 (第一个被淘汰mu={mu_estimated[first_eliminated]:.4f}, 冠军mu={mu_estimated[winner_idx]:.4f})")
    
    # 阶段2: 将mu映射到投票数量
    def objective_votes(params):
        a, b = params
        votes = a * np.exp(mu_estimated) + b
        votes = np.clip(votes, 1, None)
        loss = pl_elimination_loss(votes, elimination_order)
        reg = 0.01 * ((a - 1000) ** 2 + b ** 2)
        return loss + reg
    
    params0 = np.array([1000.0, 100.0])
    result_votes = minimize(objective_votes, params0, method='L-BFGS-B',
                           bounds=[(1, 10000), (0, 10000)], options={'maxiter': 500})
    
    a, b = result_votes.x
    votes_predicted = a * np.exp(mu_estimated) + b
    votes_predicted = np.clip(votes_predicted, 1, None)
    
    mu_std = np.std(mu_estimated) if len(mu_estimated) > 1 else 0.5
    votes_std = votes_predicted * 0.1 * (1 + mu_std)
    
    return {
        'votes': votes_predicted,
        'votes_std': votes_std,
        'mu': mu_estimated,
        'params': (a, b),
        'success': result_votes.success
    }

def regression_vote_prediction(X, elimination_order, mu_from_model):
    """
    基于特征的回归方法
    """
    return two_stage_vote_prediction(X, elimination_order, mu_from_features=mu_from_model)

def evaluate_round_by_round_elimination(X, elimination_order, mu_from_model=None, method='two_stage', vote_params=None):
    """
    逐轮淘汰评估：每轮预测投票数，淘汰最低的，直到只剩一人
    
    参数:
    - X: 特征矩阵 (n_players, n_features)
    - elimination_order: 实际淘汰顺序
    - mu_from_model: 从特征模型得到的mu（可选）
    - method: 预测方法
    - vote_params: 投票数映射参数 (a, b)，如果为None则使用默认值
    
    返回:
    - 预测的淘汰顺序和一致性指标
    """
    n_players = len(elimination_order) + 1
    alive_indices = list(range(n_players))
    predicted_elim_order = []
    
    # 使用提供的参数，或使用默认值
    if vote_params is not None:
        a, b = vote_params
    else:
        a, b = 1000.0, 100.0
    
    # 逐轮淘汰
    for round_num in range(len(elimination_order)):
        if len(alive_indices) <= 1:
            break
        
        # 当前轮次的选手
        if mu_from_model is not None:
            mu_alive = mu_from_model[alive_indices]
        else:
            # 如果没有mu，使用简单的初始化
            mu_alive = np.zeros(len(alive_indices))
        
        # 根据mu预测投票数：votes = a * exp(mu) + b
        votes_alive = a * np.exp(mu_alive) + b
        votes_alive = np.clip(votes_alive, 1, None)
        
        # 找到投票数最低的选手（本轮被淘汰）
        min_vote_idx = np.argmin(votes_alive)
        eliminated_idx = alive_indices[min_vote_idx]
        
        predicted_elim_order.append(eliminated_idx)
        alive_indices.remove(eliminated_idx)
    
    # 计算一致性
    if len(predicted_elim_order) == len(elimination_order):
        pred_ranks = np.zeros(n_players)
        actual_ranks = np.zeros(n_players)
        
        for i, idx in enumerate(predicted_elim_order):
            pred_ranks[idx] = i + 1
        for i, idx in enumerate(elimination_order):
            actual_ranks[idx] = i + 1
        
        pred_winner = alive_indices[0]
        actual_winner = [i for i in range(n_players) if i not in elimination_order][0]
        pred_ranks[pred_winner] = len(elimination_order) + 1
        actual_ranks[actual_winner] = len(elimination_order) + 1
        
        tau, p_value = kendalltau(pred_ranks, actual_ranks)
        
        return {
            'predicted_elim_order': predicted_elim_order,
            'elimination_order_tau': tau,
            'elimination_order_p': p_value,
            'elimination_order_match': (predicted_elim_order == elimination_order)
        }
    
    return {
        'predicted_elim_order': predicted_elim_order,
        'elimination_order_tau': 0.0,
        'elimination_order_p': 1.0,
        'elimination_order_match': False
    }

def evaluate_vote_prediction(votes_pred, votes_true=None, elimination_order=None, 
                            X=None, mu_from_model=None, method='two_stage', vote_params=None):
    """
    评估投票数量预测的准确性
    
    如果提供了X和mu_from_model，使用逐轮淘汰评估
    否则使用一次性排序评估（向后兼容）
    """
    results = {}
    
    if votes_true is not None:
        mse = np.mean((votes_pred - votes_true) ** 2)
        mae = np.mean(np.abs(votes_pred - votes_true))
        rmse = np.sqrt(mse)
        ss_res = np.sum((votes_true - votes_pred) ** 2)
        ss_tot = np.sum((votes_true - np.mean(votes_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        relative_error = np.mean(np.abs(votes_pred - votes_true) / (votes_true + 1e-10))
        
        results.update({
            'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'relative_error': relative_error
        })
    
    if elimination_order is not None:
        # 如果提供了X和mu_from_model，使用逐轮淘汰评估
        if X is not None and mu_from_model is not None:
            round_by_round_result = evaluate_round_by_round_elimination(
                X, elimination_order, mu_from_model, method, vote_params
            )
            results.update(round_by_round_result)
            
            if round_by_round_result['elimination_order_tau'] < 0:
                print(f"    ⚠ 警告: Kendall's tau为负 ({round_by_round_result['elimination_order_tau']:.4f})，预测顺序可能反了")
                print(f"    预测顺序: {round_by_round_result['predicted_elim_order']}")
                print(f"    实际顺序: {elimination_order}")
        else:
            # 向后兼容：一次性排序评估
            votes_sorted_indices = np.argsort(votes_pred)
            predicted_elim_order = votes_sorted_indices[:-1].tolist()
            
            if len(predicted_elim_order) == len(elimination_order):
                pred_ranks = np.zeros(len(votes_pred))
                actual_ranks = np.zeros(len(votes_pred))
                
                for i, idx in enumerate(predicted_elim_order):
                    pred_ranks[idx] = i + 1
                for i, idx in enumerate(elimination_order):
                    actual_ranks[idx] = i + 1
                
                pred_winner = votes_sorted_indices[-1]
                actual_winner = [i for i in range(len(votes_pred)) if i not in elimination_order][0]
                pred_ranks[pred_winner] = len(elimination_order) + 1
                actual_ranks[actual_winner] = len(elimination_order) + 1
                
                tau, p_value = kendalltau(pred_ranks, actual_ranks)
                results['elimination_order_tau'] = tau
                results['elimination_order_p'] = p_value
                results['elimination_order_match'] = (predicted_elim_order == elimination_order)
                
                if tau < 0:
                    print(f"    ⚠ 警告: Kendall's tau为负 ({tau:.4f})，预测顺序可能反了")
                    print(f"    预测顺序: {predicted_elim_order}")
                    print(f"    实际顺序: {elimination_order}")
    
    return results

def predict_votes_comprehensive(X, elimination_order, mu_from_model=None, method='two_stage'):
    """
    综合投票数量预测
    
    参数:
    - X: 特征矩阵
    - elimination_order: 淘汰顺序
    - mu_from_model: 从特征模型得到的mu（可选）
    - method: 方法选择 ('bayesian', 'mle', 'two_stage', 'regression', 'all')
    
    返回:
    - 预测结果字典
    """
    results = {}
    
    if method == 'all':
        methods_to_run = ['bayesian', 'mle', 'two_stage']
        if mu_from_model is not None:
            methods_to_run.append('regression')
    else:
        methods_to_run = [method]
    
    for m in methods_to_run:
        try:
            if m == 'bayesian':
                results['bayesian'] = bayesian_vote_prediction(X, elimination_order)
            elif m == 'mle':
                results['mle'] = mle_vote_prediction(elimination_order)
            elif m == 'two_stage':
                results['two_stage'] = two_stage_vote_prediction(X, elimination_order, mu_from_model)
            elif m == 'regression' and mu_from_model is not None:
                results['regression'] = regression_vote_prediction(X, elimination_order, mu_from_model)
        except Exception as e:
            print(f"方法 {m} 失败: {str(e)}")
            results[m] = {'error': str(e)}
    
    # 评估所有方法（使用逐轮淘汰评估）
    if elimination_order is not None:
        for method_name, result in results.items():
            if 'error' not in result:
                votes = result.get('votes', result.get('votes_map', result.get('votes_mean')))
                if votes is not None:
                    # 使用逐轮淘汰评估
                    # 优先使用result中的mu（经过修正的），否则使用mu_from_model
                    mu_for_eval = result.get('mu', mu_from_model)
                    # 获取投票数映射参数（如果存在）
                    vote_params = result.get('params', None)
                    eval_result = evaluate_vote_prediction(
                        votes, 
                        elimination_order=elimination_order,
                        X=X,
                        mu_from_model=mu_for_eval,
                        method=method_name if method_name != 'regression' else 'two_stage',
                        vote_params=vote_params
                    )
                    results[method_name]['evaluation'] = eval_result
    
    return results

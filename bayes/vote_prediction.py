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
    
    原理说明：
    ---------
    Plackett-Luce模型假设：在每一轮淘汰中，某个选手被淘汰的概率与其"实力"成反比。
    实力用投票数表示：投票数越多，实力越强，被淘汰的概率越小。
    
    数学公式：
    - 第i轮淘汰中，选手j被淘汰的概率 = exp(-mu_j) / sum(exp(-mu_k)) for all k in alive
    - 其中 mu_j = log(votes_j)，表示选手j的实力（对数空间）
    - 负对数似然 = -log(概率)，我们最小化这个值
    
    计算过程：
    1. 将投票数转换为对数空间：mu = log(votes)
    2. 逐轮计算每轮淘汰的概率
    3. 累加所有轮次的负对数概率
    
    参数:
    - votes: (n_players,) 投票数量数组
    - elimination_order: list, 淘汰顺序（索引列表），例如[2,0,1]表示：先淘汰索引2，再淘汰索引0，最后淘汰索引1
    
    返回:
    - loss: 负对数似然（越小越好）
    
    示例:
    -----
    假设有4个选手，投票数=[1000, 2000, 500, 3000]，淘汰顺序=[2, 0, 1]
    第1轮：从[0,1,2,3]中淘汰2，概率 = exp(-log(500)) / sum(exp(-log(votes)))
    第2轮：从[0,1,3]中淘汰0，概率 = exp(-log(1000)) / sum(exp(-log(votes)))
    第3轮：从[1,3]中淘汰1，概率 = exp(-log(2000)) / sum(exp(-log(votes)))
    损失 = -log(第1轮概率) - log(第2轮概率) - log(第3轮概率)
    """
    # 步骤1: 将投票数转换为对数空间（mu = log(votes)）
    # 这样做的原因：Plackett-Luce模型在对数空间中计算更稳定
    mu = np.log(votes + 1e-10)  # 1e-10防止log(0)
    
    # 步骤2: 初始化存活选手列表
    alive = list(range(len(votes)))  # 开始时所有选手都存活
    
    # 步骤3: 初始化损失（负对数似然）
    loss = 0.0
    
    # 步骤4: 逐轮计算淘汰概率并累加损失
    for eliminated in elimination_order:
        # 4.1: 找到被淘汰选手在当前存活列表中的位置
        idx = alive.index(eliminated)
        
        # 4.2: 获取所有存活选手的实力（对数空间）
        mu_alive = mu[alive]
        
        # 4.3: 计算被淘汰选手的对数概率
        # 公式：log(P(淘汰j)) = -mu_j - log(sum(exp(-mu_k))) for all k in alive
        # 这里使用 logsumexp 是为了数值稳定性
        log_prob = -mu_alive[idx] - logsumexp(-mu_alive)
        
        # 4.4: 累加负对数概率（因为我们要最小化损失，所以用负号）
        loss -= log_prob
        
        # 4.5: 从存活列表中移除被淘汰的选手
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

def mle_vote_prediction(elimination_order, X, feature_weights, 
                        votes_bounds=(100, 100000), reg_lambda=0.01):
    """
    最大似然估计投票数量（使用特征优化）
    
    参数:
    - elimination_order: 淘汰顺序（索引列表）
    - X: 特征矩阵 (n_players, n_features)
    - feature_weights: 特征权重向量 (n_features,)，用于从特征预测初始投票数
    - votes_bounds: 投票数量的取值范围
    - reg_lambda: 正则化系数，控制特征预测与MLE估计的平衡
    
    返回:
    - 字典，包含预测的投票数、损失值、训练状态
    """
    n_players = len(elimination_order) + 1
    
    # 从特征预测初始投票数
    mu_from_features = X @ feature_weights
    
    # 将mu线性映射到投票数范围（不使用exp）
    mu_min, mu_max = mu_from_features.min(), mu_from_features.max()
    if mu_max > mu_min:
        scale = (votes_bounds[1] - votes_bounds[0]) / (mu_max - mu_min)
        offset = votes_bounds[0] - scale * mu_min
    else:
        scale = (votes_bounds[1] - votes_bounds[0]) / 2
        offset = (votes_bounds[0] + votes_bounds[1]) / 2
    
    # 使用线性映射：votes = mu * scale + offset
    votes_from_features = mu_from_features * scale + offset
    votes0 = np.clip(votes_from_features, votes_bounds[0], votes_bounds[1])
    
    # 目标函数：MLE损失 + 特征正则化项
    # 正则化项：让投票数接近从特征预测的投票数
    def objective(votes):
        mle_loss = pl_elimination_loss(votes, elimination_order)
        reg_term = reg_lambda * np.sum((votes - votes_from_features) ** 2)
        return mle_loss + reg_term
    
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
    
    # 如果提供了X和mu_from_model，计算feature_weights用于MLE
    feature_weights = None
    if X is not None and mu_from_model is not None:
        # 通过线性回归从X和mu反推weights: mu = X @ w
        # 使用最小二乘法求解 w = (X^T X)^(-1) X^T mu
        try:
            X_pinv = np.linalg.pinv(X)
            feature_weights = X_pinv @ mu_from_model
        except:
            feature_weights = None
    
    for m in methods_to_run:
        try:
            if m == 'bayesian':
                results['bayesian'] = bayesian_vote_prediction(X, elimination_order)
            elif m == 'mle':
                # MLE方法现在需要X和feature_weights
                if X is not None and feature_weights is not None:
                    results['mle'] = mle_vote_prediction(elimination_order, X, feature_weights)
                else:
                    print(f"方法 {m} 跳过: 需要X和mu_from_model来计算feature_weights")
                    results[m] = {'error': '需要X和mu_from_model'}
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

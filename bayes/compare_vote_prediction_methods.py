"""
比较不同投票预测方法的一致性
使用integrated_vote_prediction.py的方式，但应用vote_prediction.py中的所有方法
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import kendalltau
import vote_prediction as vp
import integrated_vote_prediction as ivp

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

def simulate_elimination_with_method(model_result, method_result, season_data=None, current_week=1):
    """
    使用vote_prediction.py中的方法结果进行逐轮淘汰模拟
    
    参数:
    - model_result: train_model返回的模型结果
    - method_result: vote_prediction方法返回的结果（包含votes等）
    - season_data: 季节数据DataFrame（包含评委分数）
    - current_week: 当前周数
    
    返回:
    - 预测的淘汰顺序（列表）
    """
    if model_result is None or method_result is None:
        return None
    
    # 获取预测的投票数
    votes_all = method_result.get('votes', 
                                  method_result.get('votes_map', 
                                  method_result.get('votes_mean')))
    if votes_all is None:
        return None
    
    n_players = model_result['n_players']
    alive_indices = list(range(n_players))
    predicted_elim_order = []
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
        
        # 从存活列表中移除被淘汰的选手
        alive_indices.remove(eliminated_idx)
        
        # 进入下一周
        week += 1
    
    return predicted_elim_order

def evaluate_method_consistency(model_result, method_result, season_data=None):
    """
    评估特定方法的一致性
    
    参数:
    - model_result: train_model返回的模型结果
    - method_result: vote_prediction方法返回的结果
    - season_data: 季节数据DataFrame
    
    返回:
    - 一致性评估结果字典
    """
    if model_result is None:
        return None
    
    n_players = model_result['n_players']
    elimination_order = model_result['elimination_order']
    
    # 使用该方法的结果进行逐轮淘汰模拟
    predicted_elim_order = simulate_elimination_with_method(
        model_result, method_result, season_data, current_week=1
    )
    
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
    
    for i, idx in enumerate(predicted_elim_order):
        pred_ranks[idx] = i + 1
    for i, idx in enumerate(elimination_order):
        actual_ranks[idx] = i + 1
    
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

def compare_all_methods_for_season(season=1):
    """
    对指定季节比较所有投票预测方法的一致性
    
    参数:
    - season: 季节编号
    
    返回:
    - 所有方法的评估结果字典
    """
    print("="*60)
    print(f"比较所有投票预测方法 - Season {season}")
    print("="*60)
    
    # 加载数据
    fame_data = pd.read_csv('../2026_MCM_Problem_Fame_Data.csv')
    wiki_data = pd.read_csv('../celebrity_wikipedia_stats.csv')
    c_data_all = pd.read_csv('../2026_MCM_Problem_C_Data.csv')
    season_c_data = c_data_all[c_data_all['season'] == season].copy()
    
    print(f"\n加载数据完成:")
    print(f"  Season {season} 选手数量: {len(season_c_data)}")
    
    # 准备数据
    X, elimination_order, player_names, elimination_ranks = ivp.prepare_season_data(
        fame_data, wiki_data, season
    )
    
    if X is None:
        print("数据准备失败！")
        return None
    
    # 训练模型获取mu
    print(f"\n训练模型获取mu...")
    model_result = ivp.train_model(fame_data, wiki_data, season)
    
    if model_result is None:
        print("模型训练失败！")
        return None
    
    mu_from_model = model_result['mu']
    
    # 测试所有方法
    methods_to_test = ['bayesian', 'mle', 'two_stage']
    if mu_from_model is not None:
        methods_to_test.append('regression')
    
    all_results = {}
    
    for method in methods_to_test:
        print(f"\n{'='*60}")
        print(f"测试方法: {method}")
        print(f"{'='*60}")
        
        try:
            # 使用vote_prediction.py中的方法
            if method == 'bayesian':
                method_result = vp.bayesian_vote_prediction(X, elimination_order)
            elif method == 'mle':
                method_result = vp.mle_vote_prediction(elimination_order)
            elif method == 'two_stage':
                method_result = vp.two_stage_vote_prediction(X, elimination_order, mu_from_features=mu_from_model)
            elif method == 'regression':
                method_result = vp.regression_vote_prediction(X, elimination_order, mu_from_model)
            else:
                continue
            
            if 'error' in method_result:
                print(f"  方法 {method} 失败: {method_result['error']}")
                continue
            
            # 评估一致性
            eval_result = evaluate_method_consistency(
                model_result, method_result, season_c_data
            )
            
            if eval_result:
                tau = eval_result['elimination_order_tau']
                p_value = eval_result['elimination_order_p']
                match = eval_result['elimination_order_match']
                
                print(f"\n  {method} 方法结果:")
                print(f"    Kendall's tau: {tau:.4f}")
                print(f"    p值: {p_value:.4f}")
                print(f"    完全匹配: {match}")
                
                all_results[method] = {
                    'method_result': method_result,
                    'evaluation': eval_result,
                    'tau': tau,
                    'p_value': p_value,
                    'match': match
                }
        
        except Exception as e:
            print(f"  方法 {method} 执行出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results

def compare_all_methods_all_seasons():
    """
    对所有季节比较所有投票预测方法的一致性
    """
    print("="*60)
    print("比较所有投票预测方法 - 所有季节")
    print("="*60)
    
    # 加载数据
    fame_data = pd.read_csv('../2026_MCM_Problem_Fame_Data.csv')
    wiki_data = pd.read_csv('../celebrity_wikipedia_stats.csv')
    c_data_all = pd.read_csv('../2026_MCM_Problem_C_Data.csv')
    
    seasons = sorted(wiki_data['season'].unique())
    print(f"\n找到 {len(seasons)} 个季节: {seasons}")
    
    # 存储所有结果
    all_season_results = {}
    method_summaries = {
        'bayesian': [],
        'mle': [],
        'two_stage': [],
        'regression': []
    }
    
    for season in seasons:
        print(f"\n{'='*80}")
        print(f"处理 Season {season}")
        print(f"{'='*80}")
        
        try:
            season_c_data = c_data_all[c_data_all['season'] == season].copy()
            
            # 准备数据
            X, elimination_order, player_names, elimination_ranks = ivp.prepare_season_data(
                fame_data, wiki_data, season
            )
            
            if X is None:
                continue
            
            # 训练模型
            model_result = ivp.train_model(fame_data, wiki_data, season)
            if model_result is None:
                continue
            
            mu_from_model = model_result['mu']
            
            # 测试所有方法
            methods_to_test = ['bayesian', 'mle', 'two_stage']
            if mu_from_model is not None:
                methods_to_test.append('regression')
            
            season_results = {}
            
            for method in methods_to_test:
                try:
                    # 使用vote_prediction.py中的方法
                    if method == 'bayesian':
                        method_result = vp.bayesian_vote_prediction(X, elimination_order)
                    elif method == 'mle':
                        method_result = vp.mle_vote_prediction(elimination_order)
                    elif method == 'two_stage':
                        method_result = vp.two_stage_vote_prediction(X, elimination_order, mu_from_features=mu_from_model)
                    elif method == 'regression':
                        method_result = vp.regression_vote_prediction(X, elimination_order, mu_from_model)
                    else:
                        continue
                    
                    if 'error' in method_result:
                        continue
                    
                    # 评估一致性
                    eval_result = evaluate_method_consistency(
                        model_result, method_result, season_c_data
                    )
                    
                    if eval_result:
                        tau = eval_result['elimination_order_tau']
                        p_value = eval_result['elimination_order_p']
                        match = eval_result['elimination_order_match']
                        
                        season_results[method] = {
                            'tau': tau,
                            'p_value': p_value,
                            'match': match
                        }
                        
                        method_summaries[method].append({
                            'season': season,
                            'tau': tau,
                            'p_value': p_value,
                            'match': match
                        })
                
                except Exception as e:
                    continue
            
            all_season_results[season] = season_results
        
        except Exception as e:
            print(f"Season {season} 处理失败: {str(e)}")
            continue
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("所有方法的总体一致性统计")
    print(f"{'='*80}")
    
    summary_data = []
    
    for method, results in method_summaries.items():
        if len(results) == 0:
            continue
        
        df = pd.DataFrame(results)
        tau_values = df['tau'].values
        
        tau_mean = np.mean(tau_values)
        tau_std = np.std(tau_values)
        tau_median = np.median(tau_values)
        tau_min = np.min(tau_values)
        tau_max = np.max(tau_values)
        positive_count = np.sum(tau_values > 0)
        perfect_match_count = df['match'].sum()
        
        print(f"\n{method.upper()} 方法:")
        print(f"  平均值: {tau_mean:.4f} ± {tau_std:.4f}")
        print(f"  中位数: {tau_median:.4f}")
        print(f"  范围: [{tau_min:.4f}, {tau_max:.4f}]")
        print(f"  正相关季节数: {positive_count}/{len(tau_values)} ({100*positive_count/len(tau_values):.1f}%)")
        print(f"  完全匹配季节数: {perfect_match_count}/{len(tau_values)} ({100*perfect_match_count/len(tau_values):.1f}%)")
        
        summary_data.append({
            'method': method,
            'mean_tau': tau_mean,
            'std_tau': tau_std,
            'median_tau': tau_median,
            'min_tau': tau_min,
            'max_tau': tau_max,
            'positive_count': positive_count,
            'total_seasons': len(tau_values),
            'perfect_match_count': perfect_match_count
        })
    
    # 保存详细结果
    for method, results in method_summaries.items():
        if len(results) > 0:
            df = pd.DataFrame(results)
            df.to_csv(f'method_comparison_{method}_all_seasons.csv', index=False)
            print(f"\n  {method} 方法详细结果已保存到: method_comparison_{method}_all_seasons.csv")
    
    # 保存汇总结果
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('method_comparison_summary.csv', index=False)
        print(f"\n汇总结果已保存到: method_comparison_summary.csv")
    
    return all_season_results, method_summaries

if __name__ == "__main__":
    # 比较所有季节的所有方法
    compare_all_methods_all_seasons()
    
    # 或者只比较单个季节
    # compare_all_methods_for_season(season=1)


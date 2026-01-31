import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau, rankdata
from sklearn.model_selection import KFold

# ==========================================
# 1. 核心数学函数 (保持不变)
# ==========================================
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_mu_sigma(X, params, n_feats):
    w_mu = params[0:n_feats]
    b_mu = params[n_feats]
    w_sig = params[n_feats + 1: 2 * n_feats + 1]
    b_sig = params[2 * n_feats + 1]
    mu = np.dot(X, w_mu) + b_mu
    y2 = np.dot(X, w_sig) + b_sig
    sigma = np.where(y2 > 20, y2, np.log1p(np.exp(np.clip(y2, -20, 20))))
    return mu, sigma

# ==========================================
# 2. 损失函数 (训练逻辑保持不变)
# ==========================================
def objective_function(params, pools, n_feats):
    phi = params[-1]
    total_loss = 0
    num_samples = 10 
    for pool in pools:
        X, J, targets, weeks = pool['X'], pool['J'], pool['targets'], pool['week']
        mu, sigma = get_mu_sigma(X, params, n_feats)
        pool_loss = 0
        for _ in range(num_samples):
            epsilon = np.random.normal(0, 1, size=len(mu))
            u_sampled = mu + phi * weeks + sigma * epsilon
            V = softmax(u_sampled * 10)
            total_score = V + J
            current_sample_loss = 0
            if targets.sum() > 0:
                elim_scores = total_score[targets == 1]
                surv_scores = total_score[targets == 0]
                if len(surv_scores) > 0:
                    for e_s in elim_scores:
                        diff = e_s - surv_scores
                        current_sample_loss += np.log1p(np.exp(20 * diff)).mean()
            pool_loss += current_sample_loss
        total_loss += pool_loss / num_samples 
    l2_reg = 0.01 * np.sum(params**2)
    return total_loss + l2_reg

# ==========================================
# 3. 修改后的评估逻辑：验证最终排名准确度
# ==========================================
def evaluate_final_ranking_accuracy(params, season_df, static_cols):
    """
    验证预测排名与实际排名的匹配程度
    """
    n_feats = len(static_cols)
    phi = params[-1]

    # 1. 获取该赛季所有选手的最终记录（被淘汰时的最后一周或决赛周）
    # 我们按选手(index)分组，取他们最后一次出现的数据
    final_stats = season_df.sort_values('week').groupby('index').last()
    
    X = final_stats[static_cols].values
    J = final_stats['Judge_Share'].values
    actual_placement = final_stats['placement'].values
    weeks = final_stats['week'].values

    # 2. 计算模型预测的“综合实力分”
    mu, _ = get_mu_sigma(X, params, n_feats)
    # 预测强度 = 基础实力 + 时间趋势 + 评委分
    predicted_strength = mu + phi * weeks + J
    
    # 3. 将预测强度转化为排名 (1nd, 2st, 3rd...)
    # 强度越高，排名数值越小 (rank 1 是冠军)。所以对强度取负值进行排名
    predicted_ranks = rankdata(-predicted_strength, method='ordinal')
    
    # 4. 计算指标
    # 严格排名准确度：预测名次和实际名次完全一致的人数占比
    correct_positions = np.sum(predicted_ranks == actual_placement)
    accuracy = correct_positions / len(actual_placement)
    
    # 相关性指标：Kendall's Tau
    tau, _ = kendalltau(predicted_ranks, actual_placement)
    
    return accuracy, tau

# ==========================================
# 4. 交叉验证
# ==========================================
def run_cv_with_ranking_accuracy(df, k=5):
    industry_cols = [c for c in df.columns if 'industry_' in c]
    static_cols = ['log_fame', 'age_scaled'] + industry_cols
    n_feats = len(static_cols)

    seasons = sorted(df['season'].unique())
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    results = []

    print(f"开始 {k} 折交叉验证 (评估标准：最终排名准确度)...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(seasons)):
        train_seasons = [seasons[i] for i in train_idx]
        test_seasons = [seasons[i] for i in test_idx]

        # 构造训练竞争池 (训练逻辑不变)
        train_pools = []
        train_data = df[df['season'].isin(train_seasons)]
        for (s, w), group in train_data.groupby(['season', 'week']):
            if len(group) >= 2:
                train_pools.append({
                    'X': group[static_cols].values,
                    'J': group['Judge_Share'].values,
                    'targets': group['is_eliminated'].values,
                    'week': group['week'].values
                })

        # 训练 (参数设置不变)
        n_params = 2 * n_feats + 3
        init_params = np.random.randn(n_params)*0.01
        res = minimize(objective_function, init_params, args=(train_pools, n_feats),
                       method='L-BFGS-B', options={'maxiter': 500})

        # 验证测试集中的每一个赛季
        fold_accs = []
        fold_taus = []
        for s_id in test_seasons:
            season_df = df[df['season'] == s_id]
            # --- 调用新的排名验证逻辑 ---
            acc, tau = evaluate_final_ranking_accuracy(res.x, season_df, static_cols)
            fold_accs.append(acc)
            if not np.isnan(tau): fold_taus.append(tau)

        results.append({
            'fold': fold + 1,
            'mean_acc': np.mean(fold_accs),
            'mean_tau': np.mean(fold_taus)
        })
        print(f"Fold {fold + 1} | 排名准确率: {results[-1]['mean_acc']:.2%} | Kendall's tau: {results[-1]['mean_tau']:.4f}")

    # 打印最终统计
    avg_acc = np.mean([r['mean_acc'] for r in results])
    avg_tau = np.mean([r['mean_tau'] for r in results])
    std_acc = np.std([r['mean_acc'] for r in results])

    print("\n" + "=" * 40)
    print("最终排名对齐验证统计")
    print("=" * 40)
    print(f"平均排名精确匹配度: {avg_acc:.2%} (±{std_acc:.2%})")
    print(f"平均 Kendall's tau (排序一致性): {avg_tau:.4f}")
    print("=" * 40)

    # 最终在全量数据上训练
    final_pools = []
    for (s, w), group in df.groupby(['season', 'week']):
        if len(group) >= 2:
            final_pools.append({
                'X': group[static_cols].values, 'J': group['Judge_Share'].values,
                'targets': group['is_eliminated'].values, 'week': group['week'].values
            })
    final_res = minimize(objective_function, np.random.randn(2 * n_feats + 3)*0.01, args=(final_pools, n_feats),
                         method='L-BFGS-B', options={'maxiter': 500})
    
    return final_res, static_cols

# ==========================================
# 5. 主程序运行
# ==========================================
if __name__ == "__main__":
    df_all = pd.read_csv('2026_MCM_Problem_ARIMA_Data_SeasonAware.csv')
    df_train = df_all[df_all['season'].between(3, 27)].copy()

    # 执行修改后的K折验证
    final_params, static_cols = run_cv_with_ranking_accuracy(df_train, k=5)
    
    # 存为资产
    np.savez('dwts_model_assets.npz', params=final_params.x, static_cols=static_cols)
    print("\n[系统] 模型训练及排名验证完成。")


# # ==========================================
# # 5. 主程序运行
# # ==========================================
# if __name__ == "__main__":
#     # 使用预处理过的包含 placement 和 is_eliminated 的数据
#     df_all = pd.read_csv('2026_MCM_Problem_ARIMA_Data_SeasonAware.csv')

#     # 只使用赛季 3-27 进行交叉验证
#     df_train = df_all[df_all['season'].between(3, 27)].copy()

#     final_params, static_cols = run_cv_with_consistency(df_train, k=5)

#     print("\n[优化任务完成]")

# import pandas as pd
# import numpy as np
# from scipy.optimize import minimize
# from scipy.stats import kendalltau, rankdata
# from sklearn.model_selection import KFold

# # ==========================================
# # 1. 核心数学函数
# # ==========================================
# def sigmoid(x):
#     return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

# def get_mu_sigma(X, params, n_feats):
#     w_mu = params[0:n_feats]
#     b_mu = params[n_feats]
#     w_sig = params[n_feats + 1: 2 * n_feats + 1]
#     b_sig = params[2 * n_feats + 1]
#     mu = np.dot(X, w_mu) + b_mu
#     y2 = np.dot(X, w_sig) + b_sig
#     # 保证 sigma 为正
#     sigma = np.log1p(np.exp(np.clip(y2, -20, 20))) + 1e-4
#     return mu, sigma

# def compute_soft_ranks_vectorized(scores):
#     """
#     向量化可微排名函数
#     DWTS逻辑：分数越高，排名越靠前（数值越小）
#     """
#     s = scores.reshape(-1, 1)
#     # 计算差值矩阵 diff[i, j] = score_j - score_i
#     # 如果 score_j > score_i, sigmoid(diff) -> 1, 增加 i 的排名值(让其变大/靠后)
#     diff_matrix = s.T - s 
#     soft_ranks = 1 + np.sum(sigmoid(15 * diff_matrix), axis=1) - sigmoid(0)
#     return soft_ranks

# # ==========================================
# # 2. 损失函数 (基于排名差异的 MSE)
# # ==========================================
# def objective_function(params, pools, n_feats):
#     phi = params[-1]
#     total_loss = 0
#     num_samples = 5  
    
#     for pool in pools:
#         X, J, true_placements, weeks = pool['X'], pool['J'], pool['placements'], pool['week']
#         mu, sigma = get_mu_sigma(X, params, n_feats)
        
#         pool_sample_loss = 0
#         for _ in range(num_samples):
#             epsilon = np.random.normal(0, 1, size=len(mu))
#             # ARIMA-like 动态分值
#             u_sampled = mu + phi * weeks + sigma * epsilon
            
#             # 综合得分 (观众喜爱度预测 + 评委分)
#             # 假设观众分也是一种 Share，进行简单归一化
#             V = u_sampled / (np.sum(np.abs(u_sampled)) + 1e-6)
#             total_score = V + J
            
#             # 计算预测的软排名
#             pred_soft_ranks = compute_soft_ranks_vectorized(total_score)
            
#             # 损失 = 排名数值的均方误差
#             pool_sample_loss += np.mean((pred_soft_ranks - true_placements)**2)
            
#         total_loss += pool_sample_loss / num_samples
        
#     l2_reg = 0.05 * np.sum(params**2)
#     return total_loss + l2_reg

# # ==========================================
# # 3. 评估逻辑：严格排名相等 (Strict Exact Match)
# # ==========================================
# def evaluate_ranking_accuracy(params, season_df, static_cols):
#     n_feats = len(static_cols)
#     phi = params[-1]

#     # 我们关注选手在赛季中的最终排名位置
#     # 获取选手的最后状态
#     last_appearances = season_df.sort_values('week').groupby('index').last()
    
#     X = last_appearances[static_cols].values
#     J = last_appearances['Judge_Share'].values
#     true_ranks = last_appearances['placement'].values
#     weeks = last_appearances['week'].values

#     mu, _ = get_mu_sigma(X, params, n_feats)
#     # 预测总分（不带噪声的期望值）
#     final_scores = mu + phi * weeks + J
    
#     # 核心：将预测得分转为严格离散排名 (1st, 2nd, ...)
#     # 分数最高的排第1，所以对分数取负
#     pred_ranks = rankdata(-final_scores, method='ordinal') 
    
#     # 计算严格相等的人数比例
#     correct_matches = np.sum(pred_ranks == true_ranks)
#     strict_acc = correct_matches / len(true_ranks)
    
#     # 排序一致性指标
#     tau, _ = kendalltau(pred_ranks, true_ranks)
    
#     return strict_acc, tau

# # ==========================================
# # 4. 交叉验证
# # ==========================================
# def run_cv_with_ranking(df, k=5):
#     industry_cols = [c for c in df.columns if 'industry_' in c]
#     static_cols = ['log_fame', 'age_scaled'] + industry_cols
#     n_feats = len(static_cols)

#     seasons = sorted(df['season'].unique())
#     kf = KFold(n_splits=k, shuffle=True, random_state=42)

#     results = []

#     print(f"开始 {k} 折排名损失模型验证 (目标：严格排名一致)...")
#     for fold, (train_idx, test_idx) in enumerate(kf.split(seasons)):
#         train_seasons = [seasons[i] for i in train_idx]
#         test_seasons = [seasons[i] for i in test_idx]

#         train_pools = []
#         train_data = df[df['season'].isin(train_seasons)]
#         for (s, w), group in train_data.groupby(['season', 'week']):
#             if len(group) >= 2:
#                 train_pools.append({
#                     'X': group[static_cols].values,
#                     'J': group['Judge_Share'].values,
#                     'placements': group['placement'].values,
#                     'week': group['week'].values
#                 })

#         n_params = 2 * n_feats + 3
#         init_params = np.random.normal(0, 0.01, n_params) 
        
#         res = minimize(objective_function, init_params, args=(train_pools, n_feats),
#                        method='L-BFGS-B', options={'maxiter': 150})

#         fold_accs = []
#         fold_taus = []
#         for s_id in test_seasons:
#             season_df = df[df['season'] == s_id]
#             acc, tau = evaluate_ranking_accuracy(res.x, season_df, static_cols)
#             fold_accs.append(acc)
#             fold_taus.append(tau)

#         results.append({
#             'fold': fold + 1,
#             'mean_acc': np.mean(fold_accs),
#             'mean_tau': np.mean(fold_taus)
#         })
#         print(f"Fold {fold+1} | 严格排名准确率: {results[-1]['mean_acc']:.2%} | Kendall's tau: {results[-1]['mean_tau']:.4f}")

#     print("\n" + "=" * 40)
#     print("最终严格一致性统计 (Season 3-27)")
#     print(f"平均每个赛季中选手排名被‘精确预测’的比例: {np.mean([r['mean_acc'] for r in results]):.2%}")
#     print(f"平均 Kendall's tau (整体排序相关性): {np.mean([r['mean_tau'] for r in results]):.4f}")

#     return res.x, static_cols

# # ==========================================
# # 5. 主程序
# # ==========================================
# if __name__ == "__main__":
#     # 加载数据
#     try:
#         df_all = pd.read_csv('2026_MCM_Problem_ARIMA_Data_SeasonAware.csv')
#         df_train = df_all[df_all['season'].between(3, 27)].copy()

#         # 执行交叉验证
#         final_params, static_cols = run_cv_with_ranking(df_train, k=5)
        
#         # 保存资产
#         np.savez('dwts_strict_ranking_model.npz', params=final_params, cols=static_cols)
#         print("\n[系统] 模型训练完成，参数已保存。")
        
#     except FileNotFoundError:
#         print("错误：未找到数据文件 '2026_MCM_Problem_ARIMA_Data_SeasonAware.csv'")
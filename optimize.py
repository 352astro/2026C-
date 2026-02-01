import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau, rankdata
from sklearn.model_selection import KFold

# ==========================================
# 1. 数据准备
# ==========================================
df_fame = pd.read_csv('2026_MCM_Problem_Fame_Data.csv')
df_season_map = pd.read_csv('2026_MCM_Problem_ARIMA_Data_SeasonAware.csv')

# 映射赛季信息
season_mapping = df_season_map.groupby('index')['season'].first().to_dict()
df_fame['season'] = df_fame['index'].map(season_mapping)
df_train_all = df_fame[df_fame['season'].between(3, 27)].copy()

df_train_all.to_csv('2026_MCM_Problem_Train_Data')

def preprocess_features(df):
    """特征提取与标准化"""
    # 注意：这里我们保留 season 列用于后续的分组排名
    pop_raw = df['fame_1.4'].values
    age_raw = df['celebrity_age_during_season'].values
    X_judge_raw = df['score_sum'].values
    
    def scale(x): return (x - x.mean()) / (x.std() + 1e-6)
    
    X_j = scale(X_judge_raw)
    pop = scale(pop_raw)
    age = scale(age_raw)
    
    ind_cols = [c for c in df.columns if 'industry_' in c]
    Ind = df[ind_cols].values
    
    X_feats = np.column_stack([pop, age, Ind])
    y_true = df['elimination_order'].values
    y_scaled = scale(y_true)
    return X_feats, X_j, y_scaled, df['season'].values

# ==========================================
# 2. 模型核心函数
# ==========================================
def get_mu_sigma(X, params, n_f):
    w_mu = params[:n_f]
    b_mu = params[n_f]
    w_sig = params[n_f+1 : 2*n_f+1]
    b_sig = params[2*n_f+1]
    
    mu = np.dot(X, w_mu) + b_mu
    y2 = np.dot(X, w_sig) + b_sig
    sigma = np.log1p(np.exp(np.clip(y2, -20, 20))) + 1e-4
    return mu, sigma

def objective_function(params, X, X_j, y_true, n_f):
    # 现在 params 的最后只剩一个 w_judge
    w_judge = params[-1] 
    
    # 获取 mu 和 sigma (内部已包含所有特征权重)
    # params 分配: w_mu(n_f), b_mu(1), w_sig(n_f), b_sig(1), w_judge(1)
    mu, sigma = get_mu_sigma(X, params[:-1], n_f) 
    
    # 预测逻辑：评委分权重 + 观众综合实力(mu)
    # mu 已经包含了特征的原始缩放
    y_pred_mean = w_judge * X_j + mu
    
    # 此时 combined_std 直接由 sigma 调控
    combined_std = sigma + 1e-4
    
    nll = np.sum(np.log(combined_std) + 0.5 * ((y_true - y_pred_mean) / combined_std)**2)
    
    # 正则化保持不变
    l2_reg = 0.1 * (np.sum(params[:n_f]**2) + np.sum(params[n_f+1:2*n_f+1]**2))
    return nll + l2_reg

# ==========================================
# 3. 严格排名评估函数 (核心改进)
# ==========================================
def evaluate_exact_accuracy(y_pred, y_true, seasons):
    """
    按赛季对预测值进行重排名，计算与真实排名完全一致的百分比
    """
    results = pd.DataFrame({
        'pred': y_pred,
        'true': y_true,
        'season': seasons
    })
    
    correct_count = 0
    total_count = len(y_true)
    
    for s_id, group in results.groupby('season'):
        # 将预测分值转为排名 (1, 2, 3...)
        # rankdata 处理同分情况采用平均秩，但在浮点数预测中极少发生
        pred_ranks = rankdata(group['pred'])
        true_ranks = rankdata(group['true'])
        
        # 统计完全相等的个数
        correct_count += np.sum(pred_ranks == true_ranks)
        
    return correct_count / total_count

# ==========================================
# 4. K折交叉验证逻辑
# ==========================================
def run_kfold_ranking_validation(df, k=5):
    seasons = sorted(df['season'].unique())
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    all_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(seasons)):
        tr_s = [seasons[i] for i in train_idx]
        val_s = [seasons[i] for i in val_idx]
        
        df_tr = df[df['season'].isin(tr_s)]
        df_val = df[df['season'].isin(val_s)]
        
        X_tr, J_tr, y_tr, _ = preprocess_features(df_tr)
        X_val, J_val, y_val, s_val = preprocess_features(df_val)
        
        n_f = X_tr.shape[1]
        # 初始值改为小随机数，避免落入鞍点
        init_params = np.random.normal(0, 0.05, 2*(n_f+1) + 1)
        init_params[-1] = 1.0
        
        res = minimize(objective_function, init_params, args=(X_tr, J_tr, y_tr, n_f),
                       method='L-BFGS-B', options={'maxiter': 500})
        
        # 预测并评估
        mu_v, _ = get_mu_sigma(X_val, res.x, n_f)
        w_j_v = res.x[-1]
        y_pred_val = w_j_v * J_val + mu_v
        
        # 计算严格排名准确率
        exact_acc = evaluate_exact_accuracy(y_pred_val, y_val, s_val)
        # 计算排序一致性
        tau, _ = kendalltau(y_pred_val, y_val)
        
        all_metrics.append([exact_acc, tau])
        print(f"Fold {fold+1}: 严格准确率={exact_acc:.2%}, Kendall's Tau={tau:.4f}")
        
    return np.mean(all_metrics, axis=0), res.x

# ==========================================
# 5. 执行与结果
# ==========================================
print("正在执行严格排名对齐验证 (Season 3-27)...")
avg_results, final_params_raw = run_kfold_ranking_validation(df_train_all)

print("\n" + "="*40)
print(f"平均严格排名准确率 (Exact Match): {avg_results[0]:.2%}")
print(f"平均排序一致性 (Kendall's Tau): {avg_results[1]:.4f}")
print("="*40)

# 全量训练与参数解析
X_f, J_f, y_f, s_f = preprocess_features(df_train_all)
n_f = X_f.shape[1]
res_final = minimize(objective_function, final_params_raw, args=(X_f, J_f, y_f, n_f), method='L-BFGS-B')
W = res_final.x
n_f = X_f.shape[1]

print("\n[最终模型参数解析]")
print(f"评委分权重: {W[-1]:.4f}")
print(f"名声特征影响力: {W[0]:.4f}")
print(f"年龄特征影响力: {W[1]:.4f}")
# 保存
np.savez('dwts_ranking_model.npz', params=W)
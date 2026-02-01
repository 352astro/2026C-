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

def preprocess_features(df):
    """特征提取与标准化"""
    def scale(x): return (x - x.mean()) / (x.std() + 1e-6)
    
    X_j = scale(df['score_sum'].values) # 评委总分
    pop = scale(df['fame_1.4'].values)
    age = scale(df['celebrity_age_during_season'].values)
    
    # 提取周数（用于在损失函数中与 mu 相乘）
    weeks = df['weeks'].values 
    
    ind_cols = [c for c in df.columns if 'industry_' in c]
    Ind = df[ind_cols].values
    
    # 特征矩阵不再包含 X_j，X_j 独立作为输入
    X_feats = np.column_stack([pop, age, Ind])
    
    y_true = df['elimination_order'].values
    y_scaled = scale(y_true)
    
    return X_feats, X_j, weeks, y_scaled, y_true, df['season'].values

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

def objective_function(params, X, X_j, weeks, y_target, n_f):
    # params 结构: [w_mu, b_mu, w_sig, b_sig, w_judge]
    w_judge = params[-1] 
    mu, sigma = get_mu_sigma(X, params[:-1], n_f) 
    
    # 核心修改：预测值 = 评委总分贡献 + (单位周观众缘 * 周数)
    # 这使得 mu 的物理意义明确为：每多待一周，观众贡献的潜在排名增量
    y_pred_mean = w_judge * X_j + mu * weeks
    
    combined_std = sigma + 1e-4
    
    # 负对数似然损失
    nll = np.sum(np.log(combined_std) + 0.5 * ((y_target - y_pred_mean) / combined_std)**2)
    
    # 正则化（只针对特征权重）
    l2_reg = 0.1 * (np.sum(params[:n_f]**2) + np.sum(params[n_f+1:2*n_f+1]**2))
    return nll + l2_reg

# ==========================================
# 3. 严格排名评估函数
# ==========================================
def evaluate_exact_accuracy(y_pred, y_true, seasons):
    results = pd.DataFrame({'pred': y_pred, 'true': y_true, 'season': seasons})
    correct_count = 0
    total_count = len(y_true)
    for _, group in results.groupby('season'):
        pred_ranks = rankdata(group['pred'])
        true_ranks = rankdata(group['true'])
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
        tr_s, val_s = [seasons[i] for i in train_idx], [seasons[i] for i in val_idx]
        df_tr, df_val = df[df['season'].isin(tr_s)], df[df['season'].isin(val_s)]
        
        # 预处理数据
        X_tr, J_tr, W_tr, y_tr_scaled, _, _ = preprocess_features(df_tr)
        X_val, J_val, W_val, _, y_val_raw, s_val = preprocess_features(df_val)
        
        n_f = X_tr.shape[1]
        init_params = np.random.normal(0, 0.05, 2*(n_f+1) + 1)
        init_params[-1] = 1.0 # 评委权重初始设为 1
        
        res = minimize(objective_function, init_params, args=(X_tr, J_tr, W_tr, y_tr_scaled, n_f),
                       method='L-BFGS-B', options={'maxiter': 500})
        
        # 验证集预测
        mu_v, _ = get_mu_sigma(X_val, res.x[:-1], n_f)
        w_j_v = res.x[-1]
        # 验证时同样采用 mu * weeks 逻辑
        y_pred_val = w_j_v * J_val + mu_v * W_val
        
        acc = evaluate_exact_accuracy(y_pred_val, y_val_raw, s_val)
        tau, _ = kendalltau(y_pred_val, y_val_raw)
        all_metrics.append([acc, tau])
        print(f"Fold {fold+1}: 严格准确率={acc:.2%}, Kendall's Tau={tau:.4f}")
        
    return np.mean(all_metrics, axis=0), res.x

# ==========================================
# 5. 执行与参数解析
# ==========================================
print("执行 ARIMA 风格线性模型 ( mu * weeks 逻辑)...")
avg_results, final_params_raw = run_kfold_ranking_validation(df_train_all)

print("\n" + "="*40)
print(f"平均严格排名准确率 (Exact Match): {avg_results[0]:.2%}")
print(f"平均排序一致性 (Kendall's Tau): {avg_results[1]:.4f}")
print("="*40)

# 全量拟合获取最终系数
X_f, J_f, W_f, y_f_scaled, y_f_raw, s_f = preprocess_features(df_train_all)
n_f = X_f.shape[1]
res_final = minimize(objective_function, final_params_raw, args=(X_f, J_f, W_f, y_f_scaled, n_f), method='L-BFGS-B')
W = res_final.x

print("\n[最终模型参数解析]")
print(f"评委总分权重 (w_judge): {W[-1]:.4f}")
print(f"观众周票数权重基础值: 1.0000 (已通过 mu 内部权重吸收)")
print("-" * 20)
print("观众缘特征贡献（mu 层 / 周贡献）:")
print(f"名声 (Fame^1.1) 影响力: {W[0]:.4f}")
print(f"年龄 (Age) 影响力: {W[1]:.4f}")

# 保存
# 保存时必须明确写入 static_cols，否则 checkparams 无法读取
industry_cols = [c for c in df_train_all.columns if 'industry_' in c]
static_cols = ['fame_1.1', 'celebrity_age_during_season'] + industry_cols
clean_cols = [c.replace('industry_', '') for c in static_cols]
np.savez('dwts_ranking_model_weeks_weighted.npz', 
         params=final_params_raw, 
         static_cols=clean_cols)
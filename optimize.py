import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau, rankdata
from sklearn.model_selection import KFold

# ==========================================
# 1. 数据准备
# ==========================================
# 修改：读取包含 Ballroom 数据的新文件
df_fame = pd.read_csv('2026_MCM_Problem_Fame_withBallroom_Data.csv')
df_season_map = pd.read_csv('2026_MCM_Problem_ARIMA_Data_SeasonAware.csv')

# 映射赛季信息
season_mapping = df_season_map.groupby('index')['season'].first().to_dict()
df_fame['season'] = df_fame['index'].map(season_mapping)
df_train_all = df_fame[df_fame['season'].between(3, 27)].copy()

# 保存训练用的中间数据以便检查
df_train_all.to_csv('2026_MCM_Problem_Train_Data_Processed.csv', index=False)

def preprocess_features(df):
    """特征提取与标准化"""
    # 提取评委总分
    X_judge_raw = df['score_sum'].values
    
    # 修改：提取 Celebrity Fame (使用新列名)
    pop_raw = df['celebrity_fame_1'].values
    
    # 新增：提取 Ballroom Fame
    ballroom_raw = df['ballroom_fame_1'].values
    
    # 提取年龄
    age_raw = df['celebrity_age_during_season'].values
    
    # 定义标准化函数
    def scale(x): return (x - x.mean()) / (x.std() + 1e-6)
    
    X_j = scale(X_judge_raw)
    pop = scale(pop_raw)
    ballroom = scale(ballroom_raw) # 新增：标准化 Ballroom fame
    age = scale(age_raw)
    
    # 提取行业特征 (One-hot encoding cols)
    ind_cols = [c for c in df.columns if 'industry_' in c]
    Ind = df[ind_cols].values
    
    # 修改：构建特征矩阵，加入 Ballroom 特征
    # 特征顺序: [Celeb_Fame, Ballroom_Fame, Age, Industries...]
    X_feats = np.column_stack([pop, ballroom, age, Ind])
    
    y_true = df['elimination_order'].values
    y_scaled = scale(y_true)
    
    # 返回特征矩阵，评委分，目标值，赛季，以及特征名称列表以便后续使用
    feature_names = ['celebrity_fame_1.4', 'ballroom_fame_1.4', 'celebrity_age_during_season'] + ind_cols
    
    return X_feats, X_j, y_scaled, df['season'].values, feature_names

# ==========================================
# 2. 模型核心函数
# ==========================================
def get_mu_sigma(X, params, n_f):
    # 参数结构: w_mu(n_f), b_mu(1), w_sig(n_f), b_sig(1)
    w_mu = params[:n_f]
    b_mu = params[n_f]
    w_sig = params[n_f+1 : 2*n_f+1]
    b_sig = params[2*n_f+1]
    
    mu = np.dot(X, w_mu) + b_mu
    y2 = np.dot(X, w_sig) + b_sig
    
    # 限制 sigma 范围防止数值溢出
    sigma = np.log1p(np.exp(np.clip(y2, -20, 20))) + 1e-4
    return mu, sigma

def objective_function(params, X, X_j, y_true, n_f):
    # 最后一个参数是评委权重
    w_judge = params[-1] 
    
    # 获取潜在能力分布 (Mu, Sigma)
    # params[:-1] 是除了 w_judge 之外的所有参数
    mu, sigma = get_mu_sigma(X, params[:-1], n_f) 
    
    # 预测逻辑：评委分权重 + 选手综合实力(mu)
    y_pred_mean = w_judge * X_j + mu
    
    # 此时 combined_std 直接由 sigma 调控
    combined_std = sigma + 1e-4
    
    # 负对数似然 (Negative Log Likelihood)
    nll = np.sum(np.log(combined_std) + 0.5 * ((y_true - y_pred_mean) / combined_std)**2)
    
    # L2 正则化
    l2_reg = 0.1 * (np.sum(params[:n_f]**2) + np.sum(params[n_f+1:2*n_f+1]**2))
    return nll + l2_reg

# ==========================================
# 3. 严格排名评估函数
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
        
        X_tr, J_tr, y_tr, _, _ = preprocess_features(df_tr)
        X_val, J_val, y_val, s_val, _ = preprocess_features(df_val)
        
        n_f = X_tr.shape[1]
        
        # 参数初始化: [w_mu, b_mu, w_sig, b_sig, w_judge]
        # 总参数量 = 2*n_f + 2 + 1
        num_params = 2 * n_f + 3
        init_params = np.random.normal(0, 0.05, num_params)
        init_params[-1] = 1.0 # 评委初始权重设为 1
        
        # 优化
        res = minimize(objective_function, init_params, args=(X_tr, J_tr, y_tr, n_f),
                       method='L-BFGS-B', options={'maxiter': 500})
        
        # 预测验证集
        mu_v, _ = get_mu_sigma(X_val, res.x[:-1], n_f)
        w_j_v = res.x[-1]
        y_pred_val = w_j_v * J_val + mu_v
        
        # 评估
        exact_acc = evaluate_exact_accuracy(y_pred_val, y_val, s_val)
        tau, _ = kendalltau(y_pred_val, y_val)
        
        all_metrics.append([exact_acc, tau])
        print(f"Fold {fold+1}: 严格准确率={exact_acc:.2%}, Kendall's Tau={tau:.4f}")
        
    return np.mean(all_metrics, axis=0), res.x

# ==========================================
# 5. 执行与结果
# ==========================================
print("正在执行严格排名对齐验证 (含 Ballroom 特征)...")
avg_results, final_params_raw = run_kfold_ranking_validation(df_train_all)

print("\n" + "="*40)
print(f"平均严格排名准确率 (Exact Match): {avg_results[0]:.2%}")
print(f"平均排序一致性 (Kendall's Tau): {avg_results[1]:.4f}")
print("="*40)

# 全量训练与参数解析
X_f, J_f, y_f, s_f, feat_names = preprocess_features(df_train_all)
n_f = X_f.shape[1]

# 重新初始化并进行全量训练
num_params = 2 * n_f + 3
init_params_final = np.random.normal(0, 0.05, num_params)
init_params_final[-1] = 1.0

res_final = minimize(objective_function, init_params_final, args=(X_f, J_f, y_f, n_f), method='L-BFGS-B')
W = res_final.x

print("\n[最终模型参数解析]")
print(f"评委分权重 (w_judge): {W[-1]:.4f}")
print("-" * 50)
print(f"{'特征名称':<30} | {'Mu权重 (Mean)':<15} | {'Sigma权重 (Var)':<15}")
print("-" * 50)

# 打印各个特征的权重
# W结构: [w_mu(n_f), b_mu, w_sig(n_f), b_sig, w_judge]
w_mu = W[:n_f]
w_sig = W[n_f+1 : 2*n_f+1]

for i, name in enumerate(feat_names):
    print(f"{name:<30} | {w_mu[i]:>15.4f} | {w_sig[i]:>15.4f}")

# 保存模型参数及特征名
np.savez('dwts_ranking_model.npz', params=W, static_cols=feat_names)
print(f"\n模型已保存至 dwts_ranking_model.npz (包含 {n_f} 个特征)")
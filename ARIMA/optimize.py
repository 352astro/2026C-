import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau
from sklearn.model_selection import KFold


# ==========================================
# 1. 核心数学函数
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
# 2. 损失函数 (训练用)
# ==========================================
def objective_function(params, pools, n_feats):
    phi = params[-1]
    total_loss = 0
    num_samples = 10  # 蒙特卡洛采样次数
    for pool in pools:
        X, J, targets, weeks = pool['X'], pool['J'], pool['targets'], pool['week']
        mu, sigma = get_mu_sigma(X, params, n_feats)
        pool_loss = 0
        for _ in range(num_samples):
            # 1. 注入随机噪声项 (Reparameterization Trick)
            epsilon = np.random.normal(0, 1, size=len(mu))
            u_sampled = mu + phi * weeks + sigma * epsilon
            # 2. 转化为百分比 V
            V = softmax(u_sampled)
            # 3. 计算排序损失
            total_score = V + J
            current_sample_loss = 0

            if targets.sum() > 0:
                elim_scores = total_score[targets == 1]  # 淘汰者得分（可能有多个，如亚军、季军）
                surv_scores = total_score[targets == 0]  # 晋级者得分（通常是冠军）

                if len(surv_scores) > 0:
                    # 对每一个淘汰者，都要比晋级者分数低
                    for e_s in elim_scores:
                        diff = e_s - surv_scores
                        # 累加当前样本的损失
                        current_sample_loss += np.log1p(np.exp(100 * diff)).mean()
            pool_loss += current_sample_loss

        total_loss += pool_loss / num_samples # 取采样平均值作为期望损失
    return total_loss


# ==========================================
# 3. 修改后的评估逻辑：验证准确率 + Kendall's Tau
# ==========================================
def evaluate_season_consistency(params, season_df, static_cols):
    """
    计算单个赛季的预测准确率和排序一致性(Kendall's Tau)
    """
    n_feats = len(static_cols)
    phi = params[-1]

    # --- 1. 计算周准确率 ---
    correct_weeks = 0
    total_weeks = 0

    # 存储每位选手的综合实力评分（用于计算赛季排名一致性）
    # 我们记录选手在最后一次出现时的预测总分（判定生死的分数）
    player_final_strengths = {}
    player_real_placements = {}

    for w_id, group in season_df.groupby('week'):
        X = group[static_cols].values
        J = group['Judge_Share'].values
        targets = group['is_eliminated'].values

        mu, _ = get_mu_sigma(X, params, n_feats)
        v_latent = mu + phi * w_id
        V = softmax(v_latent)
        total_score = V + J

        # 周准确率判定
        if targets.sum() > 0:
            pred_elim_idx = np.argmin(total_score)
            # 真实淘汰者（可能有多个，取第一个作为代表）
            true_elim_indices = np.where(targets == 1)[0]
            if pred_elim_idx in true_elim_indices:
                correct_weeks += 1
            total_weeks += 1

        # 记录每位选手在该周的得分，用于后续计算赛季总排序
        for i, idx in enumerate(group['index']):
            player_final_strengths[idx] = total_score[i]
            player_real_placements[idx] = group['placement'].iloc[i]

    # --- 2. 计算 Kendall's Tau ---
    # 真实的 placement (1st=1, 2nd=2...)，值越小排名越高
    # 模型预测的 strength，值越大排名越高
    # 所以我们需要对比 placement 和 -strength
    indices = list(player_final_strengths.keys())
    pred_ranks = [-player_final_strengths[i] for i in indices]
    true_ranks = [player_real_placements[i] for i in indices]

    tau, _ = kendalltau(pred_ranks, true_ranks)

    acc = correct_weeks / total_weeks if total_weeks > 0 else 0
    return acc, tau


# ==========================================
# 4. 交叉验证
# ==========================================
def run_cv_with_consistency(df, k=5):
    industry_cols = [c for c in df.columns if 'industry_' in c]
    static_cols = ['log_fame', 'age_scaled'] + industry_cols
    n_feats = len(static_cols)

    seasons = sorted(df['season'].unique())
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    results = []

    print(f"开始 {k} 折交叉验证...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(seasons)):
        train_seasons = [seasons[i] for i in train_idx]
        test_seasons = [seasons[i] for i in test_idx]

        # 构造训练竞争池
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

        # 训练
        init_params = np.zeros(2 * n_feats + 3)
        res = minimize(objective_function, init_params, args=(train_pools, n_feats),
                       method='L-BFGS-B', options={'maxiter': 200})

        # 验证测试集中的每一个赛季
        fold_accs = []
        fold_taus = []
        for s_id in test_seasons:
            season_df = df[df['season'] == s_id]
            acc, tau = evaluate_season_consistency(res.x, season_df, static_cols)
            fold_accs.append(acc)
            if not np.isnan(tau): fold_taus.append(tau)

        results.append({
            'fold': fold + 1,
            'mean_acc': np.mean(fold_accs),
            'mean_tau': np.mean(fold_taus)
        })
        print(f"Fold {fold + 1} | 准确率: {np.mean(fold_accs):.4f} | Kendall's tau: {np.mean(fold_taus):.4f}")

    # 打印最终统计
    avg_acc = np.mean([r['mean_acc'] for r in results])
    avg_tau = np.mean([r['mean_tau'] for r in results])
    std_tau = np.std([r['mean_tau'] for r in results])

    print("\n" + "=" * 40)
    print("总体一致性验证统计")
    print("=" * 40)
    print(f"平均周淘汰准确率: {avg_acc:.4f}")
    print(f"Kendall's tau (排序一致性):")
    print(f"  平均值: {avg_tau:.4f} ± {std_tau:.4f}")

    # 最终在全量数据上训练
    final_pools = []
    for (s, w), group in df.groupby(['season', 'week']):
        if len(group) >= 2:
            final_pools.append({
                'X': group[static_cols].values, 'J': group['Judge_Share'].values,
                'targets': group['is_eliminated'].values, 'week': group['week'].values
            })
    final_res = minimize(objective_function, np.zeros(2 * n_feats + 3), args=(final_pools, n_feats),
                         method='L-BFGS-B', options={'maxiter': 500})

    return final_res, static_cols


# ==========================================
# 5. 主程序运行
# ==========================================
if __name__ == "__main__":
    # 使用预处理过的包含 placement 和 is_eliminated 的数据
    df_all = pd.read_csv('2026_MCM_Problem_ARIMA_Data_SeasonAware.csv')

    # 只使用争议赛季 3-27 进行交叉验证
    df_train = df_all[df_all['season'].between(3, 27)].copy()

    final_params, static_cols = run_cv_with_consistency(df_train, k=5)

    print("\n[优化任务完成]")
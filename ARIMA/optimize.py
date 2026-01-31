import pandas as pd
import numpy as np
from scipy.optimize import minimize


# 1. 稳健的数学函数
def safe_exp(x):
    # 防止指数爆炸
    return np.exp(np.clip(x, -20, 20))


def softmax(x):
    # 减去最大值以提高数值稳定性
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_mu_sigma(X, params, n_feats):
    """
    根据手写笔记：
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


# 2. 损失函数 (Ranking Loss)
def objective_function(params, pools, n_feats):
    phi = params[-1]  # ARIMA 趋势系数（代表随时间推移的关注度积累）
    total_loss = 0

    for pool in pools:
        # 每个 pool 现在代表同一个赛季的某一特定周
        X = pool['X']
        J = pool['J']
        targets = pool['targets']
        weeks = pool['week']  # 该 pool 的周次（数组）

        # 计算 mu 和 sigma
        mu, _ = get_mu_sigma(X, params, n_feats)

        # 应用 ARIMA 趋势假设
        # v_latent = 基础人气 + 时间演化增益
        v_latent = mu + phi * weeks

        # 【核心修改】在本季本周的竞争池内归一化
        # 即使不同赛季的周次相同，它们也会在各自的循环中计算 Softmax
        V = softmax(v_latent)

        # 总分判定（结合评委占比）
        total_score = V + J

        if targets.sum() > 0:
            elim_score = total_score[targets == 1].min()
            surv_scores = total_score[targets == 0]

            if len(surv_scores) > 0:
                # Ranking Loss: 确保被淘汰者分数低于晋级者
                diff = elim_score - surv_scores
                # 使用 100 作为缩放因子强化排序硬约束
                loss = np.log1p(np.exp(100 * diff)).mean()
                total_loss += loss

    return total_loss


# 3. 执行优化
def solve_dwts_model(file_path):
    # 读取预处理过的数据（包含 season 和 is_eliminated）
    df = pd.read_csv(file_path)

    industry_cols = [c for c in df.columns if 'industry_' in c]
    static_cols = ['log_fame', 'age_scaled'] + industry_cols
    n_feats = len(static_cols)

    # 【重要修改】按 (season, week) 分组构建竞争池
    pools = []
    # 确保在同一个赛季、同一周的选手被放在一个 pool 中
    for (s_id, w_id), group in df.groupby(['season', 'week']):
        if len(group) < 2: continue
        pools.append({
            'X': group[static_cols].values,
            'J': group['Judge_Share'].values,
            'targets': group['is_eliminated'].values,
            'week': group['week'].values
        })

    # 初始化参数: [w_mu(n), b_mu, w_sig(n), b_sig, phi]
    init_params = np.zeros(2 * n_feats + 3)
    init_params[-1] = 0.05  # 给趋势项 phi 一个初始值

    print(f"数据读取成功: {len(df)} 行, {len(pools)} 个独立竞争池.")
    print("正在通过跨赛季数据优化通用人气权重...")

    res = minimize(
        objective_function,
        init_params,
        args=(pools, n_feats),
        method='L-BFGS-B',
        options={'maxiter': 500, 'disp': True}
    )

    return res, df, static_cols


# 结果预测与 V 值导出
def export_estimated_votes(res, df, static_cols):
    n_feats = len(static_cols)
    params = res.x
    phi = params[-1]

    estimated_list = []

    # 预测时同样必须按 (season, week) 分组，保证每一组内 V 之和为 1
    for (s_id, w_id), group in df.groupby(['season', 'week']):
        X = group[static_cols].values
        mu, sigma = get_mu_sigma(X, params, n_feats)
        v_latent = mu + phi * group['week'].values

        V_share = softmax(v_latent)

        output = group[['index', 'season', 'week', 'Judge_Share', 'is_eliminated']].copy()
        output['Estimated_Fan_Share'] = V_share
        output['Certainty_Sigma'] = sigma
        output['Combined_Score'] = V_share + group['Judge_Share']
        estimated_list.append(output)

    final_df = pd.concat(estimated_list).sort_values(['season', 'week', 'index'])
    return final_df


# 主程序
if __name__ == "__main__":
    filename = '2026_MCM_Problem_ARIMA_Data_SeasonAware.csv'

    res, original_df, static_cols = solve_dwts_model(filename)

    if res.success:
        print("\n[优化成功]")
        final_results = export_estimated_votes(res, original_df, static_cols)

        print("\n--- 学习到的模型特征权重 (mu) ---")
        for col, weight in zip(static_cols, res.x[0:len(static_cols)]):
            print(f"{col:25} : {weight: .4f}")

        print(f"\n--- 学习到的趋势系数 (phi) ---: {res.x[-1]:.4f}")

        print("\n估算的观众投票百分比预览 (前10行):")
        print(final_results[['season', 'week', 'Judge_Share', 'Estimated_Fan_Share', 'Certainty_Sigma']].head(10))

        final_results.to_csv('Estimated_Audience_Votes_Final.csv', index=False)
        print("\n完整结果已保存至: Estimated_Audience_Votes_Final.csv")
    else:
        print("优化失败:", res.message)
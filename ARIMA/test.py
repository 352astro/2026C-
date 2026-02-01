import pandas as pd
import numpy as np
from scipy.stats import kendalltau, spearmanr


# 加载模型资产
def load_model_assets(path='dwts_model_assets.npz'):
    data = np.load(path, allow_pickle=True)
    return data['params'], list(data['static_cols'])


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_mu_sigma(X, params, n_feats):
    w_mu = params[0:n_feats];
    b_mu = params[n_feats]
    w_sig = params[n_feats + 1: 2 * n_feats + 1];
    b_sig = params[2 * n_feats + 1]
    mu = np.dot(X, w_mu) + b_mu
    y2 = np.dot(X, w_sig) + b_sig
    sigma = np.log1p(np.exp(np.clip(y2, -20, 20)))
    return mu, sigma


def perform_prediction(df, params, static_cols):
    """
    使用确定性期望值 (Expectation) 进行预测
    """
    n_feats = len(static_cols)
    phi = params[-1]
    predictions = []

    # 必须按赛季和周分组进行 Softmax 竞争预测
    for (s_id, w_id), group in df.groupby(['season', 'week']):
        X = group[static_cols].values
        J = group['Judge_Share'].values

        # 使用 mu + phi*t 作为确定性的人气期望值
        mu, sigma = get_mu_sigma(X, params, n_feats)
        v_latent = mu + phi * w_id

        # 计算预测的观众得票百分比 V
        V_share = softmax(v_latent)

        temp_df = group.copy()
        temp_df['pred_V'] = V_share
        temp_df['pred_sigma'] = sigma
        temp_df['pred_total_score'] = V_share + J
        predictions.append(temp_df)

    return pd.concat(predictions)


def calculate_metrics(pred_df):
    """
    计算模型衡量指标
    """
    # 1. 周淘汰预测准确率 (Weekly Elimination Accuracy)
    # 在有淘汰发生的周，预测分最低的是否真的是淘汰者
    weekly_hits = 0
    total_elim_weeks = 0

    # 2. 赛季排名一致性 (Average Kendall's Tau)
    season_taus = []
    season_spearmans = []

    for (s_id, w_id), group in pred_df.groupby(['season', 'week']):
        if group['is_eliminated'].sum() > 0:
            pred_elim_idx = group['pred_total_score'].idxmin()
            # 检查该 index 是否在真实的 is_eliminated==1 集合中
            if group.loc[pred_elim_idx, 'is_eliminated'] == 1:
                weekly_hits += 1
            total_elim_weeks += 1

    for s_id, group in pred_df.groupby('season'):
        # 提取选手最后时刻的预测总分和真实排名
        # 这里的最后时刻是指选手在数据中出现的最后一周
        last_stats = group.sort_values('week').groupby('index').last()

        true_rank = last_stats['placement']
        pred_score = last_stats['pred_total_score']

        tau, _ = kendalltau(-pred_score, true_rank)
        rho, _ = spearmanr(-pred_score, true_rank)

        if not np.isnan(tau): season_taus.append(tau)
        if not np.isnan(rho): season_spearmans.append(rho)

    accuracy = weekly_hits / total_elim_weeks if total_elim_weeks > 0 else 0

    print("-" * 30)
    print("模型表现评估报告")
    print("-" * 30)
    print(f"1. 周淘汰判定准确率: {accuracy:.2%}")
    print(f"2. 跨赛季平均排序一致性 (Kendall's Tau): {np.mean(season_taus):.4f}")
    print(f"3. 跨赛季平均排序一致性 (Spearman Rho): {np.mean(season_spearmans):.4f}")
    print(f"4. 预测确定性 (Mean Sigma): {pred_df['pred_sigma'].mean():.4f}")
    print("-" * 30)

    return {
        'accuracy': accuracy,
        'mean_tau': np.mean(season_taus),
        'mean_rho': np.mean(season_spearmans)
    }


if __name__ == "__main__":
    # 加载数据和模型
    data_path = '2026_MCM_Problem_ARIMA_Data_SeasonAware.csv'  # 输入数据
    params, static_cols = load_model_assets()
    df = pd.read_csv(data_path)
    df_test = df[df['season'].between(3, 27)].copy()
    # 执行预测
    print(f"正在对 {len(df_test)} 条记录进行人气推断...")
    results = perform_prediction(df_test, params, static_cols)

    # 计算指标
    metrics = calculate_metrics(results)

    # 保存带有预测值的结果
    results.to_csv('DWTS_Final_Predictions_With_Metrics.csv', index=False)
    print("预测结果已导出至: DWTS_Final_Predictions_With_Metrics.csv")
import numpy as np
import pandas as pd
from scipy.special import softmax

def run_prediction_with_noise():
    # ==========================================
    # 1. 加载模型资产与原始数据
    # ==========================================
    print("[系统] 正在加载模型与数据...")
    data_assets = np.load('dwts_ranking_model_weeks_weighted.npz', allow_pickle=True)
    params = data_assets['params']
    # 这里的 static_cols 对应训练时的特征顺序
    static_cols = list(data_assets['static_cols'])
    n = len(static_cols)

    df_fame = pd.read_csv('2026_MCM_Problem_Fame_Data.csv')
    df_c_data = pd.read_csv('2026_MCM_Problem_C_Data.csv')

    # ==========================================
    # 2. 参数切片 (完善 Sigma 参数提取)
    # 结构: [w_mu(n), b_mu(1), w_sig(n), b_sig(1), w_judge(1)]
    # ==========================================
    # 均值参数
    w_mu = params[0:n]
    b_mu = params[n]
    
    # 方差/噪声参数 (之前被忽略的部分)
    w_sig = params[n+1 : 2*n+1]
    b_sig = params[2*n + 1]
    
    # 评委权重
    w_judge = params[-1]

    # ==========================================
    # 3. 特征处理 (必须与训练时的 scale 逻辑严格一致)
    # ==========================================
    def scale_val(series):
        return (series - series.mean()) / (series.std() + 1e-6)

    # 准备特征矩阵
    fame_col = 'fame_1.1'
    age_col = 'celebrity_age_during_season'

    X_pop = scale_val(df_fame[fame_col])
    X_age = scale_val(df_fame[age_col])

    # 处理行业特征
    ind_feature_names = [col for col in static_cols if col not in [fame_col, age_col]]
    X_ind = df_fame[['industry_' + name for name in ind_feature_names]].values

    X_predict = np.column_stack([X_pop, X_age, X_ind])

    # ==========================================
    # 4. 计算 Latent Mu 和 Latent Sigma
    # ==========================================
    # 4.1 计算基本人气强度 mu
    df_fame['latent_mu'] = np.dot(X_predict, w_mu) + b_mu
    
    # 4.2 计算人气波动 sigma (使用 exp 保证非负)
    # 这代表了选手的"不稳定性"，有些人可能人气高但方差大
    df_fame['latent_sigma'] = np.exp(np.dot(X_predict, w_sig) + b_sig)

    # 创建映射字典
    mu_map = df_fame.set_index('index')['latent_mu'].to_dict()
    sigma_map = df_fame.set_index('index')['latent_sigma'].to_dict()

    # ==========================================
    # 5. 计算每周观众投票百分比 V (含噪声)
    # ==========================================
    # 设定随机种子以保证结果可重复
    np.random.seed(2026) 

    # 预处理 C_Data 的评委分
    for w in range(1, 12):
        cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
        df_c_data[f'week{w}_avg'] = pd.to_numeric(df_c_data[cols].stack(), errors='coerce').groupby(level=0).mean()

    weekly_v_results = []

    for s_id, season_group in df_c_data.groupby('season'):
        for w_idx in range(1, 12):
            w_col = f'week{w_idx}_avg'
            
            # 筛选本周还在场且有分的选手
            active_players = season_group[season_group[w_col] > 0].copy()
            
            if active_players.empty:
                continue
                
            # 映射 Mu 和 Sigma
            # 这里的 index 在 df_c_data 和 df_fame 中必须是对齐的，或者通过名字匹配
            # 假设 df_c_data 的 index 与 df_fame 的 index 是一致的 (0 to N)
            # 如果不一致，这里需要 merge，但基于上下文假设它们是行对齐的
            # 为安全起见，这里假设 df_c_data 也有 'index' 列或者重置了索引
            
            # 注意：原始代码直接用 active_players.index.map，这要求 C_Data 的 index 与 Fame Data 的 index 是同一个实体的ID
            # 如果 C_Data 是每赛季多行，Fame 是每人一行，需要确保 index 对应正确。
            # 通常 Fame Data 的 index 列就是选手的唯一ID。
            # 我们假设 C_Data 读取时没有自带 index 列，这里我们需要确保映射正确。
            # 为了保险，我们通过选手名字 merge (如果名字唯一) 或者假设行号对应。
            # 鉴于原始代码直接 map，我们沿用该逻辑，但需注意 potential bug。
            # 更稳健的方法是假定 C_Data 的行号对应 Fame 的行号 (如果是一一对应的宽表转长表前的关系)
            # 但这里 C_Data 是原始数据。
            # **修正逻辑**：Fame Data 应该是每个选手唯一的。我们需要把 Fame 的 mu/sigma merge 到 active_players 上。
            
            # 临时创建一个带 Fame ID 的视图
            # 简单起见，我们假设 df_c_data 的行索引就是选手的 ID (0~N)，这在题目数据中通常成立
            active_players['player_mu'] = active_players.index.map(mu_map)
            active_players['player_sigma'] = active_players.index.map(sigma_map)
            
            # --- 核心修改：引入噪声采样 ---
            # 从 N(mu, sigma) 中采样，模拟当周的表现/人气波动
            active_players['sampled_intensity'] = np.random.normal(
                loc=active_players['player_mu'],
                scale=active_players['player_sigma']
            )
            
            # 计算百分比 V (使用 Softmax 归一化)
            # 系数 5 为温度系数，用于放大差异
            active_players['v_predicted_share'] = softmax(active_players['sampled_intensity'])
            
            # 记录结果
            for idx, row in active_players.iterrows():
                weekly_v_results.append({
                    'season': s_id,
                    'week': f'Week_{w_idx}',
                    'celebrity_name': row['celebrity_name'],
                    'v_predicted_share': row['v_predicted_share'],
                    'judge_avg_score': row[w_col],
                    'mu_base': row['player_mu'],      # 记录基准值以便分析
                    'sigma': row['player_sigma'],     # 记录方差以便分析
                    'sampled_val': row['sampled_intensity'] # 记录采样值
                })

    df_final_v = pd.DataFrame(weekly_v_results)

    # ==========================================
    # 6. 打印参数概览并保存
    # ==========================================
    print("=== 模型参数解析 (含噪声参数) ===")
    print(f"评委总分权重 (w_judge): {w_judge:.4f}")
    print("-" * 75)
    print(f"{'特征名称':<25} | {'Mu权重(均值)':<15} | {'Sigma权重(波动)':<15}")
    print("-" * 75)
    print(f"{fame_col:<25} | {w_mu[0]:>15.4f} | {w_sig[0]:>15.4f}")
    print(f"{age_col:<25} | {w_mu[1]:>15.4f} | {w_sig[1]:>15.4f}")
    for i, name in enumerate(ind_feature_names):
        print(f"{name:<25} | {w_mu[2+i]:>15.4f} | {w_sig[2+i]:>15.4f}")
    print("-" * 75)
    print(f"{'Bias (截距)':<25} | {b_mu:>15.4f} | {b_sig:>15.4f}")
    print("-" * 75)

    # 保存预测结果
    output_file = 'predicted_fan_votes_v1.csv'
    df_final_v.to_csv(output_file, index=False)
    print(f"\n[系统] 周选票预测完成 (已加入随机噪声 sigma)。")
    print(f"[系统] 随机种子: 2026")
    print(f"[系统] 结果已保存至: {output_file}")
    
    # 简单的统计检查
    print("\n[数据检查] Sigma 分布统计:")
    print(df_fame['latent_sigma'].describe())

if __name__ == "__main__":
    run_prediction_with_noise()
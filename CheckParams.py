import pandas as pd
import numpy as np
from scipy.special import softmax

def run_prediction():
    # ==========================================
    # 1. 加载模型参数与原始数据
    # ==========================================
    try:
        # 加载优化后的模型参数
        # 注意：文件名可能需要根据实际使用的参数文件调整，这里保持原名
        model_data = np.load('dwts_ranking_model.npz', allow_pickle=True)
        params = model_data['params']
        
        # 加载原始数据
        df_fame = pd.read_csv('2026_MCM_Problem_Fame_Data.csv')
        df_c = pd.read_csv('2026_MCM_Problem_C_Data.csv')
        
        print("数据加载成功。")
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}")
        return

    # ==========================================
    # 2. 特征工程 (保持与 optimize.py 一致)
    # ==========================================
    
    # 提取原始特征列
    # 假设 df_fame 和 df_c 是行对齐的（行索引 0 对应同一个选手）
    pop_raw = df_fame['fame_1.4'].values # 注意：请确认使用的 fame 版本与 CheckParams1 一致，此处沿用原代码
    age_raw = df_fame['celebrity_age_during_season'].values
    
    # 获取行业特征列
    ind_cols = [c for c in df_fame.columns if 'industry_' in c]
    ind_vals = df_fame[ind_cols].values
    
    # 定义标准化函数
    def scale(x): 
        return (x - x.mean()) / (x.std() + 1e-6)
    
    # 标准化
    pop_scaled = scale(pop_raw)
    age_scaled = scale(age_raw)
    
    # 构建特征矩阵 X (N_samples, N_features)
    X = np.column_stack([pop_scaled, age_scaled, ind_vals])
    
    # ==========================================
    # 3. 解析模型参数 (修改部分：增加 Sigma 参数解析)
    # ==========================================
    n_f = X.shape[1] # 特征数量
    
    # 参数结构假设: [w_mu(n), b_mu(1), w_sig(n), b_sig(1), w_judge(1)]
    # 指针位置追踪
    idx = 0
    
    # 提取 Mu (均值) 参数
    w_mu = params[idx : idx + n_f]
    idx += n_f
    b_mu = params[idx]
    idx += 1
    
    # 提取 Sigma (方差) 参数 - 原代码缺失部分
    w_sig = params[idx : idx + n_f]
    idx += n_f
    b_sig = params[idx]
    idx += 1
    
    # 提取评委权重 (虽然计算V时主要用不到，但为了完整性解析)
    # w_judge = params[idx]
    
    # ==========================================
    # 4. 计算潜在粉丝力量 (Mu) 和 波动性 (Sigma)
    # ==========================================
    # Mu: 基础吸票能力
    latent_mu = np.dot(X, w_mu) + b_mu
    
    # Sigma: 人气波动范围 (使用 exp 保证非负)
    latent_sigma = np.exp(np.dot(X, w_sig) + b_sig)
    
    # 将计算出的参数贴回 df_c
    df_c['player_mu'] = latent_mu
    df_c['player_sigma'] = latent_sigma
    
    # ==========================================
    # 5. 逐周生成预测数据 (引入随机采样)
    # ==========================================
    # 设置随机种子保证结果可复现 (与 CheckParams1 保持一致)
    np.random.seed(2026)
    
    output_rows = []
    
    seasons = sorted(df_c['season'].unique())
    
    for season in seasons:
        # 获取该赛季所有选手
        season_df = df_c[df_c['season'] == season].copy()
        
        # 遍历 Week 1 到 Week 11
        for week_num in range(1, 12):
            # 构建列名匹配模式
            judge_cols = [f'week{week_num}_judge{j}_score' for j in range(1, 5)]
            
            # 检查该周是否存在数据列
            existing_cols = [c for c in judge_cols if c in season_df.columns]
            if not existing_cols:
                continue
                
            # 计算当周评委平均分
            week_scores = season_df[existing_cols].apply(pd.to_numeric, errors='coerce')
            season_df['current_week_judge_avg'] = week_scores.mean(axis=1)
            
            # 筛选当周“存活”的选手 (评委分 > 0)
            active_dancers = season_df[season_df['current_week_judge_avg'] > 0].copy()
            
            if active_dancers.empty:
                continue
            
            # === 核心逻辑修改：引入噪声采样 ===
            mus = active_dancers['player_mu'].values
            sigs = active_dancers['player_sigma'].values
            
            # 从正态分布 N(mu, sigma) 中采样，模拟当周表现
            sampled_vals = np.random.normal(loc=mus, scale=sigs)
            
            # 使用 Softmax 计算份额
            shares = softmax(sampled_vals)
            
            # 保存结果 (结构与 predicted_fan_votes_v1.csv 一致)
            for idx, (original_idx, row) in enumerate(active_dancers.iterrows()):
                output_rows.append({
                    'season': row['season'],
                    'week': f'Week_{week_num}',
                    'celebrity_name': row['celebrity_name'],
                    'v_predicted_share': shares[idx],
                    'judge_avg_score': row['current_week_judge_avg'],
                    'mu_base': mus[idx],          # 新增列
                    'sigma': sigs[idx],           # 新增列
                    'sampled_val': sampled_vals[idx] # 新增列
                })

    # ==========================================
    # 6. 导出结果
    # ==========================================
    df_output = pd.DataFrame(output_rows)
    
    # 格式化输出文件名 (保持与 v1 一致或自定义)
    output_filename = 'predicted_fan_votes_v.csv'
    df_output.to_csv(output_filename, index=False)
    
    print(f"预测完成！")
    print(f"已生成文件: {output_filename}")
    print(f"包含记录数: {len(df_output)}")
    print(f"列名检查: {list(df_output.columns)}")
    print(f"样例数据:\n{df_output.head()}")

if __name__ == "__main__":
    run_prediction()
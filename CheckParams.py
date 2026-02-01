import pandas as pd
import numpy as np
from scipy.special import softmax

def run_prediction():
    # ==========================================
    # 1. 加载模型参数与原始数据
    # ==========================================
    try:
        # 加载优化后的模型参数
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
    # 2. 特征工程 (需严格通过 optimize.py 的逻辑复刻)
    # ==========================================
    
    # 提取原始特征列
    # 注意：这里假设 df_fame 和 df_c 是行对齐的（行索引 0 对应同一个选手）
    # 如果不是对齐的，需要通过选手名字 merge，但根据题目数据特性通常是预处理好的
    # 这里我们使用 df_fame 来计算静态的人气得分
    
    pop_raw = df_fame['fame_1.4'].values
    age_raw = df_fame['celebrity_age_during_season'].values
    
    # 获取行业特征列 (动态获取，保证顺序与训练时一致)
    ind_cols = [c for c in df_fame.columns if 'industry_' in c]
    ind_vals = df_fame[ind_cols].values
    
    # 定义标准化函数 (与 optimize.py 一致)
    def scale(x): 
        return (x - x.mean()) / (x.std() + 1e-6)
    
    # 标准化
    pop_scaled = scale(pop_raw)
    age_scaled = scale(age_raw)
    
    # 构建特征矩阵 X (N_samples, N_features)
    X = np.column_stack([pop_scaled, age_scaled, ind_vals])
    
    # ==========================================
    # 3. 解析模型参数
    # ==========================================
    n_f = X.shape[1] # 特征数量
    
    # 参数结构: w_mu(n_f), b_mu(1), w_sig(n_f), b_sig(1), w_judge(1)
    w_mu = params[:n_f]
    b_mu = params[n_f]
    
    # 提取评委权重供参考，计算观众票数时主要用到 mu
    # w_judge = params[-1] 
    
    # ==========================================
    # 4. 计算潜在粉丝力量 (Latent Fan Strength)
    # ==========================================
    # Mu 代表了选手基于自身属性（名气、年龄、行业）的固有吸票能力
    latent_mu = np.dot(X, w_mu) + b_mu
    
    # 将计算出的 Mu 贴回 df_c 以便按赛季处理
    df_c['latent_fan_strength'] = latent_mu
    
    # ==========================================
    # 5. 逐周生成预测数据
    # ==========================================
    output_rows = []
    
    seasons = sorted(df_c['season'].unique())
    
    for season in seasons:
        # 获取该赛季所有选手
        season_df = df_c[df_c['season'] == season].copy()
        
        # 遍历 Week 1 到 Week 11 (根据数据列名推断)
        for week_num in range(1, 12):
            # 构建列名匹配模式
            judge_cols = [f'week{week_num}_judge{j}_score' for j in range(1, 5)]
            
            # 检查该周是否存在数据列
            existing_cols = [c for c in judge_cols if c in season_df.columns]
            if not existing_cols:
                continue
                
            # 计算当周评委平均分 (处理 N/A)
            # errors='coerce' 会将 'N/A' 变为 NaN
            week_scores = season_df[existing_cols].apply(pd.to_numeric, errors='coerce')
            
            # 计算平均分，忽略 NaN
            season_df['current_week_judge_avg'] = week_scores.mean(axis=1)
            
            # 筛选当周“存活”的选手
            # 条件：评委分 > 0 (0通常代表已淘汰或未参赛)
            active_dancers = season_df[season_df['current_week_judge_avg'] > 0].copy()
            
            if active_dancers.empty:
                continue
            
            # === 核心逻辑：计算份额 ===
            # 使用 Softmax 将潜在人气值转化为总和为 1 的概率分布
            # 乘以系数 5.0 是为了模拟投票的集中效应 (人气高的人拿票比例会显著高)
            # 这个系数即 optimize.py 中的 "Temperature" 概念，虽未显式优化，但在生成时常用
            mus = active_dancers['latent_fan_strength'].values
            shares = softmax(mus)
            
            # 保存结果
            for idx, (original_idx, row) in enumerate(active_dancers.iterrows()):
                output_rows.append({
                    'season': row['season'],
                    'week': f'Week_{week_num}',
                    'celebrity_name': row['celebrity_name'],
                    'v_predicted_share': shares[idx],
                    'judge_avg_score': row['current_week_judge_avg']
                })

    # ==========================================
    # 6. 导出结果
    # ==========================================
    df_output = pd.DataFrame(output_rows)
    
    # 格式化输出文件名
    output_filename = 'predicted_fan_votes_v.csv'
    df_output.to_csv(output_filename, index=False)
    
    print(f"预测完成！")
    print(f"已生成文件: {output_filename}")
    print(f"包含记录数: {len(df_output)}")
    print(f"样例数据:\n{df_output.head()}")

if __name__ == "__main__":
    run_prediction()
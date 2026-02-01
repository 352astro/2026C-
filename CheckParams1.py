import numpy as np
import pandas as pd
from scipy.special import softmax

# ==========================================
# 1. 加载模型资产与原始数据
# ==========================================
data_assets = np.load('dwts_ranking_model_weeks_weighted.npz', allow_pickle=True)
params = data_assets['params']
# 这里的 static_cols 对应训练时的特征顺序
static_cols = list(data_assets['static_cols'])
n = len(static_cols)

df_fame = pd.read_csv('2026_MCM_Problem_Fame_Data.csv')
df_c_data = pd.read_csv('2026_MCM_Problem_C_Data.csv')

# ==========================================
# 2. 参数切片 (对应 optimize1.py 的结构)
# [w_mu(n), b_mu(1), w_sig(n), b_sig(1), w_judge(1)]
# ==========================================
w_mu = params[0:n]
b_mu = params[n]
w_judge = params[-1]

# ==========================================
# 3. 特征处理 (必须与训练时的 scale 逻辑严格一致)
# ==========================================
def scale_val(series):
    return (series - series.mean()) / (series.std() + 1e-6)

# 准备特征矩阵
# 注意：static_cols 里的名称已在保存时去掉了 'industry_' 前缀
# 我们需要匹配 df_fame 里的列名
fame_col = 'fame_1.1'
age_col = 'celebrity_age_during_season'

X_pop = scale_val(df_fame[fame_col])
X_age = scale_val(df_fame[age_col])

# 处理行业特征
ind_feature_names = [col for col in static_cols if col not in [fame_col, age_col]]
# 在 df_fame 中找回带 industry_ 前缀的列
X_ind = df_fame[['industry_' + name for name in ind_feature_names]].values

X_predict = np.column_stack([X_pop, X_age, X_ind])

# 计算潜伏观众缘强度 mu (Intensity per week)
# 这个值代表选手的基本面人气
df_fame['latent_mu'] = np.dot(X_predict, w_mu) + b_mu
mu_map = df_fame.set_index('index')['latent_mu'].to_dict()

# ==========================================
# 4. 计算每周观众投票百分比 V
# ==========================================
# 将 C_Data 转换为长表以便逐周处理
judge_cols = [f'week{i}_avg' for i in range(1, 12)]

# 预处理 C_Data 的评委分（用于判断选手是否在场）
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
            
        # 映射 mu
        active_players['player_mu'] = active_players.index.map(mu_map)
        
        # 计算百分比 V (使用 Softmax 归一化)
        # 我们使用系数 5 来放大差异，使其更符合投票分布
        active_players['v_predicted_share'] = softmax(active_players['player_mu'] * 5)
        
        # 记录结果
        for _, row in active_players.iterrows():
            weekly_v_results.append({
                'season': s_id,
                'week': f'Week_{w_idx}',
                'celebrity_name': row['celebrity_name'],
                'v_predicted_share': row['v_predicted_share'],
                'judge_avg_score': row[w_col]
            })

df_final_v = pd.DataFrame(weekly_v_results)

# ==========================================
# 5. 打印参数概览并保存
# ==========================================
print("=== 模型参数解析 ===")
print(f"评委总分权重 (w_judge): {w_judge:.4f}")
print("-" * 55)
print(f"{'特征名称':<25} | {'周人气贡献(mu)':<15}")
print("-" * 55)
print(f"{fame_col:<25} | {w_mu[0]:>15.4f}")
print(f"{age_col:<25} | {w_mu[1]:>15.4f}")
for i, name in enumerate(ind_feature_names):
    print(f"{name:<25} | {w_mu[2+i]:>15.4f}")
print("-" * 55)

# 保存预测结果
df_final_v.to_csv('predicted_fan_votes_v1.csv', index=False)
print(f"\n[系统] 周选票预测完成。共计算 {len(df_final_v)} 条记录。")
print("[系统] 结果已保存至: predicted_fan_votes_v1.csv")
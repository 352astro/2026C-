import pandas as pd
import numpy as np
from scipy.stats import kendalltau, rankdata

# ==========================================
# 1. 数据加载与对齐
# ==========================================
# 加载原始数据
df_raw = pd.read_csv('2026_MCM_Problem_C_Data.csv')
# 加载预测的 V 分额
df_v = pd.read_csv('predicted_fan_votes_v1.csv')

# 处理周次名称对齐：将 "week1_judge1_score" 转换为 1
df_v['week_num'] = df_v['week'].str.extract('(\d+)').astype(int)

# 预处理原始数据的评委分：计算每周平均分
for w in range(1, 12):
    cols = [f'week{w}_judge{i}_score' for i in range(1, 5)]
    for c in cols:
        df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
    df_raw[f'week{w}_avg'] = df_raw[cols].mean(axis=1)

# ==========================================
# 2. 逐赛季计算平均值并预测排名（类似于Judge.py的平均基准逻辑）
# ==========================================
all_accuracies = []
all_taus = []

total_players = 0
total_correct = 0

# 仅针对 df_v 中的赛季进行验证
target_seasons = sorted(df_v['season'].unique())

print(f"{'Season':<10} | {'Exact Acc':<12} | {'Kendall Tau':<12} | {'Players':<8}")
print("-" * 55)

for s_id in target_seasons:
    season_data = df_raw[df_raw['season'] == s_id].copy()
    season_v = df_v[df_v['season'] == s_id]
    
    players = season_data['celebrity_name'].unique()
    
    # 计算每个选手的平均 J_share（跨其参与周）
    player_jshares = {name: [] for name in players}
    
    for w in range(1, 12):
        avg_col = f'week{w}_avg'
        active_df = season_data[~season_data[avg_col].isna()]
        
        if active_df.empty:
            continue
        
        total_avg = active_df[avg_col].sum()
        
        for _, row in active_df.iterrows():
            name = row['celebrity_name']
            j_share = row[avg_col] / total_avg
            player_jshares[name].append(j_share)
    
    # 每个选手的平均 J_share
    avg_j_dict = {name: np.mean(player_jshares[name]) if player_jshares[name] else np.nan for name in players}
    
    # 每个选手的平均 V_share
    avg_v_dict = season_v.groupby('celebrity_name')['v_predicted_share'].mean().to_dict()
    
    # 构建统计 DataFrame
    player_stats = pd.DataFrame({
        'celebrity_name': players,
        'placement': season_data.set_index('celebrity_name')['placement'].loc[players].values
    })
    
    player_stats['avg_J'] = player_stats['celebrity_name'].map(avg_j_dict)
    player_stats['avg_V'] = player_stats['celebrity_name'].map(avg_v_dict).fillna(0)
    player_stats['avg_total'] = player_stats['avg_J'] + player_stats['avg_V']
    
    # 移除 NaN（如果有）
    player_stats = player_stats.dropna(subset=['avg_total'])
    
    # 真实排名
    true_ranks = player_stats['placement'].values
    
    # 预测排名：基于平均总分排序（越高越好，取负号）
    pred_ranks = rankdata(-player_stats['avg_total'].values, method='ordinal')
    
    # 计算严格准确度 (Exact Match)
    matches = np.sum(pred_ranks == true_ranks)
    acc = matches / len(true_ranks) if len(true_ranks) > 0 else 0
    
    # 计算一致性 (Kendall's Tau)
    tau, _ = kendalltau(pred_ranks, true_ranks)
    
    # 汇总
    all_accuracies.append(acc)
    if not np.isnan(tau):
        all_taus.append(tau)
    
    total_correct += matches
    total_players += len(true_ranks)
    
    print(f"{s_id:<10} | {acc:<12.2%} | {tau:<12.4f} | {len(true_ranks):<8}")

# ==========================================
# 3. 打印最终统计结果
# ==========================================
overall_acc = total_correct / total_players if total_players > 0 else 0
mean_tau = np.mean(all_taus) if all_taus else np.nan

print("-" * 55)
print(f"最终基准统计 (Season {target_seasons[0]}-{target_seasons[-1]}):")
print(f"平均严格排名准确率 (Exact Match): {overall_acc:.2%}")
print(f"平均排序一致性 (Kendall's Tau): {mean_tau:.4f}")
print("-" * 55)
print("注：严格排名准确率指预测名次与实际名次完全一致的概率。基于平均 (V + J_share) 预测最终排名。")
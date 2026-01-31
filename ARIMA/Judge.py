import pandas as pd
import numpy as np
from scipy.stats import kendalltau, rankdata

def run_baseline_evaluation(file_path):
    # 1. 加载数据
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return

    # 2. 筛选 Season 3-27 (基准测试区间)
    df_baseline = df[df['season'].between(3, 27)].copy()

    # 3. 按赛季和选手进行聚合
    # 计算每位选手的平均评委分份额 (Judge_Share)
    # 同时保留真实的最终排名 (placement)
    player_stats = df_baseline.groupby(['season', 'index']).agg({
        'Judge_Share': 'mean',
        'placement': 'first'
    }).reset_index()

    all_accuracies = []
    all_taus = []
    
    total_players = 0
    total_correct = 0

    # 4. 逐个赛季计算指标
    seasons = sorted(player_stats['season'].unique())
    
    print(f"{'Season':<10} | {'Exact Acc':<12} | {'Kendall Tau':<12} | {'Players':<8}")
    print("-" * 55)

    for s_id in seasons:
        season_group = player_stats[player_stats['season'] == s_id].copy()
        
        # 真实排名
        true_ranks = season_group['placement'].values
        
        # 预测排名：基于平均评委分进行排序
        # 注意：评委分越高，排名越靠前（数字越小），所以取负号
        # method='ordinal' 确保排名是唯一的整数 (1, 2, 3...)
        pred_ranks = rankdata(-season_group['Judge_Share'].values, method='ordinal')
        
        # 计算严格准确度 (Exact Match)
        matches = np.sum(pred_ranks == true_ranks)
        acc = matches / len(true_ranks)
        
        # 计算一致性 (Kendall's Tau)
        tau, _ = kendalltau(pred_ranks, true_ranks)
        
        # 汇总
        all_accuracies.append(acc)
        if not np.isnan(tau):
            all_taus.append(tau)
        
        total_correct += matches
        total_players += len(true_ranks)
        
        print(f"{s_id:<10} | {acc:<12.2%} | {tau:<12.4f} | {len(true_ranks):<8}")

    # 5. 打印最终统计结果
    overall_acc = total_correct / total_players
    mean_tau = np.mean(all_taus)

    print("-" * 55)
    print(f"最终基准统计 (Season 3-27):")
    print(f"平均严格排名准确率 (Exact Match): {overall_acc:.2%}")
    print(f"平均排序一致性 (Kendall's Tau): {mean_tau:.4f}")
    print("-" * 55)
    print("注：严格排名准确率指预测名次与实际名次完全一致的概率。")

if __name__ == "__main__":
    data_file = '2026_MCM_Problem_ARIMA_Data_SeasonAware.csv'
    run_baseline_evaluation(data_file)
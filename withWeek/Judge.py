import pandas as pd
import numpy as np
from scipy.stats import kendalltau, rankdata

def run_fame_score_evaluation(fame_file_path, meta_file_path):
    print(f"正在加载 Fame 数据: {fame_file_path}")
    print(f"正在加载 Meta 数据(用于获取赛季信息): {meta_file_path}")

    # 1. 加载数据
    try:
        df_fame = pd.read_csv(fame_file_path)
        df_meta = pd.read_csv(meta_file_path)
    except FileNotFoundError as e:
        print(f"错误：未找到文件 - {e}")
        return

    # 2. 数据合并与预处理
    # Fame Data 中只有 index，没有 season。我们需要从 Meta Data (SeasonAware) 中获取 season 信息。
    # 建立 index -> season 的映射
    season_map = df_meta[['index', 'season']].drop_duplicates()
    
    # 将赛季信息合并到 Fame 数据中
    df_eval = pd.merge(df_fame, season_map, on='index', how='inner')

    # 3. 筛选 Season 3-27 (基准测试区间)
    df_baseline = df_eval[df_eval['season'].between(3, 27)].copy()

    # 4. 准备存储结果
    all_accuracies = []
    all_taus = []
    
    total_players = 0
    total_correct = 0

    # 5. 逐个赛季计算指标
    seasons = sorted(df_baseline['season'].unique())
    
    print("-" * 65)
    print(f"{'Season':<8} | {'Exact Acc':<12} | {'Kendall Tau':<12} | {'Players':<8}")
    print("-" * 65)

    for s_id in seasons:
        season_group = df_baseline[df_baseline['season'] == s_id].copy()
        
        if len(season_group) < 2:
            continue

        # --- 真实排名 (Ground Truth) ---
        # elimination_order 数值越大，代表留得越久（名次越好/数字越小）
        # 例如：10人比赛，Winner的order是10，First out的order是1
        # 取负号 rankdata，使最大的 order 变为 rank 1
        true_ranks = rankdata(-season_group['elimination_order'].values, method='ordinal')
        
        # --- 预测排名 (Prediction) ---
        # 基于总分 score_sum 进行排序
        # 总分越高，预测名次越靠前 (Rank 1)
        pred_ranks = rankdata(-season_group['score_sum'].values, method='ordinal')
        
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
        
        print(f"{s_id:<8} | {acc:<12.2%} | {tau:<12.4f} | {len(true_ranks):<8}")

    # 6. 打印最终统计结果
    overall_acc = total_correct / total_players if total_players > 0 else 0
    mean_tau = np.mean(all_taus) if all_taus else 0

    print("-" * 65)
    print(f"基于总分(Score Sum)的基准统计 (Season 3-27):")
    print(f"平均严格排名准确率 (Exact Match): {overall_acc:.2%}")
    print(f"平均排序一致性 (Kendall's Tau): {mean_tau:.4f}")
    print("-" * 65)
    print("逻辑说明：")
    print("1. 真实排名：基于 elimination_order (数值越大，名次越靠前)")
    print("2. 预测排名：基于 score_sum (总分越高，预测名次越靠前)")

if __name__ == "__main__":
    # 请确保这两个文件都在当前目录下
    fame_data_file = '2026_MCM_Problem_Fame_Data.csv'
    season_data_file = '2026_MCM_Problem_ARIMA_Data_SeasonAware.csv'
    
    run_fame_score_evaluation(fame_data_file, season_data_file)
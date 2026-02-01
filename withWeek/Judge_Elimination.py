import pandas as pd
import numpy as np
from scipy.stats import kendalltau, rankdata
import re
import warnings

# 忽略一些pandas的切片警告
warnings.filterwarnings("ignore")

def extract_elimination_week(result_str):
    """
    从 results 列中提取淘汰周数。
    例如: "Eliminated Week 4" -> 4
    冠军、亚军等非淘汰状态返回 99 (代表坚持到了最后)
    退赛 (Withdrew) 返回 -1
    """
    if pd.isna(result_str):
        return 99
    
    result_str = str(result_str).lower()
    
    if "withdrew" in result_str:
        return -1
    if "eliminated week" in result_str:
        try:
            # 提取数字
            match = re.search(r'eliminated week (\d+)', result_str)
            if match:
                return int(match.group(1))
        except:
            return 99
    
    # 冠军、亚军、季军等
    return 99

def run_judge_baseline(file_path):
    print(f"正在加载数据: {file_path} ...")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return

    # --- 1. 数据预处理 ---
    
    # 提取淘汰周
    df['Actual_Elim_Week'] = df['results'].apply(extract_elimination_week)
    
    # 确保 placement 是数值型
    df['placement'] = pd.to_numeric(df['placement'], errors='coerce')
    
    # 填充非分数列的 N/A 为 0 以便计算，但要在计算时排除没参赛的人
    # 获取所有周的评委打分列
    score_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
    
    # 将分数转换为数值，无法转换的变为NaN
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    seasons = sorted(df['season'].unique())
    
    global_correct_preds = 0
    global_total_eliminations = 0
    global_taus = []

    print("-" * 80)
    print(f"{'Season':<8} | {'Weeks':<6} | {'Elim Acc':<12} | {'Rank Tau':<12} | {'Note'}")
    print("-" * 80)

    # --- 2. 逐赛季、逐周分析 ---
    for season in seasons:
        season_df = df[df['season'] == season].copy()
        
        season_correct = 0
        season_elims = 0
        season_taus = []
        
        # 确定该赛季最大周数 (通过列名判断，或者直接循环1-15)
        # 这里我们扫描该赛季每一周是否有分数数据
        max_week = 0
        for i in range(1, 16):
            cols = [c for c in season_df.columns if f'week{i}_' in c]
            if len(cols) > 0 and season_df[cols].sum().sum() > 0:
                max_week = i
        
        # 遍历该赛季的每一周
        for week in range(1, max_week + 1):
            # 获取该周评委分数相关列
            current_week_cols = [c for c in season_df.columns if f'week{week}_' in c and 'judge' in c]
            
            if not current_week_cols:
                continue

            # --- A. 筛选本周参赛选手 ---
            # 逻辑：该周评委总分 > 0 且 之前没有被淘汰
            # 计算该周选手的评委总分
            season_df[f'Week_{week}_Total'] = season_df[current_week_cols].sum(axis=1)
            
            # 活跃选手：分数大于0 且 (未被淘汰 OR 淘汰周 >= 当前周) 且 (未退赛)
            active_dancers = season_df[
                (season_df[f'Week_{week}_Total'] > 0) & 
                (season_df['Actual_Elim_Week'] != -1) & # 排除退赛
                (
                    (season_df['Actual_Elim_Week'] == 99) | 
                    (season_df['Actual_Elim_Week'] >= week)
                )
            ].copy()
            
            if len(active_dancers) < 2:
                continue # 决赛或无人竞争，跳过

            # --- B. 计算基准指标：百分比/分数排名 ---
            # 评委分数越高，Rank数值越小 (1为最高分)
            # method='min' 处理并列情况
            active_dancers['Judge_Rank'] = rankdata(-active_dancers[f'Week_{week}_Total'], method='min')
            
            # 计算 Kendall's Tau: 本周评委排名 vs 最终赛季排名(Placement)
            # 注意：Placement 1是最好的，Judge_Rank 1也是最好的，应该正相关
            tau, _ = kendalltau(active_dancers['Judge_Rank'], active_dancers['placement'])
            if not np.isnan(tau):
                season_taus.append(tau)
                global_taus.append(tau)

            # --- C. 预测淘汰 ---
            # 真实淘汰者：Actual_Elim_Week 正好等于当前 Week 的人
            actual_eliminated = active_dancers[active_dancers['Actual_Elim_Week'] == week]
            
            if len(actual_eliminated) > 0:
                season_elims += 1
                global_total_eliminations += 1
                
                # 预测淘汰者：评委分数最低的人 (Judge_Rank 数值最大的人)
                # 找到最低分
                min_score = active_dancers[f'Week_{week}_Total'].min()
                predicted_eliminated = active_dancers[active_dancers[f'Week_{week}_Total'] == min_score]
                
                # 检查是否命中
                # 只要实际被淘汰的人里，有一个是评委打分最低的，就算预测正确
                # (处理双人淘汰或并列低分的情况)
                hit = any(name in predicted_eliminated['celebrity_name'].values for name in actual_eliminated['celebrity_name'].values)
                
                if hit:
                    season_correct += 1
                    global_correct_preds += 1

        # 赛季汇总输出
        acc_str = "N/A"
        if season_elims > 0:
            acc = season_correct / season_elims
            acc_str = f"{acc:.2%}"
        
        tau_str = "N/A"
        if season_taus:
            avg_tau = np.mean(season_taus)
            tau_str = f"{avg_tau:.4f}"
            
        print(f"{season:<8} | {max_week:<6} | {acc_str:<12} | {tau_str:<12} | Eliminated: {season_elims}")

    # --- 3. 总体统计 ---
    print("-" * 80)
    print("最终基准统计 (Judge Score Only Baseline):")
    if global_total_eliminations > 0:
        print(f"总淘汰预测准确率 (Accuracy): {global_correct_preds / global_total_eliminations:.2%} ({global_correct_preds}/{global_total_eliminations})")
    else:
        print("无有效淘汰数据")
        
    if global_taus:
        print(f"平均排名一致性 (Kendall's Tau): {np.mean(global_taus):.4f}")
    else:
        print("无有效排名数据")
    print("-" * 80)
    print("说明:")
    print("1. Accuracy: 仅依据当周评委给出的最低分预测淘汰。如果实际淘汰者是该周评委分最低者之一，记为正确。")
    print("2. Rank Tau: 衡量当周评委打分排名与选手最终赛季排名的相关性(1.0为完全正相关)。")
    print("   Tau值越高，说明评委打分越能反映选手的最终实力/名次。")

if __name__ == "__main__":
    data_file = '2026_MCM_Problem_C_Data.csv'
    run_judge_baseline(data_file)
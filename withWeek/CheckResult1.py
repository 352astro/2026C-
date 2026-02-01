import pandas as pd
import numpy as np
from scipy.stats import kendalltau, rankdata
import re

def parse_elimination_info(row):
    """
    解析 results 列，返回淘汰发生的周数。
    如果未淘汰（冠军/亚军/季军），返回 99 (代表坚持到了最后)。
    如果退赛 (Withdrew)，返回 -1。
    """
    res = str(row['results']).lower()
    if 'withdrew' in res:
        return -1
    
    # 匹配 "Eliminated Week X"
    match = re.search(r'eliminated week (\d+)', res)
    if match:
        return int(match.group(1))
    
    # 如果是名次 (1st Place, etc.)，则认为这名选手打满了该赛季所有周
    if 'place' in res:
        return 99 # 代表最后一周
    
    return 99

def get_week_number(week_str):
    """将 'Week_1' 转换为整数 1"""
    try:
        return int(week_str.replace('Week_', ''))
    except:
        return 0

def calculate_metrics(merged_df, meta_df):
    """
    核心计算逻辑：结合评委分和预测粉丝分，判断淘汰准确性
    """
    seasons = sorted(merged_df['season'].unique())
    
    total_elimination_weeks = 0
    correct_predictions = 0     # 严格准确：预测倒数第一 = 实际淘汰
    top2_predictions = 0        # 宽松准确：实际淘汰者在预测的倒数前两名内
    
    season_taus = []            # 存储每个赛季的排名相关性
    
    print(f"{'Season':<6} | {'Week':<5} | {'Actual Eliminated':<20} | {'Pred Lowest':<20} | {'Result':<10} | {'In Bottom 2?'}")
    print("-" * 90)

    for season in seasons:
        # 获取该赛季数据
        season_data = merged_df[merged_df['season'] == season]
        # 获取该赛季的元数据（用于后续计算赛季总排名相关性）
        season_meta = meta_df[meta_df['season'] == season].copy()
        
        # 获取该赛季包含的所有周次
        weeks = sorted(season_data['week_num'].unique())
        
        # 用于累积该赛季选手的综合得分表现，计算Tau
        player_scores = {} # {name: sum_of_scores}

        for week in weeks:
            week_data = season_data[season_data['week_num'] == week].copy()
            
            # --- 1. 计算评委分数份额 ---
            # 注意：judge_avg_score 是平均分，我们需要份额
            total_judge_points = week_data['judge_avg_score'].sum()
            if total_judge_points == 0:
                continue
            
            week_data['judge_share'] = week_data['judge_avg_score'] / total_judge_points
            
            # --- 2. 计算综合得分 (Composite Score) ---
            # 假设权重 50/50，直接相加份额
            week_data['composite_score'] = week_data['judge_share'] + week_data['v_predicted_share']
            
            # 记录分数用于后续计算Tau（简单的累加，或者取最后一周的排名）
            for _, r in week_data.iterrows():
                if r['celebrity_name'] not in player_scores:
                    player_scores[r['celebrity_name']] = []
                player_scores[r['celebrity_name']].append(r['composite_score'])

            # --- 3. 找出真实淘汰者 ---
            # 在 meta_df 中找到本周被淘汰的人
            eliminated_players = season_meta[season_meta['elim_week'] == week]['celebrity_name'].tolist()
            
            # 如果本周没人淘汰（或数据缺失），跳过评估
            if not eliminated_players:
                continue

            total_elimination_weeks += 1
            
            # --- 4. 找出预测的淘汰者 (分数最低) ---
            # 按综合分数升序排列 (分数越低越危险)
            week_data = week_data.sort_values(by='composite_score', ascending=True)
            
            pred_lowest = week_data.iloc[0]['celebrity_name']
            pred_bottom2 = week_data.iloc[0:2]['celebrity_name'].tolist()
            
            # --- 5. 对比 ---
            # 只要实际淘汰名单中有一个人是预测的最低分，就算严格正确
            is_strict_correct = any(p == pred_lowest for p in eliminated_players)
            
            # 只要实际淘汰名单中有一个人在预测的倒数前两名，就算宽松正确
            is_soft_correct = any(p in pred_bottom2 for p in eliminated_players)
            
            if is_strict_correct:
                correct_predictions += 1
            if is_soft_correct:
                top2_predictions += 1
            
            actual_str = ",".join(eliminated_players)
            res_str = "HIT" if is_strict_correct else "MISS"
            soft_str = "YES" if is_soft_correct else "NO"
            
            print(f"{season:<6} | {week:<5} | {actual_str[:20]:<20} | {pred_lowest[:20]:<20} | {res_str:<10} | {soft_str}")

        # --- 计算该赛季的一致性 (Consistency / Kendall's Tau) ---
        # 逻辑：比较模型生成的“平均综合得分排名”与“实际赛季最终排名(Placement)”
        if not season_meta.empty:
            # 计算模型对每个人的平均综合得分
            model_scores = []
            real_ranks = []
            
            # 仅计算在该赛季出现过的选手
            valid_players = season_meta[season_meta['celebrity_name'].isin(player_scores.keys())]
            
            for _, row in valid_players.iterrows():
                name = row['celebrity_name']
                # 使用该选手参赛期间的平均综合得分作为排序依据
                avg_comp_score = np.mean(player_scores[name])
                model_scores.append(avg_comp_score)
                real_ranks.append(row['placement'])
            
            if len(model_scores) > 1:
                # 预测排名：分数越高，Rank数值越小（1为最高）
                # 因此取负号进行rankdata，使得高分对应 Rank 1
                pred_ranks = rankdata([-s for s in model_scores], method='ordinal')
                
                # 真实排名：Placement 已经是 1, 2, 3...
                tau, _ = kendalltau(pred_ranks, real_ranks)
                if not np.isnan(tau):
                    season_taus.append(tau)

    print("-" * 90)
    print("\n========= 最终评估报告 =========")
    if total_elimination_weeks > 0:
        acc = correct_predictions / total_elimination_weeks
        soft_acc = top2_predictions / total_elimination_weeks
        print(f"总评估周数: {total_elimination_weeks}")
        print(f"严格准确率 (Accuracy - Top 1): {acc:.2%} ({correct_predictions}/{total_elimination_weeks})")
        print(f"  -> 定义: 实际淘汰者恰好是模型预测分最低的那个人")
        print(f"宽松准确率 (Accuracy - Top 2): {soft_acc:.2%} ({top2_predictions}/{total_elimination_weeks})")
        print(f"  -> 定义: 实际淘汰者位于模型预测分最低的两人之中")
    else:
        print("未找到有效的淘汰周数据进行对比。")

    if season_taus:
        avg_tau = np.mean(season_taus)
        print(f"平均一致性 (Avg Kendall's Tau): {avg_tau:.4f}")
        print(f"  -> 定义: 赛季内选手'模型平均得分排名'与'实际最终排名'的相关性 (-1~1)")
    else:
        print("无法计算排名一致性。")
    print("================================")

def main():
    # 文件路径
    pred_file = 'predicted_fan_votes_v1.csv'
    actual_data_file = '2026_MCM_Problem_C_Data.csv'

    print("1. 加载预测数据...")
    try:
        df_pred = pd.read_csv(pred_file)
        df_pred['week_num'] = df_pred['week'].apply(get_week_number)
    except Exception as e:
        print(f"加载预测文件失败: {e}")
        return

    print("2. 加载真实数据...")
    try:
        df_actual = pd.read_csv(actual_data_file)
    except Exception as e:
        print(f"加载真实数据文件失败: {e}")
        return

    # 预处理真实数据
    # 我们需要两个信息：
    # 1. 每周该选手的淘汰状态 (用于判断 Accuracy)
    # 2. 选手的最终 Placement (用于判断 Consistency)
    
    # 提取淘汰周
    df_actual['elim_week'] = df_actual.apply(parse_elimination_info, axis=1)
    # 确保 placement 是数字
    df_actual['placement'] = pd.to_numeric(df_actual['placement'], errors='coerce')

    # 合并数据
    # 注意：df_pred 中有 judge_avg_score，我们直接用这个，也可以重新从 df_actual 计算
    # 为了方便，我们这里以 df_pred 为主，因为它已经按周展开了
    # 我们只需要把 df_actual 中的 meta 信息 (elim_week, placement) 关联到 df_pred 上
    # 关联键：season, celebrity_name
    
    # 修正名字匹配问题（去除可能存在的空格）
    df_pred['celebrity_name'] = df_pred['celebrity_name'].str.strip()
    df_actual['celebrity_name'] = df_actual['celebrity_name'].str.strip()

    merged_df = pd.merge(
        df_pred, 
        df_actual[['season', 'celebrity_name', 'elim_week', 'placement']], 
        on=['season', 'celebrity_name'], 
        how='left'
    )

    # 过滤掉无法匹配的数据
    if merged_df.isnull().any().any():
        print("警告：部分数据合并后存在缺失值，将自动忽略这些行。")
        # 实际调试时可以 print(merged_df[merged_df['placement'].isnull()]) 查看名字是否不匹配
    
    merged_df = merged_df.dropna(subset=['elim_week', 'placement'])

    # 运行评估
    print("3. 开始评估...")
    calculate_metrics(merged_df, df_actual)

if __name__ == "__main__":
    main()
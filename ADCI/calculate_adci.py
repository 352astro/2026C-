import pandas as pd
import numpy as np
import math

# 读取数据
print("正在读取数据...")
fan_votes_df = pd.read_csv('predicted_fan_votes_v2.csv')
elimination_results_df = pd.read_csv('elimination_results.csv')
controversial_df = pd.read_csv('controversial_contestants_comparison.csv')

# 提取争议案例的season和index
controversial_cases = controversial_df[['season', 'index', 'celebrity_name']].values.tolist()
print(f"找到 {len(controversial_cases)} 个争议案例")

# 创建一个字典来存储每个选手的淘汰次序
elimination_dict = {}
for _, row in elimination_results_df.iterrows():
    key = (int(row['season']), int(row['index']))
    elimination_dict[key] = {
        'elimination_order': row['elimination_order']
    }

# 处理week列，提取周数
def extract_week_number(week_str):
    """从'Week_1'格式中提取数字1"""
    if isinstance(week_str, str) and week_str.startswith('Week_'):
        return int(week_str.split('_')[1])
    return int(week_str)

fan_votes_df['week_num'] = fan_votes_df['week'].apply(extract_week_number)

# 从fan_votes_df中计算每个选手的周数（最大week_num）
print("正在计算每个选手的参赛周数...")
contestant_weeks = fan_votes_df.groupby(['season', 'index'])['week_num'].max().reset_index()
for _, row in contestant_weeks.iterrows():
    key = (int(row['season']), int(row['index']))
    if key in elimination_dict:
        elimination_dict[key]['weeks'] = int(row['week_num'])

# 计算ADCI的函数
def calculate_adci_for_week(season, week_num, fan_votes_data):
    """
    计算某一周的ADCI指标
    
    参数:
    - season: 赛季号
    - week_num: 周数
    - fan_votes_data: 该周的所有观众投票数据
    
    返回:
    - 包含每个选手ADCI值的DataFrame
    """
    # 获取该周的所有参赛选手
    week_data = fan_votes_data[
        (fan_votes_data['season'] == season) & 
        (fan_votes_data['week_num'] == week_num)
    ].copy()
    
    if len(week_data) == 0:
        return pd.DataFrame()
    
    # 计算评委评分百分比
    judge_scores = week_data['judge_avg_score'].values
    judge_total = judge_scores.sum()
    week_data['judge_percentage'] = judge_scores / judge_total if judge_total > 0 else 0
    
    # 观众投票百分比（已经存在）
    week_data['fan_percentage'] = week_data['v_predicted_share']
    
    # 计算争议基数 x = |观众投票百分比 - 评委投票百分比|
    week_data['controversy_base'] = abs(week_data['fan_percentage'] - week_data['judge_percentage'])
    
    # 确定该周被淘汰的选手
    # 根据elimination_order，找出该周应该被淘汰的选手
    # 如果elimination_order <= week_num，说明该选手在该周或之前被淘汰
    # 但我们需要找出该周具体淘汰的是谁
    
    # 获取该周所有选手的index
    week_indices = week_data['index'].values
    
    # 找出该周被淘汰的选手（elimination_order == week_num的选手）
    eliminated_this_week = []
    for idx in week_indices:
        key = (season, idx)
        if key in elimination_dict:
            elim_order = elimination_dict[key]['elimination_order']
            if elim_order == week_num:
                eliminated_this_week.append(idx)
    
    # 找出评委得分最低的分数
    min_judge_score = week_data['judge_avg_score'].min()
    
    # 计算a值：针对每个选手单独计算
    # 如果该选手是评委得分最低的，且该选手未被淘汰（当前规则），则a=5，否则a=1
    # 这样所有选手都有一个争议系数
    def calculate_a(row):
        # 判断该选手是否是评委得分最低的
        is_min_score = (row['judge_avg_score'] == min_judge_score)
        # 判断该选手是否未被淘汰
        is_not_eliminated = (row['index'] not in eliminated_this_week)
        # 如果是最低分且未被淘汰，则a=5，否则a=1
        return 5 if (is_min_score and is_not_eliminated) else 1
    
    week_data['a'] = week_data.apply(calculate_a, axis=1)
    week_data['x'] = week_data['controversy_base']
    # 计算原始ADCI = a * exp(x)
    week_data['ADCI_raw'] = week_data['a'] * np.exp(week_data['x'])
    # 先计算原始ADCI，归一化会在最后统一进行
    week_data['ADCI'] = week_data['ADCI_raw']
    
    # 添加周数和淘汰信息
    week_data['week'] = week_num
    week_data['eliminated_this_week'] = week_data['index'].isin(eliminated_this_week)
    week_data['min_judge_score'] = (week_data['judge_avg_score'] == min_judge_score)
    
    return week_data

# 计算所有争议案例的ADCI
print("\n正在计算争议案例的ADCI...")
controversial_adci_results = []

for season, index, name in controversial_cases:
    print(f"\n处理争议案例: {name} (Season {season}, Index {index})")
    
    # 获取该选手的所有周数据
    contestant_data = fan_votes_df[
        (fan_votes_df['season'] == season) & 
        (fan_votes_df['index'] == index)
    ]
    
    if len(contestant_data) == 0:
        print(f"  警告: 未找到该选手的数据")
        continue
    
    # 获取该选手的参赛周数
    key = (season, index)
    if key not in elimination_dict:
        print(f"  警告: 未找到该选手的淘汰次序信息")
        continue
    
    weeks_participated = elimination_dict[key]['weeks']
    
    # 计算该选手每一周的ADCI
    for week_num in range(1, weeks_participated + 1):
        week_adci = calculate_adci_for_week(
            season, week_num, 
            fan_votes_df
        )
        
        if len(week_adci) > 0:
            # 找到该选手在该周的数据
            contestant_week_data = week_adci[week_adci['index'] == index]
            if len(contestant_week_data) > 0:
                row = contestant_week_data.iloc[0]
                controversial_adci_results.append({
                    'season': season,
                    'index': index,
                    'celebrity_name': name,
                    'week': week_num,
                    'judge_score': row['judge_avg_score'],
                    'judge_percentage': row['judge_percentage'],
                    'fan_percentage': row['fan_percentage'],
                    'controversy_base': row['controversy_base'],
                    'a': row['a'],
                    'ADCI': row['ADCI'],
                    'eliminated_this_week': row['eliminated_this_week'],
                    'min_judge_score': row['min_judge_score']
                })

controversial_adci_df = pd.DataFrame(controversial_adci_results)

# 计算当赛季其他比赛的ADCI（用于对比）
print("\n正在计算当赛季其他比赛的ADCI...")
adci_season_detail = []

# 获取所有赛季
all_seasons = fan_votes_df['season'].unique()

for season in all_seasons:
    print(f"处理赛季 {season}...")
    
    # 获取该赛季的所有周数
    season_weeks = fan_votes_df[fan_votes_df['season'] == season]['week_num'].unique()
    
    for week_num in sorted(season_weeks):
        week_adci = calculate_adci_for_week(
            season, week_num,
            fan_votes_df
        )
        
        if len(week_adci) > 0:
            for _, row in week_adci.iterrows():
                adci_season_detail.append({
                    'season': season,
                    'index': row['index'],
                    'celebrity_name': row['celebrity_name'],
                    'week': week_num,
                    'ADCI': row['ADCI']
                })

all_seasons_adci_df = pd.DataFrame(adci_season_detail)

# 归一化ADCI到0-100范围
print("\n正在归一化ADCI到0-100范围...")
# 找到所有ADCI的最大值（只考虑ADCI>0的值）
all_adci_values = pd.concat([controversial_adci_df['ADCI'], all_seasons_adci_df['ADCI']])
max_adci = all_adci_values[all_adci_values > 0].max() if len(all_adci_values[all_adci_values > 0]) > 0 else 1.0
print(f"ADCI原始最大值: {max_adci:.4f}")

# 使用实际最大值进行归一化，使最大值映射到100
# 归一化公式: ADCI_normalized = (ADCI / max_ADCI) * 100
normalization_factor = max_adci
print(f"归一化因子: {normalization_factor:.4f}")

# 归一化到0-100
controversial_adci_df['ADCI'] = (controversial_adci_df['ADCI'] / normalization_factor) * 100
all_seasons_adci_df['ADCI'] = (all_seasons_adci_df['ADCI'] / normalization_factor) * 100

print(f"归一化后ADCI范围: [{all_seasons_adci_df['ADCI'].min():.2f}, {all_seasons_adci_df['ADCI'].max():.2f}]")

# 保存结果
print("\n正在保存结果...")
all_seasons_adci_df.to_csv('adci_season_detail.csv', index=False, encoding='utf-8-sig')

# 生成对比分析
print("\n生成对比分析...")
comparison_results = []

for season, index, name in controversial_cases:
    # 获取该争议案例的ADCI数据
    case_data = controversial_adci_df[
        (controversial_adci_df['season'] == season) & 
        (controversial_adci_df['index'] == index)
    ]
    
    if len(case_data) == 0:
        continue
    
    # 获取该赛季其他选手的ADCI数据
    season_other_data = all_seasons_adci_df[
        (all_seasons_adci_df['season'] == season) & 
        ~((all_seasons_adci_df['season'] == season) & (all_seasons_adci_df['index'] == index))
    ]
    
    # 计算统计信息
    case_avg_adci = case_data['ADCI'].mean()
    case_max_adci = case_data['ADCI'].max()
    
    other_avg_adci = season_other_data['ADCI'].mean() if len(season_other_data) > 0 else 0
    
    comparison_results.append({
        'season': season,
        'index': index,
        'celebrity_name': name,
        'case_avg_adci': case_avg_adci,
        'case_max_adci': case_max_adci,
        'season_other_avg_adci': other_avg_adci
    })

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv('adci_comparison_analysis.csv', index=False, encoding='utf-8-sig')

# 打印摘要
print("\n" + "="*80)
print("ADCI指标计算结果摘要")
print("="*80)

print("\n争议案例的ADCI统计:")
print(controversial_adci_df.groupby(['season', 'celebrity_name']).agg({
    'ADCI': ['mean', 'max', 'sum', 'count'],
    'a': 'sum'
}).round(4))

print("\n\n对比分析:")
print(comparison_df[['celebrity_name', 'case_avg_adci', 'case_max_adci', 
                     'season_other_avg_adci']].to_string(index=False))

# 生成每个赛季的ADCI均值统计
print("\n正在生成每个赛季的ADCI均值统计...")
season_adci_stats = []

for season in sorted(all_seasons_adci_df['season'].unique()):
    season_data = all_seasons_adci_df[all_seasons_adci_df['season'] == season]
    
    # 只计算均值
    season_adci_stats.append({
        'season': season,
        'avg_adci': season_data['ADCI'].mean()
    })

season_stats_df = pd.DataFrame(season_adci_stats)
season_stats_df.to_csv('adci_season_statistics.csv', index=False, encoding='utf-8-sig')

print("\n每个赛季的ADCI均值:")
print(season_stats_df.round(2).to_string(index=False))

print("\n\n结果已保存到:")
print("  - adci_season_detail.csv: 所有赛季的ADCI数据")
print("  - adci_comparison_analysis.csv: 对比分析结果")
print("  - adci_season_statistics.csv: 每个赛季的ADCI均值统计")


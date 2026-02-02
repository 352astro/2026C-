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

# 定义规则列表
rules = {
    'actual': {
        'column': 'elimination_order',
        'name': '实际淘汰次序',
        'suffix': 'actual'
    },
    'rule1': {
        'column': '淘汰次序_规则1_排名相加',
        'name': '规则1_排名相加',
        'suffix': 'rule1'
    },
    'rule2': {
        'column': '淘汰次序_规则2_百分比相加',
        'name': '规则2_百分比相加',
        'suffix': 'rule2'
    },
    'rule4': {
        'column': '淘汰次序_规则4_排名倒数两名评委决定',
        'name': '规则4_排名倒数两名评委决定',
        'suffix': 'rule4'
    }
}

# 处理week列，提取周数
def extract_week_number(week_str):
    """从'Week_1'格式中提取数字1"""
    if isinstance(week_str, str) and week_str.startswith('Week_'):
        return int(week_str.split('_')[1])
    return int(week_str)

fan_votes_df['week_num'] = fan_votes_df['week'].apply(extract_week_number)

# 为每个规则创建elimination_dict
def create_elimination_dict(rule_column):
    """为指定规则创建elimination_dict"""
    elimination_dict = {}
    for _, row in elimination_results_df.iterrows():
        key = (int(row['season']), int(row['index']))
        elimination_dict[key] = {
            'elimination_order': row[rule_column]
        }
    
    # 从fan_votes_df中计算每个选手的周数（最大week_num）
    contestant_weeks = fan_votes_df.groupby(['season', 'index'])['week_num'].max().reset_index()
    for _, row in contestant_weeks.iterrows():
        key = (int(row['season']), int(row['index']))
        if key in elimination_dict:
            elimination_dict[key]['weeks'] = int(row['week_num'])
    
    return elimination_dict

# 计算ADCI的函数
def calculate_adci_for_week(season, week_num, fan_votes_data, elimination_dict_for_rule):
    """
    计算某一周的ADCI指标
    
    参数:
    - season: 赛季号
    - week_num: 周数
    - fan_votes_data: 该周的所有观众投票数据
    - elimination_dict_for_rule: 该规则对应的淘汰次序字典
    
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
    
    # 获取该周所有选手的index
    week_indices = week_data['index'].values
    
    # 找出该周被淘汰的选手（elimination_order == week_num的选手）
    eliminated_this_week = []
    for idx in week_indices:
        key = (season, idx)
        if key in elimination_dict_for_rule:
            elim_order = elimination_dict_for_rule[key]['elimination_order']
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

# 为每个规则计算ADCI，并收集所有结果
all_seasons_adci_list = []
all_comparison_list = []
all_season_stats_list = []

for rule_key, rule_info in rules.items():
    print(f"\n{'='*80}")
    print(f"正在处理规则: {rule_info['name']}")
    print(f"{'='*80}")
    
    # 创建该规则的elimination_dict
    elimination_dict = create_elimination_dict(rule_info['column'])
    print(f"已创建 {rule_info['name']} 的淘汰次序字典")
    
    # 计算所有争议案例的ADCI
    print(f"\n正在计算争议案例的ADCI（{rule_info['name']}）...")
    controversial_adci_results = []
    
    for season, index, name in controversial_cases:
        # 获取该选手的参赛周数
        key = (season, index)
        if key not in elimination_dict:
            continue
        
        weeks_participated = elimination_dict[key]['weeks']
        
        # 计算该选手每一周的ADCI
        for week_num in range(1, weeks_participated + 1):
            week_adci = calculate_adci_for_week(
                season, week_num, 
                fan_votes_df,
                elimination_dict
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
    print(f"\n正在计算当赛季其他比赛的ADCI（{rule_info['name']}）...")
    adci_season_detail = []
    
    # 获取所有赛季
    all_seasons = fan_votes_df['season'].unique()
    
    for season in all_seasons:
        # 获取该赛季的所有周数
        season_weeks = fan_votes_df[fan_votes_df['season'] == season]['week_num'].unique()
        
        for week_num in sorted(season_weeks):
            week_adci = calculate_adci_for_week(
                season, week_num,
                fan_votes_df,
                elimination_dict
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
    print(f"\n正在归一化ADCI到0-100范围（{rule_info['name']}）...")
    # 找到所有ADCI的最大值（现在所有ADCI都>0，因为a最小为1）
    all_adci_values = pd.concat([controversial_adci_df['ADCI'], all_seasons_adci_df['ADCI']])
    max_adci = all_adci_values.max() if len(all_adci_values) > 0 else 1.0
    print(f"ADCI原始最大值: {max_adci:.4f}")
    
    # 使用实际最大值进行归一化，使最大值映射到100
    # 归一化公式: ADCI_normalized = (ADCI / max_ADCI) * 100
    normalization_factor = max_adci
    print(f"归一化因子: {normalization_factor:.4f}")
    
    # 归一化到0-100
    controversial_adci_df['ADCI'] = (controversial_adci_df['ADCI'] / normalization_factor) * 100
    all_seasons_adci_df['ADCI'] = (all_seasons_adci_df['ADCI'] / normalization_factor) * 100
    
    print(f"归一化后ADCI范围: [{all_seasons_adci_df['ADCI'].min():.2f}, {all_seasons_adci_df['ADCI'].max():.2f}]")
    
    # 添加规则标识列
    all_seasons_adci_df['rule'] = rule_info['name']
    all_seasons_adci_df['rule_key'] = rule_key
    
    # 收集结果到列表中
    all_seasons_adci_list.append(all_seasons_adci_df)
    
    # 生成对比分析
    print(f"\n生成对比分析（{rule_info['name']}）...")
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
    # 添加规则标识列
    comparison_df['rule'] = rule_info['name']
    comparison_df['rule_key'] = rule_key
    
    # 收集结果到列表中
    all_comparison_list.append(comparison_df)
    
    # 打印摘要
    print(f"\n{'='*80}")
    print(f"ADCI指标计算结果摘要（{rule_info['name']}）")
    print(f"{'='*80}")
    
    if len(controversial_adci_df) > 0:
        print("\n争议案例的ADCI统计:")
        print(controversial_adci_df.groupby(['season', 'celebrity_name']).agg({
            'ADCI': ['mean', 'max', 'sum', 'count'],
            'a': 'sum'
        }).round(4))
    
    print("\n\n对比分析:")
    print(comparison_df[['celebrity_name', 'case_avg_adci', 'case_max_adci', 
                         'season_other_avg_adci']].to_string(index=False))
    
    # 生成每个赛季的ADCI均值统计
    print(f"\n正在生成每个赛季的ADCI均值统计（{rule_info['name']}）...")
    season_adci_stats = []
    
    for season in sorted(all_seasons_adci_df['season'].unique()):
        season_data = all_seasons_adci_df[all_seasons_adci_df['season'] == season]
        
        # 只计算均值
        season_adci_stats.append({
            'season': season,
            'avg_adci': season_data['ADCI'].mean()
        })
    
    season_stats_df = pd.DataFrame(season_adci_stats)
    # 添加规则标识列
    season_stats_df['rule'] = rule_info['name']
    season_stats_df['rule_key'] = rule_key
    
    # 收集结果到列表中
    all_season_stats_list.append(season_stats_df)
    
    print("\n每个赛季的ADCI均值:")
    print(season_stats_df.round(2).to_string(index=False))

# 合并所有规则的结果
print("\n\n" + "="*80)
print("正在合并所有规则的结果...")
print("="*80)

    # 合并所有赛季详细数据，转换为宽格式
if all_seasons_adci_list:
    # 先合并所有数据
    all_data = pd.concat(all_seasons_adci_list, ignore_index=True)
    
    # 使用pivot将数据转换为宽格式，每个规则一个ADCI列
    integrated_season_detail = all_data.pivot_table(
        index=['season', 'index', 'celebrity_name', 'week'],
        columns='rule_key',
        values='ADCI',
        aggfunc='first'
    ).reset_index()
    
    # 重命名列，使ADCI列更清晰
    integrated_season_detail.columns.name = None
    integrated_season_detail = integrated_season_detail.rename(columns={
        'actual': 'ADCI_actual',
        'rule1': 'ADCI_rule1',
        'rule2': 'ADCI_rule2',
        'rule4': 'ADCI_rule4'
    })
    
    # 重新排列列的顺序
    cols = ['season', 'index', 'celebrity_name', 'week', 'ADCI_actual', 'ADCI_rule1', 'ADCI_rule2', 'ADCI_rule4']
    integrated_season_detail = integrated_season_detail[cols]
    
    # 按照每周比赛的方式排序：先按season，再按week，最后按index
    integrated_season_detail = integrated_season_detail.sort_values(['season', 'week', 'index']).reset_index(drop=True)
    
    integrated_season_detail.to_csv('adci_season_detail_integrated.csv', index=False, encoding='utf-8-sig')
    print(f"\n已保存整合的赛季详细数据: adci_season_detail_integrated.csv")
    print(f"  总记录数: {len(integrated_season_detail)}")
    print(f"  列: {list(integrated_season_detail.columns)}")
    print(f"  排序方式: 按season -> week -> index排序（每周比赛方式）")

# 合并所有对比分析数据，转换为宽格式
if all_comparison_list:
    # 先合并所有数据
    all_comparison_data = pd.concat(all_comparison_list, ignore_index=True)
    
    # 分别处理每个指标
    # case_avg_adci
    case_avg = all_comparison_data.pivot_table(
        index=['season', 'index', 'celebrity_name'],
        columns='rule_key',
        values='case_avg_adci',
        aggfunc='first'
    ).reset_index()
    case_avg.columns.name = None
    case_avg = case_avg.rename(columns={
        'actual': 'case_avg_adci_actual',
        'rule1': 'case_avg_adci_rule1',
        'rule2': 'case_avg_adci_rule2',
        'rule4': 'case_avg_adci_rule4'
    })
    
    # season_other_avg_adci
    season_other = all_comparison_data.pivot_table(
        index=['season', 'index', 'celebrity_name'],
        columns='rule_key',
        values='season_other_avg_adci',
        aggfunc='first'
    ).reset_index()
    season_other.columns.name = None
    season_other = season_other.rename(columns={
        'actual': 'season_other_avg_adci_actual',
        'rule1': 'season_other_avg_adci_rule1',
        'rule2': 'season_other_avg_adci_rule2',
        'rule4': 'season_other_avg_adci_rule4'
    })
    
    # 合并所有列（只保留平均ADCI和其他选手平均ADCI）
    integrated_comparison = case_avg.merge(season_other, on=['season', 'index', 'celebrity_name'])
    
    # 重新排列列的顺序：每个规则的case和other相邻摆放
    cols = ['season', 'index', 'celebrity_name',
            'case_avg_adci_actual', 'season_other_avg_adci_actual',
            'case_avg_adci_rule1', 'season_other_avg_adci_rule1',
            'case_avg_adci_rule2', 'season_other_avg_adci_rule2',
            'case_avg_adci_rule4', 'season_other_avg_adci_rule4']
    integrated_comparison = integrated_comparison[cols]
    
    integrated_comparison.to_csv('adci_comparison_analysis_integrated.csv', index=False, encoding='utf-8-sig')
    print(f"\n已保存整合的对比分析数据: adci_comparison_analysis_integrated.csv")
    print(f"  总记录数: {len(integrated_comparison)}")
    print(f"  列: {list(integrated_comparison.columns)}")

# 合并所有赛季统计数据，转换为宽格式
if all_season_stats_list:
    # 先合并所有数据
    all_stats_data = pd.concat(all_season_stats_list, ignore_index=True)
    
    # 使用pivot将数据转换为宽格式，每个规则一个avg_adci列
    integrated_season_stats = all_stats_data.pivot_table(
        index='season',
        columns='rule_key',
        values='avg_adci',
        aggfunc='first'
    ).reset_index()
    
    # 重命名列
    integrated_season_stats.columns.name = None
    integrated_season_stats = integrated_season_stats.rename(columns={
        'actual': 'avg_adci_actual',
        'rule1': 'avg_adci_rule1',
        'rule2': 'avg_adci_rule2',
        'rule4': 'avg_adci_rule4'
    })
    
    # 重新排列列的顺序
    cols = ['season', 'avg_adci_actual', 'avg_adci_rule1', 'avg_adci_rule2', 'avg_adci_rule4']
    integrated_season_stats = integrated_season_stats[cols]
    
    # 在最后添加一行计算所有赛季的均值
    mean_row = {
        'season': 'Overall',
        'avg_adci_actual': integrated_season_stats['avg_adci_actual'].mean(),
        'avg_adci_rule1': integrated_season_stats['avg_adci_rule1'].mean(),
        'avg_adci_rule2': integrated_season_stats['avg_adci_rule2'].mean(),
        'avg_adci_rule4': integrated_season_stats['avg_adci_rule4'].mean()
    }
    mean_df = pd.DataFrame([mean_row])
    integrated_season_stats = pd.concat([integrated_season_stats, mean_df], ignore_index=True)
    
    integrated_season_stats.to_csv('adci_season_statistics_integrated.csv', index=False, encoding='utf-8-sig')
    print(f"\n已保存整合的赛季统计数据: adci_season_statistics_integrated.csv")
    print(f"  总记录数: {len(integrated_season_stats)} (包含1行总体均值)")
    print(f"  列: {list(integrated_season_stats.columns)}")
    print(f"  总体均值: actual={mean_row['avg_adci_actual']:.2f}, rule1={mean_row['avg_adci_rule1']:.2f}, rule2={mean_row['avg_adci_rule2']:.2f}, rule4={mean_row['avg_adci_rule4']:.2f}")

print("\n\n" + "="*80)
print("所有规则的计算已完成！")
print("="*80)
print("\n生成的文件:")
print("  - adci_season_detail_integrated.csv: 所有规则的所有赛季ADCI数据（整合版）")
print("  - adci_comparison_analysis_integrated.csv: 所有规则的对比分析结果（整合版）")
print("  - adci_season_statistics_integrated.csv: 所有规则的每个赛季ADCI均值统计（整合版）")

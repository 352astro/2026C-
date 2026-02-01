"""
淘汰规则工具函数
包含三种淘汰规则：
1. 规则1：观众投票排名 + 评委评分排名，排名数值最大的淘汰
2. 规则2：观众投票百分比 + 评委评分百分比，百分比数值最小的淘汰
3. 规则3：只看粉丝投票数，投票数最小的淘汰
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def eliminate_by_rank_sum(season_data: pd.DataFrame) -> Dict[int, int]:
    """
    规则1：观众投票排名 + 评委评分排名，排名数值最大的淘汰
    
    参数:
        season_data: 包含该season所有周次数据的DataFrame
        必须包含列：season, week, index, v_predicted_share, judge_avg_score
    
    返回:
        Dict[选手index, 淘汰次序] - 淘汰次序从1开始，1表示第一个被淘汰
    """
    # 获取所有周次，按顺序排序
    weeks = sorted(season_data['week'].unique(), key=lambda x: int(x.split('_')[1]))
    
    # 获取所有选手
    all_contestants = set(season_data['index'].unique())
    active_contestants = all_contestants.copy()
    
    # 记录淘汰次序
    elimination_order = {}
    elimination_round = 1
    
    # 按周次进行比赛
    for week in weeks:
        if len(active_contestants) <= 1:
            # 只剩一人，自动成为冠军
            if len(active_contestants) == 1:
                remaining = list(active_contestants)[0]
                elimination_order[remaining] = elimination_round
            break
        
        # 获取本周参赛选手的数据
        week_data = season_data[
            (season_data['week'] == week) & 
            (season_data['index'].isin(active_contestants))
        ].copy()
        
        if len(week_data) == 0:
            continue
        
        # 计算观众投票排名（排名越小越好，1最好）
        week_data['fan_rank'] = week_data['v_predicted_share'].rank(ascending=False, method='min')
        
        # 计算评委评分排名（排名越小越好，1最好）
        week_data['judge_rank'] = week_data['judge_avg_score'].rank(ascending=False, method='min')
        
        # 计算综合排名（排名数值越大越差）
        week_data['combined_rank'] = week_data['fan_rank'] + week_data['judge_rank']
        
        # 找出综合排名最大的（最差的）
        eliminated = week_data.loc[week_data['combined_rank'].idxmax(), 'index']
        
        # 记录淘汰次序
        elimination_order[eliminated] = elimination_round
        elimination_round += 1
        
        # 从活跃选手中移除
        active_contestants.remove(eliminated)
    
    # 为未淘汰的选手（冠军）分配最后的排名
    for contestant in active_contestants:
        elimination_order[contestant] = elimination_round
    
    return elimination_order


def eliminate_by_percentage_sum(season_data: pd.DataFrame) -> Dict[int, int]:
    """
    规则2：观众投票百分比 + 评委评分百分比，百分比数值最小的淘汰
    
    参数:
        season_data: 包含该season所有周次数据的DataFrame
        必须包含列：season, week, index, v_predicted_share, judge_avg_score
    
    返回:
        Dict[选手index, 淘汰次序] - 淘汰次序从1开始，1表示第一个被淘汰
    """
    # 获取所有周次，按顺序排序
    weeks = sorted(season_data['week'].unique(), key=lambda x: int(x.split('_')[1]))
    
    # 获取所有选手
    all_contestants = set(season_data['index'].unique())
    active_contestants = all_contestants.copy()
    
    # 记录淘汰次序
    elimination_order = {}
    elimination_round = 1
    
    # 按周次进行比赛
    for week in weeks:
        if len(active_contestants) <= 1:
            # 只剩一人，自动成为冠军
            if len(active_contestants) == 1:
                remaining = list(active_contestants)[0]
                elimination_order[remaining] = elimination_round
            break
        
        # 获取本周参赛选手的数据
        week_data = season_data[
            (season_data['week'] == week) & 
            (season_data['index'].isin(active_contestants))
        ].copy()
        
        if len(week_data) == 0:
            continue
        
        # 观众投票百分比（已经是百分比形式）
        fan_percentage = week_data['v_predicted_share']
        
        # 将评委评分转换为百分比（归一化到0-1范围）
        judge_scores = week_data['judge_avg_score']
        min_score = judge_scores.min()
        max_score = judge_scores.max()
        if max_score > min_score:
            judge_percentage = (judge_scores - min_score) / (max_score - min_score)
        else:
            # 如果所有评分相同，则百分比都设为0.5
            judge_percentage = pd.Series([0.5] * len(judge_scores), index=judge_scores.index)
        
        # 计算综合百分比（数值越小越差）
        combined_percentage = fan_percentage + judge_percentage
        
        # 找出综合百分比最小的（最差的）
        eliminated = week_data.loc[combined_percentage.idxmin(), 'index']
        
        # 记录淘汰次序
        elimination_order[eliminated] = elimination_round
        elimination_round += 1
        
        # 从活跃选手中移除
        active_contestants.remove(eliminated)
    
    # 为未淘汰的选手（冠军）分配最后的排名
    for contestant in active_contestants:
        elimination_order[contestant] = elimination_round
    
    return elimination_order


def eliminate_by_fan_vote_only(season_data: pd.DataFrame) -> Dict[int, int]:
    """
    规则3：只看粉丝投票数，投票数最小的淘汰
    
    参数:
        season_data: 包含该season所有周次数据的DataFrame
        必须包含列：season, week, index, v_predicted_share, judge_avg_score
    
    返回:
        Dict[选手index, 淘汰次序] - 淘汰次序从1开始，1表示第一个被淘汰
    """
    # 获取所有周次，按顺序排序
    weeks = sorted(season_data['week'].unique(), key=lambda x: int(x.split('_')[1]))
    
    # 获取所有选手
    all_contestants = set(season_data['index'].unique())
    active_contestants = all_contestants.copy()
    
    # 记录淘汰次序
    elimination_order = {}
    elimination_round = 1
    
    # 按周次进行比赛
    for week in weeks:
        if len(active_contestants) <= 1:
            # 只剩一人，自动成为冠军
            if len(active_contestants) == 1:
                remaining = list(active_contestants)[0]
                elimination_order[remaining] = elimination_round
            break
        
        # 获取本周参赛选手的数据
        week_data = season_data[
            (season_data['week'] == week) & 
            (season_data['index'].isin(active_contestants))
        ].copy()
        
        if len(week_data) == 0:
            continue
        
        # 找出观众投票数最小的（最差的）
        eliminated = week_data.loc[week_data['v_predicted_share'].idxmin(), 'index']
        
        # 记录淘汰次序
        elimination_order[eliminated] = elimination_round
        elimination_round += 1
        
        # 从活跃选手中移除
        active_contestants.remove(eliminated)
    
    # 为未淘汰的选手（冠军）分配最后的排名
    for contestant in active_contestants:
        elimination_order[contestant] = elimination_round
    
    return elimination_order


def process_all_seasons(input_file: str, output_file: str):
    """
    处理所有season的数据，生成淘汰次序结果CSV
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
    """
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 获取所有season
    seasons = sorted(df['season'].unique())
    
    results = []
    
    for season in seasons:
        season_data = df[df['season'] == season].copy()
        
        # 使用规则1计算淘汰次序
        order_rule1 = eliminate_by_rank_sum(season_data)
        
        # 使用规则2计算淘汰次序
        order_rule2 = eliminate_by_percentage_sum(season_data)
        
        # 使用规则3计算淘汰次序
        order_rule3 = eliminate_by_fan_vote_only(season_data)
        
        # 获取该season的所有选手信息
        contestants_info = season_data[['index', 'celebrity_name']].drop_duplicates()
        
        # 添加结果
        for _, row in contestants_info.iterrows():
            index = row['index']
            name = row['celebrity_name']
            
            results.append({
                'season': season,
                'index': index,
                'celebrity_name': name,
                '淘汰次序_规则1_排名相加': order_rule1.get(index, None),
                '淘汰次序_规则2_百分比相加': order_rule2.get(index, None),
                '淘汰次序_规则3_只看粉丝投票': order_rule3.get(index, None)
            })
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(results)
    
    # 保存到CSV
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"结果已保存到: {output_file}")
    print(f"共处理 {len(seasons)} 个season")
    print(f"共 {len(result_df)} 条记录")
    
    return result_df


if __name__ == "__main__":
    # 处理数据并生成结果
    input_file = "predicted_fan_votes_v2.csv"
    output_file = "elimination_results.csv"
    
    result_df = process_all_seasons(input_file, output_file)
    
    # 显示前几行结果
    print("\n前10行结果:")
    print(result_df.head(10))


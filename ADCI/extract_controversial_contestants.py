"""
提取争议选手的数据：排名方式淘汰次序 vs 实际淘汰次序
"""

import pandas as pd

# 读取数据
predicted_df = pd.read_csv('predicted_fan_votes_v2.csv')
elimination_df = pd.read_csv('elimination_results.csv')
actual_df = pd.read_csv('../2026_MCM_Problem_Mean_Data.csv')

# 定义要查找的选手
target_contestants = [
    {'name': 'Jerry Rice', 'season': 2, 'index': 11, 'description': '第2赛季 – Jerry Rice，尽管在5周内评委评分最低，仍获得亚军'},
    {'name': 'Billy Ray Cyrus', 'season': 4, 'index': 35, 'description': '第4赛季 – Billy Ray Cyrus，尽管在6周内评委评分倒数第一，仍获得第5名'},
    {'name': 'Bristol Palin', 'season': 11, 'index': 126, 'description': '第11赛季 – Bristol Palin，曾12次获得最低评委评分，却获得第3名'},
    {'name': 'Bobby Bones', 'season': 27, 'index': 318, 'description': '第27赛季 – Bobby Bones，尽管评委评分持续走低，仍获得冠军'}
]

results = []

for contestant in target_contestants:
    season = contestant['season']
    index = contestant['index']
    name = contestant['name']
    
    # 从elimination_results.csv获取排名方式的淘汰次序
    rule1_data = elimination_df[(elimination_df['season'] == season) & 
                                  (elimination_df['index'] == index)]
    
    if len(rule1_data) > 0:
        rule1_order = rule1_data['淘汰次序_规则1_排名相加'].iloc[0]
    else:
        rule1_order = None
    
    # 从2026_MCM_Problem_Mean_Data.csv获取实际排名和淘汰次序
    actual_data = actual_df[(actual_df['season'] == season) & 
                            (actual_df['index'] == index)]
    
    if len(actual_data) > 0:
        actual_placement = actual_data['placement'].iloc[0]  # 实际排名（1=冠军，2=亚军等）
        actual_elimination_order = actual_data['elimination_order'].iloc[0]  # 实际淘汰次序
        weeks = actual_data['weeks'].iloc[0]  # 参赛周数
    else:
        actual_placement = None
        actual_elimination_order = None
        weeks = None
    
    # 计算差异：排名方式预测的淘汰次序 vs 实际淘汰次序
    # 注意：淘汰次序数值越大表示淘汰越晚（排名越好）
    if rule1_order is not None and actual_elimination_order is not None:
        order_difference = actual_elimination_order - rule1_order
        # 如果差异为正，说明实际表现比排名方式预测的更好
        # 如果差异为负，说明实际表现比排名方式预测的更差
    else:
        order_difference = None
    
    results.append({
        'season': season,
        'index': index,
        'celebrity_name': name,
        '描述': contestant['description'],
        '排名方式淘汰次序_规则1': rule1_order,
        '实际排名_placement': actual_placement,
        '实际淘汰次序_elimination_order': actual_elimination_order,
        '淘汰次序差异_实际减预测': order_difference,
        '参赛周数': weeks
    })

# 创建结果DataFrame
result_df = pd.DataFrame(results)

# 保存到CSV
output_file = 'controversial_contestants_comparison.csv'
result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"结果已保存到: {output_file}")
print(f"\n共提取 {len(results)} 位选手的数据:")
print("\n" + "="*100)
print(result_df.to_string(index=False))
print("="*100)


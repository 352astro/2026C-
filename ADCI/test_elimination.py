"""测试淘汰逻辑"""
import pandas as pd
from elimination_tools import eliminate_by_rank_sum, eliminate_by_percentage_sum, eliminate_by_fan_vote_only

# 读取数据
df = pd.read_csv('predicted_fan_votes_v2.csv')

# 测试Season 1
season1 = df[df['season'] == 1].copy()
print("Season 1 数据:")
print(season1[['week', 'index', 'celebrity_name', 'v_predicted_share', 'judge_avg_score']])
print("\n" + "="*80)

# 规则1
print("\n规则1淘汰次序 (排名相加):")
order1 = eliminate_by_rank_sum(season1)
for idx, order in sorted(order1.items(), key=lambda x: x[1]):
    name = season1[season1['index']==idx]['celebrity_name'].iloc[0]
    print(f"  淘汰次序{order}: {name} (index={idx})")

# 规则2
print("\n规则2淘汰次序 (百分比相加):")
order2 = eliminate_by_percentage_sum(season1)
for idx, order in sorted(order2.items(), key=lambda x: x[1]):
    name = season1[season1['index']==idx]['celebrity_name'].iloc[0]
    print(f"  淘汰次序{order}: {name} (index={idx})")

# 规则3
print("\n规则3淘汰次序 (只看粉丝投票):")
order3 = eliminate_by_fan_vote_only(season1)
for idx, order in sorted(order3.items(), key=lambda x: x[1]):
    name = season1[season1['index']==idx]['celebrity_name'].iloc[0]
    print(f"  淘汰次序{order}: {name} (index={idx})")


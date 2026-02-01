import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('2026_MCM_Problem_C_Data.csv')
data = data.reset_index()
# grouped = data.groupby('season')
print(data)
print(data.shape)

#给数据添加weeks标签，表明一次参与了多少周节目
data['weeks'] = 1
score_cols = [f'week{i}_judge1_score' for i in range(1, 12)]
temp_df = data[score_cols].replace(0, np.nan)
data['weeks'] = temp_df.count(axis=1)

# 将homestate和rigion合并，方便后续编码
data = data.rename(columns={'celebrity_homecountry/region':'region'})
data['region'] = data['celebrity_homestate'].fillna('') + ' ' + data['region'].fillna('')
data =data.drop(columns='celebrity_homestate')
data['region'] = data['region'].str.strip()

# 对rigion进行编码

# 对职业进行编码


#计算每周的平均分，替换繁杂的每周每个评委的分数
# avg_cols = ['index','celebrity_industry','region','celebrity_age_during_season','placement','season'] + [str(i) for i in range(1, 12)] + ['weeks']
avg_cols = ['index','celebrity_industry','celebrity_age_during_season','placement','season'] + [str(i) for i in range(1, 12)] + ['weeks']
processed_data = data.copy()
for i in range(1,12):
    temp_mean = data[i<=data['weeks']]
    cols = [f'week{i}_judge{j}_score' for j in range(1, 5)]
    temp_mean = temp_mean[cols].mean(axis=1)
    processed_data[str(i)] = temp_mean
processed_data = processed_data[avg_cols]

# 计算每周分数对应的百分比
# processed_percentage = processed_data.copy()
# cols = [str(i) for i in range(1, 12)]
# season_sums = processed_data.groupby('season')[cols].transform('sum')
# processed_percentage[cols] = processed_data[cols] / season_sums
# print(processed_percentage)

# 将排名转换为被淘汰的次序
processed_data['elimination_order'] = processed_data.groupby('season')['placement'].rank(ascending=False, method='min')
processed_data['score_sum'] = processed_data[[str(i) for i in range(1, 12)]].sum(axis=1)
plt.figure(figsize=(10, 6)) # 设置画布大小

# 画散点图
# x轴：淘汰次序 (elimination_order)
# y轴：总分 (score_sum)
# alpha=0.6：设置透明度，防止点重叠时看不清
plt.scatter(x=processed_data['score_sum'],
            y=processed_data['elimination_order'],
            color='blue',
            alpha=0.6)

plt.title('Total Score vs. Elimination Order') # 标题
plt.xlabel('Elimination Order (Higher = Stayed Longer)') # X轴标签
plt.ylabel('Score Sum') # Y轴标签
plt.grid(True, linestyle='--', alpha=0.5) # 加网格线好看一点
plt.show()


# 保存为新的csv
processed_data.to_csv('2026_MCM_Problem_Mean_Data.csv', index=False)
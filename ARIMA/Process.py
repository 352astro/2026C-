import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df_fame = pd.read_csv('2026_MCM_Problem_Fame_Data.csv')
df_raw = pd.read_csv('2026_MCM_Problem_C_Data.csv')

# 将赛季信息合并到特征表中
df_fame['season'] = df_raw['season']

# 对年龄进行标准化
scaler = StandardScaler()
df_fame['age_scaled'] = scaler.fit_transform(df_fame[['celebrity_age_during_season']])

# 提取行业特征
industry_cols = [c for c in df_fame.columns if 'industry_' in c]

# 对fame进行对数处理
selected_fame = 'fame_1.2'
df_fame['log_fame'] = np.log1p(df_fame[selected_fame])

# 汇总所有静态特征 X_i
static_feature_cols = ['log_fame', 'age_scaled'] + industry_cols

# 宽表转长表,week作为时间序列
week_cols = [str(i) for i in range(1, 12)]

# 转为长表
df_long = pd.melt(df_fame,
                  id_vars=['index','season'] + static_feature_cols,
                  value_vars=week_cols,
                  var_name='week',
                  value_name='Judge_Share')

df_long['week'] = df_long['week'].astype(int)

# 识别该选手在该周是否还在场
df_long = df_long.dropna(subset=['Judge_Share'])
df_long = df_long[df_long['Judge_Share'] > 0].copy()

# 构建时序滞后占位
df_long = df_long.sort_values(by=['season', 'index', 'week'])
# 只有在同一个 index 且同一个 season 下，前一周存在才标记为 1
df_long['Has_Previous_Week'] = df_long.groupby(['index', 'season'])['week'].shift(1).notna().astype(int)

# 找出每个选手在每个赛季中出现的最大周次，作为淘汰的标签
max_week_per_player = df_long.groupby(['index', 'season'])['week'].max().reset_index()
max_week_per_player.columns = ['index', 'season', 'last_week_appearance']
df_long = df_long.merge(max_week_per_player, on=['index', 'season'])

# 找出每个赛季整体运行到的最后一周（决赛周）
max_week_per_season = df_long.groupby('season')['week'].max().reset_index()
max_week_per_season.columns = ['season', 'season_final_week']
df_long = df_long.merge(max_week_per_season, on='season')

# 判定：如果选手的最后一周 < 该赛季的决赛周，则认为他在那周被淘汰了
df_long['is_eliminated'] = (
    (df_long['week'] == df_long['last_week_appearance']) &
    (df_long['week'] < df_long['season_final_week'])
).astype(int)

final_cols = ['index', 'season', 'week', 'Judge_Share', 'Has_Previous_Week', 'is_eliminated'] + static_feature_cols
processed_data = df_long[final_cols].sort_values(by=['season', 'week', 'index'])


processed_data.to_csv('2026_MCM_Problem_ARIMA_Data_SeasonAware.csv', index=False)
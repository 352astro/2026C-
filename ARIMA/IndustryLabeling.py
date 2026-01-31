import pandas as pd

processed_data = pd.read_csv('2026_MCM_Problem_Mean_Data.csv')
# 1. 定义归类函数 (与之前相同，将职业归为7大类)
def classify_industry(job):
    job = str(job).lower().strip()
    if job in ['actor/actress', 'comedian', 'magician']:
        return 'Actor_Performer'
    elif job in ['singer/rapper', 'musician']:
        return 'Music'
    elif job in ['athlete', 'racing driver', 'fitness instructor']:
        return 'Sports'
    elif job in ['model', 'beauty pagent', 'fashion designer']:
        return 'Model_Fashion'
    elif job in ['tv personality', 'news anchor', 'sports broadcaster', 'radio personality', 'journalist']:
        return 'TV_Media'
    elif 'social media' in job:
        return 'Influencer'
    else:
        return 'Others'

# 2. 先创建一个临时的归类列
processed_data['temp_category'] = processed_data['celebrity_industry'].apply(classify_industry)

# 3. 进行独热编码 (One-Hot Encoding)
# prefix='industry' 会让新列名变成 industry_Sports, industry_Music 等
one_hot_encoded = pd.get_dummies(processed_data['temp_category'], prefix='industry', dtype=int)

# 4. 将独热编码的新列拼接到原数据中
processed_data = pd.concat([processed_data, one_hot_encoded], axis=1)

# 5. 删除原始的 'celebrity_industry' 列 (以及临时的 'temp_category' 列)
processed_data.drop(columns=['celebrity_industry', 'temp_category'], inplace=True)

# --- 打印结果检查 ---
# 查看包含新生成的 industry_ 开头的列
new_columns = [col for col in processed_data.columns if 'industry_' in col]
print(processed_data[['season', 'placement'] + new_columns].head())

processed_data.drop(['placement','season','weeks','elimination_order','score_sum'],inplace=True, axis=1)
# processed_data.drop([str(i) for i in range(1,12)],inplace=True,axis=1)

processed_data.to_csv('2026_MCM_Problem_Labeled_Data.csv', index=False)
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, kendalltau

# 设置绘图风格 (支持中文显示需另外配置字体，这里主要保证英文字符正常)
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# ==========================================
# 1. 数据加载与预处理
# ==========================================

# 假设你的数据保存在 'mcm_data.csv' 中
# 如果你想直接运行此代码，请将你的CSV内容保存为文件，或者使用由于篇幅限制省略的 StringIO 方法
# df = pd.read_csv('mcm_data.csv')

# 为了演示，这里直接读取你提供的第一个CSV数据块（包含 fame 列的那部分）
csv_data = """index,celebrity_age_during_season,weeks,elimination_order,score_sum,industry_Actor_Performer,industry_Influencer,industry_Model_Fashion,industry_Music,industry_Others,industry_Sports,industry_TV_Media,celebrity_fame_1,celebrity_fame_1.1,celebrity_fame_1.2,celebrity_fame_1.3,celebrity_fame_1.4,ballroom_fame_1,ballroom_fame_1.1,ballroom_fame_1.2,ballroom_fame_1.3,ballroom_fame_1.4
0,50,6,5.0,48.33333333333333,1,0,0,0,0,0,0,49,72.31288491813308,106.7174147996562,157.49069663608591,232.42054329636488,0,0.0,0.0,0.0,0.0
1,29,6,6.0,42.66666666666666,1,0,0,0,0,0,0,114,183.06079178697712,293.9583639445171,472.0372882112955,757.9957871310279,0,0.0,0.0,0.0,0.0
2,42,3,2.0,15.0,0,0,0,0,0,1,0,47,69.0728926279167,101.5120105529299,149.185706497159,219.24868694676914,12,15.385066247841788,19.725021954206706,25.289230792133658,32.423040924494714
3,35,4,3.0,31.666666666666668,0,0,1,0,0,0,0,24,32.9786114475993,45.316200542155286,62.26939041505117,85.56491798236814,2,2.1435469250725863,2.2973967099940698,2.4622888266898326,2.6390158215457884
4,32,5,4.0,35.16666666666667,0,0,0,1,0,0,0,31,43.70165288902177,61.60756339459819,86.8500758325527,122.43522152965917,0,0.0,0.0,0.0,0.0
5,32,2,1.0,12.333333333333332,0,0,0,0,0,0,1,39,56.25628739463053,81.14794542121189,117.05338818203819,168.84587297651242,0,0.0,0.0,0.0,0.0
6,42,2,2.0,13.333333333333334,1,0,0,0,0,0,0,83,129.11773577858514,200.86011677817484,312.4651022484354,486.0817652065305,17,22.56803847310483,29.959785913149386,39.77256477258468,52.799339527161436
7,39,5,5.0,38.333333333333336,1,0,0,0,0,0,0,123,199.01949622581813,322.02243803234467,521.046694232577,843.0768341162217,0,0.0,0.0,0.0,0.0
8,66,6,6.0,43.33333333333333,1,0,0,0,0,0,0,0,0.0,0.0,0.0,0.0,771,1498.8556724613386,2913.8369998307753,5664.618827268594,11012.251701145025
9,42,7,7.0,56.16666666666667,1,0,0,0,0,0,0,98,155.00606210939196,245.17223765980262,387.7875826346324,613.3614910113748,34,48.37564947394369,68.82951358899608,97.9315418483329,139.33829237934694
10,26,8,8.0,73.38883333333334,0,0,0,0,0,1,0,1098,2211.375909511406,4453.718955525856,8969.80583422967,18065.22088780539,43,62.634743194944704,91.23514081620159,132.89510733435824,193.5779283663287
11,43,8,9.0,60.05553333333334,0,0,0,0,0,1,0,495,920.5888676588002,1712.0886126410317,3184.100438874209,5921.7119546163485,0,0.0,0.0,0.0,0.0
12,44,3,3.0,23.0,0,0,0,0,0,0,1,0,0.0,0.0,0.0,0.0,5,5.873094715440096,6.898648307306074,8.103282983463814,9.518269693579391
13,35,4,4.0,16.666666666666664,0,0,0,1,0,0,0,668,1280.1300981011912,2453.1932156655134,4701.207293160285,9009.216996088777,39,56.25628739463053,81.14794542121189,117.05338818203819,168.84587297651242
14,29,8,10.0,73.1666,0,0,0,1,0,0,0,97,153.2670885632895,242.17320037801238,382.6513541236297,604.6171029002984,123,199.01949622581813,322.02243803234467,521.046694232577,843.0768341162217
15,46,1,1.0,4.333333333333333,0,0,0,0,0,0,1,64,97.00586025666551,147.03338943962046,222.86094420380783,337.7940251578608,0,0.0,0.0,0.0,0.0
"""
# 注意：实际运行时请加载完整数据
# df = pd.read_csv(io.StringIO(csv_data))
# 这里为了演示方便，如果上面的string不够全，请替换为你的文件读取
df = pd.read_csv("2026_MCM_Problem_Fame_withBallroom_Data.csv")

# 模拟读取步骤（假设你已经读取了完整数据到 df）
# 下面是一行代码读取你贴出的数据（需要将数据保存为csv）
# df = pd.read_csv("data.csv")
# 这里的df使用你提供的数据列结构

# ----------------------------------------------------
# 关键步骤：特征选择 (Feature Selection)
# ----------------------------------------------------
# 我们要预测的目标是 score_sum (评委评分总和)
# 特征包括：年龄、行业、名人名气、舞伴名气

feature_cols = [
    'celebrity_age_during_season',
    'industry_Actor_Performer', 'industry_Influencer', 'industry_Model_Fashion', 
    'industry_Music', 'industry_Others', 'industry_Sports', 'industry_TV_Media',
    # 选取一个主要的名气指标即可，或者全部放入让RF自己选。
    # 这里为了全面，放入所有的 fame 代理变量
    # 'celebrity_fame_1', 'celebrity_fame_1.1', 'celebrity_fame_1.2', 'celebrity_fame_1.3', 'celebrity_fame_1.4',
    # 'ballroom_fame_1', 'ballroom_fame_1.1', 'ballroom_fame_1.2', 'ballroom_fame_1.3', 'ballroom_fame_1.4'
    'celebrity_fame_1.4',
    'ballroom_fame_1.4'
]

target_col = 'score_sum'

# 注意：我故意移除了 'weeks' (参赛周数)。
# 原因：'score_sum' 是累积值，跟参赛周数几乎是完全线性相关的。
# 如果把 'weeks' 放进去，模型会告诉你 'weeks' 是唯一重要的特征。
# 我们的目的是分析“名气/年龄”如何影响“得分能力”，所以不应该包含结果变量(weeks)。

# 准备 X 和 y
# 假设你已经有了完整的 df
# df = ... (加载你的完整数据)
# 暂时用随机生成的数据填充以便代码可运行，请替换为真实数据加载
if 'df' not in locals():
    # 这是一个占位符，请确保你加载了真实数据
    print("请先加载你的CSV文件到变量 df 中")
    # 示例结构：
    # df = pd.read_csv("cleaned_data.csv")

# 简单处理缺失值
X = df[feature_cols].fillna(0)
y = df[target_col].fillna(0)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. 建立与训练随机森林模型
# ==========================================

rf = RandomForestRegressor(
    n_estimators=200,    # 树的数量
    max_depth=10,        # 防止过拟合
    random_state=42,
    n_jobs=-1            # 并行计算
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ==========================================
# 3. 计算相关性指标 (单调性与一致性)
# ==========================================

# Spearman Correlation (单调性: 排名相关性)
spearman_corr, _ = spearmanr(y_test, y_pred)

# Kendall Tau (一致性: 序对的一致性)
kendall_corr, _ = kendalltau(y_test, y_pred)

print(f"--- 模型评估指标 ---")
print(f"R^2 Score (测试集): {rf.score(X_test, y_test):.4f}")
print(f"Spearman Correlation (单调性): {spearman_corr:.4f}")
print(f"Kendall Tau (一致性): {kendall_corr:.4f}")

# ==========================================
# 4. 特征重要性分析 (Permutation Importance)
# ==========================================
# 置换重要性通过打乱某一列特征的值，观察模型误差增加了多少，从而判断该特征的重要性。

result = permutation_importance(
    rf, X_test, y_test, 
    n_repeats=10, 
    random_state=42, 
    n_jobs=-1
)

# 整理结果
perm_sorted_idx = result.importances_mean.argsort()
importances = pd.DataFrame({
    'Feature': np.array(feature_cols)[perm_sorted_idx],
    'Importance': result.importances_mean[perm_sorted_idx],
    'Std': result.importances_std[perm_sorted_idx]
})

# ==========================================
# 5. 可视化
# ==========================================

plt.figure(figsize=(10, 8))
plt.barh(importances['Feature'], importances['Importance'], xerr=importances['Std'])
plt.xlabel("Permutation Importance (对模型预测误差的影响程度)")
plt.ylabel("特征 (Features)")
plt.title("各因素对评委评分总和(Score Sum)的影响分析")
plt.tight_layout()
plt.show()

# ==========================================
# 6. 分析结论建议
# ==========================================
print("\n--- 分析指南 ---")
print("1. 查看图表中最高的条形：那是对选手得分影响最大的因素。")
print("2. 比较 Celebrity Fame (选手名气) 和 Ballroom Fame (舞伴名气) 的重要性：")
print("   - 如果 Ballroom Fame 更高，说明'抱大腿'（有个好老师）比自带流量更重要。")
print("   - 如果 Celebrity Fame 更高，说明评委可能受到选手名气的光环效应影响，或者名气大的选手更努力。")
print("3. Industry (行业) 的影响：查看是否有特定行业（如 Athlete 运动员）表现出正向重要性。")
print("4. 关于'对粉丝投票的影响方式是否相同'：")
print("   - 你需要用同样的脚本，只需将 target_col 改为 'fan_votes' (或者你估算的粉丝投票列)。")
print("   - 然后对比两张重要性图。如果评委看重 'Ballroom Fame' 而粉丝看重 'Celebrity Fame'，则影响方式不同。")
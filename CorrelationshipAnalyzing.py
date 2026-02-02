import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr   # 用于计算相关系数

# 读取数据
data = pd.read_csv('2026_MCM_Problem_Mean_Data.csv')

# 计算 Pearson 相关系数
r, p = pearsonr(data['score_sum'], data['elimination_order'])
r_text = f"Pearson r = {r:.3f}"   # 保留3位小数
# 如果你也想显示 p 值，可以加： f"p = {p:.2e}"

plt.figure(figsize=(10, 6))

# 散点图
plt.scatter(
    data['score_sum'],
    data['elimination_order'],
    alpha=0.6,
    color='blue',
    s=50
)

# 回归直线
X = data['score_sum'].values.reshape(-1, 1)
y = data['elimination_order'].values
reg = LinearRegression().fit(X, y)
plt.plot(
    X,
    reg.predict(X),
    color='red',
    linewidth=2.5,
    label=f'Regression Line (slope = {reg.coef_[0]:.3f})'
)

# 图标题与轴标签（与参考图风格一致）
plt.title('Total Score vs. Elimination Order', fontsize=14, pad=12)
plt.xlabel('Total Score (Score Sum)', fontsize=12)
plt.ylabel('Elimination Order (Higher = Stayed Longer)', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)

# 图例：包含回归线 + 相关系数，放在左上角
plt.legend(
    [f'Regression Line (slope = {reg.coef_[0]:.3f})', r_text],
    loc='upper left',
    fontsize=10,
    frameon=True,          # 有边框，更清晰
    edgecolor='gray',
    facecolor='white',
    framealpha=0.95
)

plt.tight_layout()

plt.savefig('total_score_vs_elimination_order.png',
            dpi=300,           # 高清，300dpi 适合打印
            bbox_inches='tight')  # 自动裁剪白边

# 方法2：同时导出其他格式（可选）
# plt.savefig('total_score_vs_elimination_order.jpg', dpi=200)
# plt.savefig('total_score_vs_elimination_order.pdf', format='pdf')

# 显示图像（可选，如果你还想在运行时看到窗口）
plt.show()

print("图片已保存为：total_score_vs_elimination_order.png")


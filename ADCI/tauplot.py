import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # 引入自定义线条工具

# 设置Seaborn主题
sns.set_theme(style="ticks")

# --- 1. 模拟数据 (请保留你自己的 pd.read_csv 加载代码) ---
df = pd.read_csv(r"C:\Users\Yelo_Pasge\Documents\Tencent Files\1262866316\FileRecv\contestants_analysis.csv")
# 假设下面的 df 已经按照你的逻辑加载完毕
# 为了演示代码能跑，我这里用模拟数据，你运行时请保留你原来的数据处理部分
# ========================================================
import numpy as np

# df = pd.DataFrame({'season': np.tile(range(1, 11), 6),
#                    'tau_1-2': np.random.rand(60), 'spearman_1-2': np.random.rand(60),
#                    'tau_1-3': np.random.rand(60), 'tau_2-3': np.random.rand(60),
#                    'spearman_1-3': np.random.rand(60), 'spearman_2-3': np.random.rand(60)})

# --- 你的数据重塑代码 (保持不变) ---
df_1_2_tau = pd.DataFrame({
    'season': pd.concat([df['season']]),
    'coefficient_calc': pd.concat([df['tau_1-2']]),
    'correlation_type': ['τ_1-2'] * len(df),
    'method': ['df_1_2_tau'] * len(df)
})

df_1_2_spearman = pd.DataFrame({
    'season': pd.concat([df['season']]),
    'coefficient_calc': pd.concat([df['spearman_1-2']]),
    'correlation_type': ['ρ_1-2'] * len(df),
    'method': ['df_1_2_spearman'] * len(df)
})

df_1_2_3_tau = pd.DataFrame({
    'season': pd.concat([df['season'], df['season']]),
    'coefficient_calc': pd.concat([df['tau_1-3'], df['tau_2-3']]),
    'correlation_type': ['τ_1-3'] * len(df) + ['τ_2-3'] * len(df),
    'method': ['df_1_2_3_tau'] * len(df) * 2
})

df_1_2_3_spearman = pd.DataFrame({
    'season': pd.concat([df['season'], df['season']]),
    'coefficient_calc': pd.concat([df['spearman_1-3'], df['spearman_2-3']]),
    'correlation_type': ['ρ_1-3'] * len(df) + ['ρ_2-3'] * len(df),
    'method': ['df_1_2_3_spearman'] * len(df) * 2
})

# 合并数据
plot_data = pd.concat([df_1_2_tau, df_1_2_spearman, df_1_2_3_tau, df_1_2_3_spearman], ignore_index=True)

# ================= 关键修改开始 =================

# 1. 明确定义子图顺序 (Method Order) 和 线条类型的顺序 (Hue Order)
# 这样做是为了确保子图位置和颜色是固定的，不会因为数据排序乱掉
method_order = ['df_1_2_tau', 'df_1_2_spearman', 'df_1_2_3_tau', 'df_1_2_3_spearman']
hue_types = plot_data['correlation_type'].unique()

# 2. 创建一个颜色字典 (Color Mapping)
# 这样我们就可以随时查到 'τ_1-2' 到底对应什么颜色
colors = sns.color_palette("rocket_r", n_colors=len(hue_types))
palette_dict = dict(zip(hue_types, colors))

# 3. 绘图
g = sns.relplot(
    data=plot_data,
    x="season", y="coefficient_calc",
    hue="correlation_type",
    col="method",
    col_order=method_order,  # 强制指定子图顺序
    kind="line",
    palette=palette_dict,  # 使用我们定义好的颜色字典
    height=5, aspect=1.5, facet_kws=dict(sharex=True, sharey=True),
    linewidth=2,
    col_wrap=2,
    legend=False  # 关闭全局图例
)

# 设置总标题
g.fig.suptitle("Consistency Analysis of Different Elimination Methods",
               fontsize=16, fontweight='bold', y=0.96)
g.set_axis_labels("Seasons", "Coefficient Value")

# 自定义每个子图的标题
titles = [
    "Rank_τ vs Percentage_τ",
    "Rank_Spearman vs Percentage_ρ",
    "τ for Rank vs Percentage vs FansOnly",
    "ρ for Rank vs Percentage vs FansOnly"
]

# 4. 循环遍历子图，手动生成图例
# g.axes.flat 按照 col_order 的顺序排列，与 method_order 一一对应
for i, ax in enumerate(g.axes.flat):
    # 设置子图标题
    if i < len(titles):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')

    # 设置辅助线
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- 核心逻辑：为当前子图生成图例 ---
    current_method = method_order[i]

    # 找出在这个 method (子图) 中出现过哪些 correlation_type
    current_types = plot_data[plot_data['method'] == current_method]['correlation_type'].unique()

    legend_handles = []
    legend_labels = []

    for c_type in current_types:
        # 获取该类型对应的颜色
        color = palette_dict[c_type]
        # 手动创建一个线条对象 (Proxy Artist) 用于显示在图例中
        # Line2D(x, y, color=..., linewidth=...)
        line = Line2D([0], [0], color=color, linewidth=2)

        legend_handles.append(line)
        legend_labels.append(c_type)

    # 如果有内容，则添加图例
    if legend_handles:
        ax.legend(handles=legend_handles, labels=legend_labels,
                  loc='upper right', frameon=True, fontsize='medium')

g.set(ylim=(-1.1, 1.1))
plt.subplots_adjust(top=0.88)
plt.savefig("Consistency Analysis.png", bbox_inches='tight')
plt.show()
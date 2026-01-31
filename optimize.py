import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

data = pd.read_csv('2026_MCM_Problem_Fame_Data.csv')
# 原始经过1.2次方处理得到的人气值
pop_raw = data['fame_1.1'].values
# 各名人的年龄数据
age_raw = data['celebrity_age_during_season'].values
# 评委分周平均分在所有存活周上的分数总和
X_judge = data['score_sum'].values
# 真实淘汰次序
eta_true = data['elimination_order'].values
# 行业独热编码
ind_cols = [c for c in data.columns if 'industry_' in c]
Ind = data[ind_cols].values

# 标准化：确保优化的稳定性
def scale(x): return (x - x.mean()) / (x.std() + 1e-6)
pop = scale(pop_raw)
age = scale(age_raw)
X_judge = scale(X_judge)

# 合并特征矩阵 [人气, 年龄, 行业1...N]
Features = np.column_stack([pop, age, Ind])
num_f = Features.shape[1]



#目标函数
def end_to_end_loss(params, Features, X_judge, eta_true):
    # 拆分参数
    # params_y1: 均值权重 (num_f + 1)
    # params_y2: 方差权重 (num_f + 1)
    # params_final: [w_judge, w_audience] (2)
    idx1 = num_f + 1
    idx2 = 2 * (num_f + 1)

    p_y1 = params[:idx1]
    p_y2 = params[idx1:idx2]
    w_final = params[idx2:]  # [w1, w2]

    # 计算 mu (y1)
    mu = np.dot(Features, p_y1[:-1]) + p_y1[-1]

    # 计算 sigma (y2 -> softplus)
    y2 = np.dot(Features, p_y2[:-1]) + p_y2[-1]
    # sigma = np.log1p(np.exp(np.clip(y2, -20, 20)))
    sigma = np.log(1+np.exp(-y2))

    final_mean = w_final[0] * X_judge + w_final[1] * mu
    final_std = np.abs(w_final[1]) * sigma + 1e-6


    # 预测淘汰次序 eta_bar = w1*X + w2*mu
    # eta_pred = w_final[0] * X_judge + w_final[1] * mu
    nll = np.sum(np.log(final_std) + (eta_true - final_mean)**2 / (2 * final_std**2))

    # # 计算期望平方误差：(真-预测)^2 + (权重2^2 * sigma^2)
    # # 后半项考虑了投票数波动对淘汰次序稳定性的影响
    # mse = np.mean((eta_true - eta_pred) ** 2 + (w_final[1] ** 2 * sigma ** 2))

    # 加入 L2 正则化防止权重过大
    # return mse + 0.01 * np.sum(params ** 2)
    return nll + 0.01 * np.sum(params**2)


total_params = 2 * (num_features := num_f + 1) + 2
initial_guess = np.zeros(total_params)
initial_guess[-2:] = [0.1, 0.1]  # 给最终加权一个初始正值

res = minimize(end_to_end_loss, initial_guess,
               args=(Features, X_judge, eta_true),
               method='L-BFGS-B')

# --- 结果解析 ---
if res.success:
    final_p = res.x
    w_mu = final_p[:num_f]
    w_final = final_p[-2:]

    print("优化结果成功！")
    print(f"评委分权重 (w1): {w_final[0]:.4f}")
    print(f"观众分权重 (w2): {w_final[1]:.4f}")
    print("-" * 30)
    print("均值层（人气预测）核心贡献：")
    print(f"人气值(WE^1.2)影响力: {w_mu[0]:.4f}")
    print(f"年龄影响力: {w_mu[1]:.4f}")
    # 打印前两个行业作为示例
    for i in range(2):
        print(f"行业[{ind_cols[i]}]修正项: {w_mu[2 + i]:.4f}")
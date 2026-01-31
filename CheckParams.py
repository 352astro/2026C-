import numpy as np

# 加载数据
data = np.load('dwts_model_assets.npz', allow_pickle=True)
params = data['params']
cols = list(data['static_cols'])
n = len(cols)

# 根据模型定义的顺序进行切片:
# [w_mu(n), b_mu(1), w_sig(n), b_sig(1), phi(1)]
w_mu = params[0:n]
b_mu = params[n]
w_sig = params[n+1 : 2*n+1]
b_sig = params[2*n+1]
phi = params[-1]

print("=== 模型参数解析 ===")
print(f"{'特征名称':<25} | {'人气权重(mu)':<12} | {'波动权重(sigma)':<12}")
print("-" * 55)
for i in range(n):
    print(f"{cols[i]:<25} | {w_mu[i]:>12.4f} | {w_sig[i]:>12.4f}")

print("-" * 55)
print(f"{'偏置项 (Intercept/Bias)':<25} | {b_mu:>12.4f} | {b_sig:>12.4f}")
print("-" * 55)
print(f"ARIMA 趋势系数 (phi): {phi:.4f}")
import pandas as pd
import numpy as np
from scipy.special import softmax

def calculate_confidence_intervals(input_file, num_simulations=1000, confidence_level=0.95):
    print(f"[系统] 正在读取数据: {input_file} ...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return

    # 检查必要的列是否存在
    required_cols = ['season', 'week', 'celebrity_name', 'mu_base', 'sigma']
    if not all(col in df.columns for col in required_cols):
        print(f"错误: 输入文件缺少必要的列。需要: {required_cols}")
        print("请确保使用的是包含 'mu_base' 和 'sigma' 列的 predicted_fan_votes_v1.csv 版本。")
        return

    results = []
    
    # 按赛季和周分组处理
    # 因为 Softmax 是在同一周的所有选手中计算的，所以必须分组模拟
    grouped = df.groupby(['season', 'week'])
    
    print(f"[系统] 开始进行蒙特卡洛模拟 (Simulations={num_simulations}, CI={confidence_level:.0%})...")
    
    total_groups = len(grouped)
    processed_count = 0

    for (season, week), group in grouped:
        # 提取当前组的 mu 和 sigma
        mus = group['mu_base'].values
        sigmas = group['sigma'].values
        names = group['celebrity_name'].values
        original_shares = group['v_predicted_share'].values
        
        # 蒙特卡洛模拟
        # 生成形状为 (num_simulations, num_players) 的随机矩阵
        # 每一行是一次模拟中所有选手的人气值
        # np.random.normal 支持广播，但为了清晰，我们显式生成
        
        # 方式: normal(loc=mus, scale=sigmas, size=(num_simulations, len(mus)))
        # 注意: mus 和 sigmas 是 1D 数组，numpy 会自动广播到 size 的最后维度吗？
        # 不会自动广播到 (N, M) 如果输入是 (M,)。我们需要手动处理或利用广播规则。
        # 正确做法：生成标准正态分布，然后 * sigma + mu
        
        n_players = len(mus)
        epsilon = np.random.normal(0, 1, size=(num_simulations, n_players))
        
        # 广播计算: sigma 和 mu 需要是 (1, n_players)
        simulated_intensities = epsilon * sigmas.reshape(1, -1) + mus.reshape(1, -1)
        
        # 应用 Softmax (带温度系数 5.0，保持与预测脚本一致)
        # axis=1 表示对每一行（每一次模拟）的所有选手做 softmax
        simulated_shares = softmax(simulated_intensities, axis=1)
        
        # 计算置信区间
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        lower_bounds = np.percentile(simulated_shares, lower_percentile, axis=0)
        upper_bounds = np.percentile(simulated_shares, upper_percentile, axis=0)
        std_devs = np.std(simulated_shares, axis=0)
        
        # 整理结果
        for i in range(n_players):
            results.append({
                'season': season,
                'week': week,
                'celebrity_name': names[i],
                'v_predicted_share': original_shares[i], # 原始预测值
                'ci_lower': lower_bounds[i],             # 95% CI 下界
                'ci_upper': upper_bounds[i],             # 95% CI 上界
                'ci_std': std_devs[i],                   # 模拟分布的标准差
                'mu_base': mus[i],
                'sigma': sigmas[i]
            })
            
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"  ...已处理 {processed_count}/{total_groups} 个周次")

    # 创建结果 DataFrame
    df_ci = pd.DataFrame(results)
    
    # 导出文件
    output_file = 'predicted_fan_votes_with_CI.csv'
    df_ci.to_csv(output_file, index=False)
    
    print("-" * 60)
    print(f"[完成] 置信区间计算完毕。")
    print(f"输出文件: {output_file}")
    print(f"包含列: season, week, celebrity_name, v_predicted_share, ci_lower, ci_upper, ci_std, ...")
    
    # 打印前几行预览
    print("\n结果预览:")
    print(df_ci[['season', 'week', 'celebrity_name', 'v_predicted_share', 'ci_lower', 'ci_upper']].head().to_string())

if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    np.random.seed(2026)
    
    input_filename = 'predicted_fan_votes_v1.csv'
    calculate_confidence_intervals(input_filename)
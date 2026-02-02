"""
使用Kendall's tau分析四种淘汰规则的一致性
比较排名方式和百分比方式哪个更倾向于粉丝投票
包含规则4：排名倒数两名，评委决定
"""

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, spearmanr


def calculate_kendall_tau_analysis(results_file: str, output_file: str):
    """
    计算Kendall's tau一致性分析
    同时计算Spearman相关性
    
    参数:
        results_file: 淘汰结果CSV文件路径
        output_file: 输出分析结果CSV文件路径
    """
    # 读取淘汰结果
    df = pd.read_csv(results_file)
    
    # 获取所有season
    seasons = sorted(df['season'].unique())
    
    analysis_results = []
    
    for season in seasons:
        season_data = df[df['season'] == season].copy()
        
        # 获取四种规则的淘汰次序
        order_rule1 = season_data['淘汰次序_规则1_排名相加'].values
        order_rule2 = season_data['淘汰次序_规则2_百分比相加'].values
        order_rule3 = season_data['淘汰次序_规则3_只看粉丝投票'].values
        order_rule4 = season_data['淘汰次序_规则4_排名倒数两名评委决定'].values
        
        # 计算Kendall's tau
        # 规则1 vs 规则3（排名方式 vs 粉丝投票）
        tau_rule1_vs_rule3, p_value_rule1 = kendalltau(order_rule1, order_rule3)
        
        # 规则2 vs 规则3（百分比方式 vs 粉丝投票）
        tau_rule2_vs_rule3, p_value_rule2 = kendalltau(order_rule2, order_rule3)
        
        # 规则4 vs 规则3（排名倒数两名评委决定 vs 粉丝投票）
        tau_rule4_vs_rule3, p_value_rule4 = kendalltau(order_rule4, order_rule3)
        
        # 规则1 vs 规则2（排名方式 vs 百分比方式）
        tau_rule1_vs_rule2, p_value_rule1_rule2 = kendalltau(order_rule1, order_rule2)
        
        # 规则1 vs 规则4（排名方式 vs 排名倒数两名评委决定）
        tau_rule1_vs_rule4, p_value_rule1_rule4 = kendalltau(order_rule1, order_rule4)
        
        # 规则2 vs 规则4（百分比方式 vs 排名倒数两名评委决定）
        tau_rule2_vs_rule4, p_value_rule2_rule4 = kendalltau(order_rule2, order_rule4)
        
        # 计算Spearman相关性
        # 规则1 vs 规则3（排名方式 vs 粉丝投票）
        spearman_rule1_vs_rule3, _ = spearmanr(order_rule1, order_rule3)
        
        # 规则2 vs 规则3（百分比方式 vs 粉丝投票）
        spearman_rule2_vs_rule3, _ = spearmanr(order_rule2, order_rule3)
        
        # 规则4 vs 规则3（排名倒数两名评委决定 vs 粉丝投票）
        spearman_rule4_vs_rule3, _ = spearmanr(order_rule4, order_rule3)
        
        # 规则1 vs 规则2（排名方式 vs 百分比方式）
        spearman_rule1_vs_rule2, _ = spearmanr(order_rule1, order_rule2)
        
        # 规则1 vs 规则4（排名方式 vs 排名倒数两名评委决定）
        spearman_rule1_vs_rule4, _ = spearmanr(order_rule1, order_rule4)
        
        # 规则2 vs 规则4（百分比方式 vs 排名倒数两名评委决定）
        spearman_rule2_vs_rule4, _ = spearmanr(order_rule2, order_rule4)
        
        # 判断哪个更倾向于粉丝投票（tau值越大，一致性越高）
        # 比较规则1、规则2、规则4与规则3的一致性
        tau_values = {
            '规则1_排名相加': tau_rule1_vs_rule3,
            '规则2_百分比相加': tau_rule2_vs_rule3,
            '规则4_排名倒数两名评委决定': tau_rule4_vs_rule3
        }
        more_fan_oriented = max(tau_values, key=tau_values.get)
        tau_difference = abs(tau_rule1_vs_rule3 - tau_rule2_vs_rule3)
        
        # 计算平均tau值（与粉丝投票的一致性）
        avg_tau_with_fan = (tau_rule1_vs_rule3 + tau_rule2_vs_rule3 + tau_rule4_vs_rule3) / 3
        
        analysis_results.append({
            'season': season,
            # 规则1和其他：1-2, 1-3, 1-4
            'tau_1-2': float(f'{tau_rule1_vs_rule2:.5f}'),
            'spearman_1-2': float(f'{spearman_rule1_vs_rule2:.5f}'),
            'tau_1-3': float(f'{tau_rule1_vs_rule3:.5f}'),
            'spearman_1-3': float(f'{spearman_rule1_vs_rule3:.5f}'),
            'tau_1-4': float(f'{tau_rule1_vs_rule4:.5f}'),
            'spearman_1-4': float(f'{spearman_rule1_vs_rule4:.5f}'),
            # 规则2和剩余其他：2-3, 2-4
            'tau_2-3': float(f'{tau_rule2_vs_rule3:.5f}'),
            'spearman_2-3': float(f'{spearman_rule2_vs_rule3:.5f}'),
            'tau_2-4': float(f'{tau_rule2_vs_rule4:.5f}'),
            'spearman_2-4': float(f'{spearman_rule2_vs_rule4:.5f}'),
            # 规则3和其他：3-4 (即4-3，但用3-4表示)
            'tau_3-4': float(f'{tau_rule4_vs_rule3:.5f}'),
            'spearman_3-4': float(f'{spearman_rule4_vs_rule3:.5f}'),
            # 其他统计信息
            '更倾向粉丝': more_fan_oriented,
            'tau_diff_1-2': float(f'{tau_difference:.5f}'),
            'avg_tau_粉丝': float(f'{avg_tau_with_fan:.5f}'),
            '选手数': len(season_data)
        })
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(analysis_results)
    
    # 先计算和打印统计摘要（使用原始数值）
    print(f"分析结果已保存到: {output_file}")
    print(f"共分析 {len(seasons)} 个season")
    print("\n" + "="*80)
    print("统计摘要:")
    print("="*80)
    print(f"规则1 vs 规则3 (排名相加 vs 只看粉丝投票) 平均tau: {result_df['tau_1-3'].mean():.5f}")
    print(f"规则2 vs 规则3 (百分比相加 vs 只看粉丝投票) 平均tau: {result_df['tau_2-3'].mean():.5f}")
    print(f"规则4 vs 规则3 (倒数两名评委决定 vs 只看粉丝投票) 平均tau: {result_df['tau_3-4'].mean():.5f}")
    print(f"\n规则间比较:")
    print(f"  规则1 vs 规则2 (排名相加 vs 百分比相加) 平均tau: {result_df['tau_1-2'].mean():.5f}")
    print(f"  规则1 vs 规则4 (排名相加 vs 倒数两名评委决定) 平均tau: {result_df['tau_1-4'].mean():.5f}")
    print(f"  规则2 vs 规则4 (百分比相加 vs 倒数两名评委决定) 平均tau: {result_df['tau_2-4'].mean():.5f}")
    print(f"\nSpearman相关性统计:")
    print(f"  规则1 vs 规则3 平均Spearman: {result_df['spearman_1-3'].mean():.5f}")
    print(f"  规则2 vs 规则3 平均Spearman: {result_df['spearman_2-3'].mean():.5f}")
    print(f"  规则4 vs 规则3 平均Spearman: {result_df['spearman_3-4'].mean():.5f}")
    print(f"  规则1 vs 规则2 平均Spearman: {result_df['spearman_1-2'].mean():.5f}")
    print(f"  规则1 vs 规则4 平均Spearman: {result_df['spearman_1-4'].mean():.5f}")
    print(f"  规则2 vs 规则4 平均Spearman: {result_df['spearman_2-4'].mean():.5f}")
    print(f"\n更倾向于粉丝投票的规则统计:")
    rule_counts = result_df['更倾向粉丝'].value_counts()
    print(rule_counts)
    print(f"\n结论:")
    max_rule = rule_counts.index[0] if len(rule_counts) > 0 else None
    if max_rule:
        print(f"✓ {max_rule} 更倾向于粉丝投票")
        print(f"  在 {rule_counts[max_rule]} 个season中，{max_rule}与粉丝投票的一致性更高")
    print(f"\n平均tau值比较（与粉丝投票的一致性）:")
    print(f"  规则1（排名相加）: {result_df['tau_1-3'].mean():.5f}")
    print(f"  规则2（百分比相加）: {result_df['tau_2-3'].mean():.5f}")
    print(f"  规则4（倒数两名评委决定）: {result_df['tau_3-4'].mean():.5f}")
    
    # 找出与粉丝投票一致性最高的规则
    tau_means = {
        '规则1': result_df['tau_1-3'].mean(),
        '规则2': result_df['tau_2-3'].mean(),
        '规则4': result_df['tau_3-4'].mean()
    }
    best_rule = max(tau_means, key=tau_means.get)
    print(f"\n  → {best_rule}与粉丝投票的一致性最高 (tau = {tau_means[best_rule]:.4f})")
    
    # 格式化所有数值列为5位小数（除了season和选手数）
    numeric_cols = [col for col in result_df.columns if col not in ['season', '更倾向粉丝', '选手数']]
    for col in numeric_cols:
        result_df[col] = result_df[col].apply(lambda x: f'{float(x):.5f}' if pd.notna(x) and x != '' else '')
    
    # 保存到CSV
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    return result_df


if __name__ == "__main__":
    # 分析数据并生成结果
    input_file = "elimination_results.csv"
    output_file = "contestants_analysis.csv"
    
    result_df = calculate_kendall_tau_analysis(input_file, output_file)
    
    # 显示前几行结果
    print("\n前10行分析结果:")
    print(result_df.head(10).to_string())


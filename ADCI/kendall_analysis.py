"""
使用Kendall's tau分析三种淘汰规则的一致性
比较排名方式和百分比方式哪个更倾向于粉丝投票
"""

import pandas as pd
import numpy as np
from scipy.stats import kendalltau


def calculate_kendall_tau_analysis(results_file: str, output_file: str):
    """
    计算Kendall's tau一致性分析
    
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
        
        # 获取三种规则的淘汰次序
        order_rule1 = season_data['淘汰次序_规则1_排名相加'].values
        order_rule2 = season_data['淘汰次序_规则2_百分比相加'].values
        order_rule3 = season_data['淘汰次序_规则3_只看粉丝投票'].values
        
        # 计算Kendall's tau
        # 规则1 vs 规则3（排名方式 vs 粉丝投票）
        tau_rule1_vs_rule3, p_value_rule1 = kendalltau(order_rule1, order_rule3)
        
        # 规则2 vs 规则3（百分比方式 vs 粉丝投票）
        tau_rule2_vs_rule3, p_value_rule2 = kendalltau(order_rule2, order_rule3)
        
        # 规则1 vs 规则2（排名方式 vs 百分比方式）
        tau_rule1_vs_rule2, p_value_rule1_rule2 = kendalltau(order_rule1, order_rule2)
        
        # 判断哪个更倾向于粉丝投票（tau值越大，一致性越高）
        more_fan_oriented = "规则1_排名相加" if tau_rule1_vs_rule3 > tau_rule2_vs_rule3 else "规则2_百分比相加"
        tau_difference = abs(tau_rule1_vs_rule3 - tau_rule2_vs_rule3)
        
        # 计算平均tau值（与粉丝投票的一致性）
        avg_tau_with_fan = (tau_rule1_vs_rule3 + tau_rule2_vs_rule3) / 2
        
        analysis_results.append({
            'season': season,
            'tau_规则1_vs_规则3_排名vs粉丝': tau_rule1_vs_rule3,
            'p_value_规则1_vs_规则3': p_value_rule1,
            'tau_规则2_vs_规则3_百分比vs粉丝': tau_rule2_vs_rule3,
            'p_value_规则2_vs_规则3': p_value_rule2,
            'tau_规则1_vs_规则2_排名vs百分比': tau_rule1_vs_rule2,
            'p_value_规则1_vs_规则2': p_value_rule1_rule2,
            '更倾向于粉丝投票的规则': more_fan_oriented,
            'tau差异绝对值': tau_difference,
            '平均tau_与粉丝投票': avg_tau_with_fan,
            '选手数量': len(season_data)
        })
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(analysis_results)
    
    # 保存到CSV
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"分析结果已保存到: {output_file}")
    print(f"共分析 {len(seasons)} 个season")
    print("\n" + "="*80)
    print("统计摘要:")
    print("="*80)
    print(f"规则1 vs 规则3 (排名相加 vs 只看粉丝投票) 平均tau: {result_df['tau_规则1_vs_规则3_排名vs粉丝'].mean():.4f}")
    print(f"规则2 vs 规则3 (百分比相加 vs 只看粉丝投票) 平均tau: {result_df['tau_规则2_vs_规则3_百分比vs粉丝'].mean():.4f}")
    print(f"\n规则1 vs 规则2 (排名相加 vs 百分比相加) 平均tau: {result_df['tau_规则1_vs_规则2_排名vs百分比'].mean():.4f}")
    print(f"\n更倾向于粉丝投票的规则统计:")
    rule_counts = result_df['更倾向于粉丝投票的规则'].value_counts()
    print(rule_counts)
    print(f"\n结论:")
    if rule_counts.get('规则1_排名相加', 0) > rule_counts.get('规则2_百分比相加', 0):
        print("✓ 排名相加方式（规则1）更倾向于粉丝投票")
        print(f"  在 {rule_counts.get('规则1_排名相加', 0)} 个season中，规则1与粉丝投票的一致性更高")
    else:
        print("✓ 百分比相加方式（规则2）更倾向于粉丝投票")
        print(f"  在 {rule_counts.get('规则2_百分比相加', 0)} 个season中，规则2与粉丝投票的一致性更高")
    print(f"\n平均tau值比较:")
    print(f"  规则1与粉丝投票的平均一致性: {result_df['tau_规则1_vs_规则3_排名vs粉丝'].mean():.4f}")
    print(f"  规则2与粉丝投票的平均一致性: {result_df['tau_规则2_vs_规则3_百分比vs粉丝'].mean():.4f}")
    if result_df['tau_规则1_vs_规则3_排名vs粉丝'].mean() > result_df['tau_规则2_vs_规则3_百分比vs粉丝'].mean():
        print("  → 规则1（排名相加）与粉丝投票的一致性更高")
    else:
        print("  → 规则2（百分比相加）与粉丝投票的一致性更高")
    
    return result_df


if __name__ == "__main__":
    # 分析数据并生成结果
    input_file = "elimination_results.csv"
    output_file = "kendall_tau_analysis.csv"
    
    result_df = calculate_kendall_tau_analysis(input_file, output_file)
    
    # 显示前几行结果
    print("\n前10行分析结果:")
    print(result_df.head(10).to_string())


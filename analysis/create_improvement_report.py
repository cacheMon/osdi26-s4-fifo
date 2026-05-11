#!/usr/bin/env python3
"""
创建详细的提升报告
"""
import pandas as pd
import numpy as np

def create_detailed_report():
    """创建详细报告"""
    print("Creating detailed improvement report...\n")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Grid Optimal 提升报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 读取汇总数据
    summary_df = pd.read_csv('cleaned/update_summary.csv')
    
    report_lines.append("## 整体汇总")
    report_lines.append("")
    report_lines.append(f"{'Cache Ratio':<15} {'Traces':<10} {'提升':<10} {'百分比':<12} {'平均提升':<15}")
    report_lines.append("-" * 80)
    
    for _, row in summary_df.iterrows():
        ratio = row['ratio']
        total = row['total_traces']
        improved = row['improved_traces']
        pct = row['improved_traces'] / row['total_traces'] * 100
        avg_imp = row['avg_improvement_pct']
        report_lines.append(f"{ratio:<15} {total:<10} {improved:<10} {pct:>10.2f}%  {avg_imp:>13.3f}%")
    
    report_lines.append("")
    report_lines.append("## 详细分析")
    report_lines.append("")
    
    # 对每个ratio进行详细分析
    for ratio in ['0.001', '0.01', '0.1']:
        report_lines.append(f"\n### Cache Ratio: {ratio}")
        report_lines.append("-" * 80)
        
        # 读取对比数据
        comp_df = pd.read_csv(f'cleaned/{ratio}/optimal_comparison.csv')
        
        # 基本统计
        improved = comp_df[comp_df['improvement'] > 0]
        
        if len(improved) > 0:
            report_lines.append(f"\n提升的traces数量: {len(improved)} / {len(comp_df)} ({len(improved)/len(comp_df)*100:.2f}%)")
            report_lines.append(f"平均miss ratio: {comp_df['miss_ratio_old'].mean():.6f} -> {comp_df['miss_ratio_new'].mean():.6f}")
            report_lines.append(f"绝对提升: {comp_df['improvement'].mean():.6f}")
            report_lines.append(f"相对提升: {(comp_df['improvement'].mean() / comp_df['miss_ratio_old'].mean() * 100):.3f}%")
            
            # 提升分布
            report_lines.append(f"\n提升分布（针对提升的traces）:")
            report_lines.append(f"  最小提升: {improved['improvement'].min():.6f} ({improved['improvement_pct'].min():.3f}%)")
            report_lines.append(f"  25分位数: {improved['improvement'].quantile(0.25):.6f}")
            report_lines.append(f"  中位数:   {improved['improvement'].median():.6f}")
            report_lines.append(f"  75分位数: {improved['improvement'].quantile(0.75):.6f}")
            report_lines.append(f"  最大提升: {improved['improvement'].max():.6f} ({improved['improvement_pct'].max():.3f}%)")
            
            # 按类别统计
            report_lines.append(f"\n按提升幅度分类:")
            bins = [0, 0.001, 0.005, 0.01, 0.05, 1.0]
            labels = ['0-0.1%', '0.1-0.5%', '0.5-1%', '1-5%', '>5%']
            improved_copy = improved.copy()
            improved_copy['category'] = pd.cut(improved_copy['improvement'], bins=bins, labels=labels)
            for cat in labels:
                count = (improved_copy['category'] == cat).sum()
                if count > 0:
                    report_lines.append(f"  {cat:>10s}: {count:4d} traces ({count/len(improved)*100:5.2f}%)")
            
            # Top 20提升的traces
            report_lines.append(f"\nTop 20 提升最大的traces:")
            report_lines.append(f"{'Rank':<6} {'Trace':<55} {'Old MR':<10} {'New MR':<10} {'提升':<12} {'提升%':<10}")
            report_lines.append("-" * 110)
            top20 = improved.nlargest(20, 'improvement')
            for i, (_, row) in enumerate(top20.iterrows(), 1):
                trace_name = row['trace'][:50]
                report_lines.append(
                    f"{i:<6} {trace_name:<55} {row['miss_ratio_old']:<10.6f} "
                    f"{row['miss_ratio_new']:<10.6f} {row['improvement']:<12.6f} {row['improvement_pct']:<10.3f}%"
                )
        else:
            report_lines.append(f"\n没有traces得到提升")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("数据源说明:")
    report_lines.append("- 原始grid_full: 248,208行 (每个ratio)")
    report_lines.append("- aggregated_results: 744,624行 (所有ratio)")
    report_lines.append("- 新增配置组合: 186,156个 (每个ratio)")
    report_lines.append("- 更新已有配置: 62,052个 (每个ratio)")
    report_lines.append("- 新grid_full总计: 434,364行 (每个ratio)")
    report_lines.append("=" * 80)
    
    # 写入文件
    report_text = "\n".join(report_lines)
    with open('cleaned/improvement_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n报告已保存到: cleaned/improvement_report.txt")

if __name__ == '__main__':
    create_detailed_report()

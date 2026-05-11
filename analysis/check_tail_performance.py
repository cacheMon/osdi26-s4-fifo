#!/usr/bin/env python3
"""
快速查看tail performance结果（包括P1 best）
"""
import pandas as pd

def format_pct(val):
    """格式化百分比"""
    if abs(val) < 10:
        return f"{val:+7.2f}%"
    elif abs(val) < 100:
        return f"{val:+7.1f}%"
    else:
        return f"{val:+7.0f}%"

def display_tail_stats(ratio):
    """显示指定ratio的tail statistics"""
    print(f"\n{'='*140}")
    print(f"Cache Ratio: {ratio} - Tail Performance vs FIFO")
    print(f"{'='*140}")
    
    df = pd.read_csv(f'cleaned/tail_stats_ratio_{ratio}_vs_fifo.csv')
    
    # 找出Grid Optimal和Grid Optimal Old的行
    grid_opt_mask = df['algorithm'] == 'Grid Optimal'
    grid_opt_old_mask = df['algorithm'] == 'Grid Optimal Old'
    
    print(f"\n{'Algorithm':<20} {'Worst':<12} {'P99 Worst':<12} {'Mean':<12} {'P1 Best':<12} | 说明")
    print("-" * 140)
    
    for _, row in df.iterrows():
        algo = row['algorithm']
        worst = format_pct(row['worst_vs_fifo_pct'])
        p99_worst = format_pct(row['p99_worst_vs_fifo_pct'])
        mean = format_pct(row['mean_vs_fifo_pct'])
        p1_best = format_pct(row['p1_best_vs_fifo_pct'])
        
        # 添加说明
        note = ""
        if algo == 'Grid Optimal':
            note = "← 新optimal (基于扩展参数空间)"
        elif algo == 'Grid Optimal Old':
            note = "← 旧optimal (原始参数空间)"
        elif algo == 'Learned':
            note = "← 机器学习方法"
        
        # 特殊格式化Grid Optimal行
        if algo in ['Grid Optimal', 'Grid Optimal Old']:
            print(f"**{algo:<18}** {worst:<12} {p99_worst:<12} {mean:<12} {p1_best:<12} | {note}")
        else:
            print(f"{algo:<20} {worst:<12} {p99_worst:<12} {mean:<12} {p1_best:<12} | {note}")
    
    # 对比Grid Optimal vs Grid Optimal Old
    if grid_opt_mask.any() and grid_opt_old_mask.any():
        old_row = df[grid_opt_old_mask].iloc[0]
        new_row = df[grid_opt_mask].iloc[0]
        
        print(f"\n{'='*140}")
        print("Grid Optimal New vs Old 对比:")
        print("-" * 140)
        
        metrics = [
            ('Worst Case', 'worst_vs_fifo_pct'),
            ('P99 Worst', 'p99_worst_vs_fifo_pct'),
            ('Mean', 'mean_vs_fifo_pct'),
            ('P1 Best', 'p1_best_vs_fifo_pct')
        ]
        
        for metric_name, col_name in metrics:
            old_val = old_row[col_name]
            new_val = new_row[col_name]
            diff = new_val - old_val
            
            if abs(old_val) > 0.001:
                improvement = diff / abs(old_val) * 100
                status = "✓ 改进" if diff < 0 else ("✗ 退化" if diff > 0 else "= 相同")
                print(f"{metric_name:<15}: Old={format_pct(old_val)}  New={format_pct(new_val)}  Δ={format_pct(diff)}  ({status})")
            else:
                print(f"{metric_name:<15}: Old={format_pct(old_val)}  New={format_pct(new_val)}  Δ={format_pct(diff)}")

def main():
    print("\n" + "="*140)
    print("Tail Performance Analysis - 包括 Mean 和 P1 Best")
    print("="*140)
    print("\n说明:")
    print("  - 负值表示比FIFO更好（miss ratio更低）")
    print("  - 正值表示比FIFO更差（miss ratio更高）")
    print("  - Worst: 所有traces中表现最差的单个trace")
    print("  - P99 Worst: 99分位数（最差的1%的traces）")
    print("  - Mean: 平均性能")
    print("  - P1 Best: 1分位数（最好的1%的traces）")
    
    for ratio in ['0.001', '0.01', '0.1']:
        display_tail_stats(ratio)
    
    print("\n" + "="*140)
    print("可视化图表已保存到:")
    for ratio in ['0.001', '0.01', '0.1']:
        print(f"  - cleaned/figure_ratio_{ratio}_tail_performance_vs_fifo.png (2x2布局：Worst, P99 Worst, Mean, P1 Best)")
    print("="*140 + "\n")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""分析视频级别指标报告

该脚本用于分析训练过程中生成的视频级别指标报告，帮助识别：
- 表现异常差的视频
- 数据质量问题
- 性能分布特征
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_video_metrics(csv_path: str, top_n: int = 10):
    """分析视频级别指标报告
    
    Args:
        csv_path: CSV文件路径
        top_n: 显示前N个最差/最好的视频
    """
    # 读取CSV文件
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print(f"📊 视频级别指标分析报告")
    print(f"📁 文件: {csv_path}")
    print("=" * 80)
    
    # 基本统计信息
    print(f"\n📈 基本统计信息:")
    print(f"   总视频数: {len(df)}")
    print(f"   总片段数: {df['total_clips'].sum()}")
    print(f"   平均每个视频的片段数: {df['total_clips'].mean():.1f}")
    
    # 性能统计
    print(f"\n🎯 性能统计:")
    for metric in ['avg_precision', 'avg_recall', 'avg_f1', 'avg_accuracy']:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        median_val = df[metric].median()
        min_val = df[metric].min()
        max_val = df[metric].max()
        
        metric_name = metric.replace('avg_', '').replace('_', ' ').title()
        print(f"   {metric_name}:")
        print(f"      平均值: {mean_val:.4f} ± {std_val:.4f}")
        print(f"      中位数: {median_val:.4f}")
        print(f"      范围: [{min_val:.4f}, {max_val:.4f}]")
    
    # 表现最差的视频
    print(f"\n❌ 表现最差的 {top_n} 个视频 (按F1分数排序):")
    worst_videos = df.nsmallest(top_n, 'avg_f1')
    print(worst_videos[['session_name', 'total_clips', 'avg_precision', 'avg_recall', 
                        'avg_f1', 'avg_accuracy']].to_string(index=False))
    
    # 表现最好的视频
    print(f"\n✅ 表现最好的 {top_n} 个视频 (按F1分数排序):")
    best_videos = df.nlargest(top_n, 'avg_f1')
    print(best_videos[['session_name', 'total_clips', 'avg_precision', 'avg_recall', 
                       'avg_f1', 'avg_accuracy']].to_string(index=False))
    
    # 异常检测（使用3-sigma规则）
    print(f"\n⚠️  异常视频检测 (F1分数低于 mean - 2*std):")
    mean_f1 = df['avg_f1'].mean()
    std_f1 = df['avg_f1'].std()
    threshold = mean_f1 - 2 * std_f1
    
    outliers = df[df['avg_f1'] < threshold]
    if len(outliers) > 0:
        print(f"   检测到 {len(outliers)} 个异常视频 (F1 < {threshold:.4f}):")
        print(outliers[['session_name', 'total_clips', 'avg_f1', 'avg_accuracy']].to_string(index=False))
    else:
        print(f"   未检测到异常视频")
    
    # 片段数量分析
    print(f"\n📦 片段数量分布:")
    clip_bins = [0, 50, 100, 150, 200, float('inf')]
    clip_labels = ['0-50', '51-100', '101-150', '151-200', '200+']
    df['clip_range'] = pd.cut(df['total_clips'], bins=clip_bins, labels=clip_labels)
    
    for label in clip_labels:
        videos_in_range = df[df['clip_range'] == label]
        if len(videos_in_range) > 0:
            avg_f1 = videos_in_range['avg_f1'].mean()
            print(f"   {label} 片段: {len(videos_in_range)} 个视频, 平均F1={avg_f1:.4f}")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='分析视频级别指标报告')
    parser.add_argument('csv_path', type=str, help='CSV文件路径')
    parser.add_argument('--top-n', type=int, default=10, help='显示前N个最差/最好的视频 (默认: 10)')
    
    args = parser.parse_args()
    
    analyze_video_metrics(args.csv_path, args.top_n)

if __name__ == '__main__':
    main()


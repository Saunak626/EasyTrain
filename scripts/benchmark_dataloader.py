#!/usr/bin/env python3
"""
数据加载器性能基准测试脚本

用于测试不同配置下的数据加载速度，帮助诊断I/O瓶颈。

使用方法：
    python scripts/benchmark_dataloader.py --num_workers 4 --batch_size 8
"""

import os
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.neonatal_multilabel_simple import NeonatalMultilabelSimple


def benchmark_dataloader(num_workers, batch_size, num_batches=50, persistent_workers=True, prefetch_factor=2):
    """基准测试数据加载器性能
    
    Args:
        num_workers: worker进程数
        batch_size: 批大小
        num_batches: 测试的batch数量
        persistent_workers: 是否启用persistent_workers
        prefetch_factor: 预加载因子
    """
    print("=" * 80)
    print(f"数据加载器性能基准测试")
    print("=" * 80)
    print(f"配置:")
    print(f"  - num_workers: {num_workers}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - persistent_workers: {persistent_workers}")
    print(f"  - prefetch_factor: {prefetch_factor}")
    print(f"  - 测试batch数: {num_batches}")
    print("-" * 80)
    
    # 创建数据集
    print("正在创建数据集...")
    dataset = NeonatalMultilabelSimple(
        frames_dir='../Neonate-Feeding-Assessment/data/cpu_processed_627/frames_segments',
        labels_file='../Neonate-Feeding-Assessment/result_xlsx/shanghai/multi_hot_labels.xlsx',
        split='test',  # 使用测试集（较小）
        clip_len=16
    )
    print(f"数据集大小: {len(dataset)} 样本")
    
    # 创建数据加载器
    print("正在创建数据加载器...")
    use_persistent = persistent_workers and num_workers > 0
    use_prefetch = prefetch_factor if num_workers > 0 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=use_prefetch
    )
    
    # 预热（第一次迭代通常较慢）
    print("预热中...")
    for i, (frames, labels) in enumerate(dataloader):
        if i >= 2:
            break
    
    # 基准测试
    print(f"\n开始基准测试（{num_batches} batches）...")
    batch_times = []
    total_start = time.time()
    
    for i, (frames, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        # 模拟GPU计算（将数据移到GPU）
        if torch.cuda.is_available():
            frames = frames.cuda()
            labels = labels.cuda()
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        if (i + 1) % 10 == 0:
            avg_time = sum(batch_times[-10:]) / 10
            print(f"  Batch {i+1}/{num_batches} | 平均时间: {avg_time:.3f}s/batch")
    
    total_time = time.time() - total_start
    
    # 统计结果
    print("\n" + "=" * 80)
    print("基准测试结果:")
    print("=" * 80)
    print(f"总时间: {total_time:.2f}s")
    print(f"平均时间: {sum(batch_times) / len(batch_times):.3f}s/batch")
    print(f"最快: {min(batch_times):.3f}s/batch")
    print(f"最慢: {max(batch_times):.3f}s/batch")
    print(f"吞吐量: {batch_size * len(batch_times) / total_time:.1f} 样本/秒")
    
    # 预估完整epoch时间
    total_batches = len(dataloader)
    estimated_epoch_time = (sum(batch_times) / len(batch_times)) * total_batches
    print(f"\n预估完整epoch时间: {estimated_epoch_time / 60:.1f} 分钟 ({total_batches} batches)")
    
    return {
        'avg_time': sum(batch_times) / len(batch_times),
        'total_time': total_time,
        'throughput': batch_size * len(batch_times) / total_time
    }


def compare_configurations():
    """比较不同配置的性能"""
    print("\n" + "🔬 " * 20)
    print("多配置对比测试")
    print("🔬 " * 20 + "\n")
    
    configs = [
        {'num_workers': 0, 'batch_size': 8, 'persistent_workers': False, 'prefetch_factor': None},
        {'num_workers': 2, 'batch_size': 8, 'persistent_workers': False, 'prefetch_factor': 2},
        {'num_workers': 4, 'batch_size': 8, 'persistent_workers': True, 'prefetch_factor': 2},
        {'num_workers': 8, 'batch_size': 8, 'persistent_workers': True, 'prefetch_factor': 2},
        {'num_workers': 12, 'batch_size': 8, 'persistent_workers': True, 'prefetch_factor': 2},
    ]
    
    results = []
    for config in configs:
        result = benchmark_dataloader(num_batches=30, **config)
        results.append({**config, **result})
        time.sleep(2)  # 短暂休息
    
    # 打印对比表
    print("\n" + "=" * 100)
    print("配置对比总结")
    print("=" * 100)
    print(f"{'Workers':<10} {'Batch':<8} {'Persistent':<12} {'Prefetch':<10} {'平均时间':<12} {'吞吐量':<15}")
    print("-" * 100)
    for r in results:
        print(f"{r['num_workers']:<10} {r['batch_size']:<8} {str(r['persistent_workers']):<12} "
              f"{str(r['prefetch_factor']):<10} {r['avg_time']:.3f}s/batch  {r['throughput']:.1f} 样本/秒")
    
    # 找出最佳配置
    best = min(results, key=lambda x: x['avg_time'])
    print("\n🏆 最佳配置:")
    print(f"  num_workers={best['num_workers']}, batch_size={best['batch_size']}, "
          f"persistent_workers={best['persistent_workers']}, prefetch_factor={best['prefetch_factor']}")
    print(f"  平均时间: {best['avg_time']:.3f}s/batch")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据加载器性能基准测试')
    parser.add_argument('--num_workers', type=int, default=4, help='worker进程数')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--num_batches', type=int, default=50, help='测试的batch数量')
    parser.add_argument('--persistent_workers', action='store_true', help='启用persistent_workers')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='预加载因子')
    parser.add_argument('--compare', action='store_true', help='对比多种配置')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_configurations()
    else:
        benchmark_dataloader(
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor
        )


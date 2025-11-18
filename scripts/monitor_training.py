#!/usr/bin/env python3
"""
训练性能实时监控脚本

监控GPU利用率、显存占用、训练速度等关键指标，帮助诊断性能问题。

使用方法：
    python scripts/monitor_training.py
"""

import subprocess
import time
import re
from datetime import datetime


def get_gpu_stats():
    """获取GPU统计信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        gpu_stats = []
        
        for line in lines:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 6:
                gpu_stats.append({
                    'index': int(parts[0]),
                    'utilization': float(parts[1]),
                    'memory_used': float(parts[2]),
                    'memory_total': float(parts[3]),
                    'power': float(parts[4]),
                    'temperature': float(parts[5])
                })
        
        return gpu_stats
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return []


def format_gpu_stats(stats):
    """格式化GPU统计信息"""
    if not stats:
        return "无GPU信息"
    
    lines = []
    for gpu in stats:
        memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
        lines.append(
            f"GPU {gpu['index']}: "
            f"利用率={gpu['utilization']:5.1f}% | "
            f"显存={gpu['memory_used']:6.0f}/{gpu['memory_total']:6.0f}MB ({memory_percent:5.1f}%) | "
            f"功耗={gpu['power']:6.1f}W | "
            f"温度={gpu['temperature']:4.1f}°C"
        )
    
    return '\n'.join(lines)


def diagnose_performance(stats):
    """诊断性能问题"""
    if not stats:
        return []
    
    issues = []
    
    for gpu in stats:
        # 检查GPU利用率
        if gpu['utilization'] < 20:
            issues.append(f"⚠️  GPU {gpu['index']} 利用率过低 ({gpu['utilization']:.1f}%) - 可能存在数据加载瓶颈")
        
        # 检查显存使用
        memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
        if memory_percent > 90:
            issues.append(f"⚠️  GPU {gpu['index']} 显存使用过高 ({memory_percent:.1f}%) - 可能需要降低batch_size")
        elif memory_percent < 30:
            issues.append(f"💡 GPU {gpu['index']} 显存使用较低 ({memory_percent:.1f}%) - 可以尝试增大batch_size")
        
        # 检查功耗
        if gpu['power'] < 100:
            issues.append(f"⚠️  GPU {gpu['index']} 功耗过低 ({gpu['power']:.1f}W) - GPU处于空闲状态")
        
        # 检查温度
        if gpu['temperature'] > 85:
            issues.append(f"🔥 GPU {gpu['index']} 温度过高 ({gpu['temperature']:.1f}°C) - 注意散热")
    
    return issues


def monitor_training(interval=5, duration=300):
    """监控训练性能
    
    Args:
        interval: 监控间隔（秒）
        duration: 监控时长（秒），None表示持续监控
    """
    print("=" * 100)
    print("训练性能实时监控")
    print("=" * 100)
    print(f"监控间隔: {interval}秒")
    print(f"监控时长: {duration}秒" if duration else "持续监控（按Ctrl+C停止）")
    print("=" * 100)
    print()
    
    start_time = time.time()
    iteration = 0
    
    # 记录历史数据用于趋势分析
    history = {
        'utilization': [],
        'memory_used': [],
        'power': [],
        'temperature': []
    }
    
    try:
        while True:
            iteration += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - start_time
            
            # 获取GPU统计
            stats = get_gpu_stats()
            
            # 清屏（可选）
            # print("\033[2J\033[H")
            
            print(f"\n{'='*100}")
            print(f"[{current_time}] 监控迭代 #{iteration} (已运行 {elapsed:.0f}秒)")
            print(f"{'='*100}")
            
            # 显示GPU状态
            print("\n📊 GPU状态:")
            print(format_gpu_stats(stats))
            
            # 记录历史数据
            if stats:
                for gpu in stats:
                    history['utilization'].append(gpu['utilization'])
                    history['memory_used'].append(gpu['memory_used'])
                    history['power'].append(gpu['power'])
                    history['temperature'].append(gpu['temperature'])
            
            # 诊断问题
            issues = diagnose_performance(stats)
            if issues:
                print("\n🔍 性能诊断:")
                for issue in issues:
                    print(f"  {issue}")
            else:
                print("\n✅ 性能正常")
            
            # 显示趋势（最近5次平均）
            if len(history['utilization']) >= 5:
                recent_util = sum(history['utilization'][-5:]) / 5
                recent_power = sum(history['power'][-5:]) / 5
                print(f"\n📈 最近趋势（5次平均）:")
                print(f"  平均GPU利用率: {recent_util:.1f}%")
                print(f"  平均功耗: {recent_power:.1f}W")
            
            # 检查是否达到监控时长
            if duration and elapsed >= duration:
                print(f"\n监控完成（已运行{duration}秒）")
                break
            
            # 等待下一次监控
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n监控已停止（用户中断）")
    
    # 显示总结
    if history['utilization']:
        print("\n" + "=" * 100)
        print("监控总结")
        print("=" * 100)
        print(f"监控时长: {elapsed:.0f}秒")
        print(f"监控次数: {iteration}")
        print(f"\nGPU利用率:")
        print(f"  平均: {sum(history['utilization']) / len(history['utilization']):.1f}%")
        print(f"  最大: {max(history['utilization']):.1f}%")
        print(f"  最小: {min(history['utilization']):.1f}%")
        print(f"\n功耗:")
        print(f"  平均: {sum(history['power']) / len(history['power']):.1f}W")
        print(f"  最大: {max(history['power']):.1f}W")
        print(f"  最小: {min(history['power']):.1f}W")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练性能实时监控')
    parser.add_argument('--interval', type=int, default=5, help='监控间隔（秒）')
    parser.add_argument('--duration', type=int, default=None, help='监控时长（秒），不指定则持续监控')
    
    args = parser.parse_args()
    
    monitor_training(interval=args.interval, duration=args.duration)


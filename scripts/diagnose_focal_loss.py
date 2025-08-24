#!/usr/bin/env python3
"""
Focal Loss过度预测问题诊断脚本

分析Focal Loss在多标签分类中的表现，识别过度预测问题并提供修复建议。
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_metrics_file(metrics_file):
    """分析单个指标文件"""
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        epoch = metrics.get('epoch', 0)
        class_metrics = metrics.get('class_metrics', {})
        
        # 分析每个类别的精确率和召回率
        analysis = {
            'epoch': epoch,
            'classes': {},
            'overall': {}
        }
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_name, class_data in class_metrics.items():
            precision = class_data.get('precision', 0)
            recall = class_data.get('recall', 0)
            f1 = class_data.get('f1', 0)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            
            analysis['classes'][class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'imbalance_ratio': recall / max(precision, 0.001)  # 召回率/精确率比例
            }
        
        # 整体分析
        analysis['overall'] = {
            'avg_precision': np.mean(precisions),
            'avg_recall': np.mean(recalls),
            'avg_f1': np.mean(f1_scores),
            'precision_recall_ratio': np.mean(recalls) / max(np.mean(precisions), 0.001),
            'min_precision': np.min(precisions),
            'max_recall': np.max(recalls)
        }
        
        return analysis
        
    except Exception as e:
        print(f"❌ 分析文件 {metrics_file} 失败: {e}")
        return None

def diagnose_overprediction(analysis):
    """诊断过度预测问题"""
    overall = analysis['overall']
    classes = analysis['classes']
    
    # 过度预测的判断标准
    issues = []
    severity = "正常"
    
    # 1. 精确率vs召回率失衡
    pr_ratio = overall['precision_recall_ratio']
    if pr_ratio > 2.0:
        issues.append(f"严重的精确率-召回率失衡 (比例: {pr_ratio:.2f})")
        severity = "严重"
    elif pr_ratio > 1.5:
        issues.append(f"中等的精确率-召回率失衡 (比例: {pr_ratio:.2f})")
        severity = "中等" if severity == "正常" else severity
    
    # 2. 整体精确率过低
    if overall['avg_precision'] < 0.3:
        issues.append(f"整体精确率过低 ({overall['avg_precision']:.3f})")
        severity = "严重"
    elif overall['avg_precision'] < 0.5:
        issues.append(f"整体精确率偏低 ({overall['avg_precision']:.3f})")
        severity = "中等" if severity == "正常" else severity
    
    # 3. 最低精确率过低
    if overall['min_precision'] < 0.1:
        issues.append(f"存在极低精确率类别 ({overall['min_precision']:.3f})")
        severity = "严重"
    
    # 4. 召回率过高
    if overall['max_recall'] > 0.95:
        issues.append(f"存在过高召回率类别 ({overall['max_recall']:.3f})")
        severity = "中等" if severity == "正常" else severity
    
    # 5. 类别级别分析
    problematic_classes = []
    for class_name, class_data in classes.items():
        if class_data['imbalance_ratio'] > 3.0:
            problematic_classes.append(f"{class_name} (比例: {class_data['imbalance_ratio']:.2f})")
    
    if problematic_classes:
        issues.append(f"问题类别: {', '.join(problematic_classes)}")
    
    return {
        'severity': severity,
        'issues': issues,
        'metrics': overall
    }

def generate_recommendations(diagnosis, current_config=None):
    """生成修复建议"""
    severity = diagnosis['severity']
    recommendations = []
    
    if severity == "严重":
        recommendations.extend([
            "🚨 立即停止当前训练，问题严重",
            "📉 使用保守参数重新开始：gamma=1.0, alpha=1.0, pos_weight=3.0",
            "🔄 考虑切换到 focal_multilabel_balanced 损失函数",
            "📚 降低学习率到 0.0005",
            "⏱️ 增加训练epoch数，给模型更多时间收敛"
        ])
    elif severity == "中等":
        recommendations.extend([
            "⚠️ 需要调整参数，但可以继续观察",
            "📊 降低 gamma 参数 (当前值 → 当前值*0.7)",
            "⚖️ 如果使用了alpha权重，考虑设置为1.0",
            "📉 适当降低 pos_weight (当前值 → 当前值*0.8)",
            "📈 监控后续几个epoch的表现"
        ])
    else:
        recommendations.extend([
            "✅ 当前表现正常，继续训练",
            "📊 可以考虑轻微调优参数以获得更好效果",
            "📈 持续监控精确率和召回率的平衡"
        ])
    
    # 基于当前配置的具体建议
    if current_config:
        loss_params = current_config.get('loss', {}).get('params', {})
        gamma = loss_params.get('gamma', 2.0)
        alpha = loss_params.get('alpha', 1.0)
        pos_weight = loss_params.get('pos_weight', 1.0)
        
        recommendations.append("\n🔧 具体参数调整建议:")
        
        if severity == "严重":
            recommendations.append(f"  gamma: {gamma} → 1.0")
            recommendations.append(f"  alpha: {alpha} → 1.0")
            recommendations.append(f"  pos_weight: {pos_weight} → 3.0")
        elif severity == "中等":
            new_gamma = max(0.5, gamma * 0.7)
            new_pos_weight = max(1.0, pos_weight * 0.8)
            recommendations.append(f"  gamma: {gamma} → {new_gamma:.1f}")
            if alpha != 1.0:
                recommendations.append(f"  alpha: {alpha} → 1.0")
            recommendations.append(f"  pos_weight: {pos_weight} → {new_pos_weight:.1f}")
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='诊断Focal Loss过度预测问题')
    parser.add_argument('--metrics_dir', type=str, default='runs/neonatal_multilabel',
                       help='指标文件目录')
    parser.add_argument('--config', type=str, help='配置文件路径（可选）')
    parser.add_argument('--latest_only', action='store_true',
                       help='只分析最新的指标文件')
    
    args = parser.parse_args()
    
    print("🔍 Focal Loss过度预测问题诊断")
    print("=" * 60)
    
    # 查找指标文件
    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.exists():
        print(f"❌ 指标目录不存在: {metrics_dir}")
        return 1
    
    # 查找epoch指标文件
    epoch_files = list(metrics_dir.glob("epoch_*_metrics.json"))
    if not epoch_files:
        print(f"❌ 在 {metrics_dir} 中未找到epoch指标文件")
        return 1
    
    # 按epoch排序
    epoch_files.sort(key=lambda x: int(x.stem.split('_')[1]))
    
    if args.latest_only:
        epoch_files = [epoch_files[-1]]
    
    print(f"📊 找到 {len(epoch_files)} 个指标文件")
    
    # 加载配置文件（如果提供）
    current_config = None
    if args.config:
        try:
            import yaml
            with open(args.config, 'r', encoding='utf-8') as f:
                current_config = yaml.safe_load(f)
            print(f"📋 已加载配置文件: {args.config}")
        except Exception as e:
            print(f"⚠️ 无法加载配置文件: {e}")
    
    # 分析每个epoch
    all_diagnoses = []
    
    for epoch_file in epoch_files:
        print(f"\n📈 分析 {epoch_file.name}")
        print("-" * 40)
        
        analysis = analyze_metrics_file(epoch_file)
        if analysis is None:
            continue
        
        diagnosis = diagnose_overprediction(analysis)
        all_diagnoses.append((analysis['epoch'], diagnosis))
        
        # 显示诊断结果
        epoch = analysis['epoch']
        severity = diagnosis['severity']
        metrics = diagnosis['metrics']
        
        # 根据严重程度选择emoji
        severity_emoji = {"正常": "✅", "中等": "⚠️", "严重": "🚨"}
        
        print(f"Epoch {epoch} - {severity_emoji[severity]} {severity}")
        print(f"  平均精确率: {metrics['avg_precision']:.3f}")
        print(f"  平均召回率: {metrics['avg_recall']:.3f}")
        print(f"  平均F1分数: {metrics['avg_f1']:.3f}")
        print(f"  精确率/召回率比例: {1/metrics['precision_recall_ratio']:.3f}")
        
        if diagnosis['issues']:
            print("  问题:")
            for issue in diagnosis['issues']:
                print(f"    • {issue}")
    
    # 生成总体建议
    if all_diagnoses:
        print("\n" + "="*60)
        print("📋 总体诊断和建议")
        print("="*60)
        
        # 分析趋势
        latest_diagnosis = all_diagnoses[-1][1]
        latest_severity = latest_diagnosis['severity']
        
        print(f"🎯 当前状态: {latest_severity}")
        
        # 生成建议
        recommendations = generate_recommendations(latest_diagnosis, current_config)
        
        print("\n💡 修复建议:")
        for rec in recommendations:
            print(f"  {rec}")
        
        # 趋势分析
        if len(all_diagnoses) > 1:
            print(f"\n📈 趋势分析:")
            severities = [d[1]['severity'] for d in all_diagnoses[-3:]]  # 最近3个epoch
            if all(s == "严重" for s in severities[-2:]):
                print("  📉 问题持续恶化，建议立即停止训练")
            elif latest_severity == "正常":
                print("  📈 表现良好，可以继续训练")
            else:
                print("  📊 表现不稳定，需要密切监控")
        
        # 返回状态码
        if latest_severity == "严重":
            return 2
        elif latest_severity == "中等":
            return 1
        else:
            return 0
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

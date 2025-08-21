"""
多标签分类评估指标
提供专门针对多标签分类任务的评估指标计算
"""

import torch
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_multilabel_metrics(outputs: torch.Tensor, targets: torch.Tensor, 
                                threshold: float = 0.5) -> Dict[str, float]:
    """计算多标签分类的各种评估指标
    
    Args:
        outputs (torch.Tensor): 模型输出的logits，形状为(batch_size, num_classes)
        targets (torch.Tensor): 真实标签，形状为(batch_size, num_classes)
        threshold (float): 二值化阈值，默认0.5
        
    Returns:
        Dict[str, float]: 包含各种指标的字典
    """
    # 获取预测
    predictions = torch.sigmoid(outputs) > threshold
    targets_bool = targets.bool()
    
    batch_size, num_classes = targets.shape
    
    # 1. 汉明准确率（Hamming Accuracy）- 推荐作为主要指标
    hamming_accuracy = (predictions == targets_bool).float().mean().item()
    
    # 2. 完全匹配准确率（Exact Match Accuracy / Subset Accuracy）
    exact_match_accuracy = (predictions == targets_bool).all(dim=1).float().mean().item()
    
    # 3. 计算每个样本的精确率、召回率、F1分数
    tp = (predictions * targets_bool).sum(dim=1).float()  # True Positives
    fp = (predictions * (~targets_bool)).sum(dim=1).float()  # False Positives
    fn = ((~predictions) * targets_bool).sum(dim=1).float()  # False Negatives
    
    # 避免除零错误
    epsilon = 1e-8
    
    # 样本级精确率、召回率、F1
    precision_per_sample = tp / (tp + fp + epsilon)
    recall_per_sample = tp / (tp + fn + epsilon)
    f1_per_sample = 2 * precision_per_sample * recall_per_sample / (precision_per_sample + recall_per_sample + epsilon)
    
    # 平均指标
    precision = precision_per_sample.mean().item()
    recall = recall_per_sample.mean().item()
    f1_score = f1_per_sample.mean().item()
    
    # 4. 标签级指标（Label-wise metrics）
    # 每个标签的精确率、召回率、F1
    tp_label = (predictions * targets_bool).sum(dim=0).float()
    fp_label = (predictions * (~targets_bool)).sum(dim=0).float()
    fn_label = ((~predictions) * targets_bool).sum(dim=0).float()
    
    precision_label = tp_label / (tp_label + fp_label + epsilon)
    recall_label = tp_label / (tp_label + fn_label + epsilon)
    f1_label = 2 * precision_label * recall_label / (precision_label + recall_label + epsilon)
    
    # 标签级平均指标
    macro_precision = precision_label.mean().item()
    macro_recall = recall_label.mean().item()
    macro_f1 = f1_label.mean().item()
    
    # 5. 覆盖率相关指标
    # 平均激活标签数
    avg_predicted_labels = predictions.sum(dim=1).float().mean().item()
    avg_true_labels = targets_bool.sum(dim=1).float().mean().item()
    
    # 标签覆盖率
    label_coverage = (predictions.sum(dim=0) > 0).float().mean().item()
    
    return {
        # 主要指标
        'hamming_accuracy': hamming_accuracy,
        'exact_match_accuracy': exact_match_accuracy,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall,
        
        # 标签级指标
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        
        # 覆盖率指标
        'avg_predicted_labels': avg_predicted_labels,
        'avg_true_labels': avg_true_labels,
        'label_coverage': label_coverage,
    }


def calculate_per_label_metrics(outputs: torch.Tensor, targets: torch.Tensor, 
                               label_names: list = None, threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """计算每个标签的详细指标
    
    Args:
        outputs (torch.Tensor): 模型输出的logits
        targets (torch.Tensor): 真实标签
        label_names (list): 标签名称列表
        threshold (float): 二值化阈值
        
    Returns:
        Dict[str, Dict[str, float]]: 每个标签的指标字典
    """
    predictions = torch.sigmoid(outputs) > threshold
    targets_bool = targets.bool()
    
    num_classes = targets.shape[1]
    if label_names is None:
        label_names = [f'label_{i}' for i in range(num_classes)]
    
    per_label_metrics = {}
    
    for i, label_name in enumerate(label_names):
        pred_i = predictions[:, i]
        target_i = targets_bool[:, i]
        
        tp = (pred_i * target_i).sum().float().item()
        fp = (pred_i * (~target_i)).sum().float().item()
        fn = ((~pred_i) * target_i).sum().float().item()
        tn = ((~pred_i) * (~target_i)).sum().float().item()
        
        epsilon = 1e-8
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        
        per_label_metrics[label_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'support': int(target_i.sum().item()),  # 正样本数
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    return per_label_metrics


def print_multilabel_report(outputs: torch.Tensor, targets: torch.Tensor, 
                           label_names: list = None, threshold: float = 0.5):
    """打印多标签分类的详细报告
    
    Args:
        outputs (torch.Tensor): 模型输出的logits
        targets (torch.Tensor): 真实标签
        label_names (list): 标签名称列表
        threshold (float): 二值化阈值
    """
    # 计算总体指标
    overall_metrics = calculate_multilabel_metrics(outputs, targets, threshold)
    
    print("🏷️  多标签分类评估报告")
    print("=" * 60)
    
    print("📊 总体指标:")
    print(f"   汉明准确率: {overall_metrics['hamming_accuracy']:.1%}")
    print(f"   完全匹配准确率: {overall_metrics['exact_match_accuracy']:.1%}")
    print(f"   F1分数: {overall_metrics['f1_score']:.1%}")
    print(f"   精确率: {overall_metrics['precision']:.1%}")
    print(f"   召回率: {overall_metrics['recall']:.1%}")
    
    print(f"\n📈 标签级指标:")
    print(f"   宏平均精确率: {overall_metrics['macro_precision']:.1%}")
    print(f"   宏平均召回率: {overall_metrics['macro_recall']:.1%}")
    print(f"   宏平均F1: {overall_metrics['macro_f1']:.1%}")
    
    print(f"\n📋 覆盖率统计:")
    print(f"   平均预测标签数: {overall_metrics['avg_predicted_labels']:.1f}")
    print(f"   平均真实标签数: {overall_metrics['avg_true_labels']:.1f}")
    print(f"   标签覆盖率: {overall_metrics['label_coverage']:.1%}")
    
    # 计算每个标签的指标
    if label_names is not None:
        per_label_metrics = calculate_per_label_metrics(outputs, targets, label_names, threshold)
        
        print(f"\n🏷️  每个标签的详细指标:")
        print(f"{'标签名':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<6}")
        print("-" * 50)
        
        for label_name, metrics in per_label_metrics.items():
            print(f"{label_name:<12} {metrics['precision']:<8.1%} {metrics['recall']:<8.1%} "
                  f"{metrics['f1_score']:<8.1%} {metrics['support']:<6}")


def get_best_threshold(outputs: torch.Tensor, targets: torch.Tensor, 
                      metric: str = 'f1_score', thresholds: np.ndarray = None) -> Tuple[float, float]:
    """寻找最佳阈值
    
    Args:
        outputs (torch.Tensor): 模型输出的logits
        targets (torch.Tensor): 真实标签
        metric (str): 优化的指标名称
        thresholds (np.ndarray): 要测试的阈值数组
        
    Returns:
        Tuple[float, float]: (最佳阈值, 最佳指标值)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        metrics = calculate_multilabel_metrics(outputs, targets, threshold)
        score = metrics.get(metric, 0.0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

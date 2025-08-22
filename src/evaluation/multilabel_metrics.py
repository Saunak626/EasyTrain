"""
多标签分类评估指标模块

提供详细的多标签分类性能评估，包括：
- 每个类别的精确率、召回率、F1分数
- 类别不平衡分析
- 实时训练监控
- 结果保存和可视化
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


class MultilabelMetricsCalculator:
    """多标签分类指标计算器
    
    负责计算和管理多标签分类的详细指标，包括每个类别的性能指标。
    """
    
    def __init__(self, class_names: List[str], output_dir: str = "runs/neonatal"):
        """初始化指标计算器
        
        Args:
            class_names: 类别名称列表
            output_dir: 输出目录
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化最佳指标追踪
        self.best_metrics = {
            'epoch': 0,
            'macro_avg_f1': 0.0,
            'class_metrics': {}
        }
        
        # 历史指标记录
        self.metrics_history = []
        
    def calculate_detailed_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                                 threshold: float = 0.5) -> Dict[str, Any]:
        """计算详细的多标签分类指标
        
        Args:
            predictions: 模型预测概率，形状为 (N, num_classes)
            targets: 真实标签，形状为 (N, num_classes)
            threshold: 二值化阈值
            
        Returns:
            包含详细指标的字典
        """
        # 二值化预测
        pred_binary = (predictions > threshold).astype(int)
        targets_binary = targets.astype(int)
        
        # 计算每个类别的指标
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            # 提取当前类别的预测和真实标签
            class_pred = pred_binary[:, i]
            class_true = targets_binary[:, i]
            
            # 计算基本指标
            precision = precision_score(class_true, class_pred, zero_division=0)
            recall = recall_score(class_true, class_pred, zero_division=0)
            f1 = f1_score(class_true, class_pred, zero_division=0)
            
            # 计算样本数量
            pos_samples = int(np.sum(class_true))
            neg_samples = int(len(class_true) - pos_samples)
            
            # 计算准确率（正确预测的比例）
            accuracy = np.mean(class_pred == class_true)
            
            class_metrics[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'pos_samples': pos_samples,
                'neg_samples': neg_samples,
                'total_samples': pos_samples + neg_samples
            }
        
        # 计算宏平均指标
        macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
        macro_accuracy = np.mean([m['accuracy'] for m in class_metrics.values()])
        
        # 计算微平均指标
        micro_precision = precision_score(targets_binary, pred_binary, average='micro', zero_division=0)
        micro_recall = recall_score(targets_binary, pred_binary, average='micro', zero_division=0)
        micro_f1 = f1_score(targets_binary, pred_binary, average='micro', zero_division=0)
        
        # 计算加权平均指标
        weighted_precision = precision_score(targets_binary, pred_binary, average='weighted', zero_division=0)
        weighted_recall = recall_score(targets_binary, pred_binary, average='weighted', zero_division=0)
        weighted_f1 = f1_score(targets_binary, pred_binary, average='weighted', zero_division=0)
        
        return {
            'class_metrics': class_metrics,
            'macro_avg': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1': float(macro_f1),
                'accuracy': float(macro_accuracy)
            },
            'micro_avg': {
                'precision': float(micro_precision),
                'recall': float(micro_recall),
                'f1': float(micro_f1)
            },
            'weighted_avg': {
                'precision': float(weighted_precision),
                'recall': float(weighted_recall),
                'f1': float(weighted_f1)
            },
            'threshold': threshold,
            'total_samples': len(targets)
        }
    
    def format_metrics_display(self, metrics: Dict[str, Any], epoch: int, 
                             val_loss: float, train_batches: int) -> str:
        """格式化指标显示
        
        Args:
            metrics: 详细指标字典
            epoch: 当前epoch
            val_loss: 验证损失
            train_batches: 训练批次数
            
        Returns:
            格式化的显示字符串
        """
        macro_acc = metrics['macro_avg']['accuracy'] * 100
        macro_f1 = metrics['macro_avg']['f1'] * 100
        
        # 主要指标行
        main_line = (f"Epoch {epoch:03d} | val_loss={val_loss:.4f} | "
                    f"val_acc={macro_acc:.2f}% (宏平均) | val_f1={macro_f1:.2f}% | "
                    f"train_batches={train_batches}")
        
        # 详细类别指标表格
        detail_lines = ["\n各类别详细指标:"]
        detail_lines.append("类别名称          精确率   召回率   F1分数   准确率   正样本   负样本")
        detail_lines.append("-" * 75)
        
        for class_name, class_metric in metrics['class_metrics'].items():
            line = (f"{class_name:<12} "
                   f"{class_metric['precision']:>7.3f}  "
                   f"{class_metric['recall']:>7.3f}  "
                   f"{class_metric['f1']:>7.3f}  "
                   f"{class_metric['accuracy']:>7.3f}  "
                   f"{class_metric['pos_samples']:>6d}  "
                   f"{class_metric['neg_samples']:>6d}")
            detail_lines.append(line)
        
        # 平均指标汇总
        detail_lines.append("-" * 75)
        detail_lines.append(f"宏平均           "
                           f"{metrics['macro_avg']['precision']:>7.3f}  "
                           f"{metrics['macro_avg']['recall']:>7.3f}  "
                           f"{metrics['macro_avg']['f1']:>7.3f}  "
                           f"{metrics['macro_avg']['accuracy']:>7.3f}  "
                           f"{'':>6s}  {'':>6s}")
        
        detail_lines.append(f"微平均           "
                           f"{metrics['micro_avg']['precision']:>7.3f}  "
                           f"{metrics['micro_avg']['recall']:>7.3f}  "
                           f"{metrics['micro_avg']['f1']:>7.3f}  "
                           f"{'':>7s}  {'':>6s}  {'':>6s}")
        
        return main_line + "\n" + "\n".join(detail_lines)
    
    def update_best_metrics(self, metrics: Dict[str, Any], epoch: int) -> bool:
        """更新最佳指标记录
        
        Args:
            metrics: 当前指标
            epoch: 当前epoch
            
        Returns:
            是否更新了最佳指标
        """
        current_f1 = metrics['macro_avg']['f1']
        
        if current_f1 > self.best_metrics['macro_avg_f1']:
            self.best_metrics = {
                'epoch': epoch,
                'macro_avg_f1': current_f1,
                'class_metrics': metrics['class_metrics'].copy(),
                'macro_avg': metrics['macro_avg'].copy(),
                'micro_avg': metrics['micro_avg'].copy(),
                'weighted_avg': metrics['weighted_avg'].copy()
            }
            return True
        return False
    
    def save_metrics(self, metrics: Dict[str, Any], epoch: int, 
                    val_loss: float, is_best: bool = False):
        """保存指标到文件
        
        Args:
            metrics: 指标字典
            epoch: 当前epoch
            val_loss: 验证损失
            is_best: 是否为最佳指标
        """
        # 添加到历史记录
        record = {
            'epoch': epoch,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best,
            **metrics
        }
        self.metrics_history.append(record)
        
        # 保存历史记录
        history_file = os.path.join(self.output_dir, 'metrics_history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, ensure_ascii=False, indent=2)
        
        # 保存最佳指标
        best_file = os.path.join(self.output_dir, 'best_metrics.json')
        with open(best_file, 'w', encoding='utf-8') as f:
            json.dump(self.best_metrics, f, ensure_ascii=False, indent=2)
        
        # 保存当前epoch的详细指标
        epoch_file = os.path.join(self.output_dir, f'epoch_{epoch:03d}_metrics.json')
        with open(epoch_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式的类别指标（便于分析）
        self._save_class_metrics_csv(metrics, epoch)
    
    def _save_class_metrics_csv(self, metrics: Dict[str, Any], epoch: int):
        """保存类别指标到CSV文件"""
        csv_file = os.path.join(self.output_dir, 'class_metrics_history.csv')
        
        # 准备数据
        rows = []
        for class_name, class_metric in metrics['class_metrics'].items():
            row = {
                'epoch': epoch,
                'class_name': class_name,
                'precision': class_metric['precision'],
                'recall': class_metric['recall'],
                'f1': class_metric['f1'],
                'accuracy': class_metric['accuracy'],
                'pos_samples': class_metric['pos_samples'],
                'neg_samples': class_metric['neg_samples']
            }
            rows.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(rows)
        
        # 追加到CSV文件
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8')
        else:
            df.to_csv(csv_file, index=False, encoding='utf-8')
    
    def get_summary_report(self) -> str:
        """获取训练总结报告"""
        if not self.best_metrics['class_metrics']:
            return "暂无最佳指标记录"
        
        lines = [
            f"\n🏆 训练总结报告 (最佳epoch: {self.best_metrics['epoch']})",
            "=" * 80,
            f"最佳宏平均F1分数: {self.best_metrics['macro_avg_f1']:.4f}",
            f"最佳宏平均准确率: {self.best_metrics['macro_avg']['accuracy']:.4f}",
            "",
            "各类别最佳指标:",
            "类别名称          精确率   召回率   F1分数   准确率   正样本   负样本",
            "-" * 75
        ]
        
        for class_name, class_metric in self.best_metrics['class_metrics'].items():
            line = (f"{class_name:<12} "
                   f"{class_metric['precision']:>7.3f}  "
                   f"{class_metric['recall']:>7.3f}  "
                   f"{class_metric['f1']:>7.3f}  "
                   f"{class_metric['accuracy']:>7.3f}  "
                   f"{class_metric['pos_samples']:>6d}  "
                   f"{class_metric['neg_samples']:>6d}")
            lines.append(line)
        
        lines.extend([
            "-" * 75,
            f"📊 指标文件保存位置: {self.output_dir}",
            f"📈 历史记录: {len(self.metrics_history)} 个epoch",
            "=" * 80
        ])
        
        return "\n".join(lines)

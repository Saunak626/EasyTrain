"""
多标签分类评估指标模块 (已使用 sklearn.metrics 优化)

提供详细的多标签分类性能评估，包括：
- 每个类别的精确率、召回率、F1分数
- 类别不平衡分析
- 实时训练监控
- 结果保存和可视化
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
# 优化：引入 classification_report 和 accuracy_score
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             classification_report, accuracy_score)


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
            'macro_avg_accuracy': 0.0,
            'micro_avg_f1': 0.0,
            'weighted_avg_f1': 0.0,
            'class_metrics': {},
            'macro_avg': {},
            'micro_avg': {},
            'weighted_avg': {}
        }

        # 为每个类别单独追踪最佳指标
        self.best_class_metrics = {}
        for class_name in class_names:
            self.best_class_metrics[class_name] = {
                'best_precision': {'value': 0.0, 'epoch': 0},
                'best_recall': {'value': 0.0, 'epoch': 0},
                'best_f1': {'value': 0.0, 'epoch': 0},
                'best_accuracy': {'value': 0.0, 'epoch': 0}
            }

        # 历史指标记录
        self.metrics_history = []
    
    def calculate_detailed_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                                     threshold: float = 0.5) -> Dict[str, Any]:
        """
        (已优化) 计算详细的多标签分类指标
        
        此版本使用 sklearn.metrics.classification_report 和向量化操作进行优化，
        替代了原有的 for 循环，提高了计算效率和代码简洁性，同时保持输出结构不变。
        
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
        
        # --- 优化核心 ---
        # 1. 使用 classification_report 一次性计算大多数指标
        report = classification_report(
            targets_binary, 
            pred_binary, 
            target_names=self.class_names, 
            zero_division=0,
            output_dict=True
        )
        
        # 2. 向量化计算每个类别的准确率 (classification_report 不提供此项)
        per_class_accuracy = (pred_binary == targets_binary).mean(axis=0)
        
        # 3. 按照原函数接口要求，重新组织 class_metrics 字典
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_report = report[class_name]
            support = int(class_report['support'])
            class_metrics[class_name] = {
                'precision': float(class_report['precision']),
                'recall': float(class_report['recall']),
                'f1': float(class_report['f1-score']), # 键名映射
                'accuracy': float(per_class_accuracy[i]),
                'pos_samples': support,
                'neg_samples': len(targets) - support,
                'total_samples': len(targets)
            }
            
        # 4. 提取或计算平均指标
        macro_avg_report = report['macro avg']
        macro_avg = {
            'precision': float(macro_avg_report['precision']),
            'recall': float(macro_avg_report['recall']),
            'f1': float(macro_avg_report['f1-score']),
            'accuracy': float(np.mean(per_class_accuracy))
        }
        
        # 🔧 修复：micro_accuracy使用全局accuracy，与micro的precision/recall一致
        # 原来使用accuracy_score计算的是subset accuracy（所有类别都正确才算正确）
        # 现在改为全局accuracy（所有预测中正确的比例）
        micro_accuracy = float((pred_binary == targets_binary).mean())

        # 可选：保留subset accuracy作为额外指标
        subset_accuracy = float(accuracy_score(targets_binary, pred_binary))

        micro_avg = {
            'precision': float(precision_score(targets_binary, pred_binary, average='micro', zero_division=0)),
            'recall': float(recall_score(targets_binary, pred_binary, average='micro', zero_division=0)),
            'f1': float(f1_score(targets_binary, pred_binary, average='micro', zero_division=0)),
            'accuracy': micro_accuracy,  # 全局准确率（与micro的precision/recall一致）
            'subset_accuracy': subset_accuracy  # 子集准确率（所有类别都正确）
        }
        
        weighted_avg_report = report['weighted avg']
        class_supports = np.array([m['pos_samples'] for m in class_metrics.values()])
        
        # 修正加权准确率的计算
        if np.sum(class_supports) > 0:
            weighted_accuracy = np.average(per_class_accuracy, weights=class_supports)
        else:
            weighted_accuracy = macro_avg['accuracy']

        weighted_avg = {
            'precision': float(weighted_avg_report['precision']),
            'recall': float(weighted_avg_report['recall']),
            'f1': float(weighted_avg_report['f1-score']),
            'accuracy': float(weighted_accuracy)
        }

        # 5. 组装成与原函数完全相同的返回结构
        return {
            'class_metrics': class_metrics,
            'macro_avg': macro_avg,
            'micro_avg': micro_avg,
            'weighted_avg': weighted_avg,
            'threshold': threshold,
            'total_samples': len(targets)
        }
        
    def format_metrics_display(self, metrics: Dict[str, Any], epoch: int,
                               val_loss: float, train_batches: int) -> str:
        """格式化指标显示（突出显示加权平均指标）

        (无需修改)
        """
        macro_acc = metrics['macro_avg']['accuracy'] * 100
        macro_f1 = metrics['macro_avg']['f1'] * 100
        
        micro_acc = metrics['micro_avg']['accuracy'] * 100
        weighted_acc = metrics['weighted_avg']['accuracy'] * 100

        main_line = (f"Epoch {epoch:03d} | val_loss={val_loss:.4f} | "
                     f"macro_acc={macro_acc:.2f}% | micro_acc={micro_acc:.2f}% | weighted_acc={weighted_acc:.2f}% | "
                     f"val_f1={macro_f1:.2f}% | train_batches={train_batches}")
        
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
        
        detail_lines.append("-" * 75)
        detail_lines.append(f"宏平均            "
                            f"{metrics['macro_avg']['precision']:>7.3f}  "
                            f"{metrics['macro_avg']['recall']:>7.3f}  "
                            f"{metrics['macro_avg']['f1']:>7.3f}  "
                            f"{metrics['macro_avg']['accuracy']:>7.3f}  "
                            f"{'':>6s}  {'':>6s}")
        
        detail_lines.append(f"微平均            "
                            f"{metrics['micro_avg']['precision']:>7.3f}  "
                            f"{metrics['micro_avg']['recall']:>7.3f}  "
                            f"{metrics['micro_avg']['f1']:>7.3f}  "
                            f"{metrics['micro_avg']['accuracy']:>7.3f}  "
                            f"{'':>6s}  {'':>6s}")

        detail_lines.append(f"🎯加权平均        "
                            f"{metrics['weighted_avg']['precision']:>7.3f}  "
                            f"{metrics['weighted_avg']['recall']:>7.3f}  "
                            f"{metrics['weighted_avg']['f1']:>7.3f}  "
                            f"{metrics['weighted_avg']['accuracy']:>7.3f}  "
                            f"{'':>6s}  {'':>6s}")

        detail_lines.append("")
        detail_lines.append("📊 指标说明:")
        detail_lines.append("  • 宏平均: 每个类别权重相等，对稀有类别敏感")
        detail_lines.append("  • 微平均: 按样本数量加权，对主要类别敏感")
        detail_lines.append("  • 🎯加权平均: 按类别样本数加权，适合不平衡数据（推荐）")

        return main_line + "\n" + "\n".join(detail_lines)
    
    def update_best_metrics(self, metrics: Dict[str, Any], epoch: int) -> bool:
        """更新最佳指标记录（包括每个类别的最佳指标）

        (无需修改)
        """
        current_f1 = metrics['macro_avg']['f1']
        is_best_overall = False

        if current_f1 > self.best_metrics['macro_avg_f1']:
            self.best_metrics = {
                'epoch': epoch,
                'macro_avg_f1': current_f1,
                'macro_avg_accuracy': metrics['macro_avg']['accuracy'],
                'micro_avg_f1': metrics['micro_avg']['f1'],
                'weighted_avg_f1': metrics['weighted_avg']['f1'],
                'class_metrics': metrics['class_metrics'].copy(),
                'macro_avg': metrics['macro_avg'].copy(),
                'micro_avg': metrics['micro_avg'].copy(),
                'weighted_avg': metrics['weighted_avg'].copy()
            }
            is_best_overall = True

        for class_name, class_metric in metrics['class_metrics'].items():
            if class_name in self.best_class_metrics:
                if class_metric['precision'] > self.best_class_metrics[class_name]['best_precision']['value']:
                    self.best_class_metrics[class_name]['best_precision'] = {'value': class_metric['precision'], 'epoch': epoch}
                if class_metric['recall'] > self.best_class_metrics[class_name]['best_recall']['value']:
                    self.best_class_metrics[class_name]['best_recall'] = {'value': class_metric['recall'], 'epoch': epoch}
                if class_metric['f1'] > self.best_class_metrics[class_name]['best_f1']['value']:
                    self.best_class_metrics[class_name]['best_f1'] = {'value': class_metric['f1'], 'epoch': epoch}
                if class_metric['accuracy'] > self.best_class_metrics[class_name]['best_accuracy']['value']:
                    self.best_class_metrics[class_name]['best_accuracy'] = {'value': class_metric['accuracy'], 'epoch': epoch}

        self.save_best_metrics_files()
        return is_best_overall

    def save_best_metrics_files(self):
        """保存最佳指标到文件

        (无需修改)
        """
        csv_data = []
        for class_name in self.class_names:
            if class_name in self.best_class_metrics:
                class_best = self.best_class_metrics[class_name]
                csv_data.append({
                    '类别名称': class_name,
                    '最佳精确率': f"{class_best['best_precision']['value']:.4f}",
                    '最佳精确率Epoch': class_best['best_precision']['epoch'],
                    '最佳召回率': f"{class_best['best_recall']['value']:.4f}",
                    '最佳召回率Epoch': class_best['best_recall']['epoch'],
                    '最佳F1分数': f"{class_best['best_f1']['value']:.4f}",
                    '最佳F1分数Epoch': class_best['best_f1']['epoch'],
                    '最佳准确率': f"{class_best['best_accuracy']['value']:.4f}",
                    '最佳准确率Epoch': class_best['best_accuracy']['epoch']
                })

        if 'macro_avg' in self.best_metrics and self.best_metrics['macro_avg']:
            csv_data.append({
                '类别名称': '🏆整体最佳',
                '最佳精确率': f"{self.best_metrics['macro_avg']['precision']:.4f}",
                '最佳精确率Epoch': self.best_metrics['epoch'],
                '最佳召回率': f"{self.best_metrics['macro_avg']['recall']:.4f}",
                '最佳召回率Epoch': self.best_metrics['epoch'],
                '最佳F1分数': f"{self.best_metrics['macro_avg_f1']:.4f}",
                '最佳F1分数Epoch': self.best_metrics['epoch'],
                '最佳准确率': f"{self.best_metrics['macro_avg_accuracy']:.4f}",
                '最佳准确率Epoch': self.best_metrics['epoch']
            })
        else:
            csv_data.append({
                '类别名称': '🏆整体最佳', '最佳精确率': '待更新', '最佳精确率Epoch': 0,
                '最佳召回率': '待更新', '最佳召回率Epoch': 0, '最佳F1分数': '待更新',
                '最佳F1分数Epoch': 0, '最佳准确率': '待更新', '最佳准确率Epoch': 0
            })

        df = pd.DataFrame(csv_data)
        best_metrics_csv = os.path.join(self.output_dir, "best_metrics_summary.csv")
        df.to_csv(best_metrics_csv, index=False, encoding='utf-8-sig')

    def save_metrics(self, metrics: Dict[str, Any], epoch: int,
                     val_loss: float, is_best: bool = False):
        """保存指标到文件

        (无需修改)
        """
        record = {
            'epoch': epoch,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best,
            **metrics
        }
        self.metrics_history.append(record)
        self._save_class_metrics_csv(metrics, epoch)
    
    def _save_class_metrics_csv(self, metrics: Dict[str, Any], epoch: int):
        """保存类别指标到CSV文件

        (无需修改)
        """
        csv_file = os.path.join(self.output_dir, 'class_metrics_history.csv')
        rows = []
        for class_name, class_metric in metrics['class_metrics'].items():
            row = {
                'epoch': epoch, 'class_name': class_name,
                'precision': class_metric['precision'], 'recall': class_metric['recall'],
                'f1': class_metric['f1'], 'accuracy': class_metric['accuracy'],
                'pos_samples': class_metric['pos_samples'], 'neg_samples': class_metric['neg_samples']
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8')
        else:
            df.to_csv(csv_file, index=False, encoding='utf-8')

    def save_train_metrics(self, metrics: Dict[str, Any], epoch: int, train_loss: float):
        """保存训练集指标到单独的CSV文件

        (无需修改)
        """
        csv_file = os.path.join(self.output_dir, 'train_metrics_history.csv')
        rows = []
        for class_name, class_metric in metrics['class_metrics'].items():
            row = {
                'epoch': epoch, 'class_name': class_name,
                'precision': class_metric['precision'], 'recall': class_metric['recall'],
                'f1': class_metric['f1'], 'accuracy': class_metric['accuracy'],
                'pos_samples': class_metric['pos_samples'], 'neg_samples': class_metric['neg_samples']
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8')
        else:
            df.to_csv(csv_file, index=False, encoding='utf-8')

    def save_test_metrics(self, metrics: Dict[str, Any], epoch: int, test_loss: float):
        """保存测试集指标到单独的CSV文件

        (无需修改)
        """
        csv_file = os.path.join(self.output_dir, 'test_metrics_history.csv')
        rows = []
        for class_name, class_metric in metrics['class_metrics'].items():
            row = {
                'epoch': epoch, 'class_name': class_name,
                'precision': class_metric['precision'], 'recall': class_metric['recall'],
                'f1': class_metric['f1'], 'accuracy': class_metric['accuracy'],
                'pos_samples': class_metric['pos_samples'], 'neg_samples': class_metric['neg_samples']
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8')
        else:
            df.to_csv(csv_file, index=False, encoding='utf-8')
    
    def get_summary_report(self) -> str:
        """获取训练总结报告
        
        (无需修改)
        """
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
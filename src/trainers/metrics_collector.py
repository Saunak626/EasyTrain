"""统一的指标收集器

本模块提供统一的指标收集接口，处理多GPU环境下的指标汇总。
"""
import torch
import numpy as np
from typing import Optional, Dict, Any
from accelerate import Accelerator


class MetricsCollector:
    """统一的指标收集器
    
    负责在训练和测试过程中收集预测和标签，并在epoch结束时汇总计算指标。
    自动处理多GPU环境下的数据同步。
    """
    
    def __init__(self, accelerator: Accelerator, metrics_calculator):
        """初始化指标收集器
        
        Args:
            accelerator: Accelerate实例，用于多GPU数据汇总
            metrics_calculator: 指标计算器实例
        """
        self.accelerator = accelerator
        self.metrics_calculator = metrics_calculator
        self.predictions = []
        self.targets = []
    
    def collect(self, predictions: torch.Tensor, targets: torch.Tensor):
        """收集单个batch的预测和标签
        
        Args:
            predictions: 预测结果tensor（已经过sigmoid或softmax）
            targets: 真实标签tensor
        """
        self.predictions.append(predictions.detach())
        self.targets.append(targets.detach())
    
    def compute_and_reset(self, epoch: int, phase: str, loss: float) -> Optional[Dict[str, Any]]:
        """计算指标并重置收集器
        
        在epoch结束时调用，汇总所有GPU的数据并计算指标。
        
        Args:
            epoch: epoch编号
            phase: 阶段标识，'train' 或 'test'
            loss: 平均损失
            
        Returns:
            指标字典（仅主进程返回，其他进程返回None）
        """
        if not self.predictions:
            return None
        
        # 合并本地数据
        local_pred = torch.cat(self.predictions, dim=0)
        local_target = torch.cat(self.targets, dim=0)
        
        # 跨GPU汇总
        global_pred = self.accelerator.gather_for_metrics(local_pred)
        global_target = self.accelerator.gather_for_metrics(local_target)
        
        # 重置收集器
        self.predictions = []
        self.targets = []
        
        # 只在主进程计算指标
        if not self.accelerator.is_main_process:
            return None
        
        # 转换为numpy并计算指标
        pred_array = global_pred.cpu().numpy()
        target_array = global_target.cpu().numpy()
        
        metrics = self.metrics_calculator.calculate_detailed_metrics(
            pred_array, target_array, threshold=0.5
        )
        
        # 保存指标
        if phase == 'train':
            self.metrics_calculator.save_train_metrics(metrics, epoch, loss)
        else:
            is_best = self.metrics_calculator.update_best_metrics(metrics, epoch)
            self.metrics_calculator.save_metrics(metrics, epoch, loss, is_best)
            self.metrics_calculator.save_test_metrics(metrics, epoch, loss)
        
        return metrics
    
    def reset(self):
        """重置收集器，清空已收集的数据"""
        self.predictions = []
        self.targets = []


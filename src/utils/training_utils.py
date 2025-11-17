"""训练工具函数

提供训练相关的工具函数，包括学习率信息获取、SwanLab 日志记录等。
"""

from typing import Dict, Any
from accelerate import Accelerator


def get_learning_rate_info(optimizer, lr_scheduler, scheduler_config, initial_lr):
    """获取学习率监控信息
    
    Args:
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        scheduler_config: 调度器配置
        initial_lr: 初始学习率
        
    Returns:
        dict: 包含学习率信息的字典
    """
    current_lr = optimizer.param_groups[0]['lr']
    scheduler_name = scheduler_config.get('name', 'default')
    
    return {
        'initial_lr': initial_lr,
        'current_lr': current_lr,
        'scheduler_name': scheduler_name
    }


def log_multilabel_metrics_to_swanlab(accelerator: Accelerator, metrics: Dict[str, Any],
                                      prefix: str, epoch: int):
    """记录多标签指标到SwanLab
    
    统一处理多标签分类指标的日志记录，避免重复代码。
    
    Args:
        accelerator: Accelerator实例
        metrics: 指标字典（包含 macro_avg, micro_avg, weighted_avg, class_metrics）
        prefix: 日志前缀（'train' 或 'test'）
        epoch: 当前epoch
    """
    # 记录平均指标
    accelerator.log({
        f"{prefix}/macro_accuracy": metrics['macro_avg']['accuracy'],
        f"{prefix}/micro_accuracy": metrics['micro_avg']['accuracy'],
        f"{prefix}/weighted_accuracy": metrics['weighted_avg']['accuracy'],
        f"{prefix}/macro_f1": metrics['macro_avg']['f1'],
        f"{prefix}/micro_f1": metrics['micro_avg']['f1'],
        f"{prefix}/weighted_f1": metrics['weighted_avg']['f1'],
        f"{prefix}/macro_precision": metrics['macro_avg']['precision'],
        f"{prefix}/macro_recall": metrics['macro_avg']['recall']
    }, step=epoch)
    
    # 记录每个类别的指标
    for class_name, class_metrics in metrics['class_metrics'].items():
        accelerator.log({
            f"{prefix}_class/{class_name}/f1": class_metrics['f1'],
            f"{prefix}_class/{class_name}/precision": class_metrics['precision'],
            f"{prefix}_class/{class_name}/recall": class_metrics['recall'],
            f"{prefix}_class/{class_name}/accuracy": class_metrics['accuracy']
        }, step=epoch)


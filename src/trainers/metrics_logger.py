"""统一的指标日志记录器

本模块提供统一的指标日志记录接口，消除训练和测试阶段的重复日志代码。
"""
from typing import Dict, Any
from accelerate import Accelerator


class MetricsLogger:
    """统一的指标日志记录器
    
    负责将训练和测试指标记录到实验追踪系统（如SwanLab）。
    支持多标签分类的平均指标和类别指标记录。
    """
    
    def __init__(self, accelerator: Accelerator):
        """初始化指标日志记录器
        
        Args:
            accelerator: Accelerate实例，用于日志记录
        """
        self.accelerator = accelerator
    
    def log_multilabel_metrics(self, metrics: Dict[str, Any], phase: str, epoch: int):
        """记录多标签分类指标
        
        Args:
            metrics: 指标字典，包含 macro_avg, micro_avg, weighted_avg, class_metrics
            phase: 阶段标识，'train' 或 'test'
            epoch: epoch编号
        """
        # 记录平均指标
        avg_metrics = {
            f"{phase}/macro_accuracy": metrics['macro_avg']['accuracy'],
            f"{phase}/micro_accuracy": metrics['micro_avg']['accuracy'],
            f"{phase}/weighted_accuracy": metrics['weighted_avg']['accuracy'],
            f"{phase}/macro_f1": metrics['macro_avg']['f1'],
            f"{phase}/micro_f1": metrics['micro_avg']['f1'],
            f"{phase}/weighted_f1": metrics['weighted_avg']['f1'],
            f"{phase}/macro_precision": metrics['macro_avg']['precision'],
            f"{phase}/macro_recall": metrics['macro_avg']['recall']
        }
        self.accelerator.log(avg_metrics, step=epoch)
        
        # 记录类别指标
        for class_name, class_metrics in metrics['class_metrics'].items():
            class_log = {
                f"{phase}_class/{class_name}/f1": class_metrics['f1'],
                f"{phase}_class/{class_name}/precision": class_metrics['precision'],
                f"{phase}_class/{class_name}/recall": class_metrics['recall'],
                f"{phase}_class/{class_name}/accuracy": class_metrics['accuracy']
            }
            self.accelerator.log(class_log, step=epoch)
    
    def log_test_loss(self, loss: float, epoch: int):
        """记录测试损失
        
        Args:
            loss: 测试损失值
            epoch: epoch编号
        """
        self.accelerator.log({"test/loss": loss}, step=epoch)
    
    def log_simple_metrics(self, loss: float, accuracy: float, phase: str, epoch: int):
        """记录简单指标（单标签分类）
        
        Args:
            loss: 损失值
            accuracy: 准确率
            phase: 阶段标识，'train' 或 'test'
            epoch: epoch编号
        """
        self.accelerator.log({
            f"{phase}/loss": loss,
            f"{phase}/accuracy": accuracy
        }, step=epoch)


"""
评估模块

提供多种评估指标和工具：
- 多标签分类详细指标
- 训练过程监控
- 结果可视化和保存
"""

from .multilabel_metrics import MultilabelMetricsCalculator

__all__ = ['MultilabelMetricsCalculator']

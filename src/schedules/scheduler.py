"""
学习率调度器模块
包含常用学习率调度策略的定义和获取函数
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, scheduler_name, **kwargs):
    """
    学习率调度器工厂函数，创建并配置学习率调度器实例
    
    Args:
        optimizer (torch.optim.Optimizer): 优化器实例
        scheduler_name (str): 调度器名称，支持'onecycle', 'step', 'cosine', 'plateau'
        **kwargs: 调度器参数，如max_lr, epochs, step_size等
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: 配置好的学习率调度器实例
        
    Raises:
        ValueError: 当指定的调度器名称不支持时
        
    示例：
        >>> scheduler = get_scheduler(optimizer, 'onecycle', max_lr=0.1, epochs=100)
        >>> scheduler = get_scheduler(optimizer, 'step', step_size=30, gamma=0.1)
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "onecycle":
        return lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=kwargs.get('max_lr', 0.1),
            epochs=kwargs.get('epochs', 100),
            steps_per_epoch=kwargs.get('steps_per_epoch', 100),
            pct_start=kwargs.get('pct_start', 0.3),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos'),
            div_factor=kwargs.get('div_factor', 25.0),
            final_div_factor=kwargs.get('final_div_factor', 1e4)
        )
    elif scheduler_name == "step":
        return lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_name == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            threshold=kwargs.get('threshold', 1e-4),
            min_lr=kwargs.get('min_lr', 0)
        )
    else:
        raise ValueError(f"不支持的调度器: {scheduler_name}。支持的调度器: onecycle, step, cosine, plateau")

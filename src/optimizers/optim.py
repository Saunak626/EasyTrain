"""
优化器模块
包含常用优化器的定义和获取函数
"""

import torch
import torch.optim as optim


def get_optimizer(model, optimizer_name, learning_rate, **kwargs):
    """
    优化器工厂函数，创建并配置优化器实例
    
    Args:
        model (torch.nn.Module): 神经网络模型，用于获取可训练参数
        optimizer_name (str): 优化器名称，支持'adam', 'adamw', 'sgd'
        learning_rate (float): 学习率
        **kwargs: 其他优化器参数，如weight_decay, momentum, betas等
        
    Returns:
        torch.optim.Optimizer: 配置好的优化器实例
        
    Raises:
        ValueError: 当指定的优化器名称不支持时
    
    示例：
        >>> optimizer = get_optimizer(model, 'adam', 0.001, weight_decay=0.01)
        >>> optimizer = get_optimizer(model, 'sgd', 0.1, momentum=0.9)
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=kwargs.get('weight_decay', 0),
            betas=kwargs.get('betas', (0.9, 0.999))
        )
    elif optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=kwargs.get('weight_decay', 0.01),
            betas=kwargs.get('betas', (0.9, 0.999))
        )
    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0),
            nesterov=kwargs.get('nesterov', False)
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}。支持的优化器: adam, adamw, sgd")

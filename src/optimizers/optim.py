"""
优化器模块
包含常用优化器的定义和获取函数
"""

import torch
import torch.optim as optim


def get_optimizer(model, optimizer_name, learning_rate, **kwargs):
    """
    获取优化器
    
    Args:
        model: 模型
        optimizer_name: 优化器名称
        learning_rate: 学习率
        **kwargs: 其他优化器参数
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

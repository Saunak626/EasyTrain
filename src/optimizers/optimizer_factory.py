"""
优化器工厂模块
包含常用优化器的定义和工厂函数
"""

import torch
import torch.optim as optim


def get_optimizer(model, optimizer_config=None, learning_rate=None, optimizer_name=None, **kwargs):
    """
    优化器工厂函数，创建并配置优化器实例

    Args:
        model (torch.nn.Module): 神经网络模型
        optimizer_config (dict, optional): 优化器配置字典
        learning_rate (float, optional): 学习率
        optimizer_name (str, optional): 优化器名称，用于向后兼容
        **kwargs: 其他优化器参数，用于向后兼容

    Returns:
        torch.optim.Optimizer: 配置好的优化器实例

    示例：
        >>> optimizer = get_optimizer(model, {'type': 'adam', 'weight_decay': 0.01}, 0.001)
        >>> optimizer = get_optimizer(model, optimizer_name='adam', learning_rate=0.001, weight_decay=0.01)
    """
    # 简化的配置解析
    if optimizer_config:
        optimizer_name = optimizer_config.get('type') or optimizer_config.get('name', 'adam')
        params = optimizer_config.get('params', {}) if 'params' in optimizer_config else {k: v for k, v in optimizer_config.items() if k not in ['type', 'name']}

        # 处理学习率
        if 'lr' in params and learning_rate is None:
            learning_rate = params.pop('lr')
    else:
        optimizer_name = optimizer_name or 'adam'
        params = kwargs

    if learning_rate is None:
        raise ValueError("必须提供学习率 (learning_rate)")

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=params.get('weight_decay', 0),
            betas=params.get('betas', (0.9, 0.999)),
            eps=params.get('eps', 1e-8),
            amsgrad=params.get('amsgrad', False)
        )
    elif optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=params.get('weight_decay', 0.01),
            betas=params.get('betas', (0.9, 0.999)),
            eps=params.get('eps', 1e-8),
            amsgrad=params.get('amsgrad', False)
        )
    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=params.get('momentum', 0.9),
            weight_decay=params.get('weight_decay', 0),
            nesterov=params.get('nesterov', False),
            dampening=params.get('dampening', 0)
        )
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            alpha=params.get('alpha', 0.99),
            eps=params.get('eps', 1e-8),
            weight_decay=params.get('weight_decay', 0),
            momentum=params.get('momentum', 0),
            centered=params.get('centered', False)
        )
    elif optimizer_name == "adagrad":
        return optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            lr_decay=params.get('lr_decay', 0),
            weight_decay=params.get('weight_decay', 0),
            eps=params.get('eps', 1e-10)
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}。支持的优化器: adam, adamw, sgd, rmsprop, adagrad")

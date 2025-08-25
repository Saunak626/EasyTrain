"""
学习率调度器工厂模块
包含常用学习率调度策略的定义和工厂函数
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, scheduler_config=None, hyperparams=None, scheduler_name=None, **kwargs):
    """
    学习率调度器工厂函数，创建并配置学习率调度器实例

    Args:
        optimizer (torch.optim.Optimizer): 优化器实例
        scheduler_config (dict, optional): 调度器配置字典
        hyperparams (dict, optional): 超参数字典，包含epochs, learning_rate等信息
        scheduler_name (str, optional): 调度器名称，用于向后兼容
        **kwargs: 调度器参数，用于向后兼容

    Returns:
        torch.optim.lr_scheduler._LRScheduler: 配置好的学习率调度器实例

    示例：
        >>> scheduler = get_scheduler(optimizer, {'type': 'onecycle', 'max_lr': 0.1}, hyperparams)
        >>> scheduler = get_scheduler(optimizer, scheduler_name='cosine', T_max=100)
    """
    # 简化的配置解析
    if scheduler_config:
        scheduler_name = scheduler_config.get('type') or scheduler_config.get('name', 'onecycle')
        params = scheduler_config.get('params', {}) if 'params' in scheduler_config else {k: v for k, v in scheduler_config.items() if k not in ['type', 'name']}
    else:
        scheduler_name = scheduler_name or 'onecycle'
        params = kwargs

    scheduler_name = scheduler_name.lower()

    # 获取超参数
    epochs = hyperparams.get('epochs', 100) if hyperparams else 100
    learning_rate = hyperparams.get('learning_rate', 0.001) if hyperparams else 0.001

    if scheduler_name == "onecycle":
        return lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=params.get('max_lr', 5 * learning_rate),
            epochs=params.get('epochs', epochs),
            steps_per_epoch=params.get('steps_per_epoch', 100),
            pct_start=params.get('pct_start', 0.3),
            anneal_strategy=params.get('anneal_strategy', 'cos'),
            div_factor=params.get('div_factor', 25.0),
            final_div_factor=params.get('final_div_factor', 1e4),
            cycle_momentum=params.get('cycle_momentum', True),
            base_momentum=params.get('base_momentum', 0.85),
            max_momentum=params.get('max_momentum', 0.95)
        )
    elif scheduler_name == "step":
        return lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=params.get('step_size', 30),
            gamma=params.get('gamma', 0.1),
            last_epoch=params.get('last_epoch', -1)
        )
    elif scheduler_name == "cosine":
        # 支持eta_min_factor参数，计算最小学习率
        eta_min_factor = params.get('eta_min_factor', 0.0)
        eta_min = params.get('eta_min', learning_rate * eta_min_factor)

        # 🔧 修复：确保T_max不为0
        T_max = params.get('T_max', epochs)
        if T_max <= 0:
            print(f"⚠️ 警告：T_max={T_max} 无效，cosine调度器退化为常数学习率")
            return lr_scheduler.ConstantLR(
                optimizer=optimizer,
                factor=1.0,  # 保持原始学习率
                total_iters=max(1, epochs),
                last_epoch=params.get('last_epoch', -1)
            )

        return lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=params.get('last_epoch', -1)
        )
    elif scheduler_name == "exponential":
        return lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=params.get('gamma', 0.95),
            last_epoch=params.get('last_epoch', -1)
        )
    elif scheduler_name == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=params.get('mode', 'min'),
            factor=params.get('factor', 0.1),
            patience=params.get('patience', 10),
            threshold=params.get('threshold', 1e-4),
            min_lr=params.get('min_lr', 0),
            cooldown=params.get('cooldown', 0),
            eps=params.get('eps', 1e-8)
        )
    elif scheduler_name == "linear":
        # Linear decay scheduler (LinearLR)
        return lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=params.get('start_factor', 1.0),
            end_factor=params.get('end_factor', 0.1),
            total_iters=params.get('total_iters', epochs),
            last_epoch=params.get('last_epoch', -1)
        )
    elif scheduler_name == "multistep":
        return lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=params.get('milestones', [30, 60, 90]),
            gamma=params.get('gamma', 0.1),
            last_epoch=params.get('last_epoch', -1)
        )
    elif scheduler_name == "warmup_cosine":
        # Warmup + Cosine Annealing scheduler
        warmup_epochs = params.get('warmup_epochs', max(1, epochs // 10))  # 默认10%的epoch用于warmup

        # 🔧 修复：当总epoch数过少时的处理逻辑
        if epochs <= 1:
            # 当只有1个epoch时，直接使用常数学习率调度器
            print(f"⚠️ 警告：epochs={epochs} 过少，warmup_cosine调度器退化为常数学习率")
            return lr_scheduler.ConstantLR(
                optimizer=optimizer,
                factor=1.0,  # 保持原始学习率
                total_iters=epochs,
                last_epoch=-1
            )

        # 确保warmup_epochs不会超过总epochs
        warmup_epochs = min(warmup_epochs, epochs - 1)
        cosine_epochs = epochs - warmup_epochs

        # 如果cosine阶段的epoch数为0，只使用warmup
        if cosine_epochs <= 0:
            print(f"⚠️ 警告：cosine阶段epoch数为{cosine_epochs}，只使用warmup调度器")
            return lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=params.get('warmup_start_factor', 0.1),
                end_factor=1.0,
                total_iters=epochs,
                last_epoch=-1
            )

        # 创建组合调度器：先warmup，再cosine annealing
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=params.get('warmup_start_factor', 0.1),  # 从10%学习率开始
            end_factor=1.0,  # 到达完整学习率
            total_iters=warmup_epochs,
            last_epoch=-1
        )

        # 支持eta_min_factor参数，计算最小学习率
        eta_min_factor = params.get('eta_min_factor', 0.01)
        eta_min = params.get('eta_min', learning_rate * eta_min_factor)

        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cosine_epochs,  # 使用修复后的cosine_epochs
            eta_min=eta_min,  # 最小学习率
            last_epoch=-1
        )

        # 使用SequentialLR组合两个调度器
        return lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
            last_epoch=-1
        )
    else:
        raise ValueError(f"不支持的调度器: {scheduler_name}。支持的调度器: onecycle, step, cosine, exponential, plateau, linear, multistep, warmup_cosine")

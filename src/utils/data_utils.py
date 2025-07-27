"""
数据工具函数
包含随机种子设置等数据相关工具函数
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    设置所有随机种子以确保可重复性

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

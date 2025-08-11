"""GPU显存管理工具模块

提供GPU显存清理和统计重置功能，用于训练过程中的显存管理。
"""

import torch


def cleanup_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_gpu_memory_stats():
    """重置GPU内存统计"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
            except RuntimeError:
                # 忽略无效设备错误
                pass

"""训练相关的共享工具函数

本模块提供训练过程中常用的工具函数，包括：
- GPU配置管理
- 任务输出目录管理
"""
import os
import yaml
from typing import Tuple


def get_gpu_config_from_file(config_path: str) -> Tuple[str, int]:
    """从配置文件读取GPU配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Tuple[GPU IDs字符串, GPU数量]
        
    Example:
        >>> gpu_ids, num_gpus = get_gpu_config_from_file('config.yaml')
        >>> print(f"GPU IDs: {gpu_ids}, Count: {num_gpus}")
        GPU IDs: 2,3, Count: 2
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    gpu_ids = config.get('gpu', {}).get('device_ids', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    gpu_ids_str = str(gpu_ids)
    
    # 解析GPU数量
    visible_ids = [gid.strip() for gid in gpu_ids_str.split(',') if gid.strip()]
    num_gpus = max(1, len(visible_ids))
    
    return ','.join(visible_ids), num_gpus


def setup_gpu_environment(gpu_ids: str) -> dict:
    """设置GPU环境变量
    
    清理分布式训练相关的环境变量，并设置CUDA_VISIBLE_DEVICES。
    
    Args:
        gpu_ids: GPU IDs字符串(如 "0,1,2")
        
    Returns:
        更新后的环境变量字典
        
    Example:
        >>> env = setup_gpu_environment("2,3")
        >>> print(env['CUDA_VISIBLE_DEVICES'])
        2,3
    """
    env = os.environ.copy()
    
    # 清理分布式训练相关的环境变量，避免冲突
    for k in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(k, None)
    
    # 设置GPU可见性
    env['CUDA_VISIBLE_DEVICES'] = gpu_ids
    
    return env


def get_task_output_dir(task_tag: str, dataset_type: str, base_dir: str = "runs") -> str:
    """根据任务类型获取输出目录
    
    根据任务标签和数据集类型自动确定输出目录路径。
    
    Args:
        task_tag: 任务标签
        dataset_type: 数据集类型
        base_dir: 基础输出目录，默认为 "runs"
        
    Returns:
        任务对应的输出目录路径
        
    Example:
        >>> output_dir = get_task_output_dir("multilabel", "neonatal_multilabel")
        >>> print(output_dir)
        runs/neonatal_multilabel
    """
    # 根据任务类型确定子目录名
    if 'multilabel' in task_tag.lower() or 'multilabel' in dataset_type.lower():
        if 'neonatal' in dataset_type.lower():
            task_subdir = "neonatal_multilabel"
        else:
            task_subdir = "multilabel_classification"
    elif 'video' in task_tag.lower():
        task_subdir = "video_classification"
    elif 'image' in task_tag.lower():
        task_subdir = "image_classification"
    elif 'text' in task_tag.lower():
        task_subdir = "text_classification"
    else:
        # 默认使用数据集类型作为子目录名
        task_subdir = dataset_type.replace('_', '_').lower() or "general"
    
    output_dir = os.path.join(base_dir, task_subdir)
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


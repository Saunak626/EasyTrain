"""数据集工具函数

提供数据集相关的工具函数，包括 Subset 解包、元数据获取等。
"""

from typing import Dict, Any


def unwrap_subset_dataset(dataset):
    """解包 Subset 包装的数据集，返回原始数据集
    
    当使用 data_percentage 参数时，数据集会被 torch.utils.data.Subset 包装。
    此函数用于获取原始数据集实例，以便访问数据集的自定义方法（如 get_class_names）。
    
    Args:
        dataset: 可能被 Subset 包装的数据集
        
    Returns:
        原始数据集（如果是 Subset 则返回内部数据集，否则返回原数据集）
    """
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        return dataset.dataset
    return dataset


def get_dataset_metadata(dataset, dataset_type: str) -> Dict[str, Any]:
    """从数据集获取元数据（类别数量、类别名称等）
    
    统一的数据集元数据获取接口，支持 Subset 包装的数据集。
    
    Args:
        dataset: 数据集实例（可能被 Subset 包装）
        dataset_type: 数据集类型字符串（如 'neonatal_multilabel'）
        
    Returns:
        包含以下键的字典：
        - 'num_classes': 类别数量（int 或 None）
        - 'classes': 类别名称列表（list 或 None）
        - 'is_multilabel': 是否为多标签任务（bool）
    """
    # 解包 Subset
    actual_dataset = unwrap_subset_dataset(dataset)
    
    metadata = {
        'num_classes': None,
        'classes': None,
        'is_multilabel': False
    }
    
    # 检测是否为多标签任务
    metadata['is_multilabel'] = 'multilabel' in dataset_type.lower()
    
    # 获取类别数量
    if hasattr(actual_dataset, 'get_num_classes'):
        metadata['num_classes'] = actual_dataset.get_num_classes()
    
    # 获取类别名称
    if hasattr(actual_dataset, 'get_class_names'):
        metadata['classes'] = actual_dataset.get_class_names()
        # 如果有类别名称，优先使用其长度作为类别数量
        if metadata['classes']:
            metadata['num_classes'] = len(metadata['classes'])
    
    return metadata


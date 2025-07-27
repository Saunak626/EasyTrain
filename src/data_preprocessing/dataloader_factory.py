"""数据加载器工厂模块

提供统一的数据加载器创建接口，支持多种数据集类型。
通过配置文件参数自动选择和配置相应的数据加载器。
"""

from .cifar10_dataset import get_cifar10_dataloaders
from .dataset import get_custom_dataloaders


def create_dataloaders(data_config, batch_size, num_workers=4):
    """
    数据加载器工厂函数
    
    根据数据配置自动创建相应的数据加载器，实现不同数据集的统一接口。
    
    Args:
        data_config (dict): 数据配置字典，包含数据集类型和相关参数
        batch_size (int): 批次大小
        num_workers (int): 数据加载进程数
        
    Returns:
        tuple: (train_dataloader, test_dataloader)
        
    Raises:
        ValueError: 当数据集类型不支持时抛出异常
        
    支持的数据集类型:
        - cifar10: CIFAR-10数据集
        - custom: 自定义数据集
    """
    dataset_type = data_config.get('type', 'cifar10')
    
    # 通用参数
    common_params = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'augment': data_config.get('augment', True)
    }
    
    if dataset_type == 'cifar10':
        # CIFAR-10数据集参数
        cifar10_params = {
            'root': data_config.get('root', './data'),
            'download': data_config.get('download', True)
        }
        return get_cifar10_dataloaders(**common_params, **cifar10_params)
        
    elif dataset_type == 'custom':
        # 自定义数据集参数
        custom_params = {
            'data_dir': data_config.get('data_dir', './data/custom'),
            'csv_file': data_config.get('csv_file', None),
            'image_size': data_config.get('image_size', 224),
            'train_split': data_config.get('train_split', 0.8)
        }
        return get_custom_dataloaders(**common_params, **custom_params)
        
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")


def get_dataset_info(data_config):
    """
    获取数据集信息
    
    Args:
        data_config (dict): 数据配置字典
        
    Returns:
        dict: 包含数据集基本信息的字典
    """
    dataset_type = data_config.get('type', 'cifar10')
    
    if dataset_type == 'cifar10':
        return {
            'num_classes': 10,
            'input_size': 32,
            'channels': 3,
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        }
    elif dataset_type == 'custom':
        return {
            'num_classes': data_config.get('num_classes', 2),
            'input_size': data_config.get('image_size', 224),
            'channels': 3,
            'class_names': data_config.get('class_names', None)
        }
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
"""配置工具函数

提供统一的配置解析和参数提取功能，支持简化的配置结构和向后兼容性。
"""


def extract_component_config(config, component_type, default_type=None):
    """统一的组件配置提取函数
    
    支持两种配置格式：
    1. 简化格式: {type: "component_name", param1: value1, param2: value2}
    2. 传统格式: {name: "component_name", params: {param1: value1, param2: value2}}
    
    Args:
        config (dict): 完整配置字典
        component_type (str): 组件类型 ('loss', 'optimizer', 'scheduler', 'model', 'data')
        default_type (str, optional): 默认组件类型
        
    Returns:
        tuple: (component_name, params_dict)
        
    Examples:
        # 简化格式
        config = {'loss': {'type': 'crossentropy', 'label_smoothing': 0.1}}
        name, params = extract_component_config(config, 'loss')
        # 返回: ('crossentropy', {'label_smoothing': 0.1})
        
        # 传统格式
        config = {'loss': {'name': 'crossentropy', 'params': {'label_smoothing': 0.1}}}
        name, params = extract_component_config(config, 'loss')
        # 返回: ('crossentropy', {'label_smoothing': 0.1})
    """
    component_config = config.get(component_type, {})
    
    if not component_config:
        if default_type:
            return default_type, {}
        else:
            raise ValueError(f"配置中缺少 {component_type} 部分")
    
    # 检查是否使用简化格式 (type字段直接指定)
    if 'type' in component_config:
        # 简化格式: {type: "component_name", param1: value1, ...}
        component_config_copy = component_config.copy()
        component_name = component_config_copy.pop('type')
        params = component_config_copy  # 其余都是参数
        
    elif 'name' in component_config:
        # 传统格式: {name: "component_name", params: {param1: value1, ...}}
        component_name = component_config.get('name')
        params = component_config.get('params', {})
        
    else:
        # 兼容旧格式，直接使用默认值
        if default_type:
            component_name = default_type
            params = component_config
        else:
            raise ValueError(f"{component_type} 配置中必须包含 'type' 或 'name' 字段")
    
    return component_name, params


def validate_component_config(component_name, params, component_type, supported_components=None):
    """验证组件配置的有效性

    Args:
        component_name (str): 组件名称
        params (dict): 组件参数
        component_type (str): 组件类型
        supported_components (list, optional): 支持的组件列表

    Raises:
        ValueError: 当组件不支持时
    """
    if supported_components and component_name not in supported_components:
        raise ValueError(
            f"不支持的{component_type}: {component_name}。"
            f"支持的{component_type}: {supported_components}"
        )

    # 使用组件注册表进行参数验证
    try:
        from src.components import COMPONENT_REGISTRY
        # 组件类型映射到注册表中的键名
        registry_type_map = {
            'loss': 'losses',
            'optimizer': 'optimizers',
            'scheduler': 'schedulers',
            'model': 'models',
            'data': 'datasets'
        }
        registry_type = registry_type_map.get(component_type, f"{component_type}s")
        COMPONENT_REGISTRY.validate_component_params(
            registry_type, component_name, params
        )
    except ImportError:
        # 如果组件注册表不可用，跳过验证
        pass


def merge_config_with_defaults(config, defaults):
    """将配置与默认值合并
    
    Args:
        config (dict): 用户配置
        defaults (dict): 默认配置
        
    Returns:
        dict: 合并后的配置
    """
    merged = defaults.copy()
    
    for key, value in config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # 递归合并嵌套字典
            merged[key] = merge_config_with_defaults(value, merged[key])
        else:
            # 直接覆盖
            merged[key] = value
    
    return merged


def get_nested_config(config, path, default=None):
    """获取嵌套配置值
    
    Args:
        config (dict): 配置字典
        path (str): 配置路径，用点号分隔，如 'data.params.clip_len'
        default: 默认值
        
    Returns:
        配置值或默认值
        
    Examples:
        config = {'data': {'params': {'clip_len': 16}}}
        value = get_nested_config(config, 'data.params.clip_len', 8)
        # 返回: 16
    """
    keys = path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_nested_config(config, path, value):
    """设置嵌套配置值
    
    Args:
        config (dict): 配置字典
        path (str): 配置路径，用点号分隔
        value: 要设置的值
        
    Examples:
        config = {}
        set_nested_config(config, 'data.params.clip_len', 16)
        # 结果: {'data': {'params': {'clip_len': 16}}}
    """
    keys = path.split('.')
    current = config
    
    # 创建嵌套结构
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # 设置最终值
    current[keys[-1]] = value


# 常用的默认配置
DEFAULT_CONFIGS = {
    'loss': {
        'type': 'crossentropy',
        'reduction': 'mean'
    },
    'optimizer': {
        'type': 'adam',
        'weight_decay': 0.0001
    },
    'scheduler': {
        'type': 'cosine',
        'T_max': 50
    },
    'model': {
        'type': 'resnet18',
        'pretrained': True
    },
    'data': {
        'type': 'cifar10',
        'num_workers': 8,
        'pin_memory': True
    }
}


def get_default_config(component_type):
    """获取组件的默认配置
    
    Args:
        component_type (str): 组件类型
        
    Returns:
        dict: 默认配置
    """
    return DEFAULT_CONFIGS.get(component_type, {}).copy()


def normalize_config_structure(config):
    """标准化配置结构，将传统格式转换为简化格式

    Args:
        config (dict): 原始配置

    Returns:
        dict: 标准化后的配置
    """
    normalized = config.copy()

    # 需要标准化的组件类型
    component_types = ['loss', 'optimizer', 'scheduler', 'model', 'data']

    for comp_type in component_types:
        if comp_type in normalized:
            comp_config = normalized[comp_type]

            # 如果使用传统格式，转换为简化格式
            if 'name' in comp_config and 'params' in comp_config:
                new_config = {'type': comp_config['name']}
                new_config.update(comp_config['params'])
                normalized[comp_type] = new_config

    return normalized


def validate_config_file(config):
    """验证配置文件的完整性和有效性

    Args:
        config (dict): 配置字典

    Raises:
        ValueError: 当配置无效时
    """
    # 必需的顶级配置节
    required_sections = ['task', 'training', 'swanlab', 'data', 'hp']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必需的部分: {section}")

    # 验证task配置
    if 'tag' not in config['task']:
        raise ValueError("配置文件中必须指定 task.tag")

    # 验证training配置
    if 'exp_name' not in config['training']:
        raise ValueError("配置文件中必须指定 training.exp_name")

    # 验证hp配置
    required_hp = ['batch_size', 'learning_rate', 'epochs']
    for hp in required_hp:
        if hp not in config['hp']:
            raise ValueError(f"配置文件中缺少必需的超参数: hp.{hp}")

    # 验证各组件配置
    component_types = ['loss', 'optimizer', 'scheduler', 'model', 'data']
    for comp_type in component_types:
        if comp_type in config:
            try:
                comp_name, comp_params = extract_component_config(config, comp_type)
                validate_component_config(comp_name, comp_params, comp_type)
            except Exception as e:
                raise ValueError(f"配置文件中 {comp_type} 部分无效: {e}")

    return True


def get_config_template():
    """获取标准配置文件模板

    Returns:
        dict: 配置文件模板
    """
    return {
        'task': {
            'tag': 'image_classification',  # 或 'video_classification'
            'description': '任务描述'
        },
        'training': {
            'exp_name': 'my_experiment',
            'save_model': True,
            'model_save_path': 'models/my_model.pth'
        },
        'swanlab': {
            'project_name': 'MyProject',
            'description': '实验描述'
        },
        'data': {
            'type': 'cifar10',
            'root': './data',
            'num_workers': 8
        },
        'model': {
            'type': 'resnet18',
            'pretrained': True
        },
        'hp': {
            'batch_size': 128,
            'learning_rate': 0.001,
            'epochs': 10,
            'dropout': 0.1
        },
        'optimizer': {
            'type': 'adam',
            'weight_decay': 0.0001
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 10
        },
        'loss': {
            'type': 'crossentropy'
        },
        'gpu': {
            'device_ids': '0',
            'multi_gpu': False
        }
    }

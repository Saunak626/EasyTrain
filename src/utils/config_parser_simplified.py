"""简化的配置解析器

基于原有config_parser.py的简化版本，移除过度工程化的功能，
保留核心功能，提高可维护性和可读性。

主要简化：
1. 移除复杂的嵌套参数处理
2. 简化参数映射逻辑
3. 减少冗余的兼容性代码
4. 优化注释和文档
"""

import argparse
import yaml
import os


def create_base_parser(description):
    """创建基础参数解析器
    
    Args:
        description (str): 解析器描述
        
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(description=description)
    
    # 基础参数
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--exp_name", type=str, help="实验名称")
    parser.add_argument("--model_name", type=str, help="模型名称")
    
    # 超参数
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--batch_size", type=int, help="批大小")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--dropout", type=float, help="Dropout率")
    parser.add_argument("--data_percentage", type=float, help="数据使用比例")
    
    # GPU配置
    parser.add_argument("--gpu", type=str, help="GPU设备ID")
    parser.add_argument("--multi_gpu", action="store_true", help="是否使用多GPU")
    
    # 网格搜索参数
    parser.add_argument("--max_experiments", type=int, default=50, help="最大实验数量")
    
    return parser


def setup_gpu_config(config):
    """设置GPU配置
    
    Args:
        config (dict): 配置字典
    """
    if 'gpu' in config and 'device_ids' in config['gpu']:
        gpu_ids = config['gpu']['device_ids']
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
        print(f"通过配置文件设置GPU: {gpu_ids}")


def apply_common_overrides(config, args, mode):
    """应用常用参数的覆盖逻辑
    
    只处理90%使用场景的常用参数，移除复杂的嵌套处理。
    
    Args:
        config (dict): 配置字典
        args: 命令行参数
        mode (str): 运行模式
        
    Returns:
        dict: 更新后的配置字典
    """
    # 确保hp节点存在
    hp = config.setdefault('hp', {})
    
    # 单实验模式：从网格配置提取默认值
    if mode == "single_experiment" and "grid_search" in config:
        grid = config["grid_search"].get("grid", {})
        
        # 提取网格搜索中的默认值
        grid_params = {
            'hp.learning_rate': 'learning_rate',
            'hp.batch_size': 'batch_size',
            'hp.epochs': 'epochs',
            'hp.dropout': 'dropout',
        }
        
        for grid_key, hp_key in grid_params.items():
            if grid_key in grid and isinstance(grid[grid_key], list) and hp_key not in hp:
                hp[hp_key] = grid[grid_key][0]  # 取第一个值作为默认值
    
    # 应用命令行参数覆盖
    param_overrides = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'dropout': args.dropout,
        'data_percentage': args.data_percentage,
    }
    
    for param_name, param_value in param_overrides.items():
        if param_value is not None:
            hp[param_name] = param_value
    
    # 处理其他配置
    if mode == "single_experiment":
        if args.exp_name:
            config.setdefault('training', {})['exp_name'] = args.exp_name
        if args.model_name:
            config.setdefault('model', {})['name'] = args.model_name
    
    return config


def parse_arguments_simplified(mode="grid_search"):
    """简化的参数解析函数
    
    移除过度工程化的功能，专注于核心需求：
    - 支持常用参数的命令行覆盖
    - 保持双模式支持
    - 简化的错误处理
    
    Args:
        mode (str): 运行模式，'grid_search' 或 'single_experiment'
        
    Returns:
        tuple: (args, config) 命令行参数和融合后的配置字典
    """
    # 创建参数解析器
    description = "网格搜索训练" if mode == "grid_search" else "单实验训练"
    parser = create_base_parser(description)
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 加载配置文件
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式错误: {e}")
    
    # 应用参数覆盖
    config = apply_common_overrides(config, args, mode)
    
    # 配置GPU环境
    setup_gpu_config(config)
    
    return args, config


def parse_arguments_with_fallback(mode="grid_search", use_simplified=False):
    """带回退的参数解析函数
    
    提供简化版本和原版本之间的选择，便于渐进式迁移。
    
    Args:
        mode (str): 运行模式
        use_simplified (bool): 是否使用简化版本
        
    Returns:
        tuple: (args, config)
    """
    if use_simplified:
        try:
            return parse_arguments_simplified(mode)
        except Exception as e:
            print(f"⚠️ 简化解析器失败，回退到原版本: {e}")
            from .config_parser import parse_arguments
            return parse_arguments(mode)
    else:
        from .config_parser import parse_arguments
        return parse_arguments(mode)


# 为了方便测试和比较，提供统一接口
def parse_arguments(mode="grid_search"):
    """统一的参数解析接口
    
    当前使用简化版本，如有问题会自动回退到原版本。
    """
    return parse_arguments_with_fallback(mode, use_simplified=True)


if __name__ == "__main__":
    # 测试简化版本的功能
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python config_parser_simplified.py --config <config_file>")
        sys.exit(1)
    
    try:
        args, config = parse_arguments_simplified("single_experiment")
        print("✅ 简化配置解析器测试成功")
        print(f"实验名称: {config.get('training', {}).get('exp_name', '未设置')}")
        print(f"任务类型: {config.get('task', {}).get('tag', '未设置')}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)

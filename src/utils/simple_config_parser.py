"""简化的配置解析器

提供更简洁、易维护的配置解析功能，替代过度复杂的原有实现。
"""

import argparse
import yaml
import os
from copy import deepcopy


class SimpleConfigParser:
    """简化的配置解析器类"""
    
    def __init__(self):
        self.base_config = {}
        self.overrides = {}
    
    def create_parser(self, description="训练配置解析"):
        """创建基础参数解析器"""
        parser = argparse.ArgumentParser(description=description)
        
        # 基础参数
        parser.add_argument('--config', type=str, required=True,
                          help='配置文件路径')
        parser.add_argument('--exp_name', type=str,
                          help='实验名称（覆盖配置文件中的设置）')
        parser.add_argument('--epochs', type=int,
                          help='训练轮数（覆盖配置文件中的设置）')
        parser.add_argument('--batch_size', type=int,
                          help='批大小（覆盖配置文件中的设置）')
        parser.add_argument('--learning_rate', type=float,
                          help='学习率（覆盖配置文件中的设置）')
        
        # GPU配置
        parser.add_argument('--gpu', type=str,
                          help='GPU设备ID，如"0,1"')
        parser.add_argument('--multi_gpu', action='store_true',
                          help='是否使用多GPU训练')
        
        # 网格搜索特定参数
        parser.add_argument('--max_experiments', type=int, default=50,
                          help='最大实验数量（仅网格搜索）')
        
        return parser
    
    def load_yaml_config(self, config_path):
        """加载YAML配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def apply_cli_overrides(self, config, args):
        """应用命令行参数覆盖"""
        # 简单的覆盖逻辑，只处理常用参数
        if args.exp_name:
            config.setdefault('training', {})['exp_name'] = args.exp_name
        
        if args.epochs:
            config.setdefault('hp', {})['epochs'] = args.epochs
            
        if args.batch_size:
            config.setdefault('hp', {})['batch_size'] = args.batch_size
            
        if args.learning_rate:
            config.setdefault('hp', {})['learning_rate'] = args.learning_rate
        
        return config
    
    def setup_gpu_config(self, config, args):
        """设置GPU配置"""
        if args.gpu:
            # 设置GPU环境变量
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            print(f"通过命令行设置GPU: {args.gpu}")
        elif 'gpu' in config and 'device_ids' in config['gpu']:
            # 使用配置文件中的GPU设置
            gpu_ids = config['gpu']['device_ids']
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
            print(f"通过配置文件设置GPU: {gpu_ids}")
        
        return config
    
    def validate_config(self, config):
        """验证配置完整性"""
        # 基础必需部分
        required_sections = ['training', 'swanlab', 'data']

        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置文件缺少必需的部分: {section}")

        # 验证关键参数
        if 'exp_name' not in config['training']:
            raise ValueError("配置文件中缺少 training.exp_name")

        # 对于网格搜索配置，hp可能在grid_search.grid中
        if 'hp' not in config and 'grid_search' not in config:
            raise ValueError("配置文件中缺少 hp 或 grid_search 部分")

        return True
    
    def parse(self, mode="single_experiment"):
        """解析配置的主函数
        
        Args:
            mode (str): 解析模式，'single_experiment' 或 'grid_search'
            
        Returns:
            tuple: (args, config)
        """
        # 创建解析器
        if mode == "single_experiment":
            parser = self.create_parser("单实验训练")
        else:
            parser = self.create_parser("网格搜索训练")
        
        # 解析命令行参数
        args = parser.parse_args()
        
        # 加载配置文件
        config = self.load_yaml_config(args.config)
        
        # 应用命令行覆盖
        config = self.apply_cli_overrides(config, args)
        
        # 设置GPU配置
        config = self.setup_gpu_config(config, args)
        
        # 验证配置
        self.validate_config(config)
        
        return args, config


def parse_arguments_simple(mode="single_experiment"):
    """简化的参数解析函数
    
    这是原有parse_arguments函数的简化版本，提供相同的接口但更易维护。
    
    Args:
        mode (str): 运行模式
        
    Returns:
        tuple: (args, config)
    """
    parser = SimpleConfigParser()
    return parser.parse(mode)


# 为了保持向后兼容性，可以选择性地替换原有函数
def parse_arguments_with_fallback(mode="grid_search", use_simple=True):
    """带回退的参数解析函数
    
    Args:
        mode (str): 运行模式
        use_simple (bool): 是否使用简化版本
        
    Returns:
        tuple: (args, config)
    """
    if use_simple:
        try:
            return parse_arguments_simple(mode)
        except Exception as e:
            print(f"简化解析器失败，回退到原有实现: {e}")
            # 这里可以回退到原有的parse_arguments函数
            from .config_parser import parse_arguments
            return parse_arguments(mode)
    else:
        from .config_parser import parse_arguments
        return parse_arguments(mode)


"""
配置解析器 - 处理YAML配置和命令行参数
"""

import argparse
import yaml
import os

def setup_gpu_config(args, config):
    """处理GPU配置的公共函数"""
    if args.use_cpu:
        # 强制使用CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # 只在非分布式环境中打印，避免重复输出
        if os.environ.get('LOCAL_RANK') is None:
            print("🖥️  强制使用CPU训练")
    else:
        # 设置GPU环境变量
        gpu_config = config.get('gpu', {})

        # 命令行参数优先级最高
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
            # 只在非分布式环境中打印，避免重复输出
            if os.environ.get('LOCAL_RANK') is None:
                print(f"通过命令行设置GPU: {args.gpu_ids}")
        # 其次是配置文件中的设置
        elif gpu_config.get('device_ids'):
            device_ids = str(gpu_config['device_ids'])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
            # 只在非分布式环境中打印，避免重复输出
            if os.environ.get('LOCAL_RANK') is None:
                print(f"通过配置文件设置GPU: {device_ids}")

def create_base_parser(description):
    """创建基础参数解析器，包含公共参数"""
    parser = argparse.ArgumentParser(description=description)
    
    # 配置文件
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    # GPU配置
    parser.add_argument("--gpu_ids", type=str, help="指定GPU ID，如 '0,1,2'")
    parser.add_argument("--use_cpu", action="store_true", help="强制使用CPU训练")
    parser.add_argument("--multi_gpu", action="store_true", help="使用多卡训练")
    parser.add_argument("--accelerate_args", type=str, default="", help="传递给accelerate launch的额外参数")
    
    return parser

def parse_single_training_arguments():
    """解析单次训练的参数"""
    parser = create_base_parser("单次训练参数解析")
    
    # 设置默认配置文件
    parser.set_defaults(config="config/base.yaml")

    # 超参数
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--batch_size", type=int, help="批大小")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--dropout", type=float, help="Dropout率")

    # 模型配置
    parser.add_argument("--model_type", type=str, help="模型类型")

    # 实验配置
    parser.add_argument("--experiment_name", type=str, help="实验名称")

    args = parser.parse_args()

    # 加载YAML配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖配置
    hp = config["hyperparameters"]
    if args.learning_rate is not None:
        hp["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        hp["batch_size"] = args.batch_size
    if args.epochs is not None:
        hp["epochs"] = args.epochs
    if args.dropout is not None:
        hp["dropout"] = args.dropout
    
    # 模型配置覆盖
    if args.model_type is not None:
        config["model"]["type"] = args.model_type
    
    # 实验配置覆盖
    if args.experiment_name is not None:
        config["training"]["experiment_name"] = args.experiment_name

    # 处理GPU配置
    setup_gpu_config(args, config)

    return args, config


def parse_grid_search_arguments():
    """解析网格搜索的参数"""
    parser = create_base_parser("深度学习训练框架 - 网格搜索")
    
    # 设置默认配置文件
    parser.set_defaults(config="config/grid.yaml")

    # 网格搜索配置
    parser.add_argument("--max_experiments", type=int, default=50,
                       help="最大实验数量限制")
    
    # 结果保存配置
    parser.add_argument("--save_results", action="store_true", default=True, help="保存实验结果表格")
    parser.add_argument("--results_file", type=str, default="grid_search_results.csv", help="结果保存文件名")

    args = parser.parse_args()

    # 加载YAML配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 处理GPU配置
    setup_gpu_config(args, config)

    return args, config


"""
配置解析器 - 处理YAML配置和命令行参数
"""

import argparse
import yaml
import os

def setup_gpu_config(args, config):
    """处理GPU配置"""
    if args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if os.environ.get('LOCAL_RANK') is None:
            print("🖥️  强制使用CPU训练")
    else:
        gpu_config = config.get('gpu', {})
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
            if os.environ.get('LOCAL_RANK') is None:
                print(f"通过命令行设置GPU: {args.gpu_ids}")
        elif gpu_config.get('device_ids'):
            device_ids = str(gpu_config['device_ids'])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
            if os.environ.get('LOCAL_RANK') is None:
                print(f"通过配置文件设置GPU: {device_ids}")

def create_base_parser(description):
    """创建基础参数解析器"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="config/grid.yaml", help="配置文件路径")
    parser.add_argument("--gpu_ids", type=str, help="指定GPU ID")
    parser.add_argument("--use_cpu", action="store_true", help="强制使用CPU训练")
    parser.add_argument("--multi_gpu", action="store_true", help="使用多卡训练")
    parser.add_argument("--accelerate_args", type=str, default="", help="accelerate参数")
    return parser

def parse_arguments(mode="train"):
    """统一的参数解析函数"""
    if mode == "train":
        parser = create_base_parser("单次训练")
        # 训练特定参数
        parser.add_argument("--learning_rate", type=float, help="学习率")
        parser.add_argument("--batch_size", type=int, help="批大小")
        parser.add_argument("--epochs", type=int, help="训练轮数")
        parser.add_argument("--dropout", type=float, help="Dropout率")
        parser.add_argument("--model_name", type=str, help="模型名称")
        parser.add_argument("--experiment_name", type=str, help="实验名称")
        parser.add_argument('--is_grid_search', action='store_true', help='Flag to indicate if this is a grid search run.')
    else:  # grid_search
        parser = create_base_parser("网格搜索")
        # 网格搜索特定参数
        parser.add_argument("--max_experiments", type=int, default=50, help="最大实验数量")
        parser.add_argument("--save_results", action="store_true", default=True, help="保存结果")
        parser.add_argument("--results_file", type=str, default="grid_search_results.csv", help="结果文件名")

    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖配置（仅训练模式）
    if mode == "train":
        hp = config["hyperparameters"]
        if args.learning_rate is not None:
            hp["learning_rate"] = args.learning_rate
        if args.batch_size is not None:
            hp["batch_size"] = args.batch_size
        if args.epochs is not None:
            hp["epochs"] = args.epochs
        if args.dropout is not None:
            hp["dropout"] = args.dropout
        if args.model_name is not None:
            config["model"]["name"] = args.model_name
        if args.experiment_name is not None:
            config["training"]["experiment_name"] = args.experiment_name

    setup_gpu_config(args, config)
    return args, config

def parse_single_training_arguments():
    """解析单次训练参数（保持向后兼容）"""
    return parse_arguments("train")

def parse_grid_search_arguments():
    """解析网格搜索参数（保持向后兼容）"""
    return parse_arguments("grid_search")

"""
配置解析器 - 处理 YAML 配置和命令行参数
"""
import argparse
import yaml
import os

def setup_gpu_config(config):
    """仅在非 Accelerate 子进程时，从 YAML 设置 CUDA_VISIBLE_DEVICES。"""
    # 在 Accelerate/torchrun 子进程中，device mapping 由框架接管，不能再动
    if os.environ.get("LOCAL_RANK") is not None:
        return

    gpu_cfg = (config or {}).get("gpu", {}) or {}
    if gpu_cfg.get("device_ids"):
        device_ids = str(gpu_cfg["device_ids"])
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        print(f"通过配置文件设置GPU: {device_ids}")
        
def create_base_parser(description):
    """
    创建基础参数解析器（统一网格搜索模式）
    
    Args:
        description (str): 解析器描述信息
        
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="config/grid.yaml", help="配置文件路径")
    parser.add_argument("--multi_gpu", action="store_true", help="使用多卡训练（由调度器/子训练决定）")

    # 网格搜索相关参数
    parser.add_argument("--max_experiments", type=int, default=50, help="最大实验数量")
    parser.add_argument("--save_results", action="store_true", default=True, help="保存结果")
    parser.add_argument("--results_file", type=str, default="grid_search_results.csv", help="结果文件名")
    parser.add_argument("--top_n", type=int, default=3, help="显示前n名实验结果")
    
    # 单个实验参数覆盖（用于网格搜索中的单个实验）
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--batch_size", type=int, help="批大小")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--dropout", type=float, help="Dropout率")
    parser.add_argument("--model_name", type=str, help="模型名称")
    parser.add_argument("--model.type", type=str, help="模型类型")
    
    parser.add_argument("--optimizer.name", type=str, help="优化器名称")
    parser.add_argument("--optimizer.params.weight_decay", type=float, help="权重衰减")
    parser.add_argument("--weight_decay", type=float, help="权重衰减")
    parser.add_argument("--scheduler.name", type=str, help="调度器名称")
    parser.add_argument("--loss", type=str, help="损失函数类型")
    parser.add_argument("--loss.name", type=str, help="损失函数名称")
    parser.add_argument("--hp.learning_rate", type=float, help="学习率")
    parser.add_argument("--hp.batch_size", type=int, help="批大小")
    parser.add_argument("--hp.epochs", type=int, help="训练轮数")
    parser.add_argument("--experiment_name", type=str, help="实验名称")
    
    return parser


def parse_arguments(mode="grid_search"):
    """
    统一的参数解析函数，处理命令行参数和配置文件（统一网格搜索模式）
    
    Args:
        mode (str, optional): 运行模式，现在统一为'grid_search'或'single_experiment'
        
    Returns:
        tuple: (args, config) 解析后的命令行参数和配置字典
    """
    if mode == "single_experiment":
        parser = create_base_parser("单个实验训练（网格搜索中的一个实验）")
    else:  # grid_search
        parser = create_base_parser("网格搜索训练")

    args = parser.parse_args()

    # 加载配置文件
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 处理嵌套参数覆盖
    def set_nested_value(config_dict, key_path, value):
        """设置嵌套字典的值，支持点号分隔的路径"""
        keys = key_path.split('.')
        current = config_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    # 应用所有命令行参数覆盖（包括嵌套参数）
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and '.' in arg_name:
            set_nested_value(config, arg_name, arg_value)

    # 为单个实验应用参数覆盖（用于网格搜索中的单个实验）
    if mode == "single_experiment":
        # 创建临时的hp节点用于兼容性
        if "hp" not in config:
            config["hp"] = {}
        
        hp = config["hp"]
        
        # 如果hp为空且存在grid_search配置，从grid中获取默认值
        if not hp and "grid_search" in config and "grid" in config["grid_search"]:
            grid = config["grid_search"]["grid"]
            # 从grid中提取第一个值作为默认值
            if "batch_size" in grid and isinstance(grid["batch_size"], list):
                hp["batch_size"] = grid["batch_size"][0]
            if "learning_rate" in grid and isinstance(grid["learning_rate"], list):
                hp["learning_rate"] = grid["learning_rate"][0]
            if "epochs" in grid and isinstance(grid["epochs"], list):
                hp["epochs"] = grid["epochs"][0]
            if "dropout" in grid and isinstance(grid["dropout"], list):
                hp["dropout"] = grid["dropout"][0]
        
        # 应用命令行参数覆盖
        if args.learning_rate is not None:
            hp["learning_rate"] = args.learning_rate
        if args.batch_size is not None:
            hp["batch_size"] = args.batch_size
        if args.epochs is not None:
            hp["epochs"] = args.epochs
        if args.dropout is not None:
            hp["dropout"] = args.dropout
            
        # 为其他配置节点设置默认值（如果不存在且grid中有配置）
        if "grid_search" in config and "grid" in config["grid_search"]:
            grid = config["grid_search"]["grid"]
            
            # 设置optimizer默认值
            if "optimizer" not in config and "optimizer" in grid and "name" in grid["optimizer"]:
                optimizer_name = grid["optimizer"]["name"][0] if isinstance(grid["optimizer"]["name"], list) else grid["optimizer"]["name"]
                config["optimizer"] = {"name": optimizer_name, "params": {}}
            if "optimizer" in config and "weight_decay" in grid and "params" not in config["optimizer"]:
                config["optimizer"]["params"] = {}
            if "optimizer" in config and "weight_decay" in grid and isinstance(grid["weight_decay"], list):
                if "params" not in config["optimizer"]:
                    config["optimizer"]["params"] = {}
                config["optimizer"]["params"]["weight_decay"] = grid["weight_decay"][0]
                
            # 设置scheduler默认值
            if "scheduler" not in config and "scheduler" in grid and "name" in grid["scheduler"]:
                scheduler_name = grid["scheduler"]["name"][0] if isinstance(grid["scheduler"]["name"], list) else grid["scheduler"]["name"]
                config["scheduler"] = {"name": scheduler_name}
                
            # 设置loss默认值
            if "loss" not in config and "loss" in grid:
                config["loss"] = {"type": grid["loss"][0] if isinstance(grid["loss"], list) else grid["loss"]}
        
        # 应用模型和优化器配置
        if args.model_name is not None:
            if "model" not in config:
                config["model"] = {}
            config["model"]["name"] = args.model_name
            # config["model"]["type"] = args.model_name
            
        # 处理optimizer.name参数
        optimizer_name = getattr(args, 'optimizer.name', None)
        if optimizer_name is not None:
            if "optimizer" not in config:
                config["optimizer"] = {"params": {}}
            config["optimizer"]["name"] = optimizer_name
        if args.weight_decay is not None:
            if "optimizer" not in config:
                config["optimizer"] = {"params": {}}
            if "params" not in config["optimizer"]:
                config["optimizer"]["params"] = {}
            config["optimizer"]["params"]["weight_decay"] = args.weight_decay
        # 处理scheduler.name参数
        scheduler_name = getattr(args, 'scheduler.name', None)
        if scheduler_name is not None:
            if "scheduler" not in config:
                config["scheduler"] = {}
            config["scheduler"]["name"] = scheduler_name
        if args.loss is not None:
            if "loss" not in config:
                config["loss"] = {}
            config["loss"]["type"] = args.loss
        if args.experiment_name is not None:
            config["training"]["experiment_name"] = args.experiment_name

    setup_gpu_config(config)
    
    return args, config



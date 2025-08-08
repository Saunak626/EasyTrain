"""配置解析器 - 处理 YAML 配置和命令行参数

设计思路：
1. 统一配置管理：将YAML配置文件和命令行参数统一处理，实现配置的灵活性和可扩展性
2. 分层覆盖机制：YAML配置作为基础，命令行参数可以覆盖配置文件中的值
3. 嵌套参数支持：支持点号分隔的嵌套参数（如optimizer.name），便于精确控制配置
4. 模式适配：支持网格搜索和单实验两种模式，满足不同的训练需求
5. GPU环境管理：智能处理GPU设备分配，避免与分布式训练框架冲突

核心功能：
- 解析命令行参数和YAML配置文件
- 处理嵌套配置参数的覆盖逻辑
- 为网格搜索提供参数组合支持
- 管理GPU设备分配和环境变量
"""
import argparse
import yaml
import os

def setup_gpu_config(config):
    """GPU环境配置管理函数
    
    设计思路：
    - 智能检测分布式训练环境，避免与Accelerate/torchrun框架冲突
    - 仅在主进程中设置GPU设备，子进程由框架自动管理
    - 通过环境变量CUDA_VISIBLE_DEVICES控制GPU可见性
    
    Args:
        config (dict): 包含GPU配置的字典，格式为 {"gpu": {"device_ids": "0,1,2"}}
    
    注意事项：
    - 在分布式训练中，LOCAL_RANK环境变量表示当前进程是子进程
    - 子进程的GPU分配由Accelerate框架自动处理，不应手动干预
    """
    # 检测是否为分布式训练的子进程
    # 在 Accelerate/torchrun 子进程中，device mapping 由框架接管，不能再动
    if os.environ.get("LOCAL_RANK") is not None:
        return

    # 安全获取GPU配置，避免KeyError
    gpu_cfg = (config or {}).get("gpu", {}) or {}
    if gpu_cfg.get("device_ids"):
        device_ids = str(gpu_cfg["device_ids"])
        # 设置环境变量，限制CUDA可见的GPU设备
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        print(f"通过配置文件设置GPU: {device_ids}")
        
def create_base_parser(description):
    """
    创建基础参数解析器（统一网格搜索模式）
    
    设计思路：
    - 统一参数定义：为网格搜索和单实验模式提供统一的参数接口
    - 分层参数设计：支持基础参数、网格搜索参数和嵌套参数三个层次
    - 灵活覆盖机制：命令行参数可以覆盖配置文件中的任何设置
    - 嵌套参数支持：使用点号分隔的参数名支持深层配置覆盖
    
    参数分类：
    1. 基础参数：config, multi_gpu等控制训练环境的参数
    2. 网格搜索参数：max_experiments, save_results等控制搜索行为的参数
    3. 实验参数：learning_rate, batch_size等可被网格搜索的超参数
    4. 嵌套参数：optimizer.name, model.type等支持精确配置的参数
    
    Args:
        description (str): 解析器描述信息，用于帮助文档显示
        
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器，包含所有必要的参数定义
    """
    parser = argparse.ArgumentParser(description=description)
    
    # === 基础环境参数 ===
    # 控制训练环境和配置文件的基础参数
    parser.add_argument("--config", type=str, default="config/grid.yaml", help="配置文件路径")
    parser.add_argument("--multi_gpu", action="store_true", help="使用多卡训练（由调度器/子训练决定）")

    # === 网格搜索控制参数 ===
    # 控制网格搜索行为和结果输出的参数
    parser.add_argument("--max_experiments", type=int, default=50, help="最大实验数量")
    parser.add_argument("--save_results", action="store_true", default=True, help="保存结果")
    parser.add_argument("--results_file", type=str, default="grid_search_results.csv", help="结果文件名")
    parser.add_argument("--top_n", type=int, default=3, help="显示前n名实验结果")
    
    # === 核心超参数覆盖 ===
    # 用于网格搜索中单个实验的参数覆盖，这些参数可以覆盖配置文件中的值
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--batch_size", type=int, help="批大小")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--dropout", type=float, help="Dropout率")
    parser.add_argument("--model_name", type=str, help="模型名称")
    parser.add_argument("--exp_name", type=str, help="实验名称")
    parser.add_argument("--data_percentage", type=float, default=None, help="使用数据的百分比 (0.0-1.0)")
    
    # === 嵌套配置参数 ===
    # 支持点号分隔的深层配置覆盖，实现精确的配置控制
    parser.add_argument("--model.type", type=str, help="模型类型")
    parser.add_argument("--optimizer.name", type=str, help="优化器名称")
    parser.add_argument("--optimizer.params.weight_decay", type=float, help="权重衰减")
    parser.add_argument("--weight_decay", type=float, help="权重衰减")  # 兼容性参数
    parser.add_argument("--scheduler.name", type=str, help="调度器名称")
    parser.add_argument("--loss", type=str, help="损失函数类型")
    parser.add_argument("--loss.name", type=str, help="损失函数名称")
    
    # === 超参数命名空间 ===
    # 使用hp前缀的参数，与配置文件中的hp节点对应
    parser.add_argument("--hp.learning_rate", type=float, help="学习率")
    parser.add_argument("--hp.batch_size", type=int, help="批大小")
    parser.add_argument("--hp.epochs", type=int, help="训练轮数")
    
    return parser


def parse_arguments(mode="grid_search"):
    """
    统一的参数解析函数，处理命令行参数和配置文件（统一网格搜索模式）
    
    设计思路：
    - 双模式支持：支持网格搜索调度器模式和单实验执行模式
    - 配置融合：将YAML配置文件和命令行参数智能融合，实现灵活的配置管理
    - 嵌套参数处理：支持点号分隔的嵌套参数，可以精确覆盖配置文件中的任意层级
    - 默认值回退：为单实验模式提供从网格配置中提取默认值的机制
    - 参数标准化：统一处理不同格式的参数，确保配置的一致性
    
    处理流程：
    1. 根据模式创建对应的参数解析器
    2. 解析命令行参数
    3. 加载YAML配置文件
    4. 处理嵌套参数覆盖
    5. 为单实验模式设置默认值和参数覆盖
    6. 配置GPU环境
    
    Args:
        mode (str, optional): 运行模式
            - 'grid_search': 网格搜索调度器模式，用于启动多个实验
            - 'single_experiment': 单实验模式，用于执行具体的训练任务
        
    Returns:
        tuple: (args, config) 
            - args: 解析后的命令行参数对象
            - config: 融合后的完整配置字典，包含所有训练所需的配置信息
    """
    # === 第1步：创建参数解析器 ===
    # 根据运行模式选择合适的解析器描述
    if mode == "single_experiment":
        parser = create_base_parser("单个实验训练（网格搜索中的一个实验）")
    else:  # grid_search
        parser = create_base_parser("网格搜索训练")

    # === 第2步：解析命令行参数 ===
    args = parser.parse_args()

    # === 第3步：加载YAML配置文件 ===
    # 从指定路径加载配置文件，作为基础配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # === 第4步：处理嵌套参数覆盖 ===
    # 定义嵌套字典值设置函数，支持点号分隔的深层路径访问
    def set_nested_value(config_dict, key_path, value):
        """设置嵌套字典的值，支持点号分隔的路径
        
        例如：set_nested_value(config, 'optimizer.params.weight_decay', 0.01)
        将设置 config['optimizer']['params']['weight_decay'] = 0.01
        """
        keys = key_path.split('.')
        current = config_dict
        # 逐层创建或访问嵌套字典
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        # 设置最终的值
        current[keys[-1]] = value

    # 应用所有命令行中的嵌套参数覆盖
    # 只处理包含点号的参数名，这些是嵌套配置参数
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and '.' in arg_name:
            set_nested_value(config, arg_name, arg_value)

    # === 第5步：单实验模式的特殊处理 ===
    # 为单个实验应用参数覆盖（用于网格搜索中的单个实验）
    if mode == "single_experiment":
        # 确保hp节点存在，用于存储超参数配置
        if "hp" not in config:
            config["hp"] = {}
        
        hp = config["hp"]
        
        # === 5.1：从网格配置中提取默认值 ===
        # 如果hp为空且存在grid_search配置，从grid中获取默认值
        # 这是为了确保单实验有合理的默认参数，即使配置文件中hp节点为空
        if not hp and "grid_search" in config and "grid" in config["grid_search"]:
            grid = config["grid_search"]["grid"]
            # 从grid中提取第一个值作为默认值（网格搜索的起始点）
            # 注意：YAML中使用hp.前缀的参数需要正确提取
            if "hp.batch_size" in grid and isinstance(grid["hp.batch_size"], list):
                hp["batch_size"] = grid["hp.batch_size"][0]
            if "hp.learning_rate" in grid and isinstance(grid["hp.learning_rate"], list):
                hp["learning_rate"] = grid["hp.learning_rate"][0]
            if "hp.epochs" in grid and isinstance(grid["hp.epochs"], list):
                hp["epochs"] = grid["hp.epochs"][0]
            if "hp.dropout" in grid and isinstance(grid["hp.dropout"], list):
                hp["dropout"] = grid["hp.dropout"][0]
            if "hp.data_percentage" in grid and isinstance(grid["hp.data_percentage"], list):
                hp["data_percentage"] = grid["hp.data_percentage"][0]
            # 兼容旧格式（不带hp.前缀）
            if "batch_size" in grid and isinstance(grid["batch_size"], list):
                hp["batch_size"] = grid["batch_size"][0]
            if "learning_rate" in grid and isinstance(grid["learning_rate"], list):
                hp["learning_rate"] = grid["learning_rate"][0]
            if "epochs" in grid and isinstance(grid["epochs"], list):
                hp["epochs"] = grid["epochs"][0]
            if "dropout" in grid and isinstance(grid["dropout"], list):
                hp["dropout"] = grid["dropout"][0]
            if "data_percentage" in grid and isinstance(grid["data_percentage"], list):
                hp["data_percentage"] = grid["data_percentage"][0]
        
        # === 5.2：应用命令行超参数覆盖 ===
        # 命令行参数具有最高优先级，可以覆盖配置文件和默认值
        if args.learning_rate is not None:
            hp["learning_rate"] = args.learning_rate
        if args.batch_size is not None:
            hp["batch_size"] = args.batch_size
        if args.epochs is not None:
            hp["epochs"] = args.epochs
        if args.dropout is not None:
            hp["dropout"] = args.dropout
        if args.data_percentage is not None:
            hp["data_percentage"] = args.data_percentage
            
        # === 5.3：为其他配置节点设置默认值 ===
        # 从网格配置中为optimizer、scheduler、loss等组件设置默认值
        # 确保单实验模式下所有必要的配置都有合理的默认值
        if "grid_search" in config and "grid" in config["grid_search"]:
            grid = config["grid_search"]["grid"]
            
            # 设置optimizer默认值
            # 如果配置中没有optimizer且grid中有定义，则创建默认optimizer配置
            if "optimizer" not in config and "optimizer" in grid and "name" in grid["optimizer"]:
                optimizer_name = grid["optimizer"]["name"][0] if isinstance(grid["optimizer"]["name"], list) else grid["optimizer"]["name"]
                config["optimizer"] = {"name": optimizer_name, "params": {}}
            # 确保optimizer有params节点
            if "optimizer" in config and "weight_decay" in grid and "params" not in config["optimizer"]:
                config["optimizer"]["params"] = {}
            # 设置weight_decay默认值
            if "optimizer" in config and "weight_decay" in grid and isinstance(grid["weight_decay"], list):
                if "params" not in config["optimizer"]:
                    config["optimizer"]["params"] = {}
                config["optimizer"]["params"]["weight_decay"] = grid["weight_decay"][0]
                
            # 设置scheduler默认值
            # 如果配置中没有scheduler且grid中有定义，则创建默认scheduler配置
            if "scheduler" not in config and "scheduler" in grid and "name" in grid["scheduler"]:
                scheduler_name = grid["scheduler"]["name"][0] if isinstance(grid["scheduler"]["name"], list) else grid["scheduler"]["name"]
                config["scheduler"] = {"name": scheduler_name}
                
            # 设置loss默认值
            # 如果配置中没有loss且grid中有定义，则创建默认loss配置
            if "loss" not in config and "loss" in grid:
                config["loss"] = {"type": grid["loss"][0] if isinstance(grid["loss"], list) else grid["loss"]}
        
        # === 5.4：应用其他命令行参数覆盖 ===
        # 处理模型配置
        if args.model_name is not None:
            if "model" not in config:
                config["model"] = {}
            config["model"]["name"] = args.model_name
            
        # 处理optimizer相关参数
        # 处理optimizer.name参数（嵌套参数格式）
        optimizer_name = getattr(args, 'optimizer.name', None)
        if optimizer_name is not None:
            if "optimizer" not in config:
                config["optimizer"] = {"params": {}}
            config["optimizer"]["name"] = optimizer_name
        # 处理weight_decay参数（兼容性参数）
        if args.weight_decay is not None:
            if "optimizer" not in config:
                config["optimizer"] = {"params": {}}
            if "params" not in config["optimizer"]:
                config["optimizer"]["params"] = {}
            config["optimizer"]["params"]["weight_decay"] = args.weight_decay
            
        # 处理scheduler相关参数
        # 处理scheduler.name参数（嵌套参数格式）
        scheduler_name = getattr(args, 'scheduler.name', None)
        if scheduler_name is not None:
            if "scheduler" not in config:
                config["scheduler"] = {}
            config["scheduler"]["name"] = scheduler_name
            
        # 处理loss相关参数
        if args.loss is not None:
            if "loss" not in config:
                config["loss"] = {}
            config["loss"]["type"] = args.loss
            
        # 处理实验名称
        if args.exp_name is not None:
            config["training"]["exp_name"] = args.exp_name

    # === 第6步：配置GPU环境 ===
    # 根据配置设置GPU环境，包括设备选择和分布式训练配置
    setup_gpu_config(config)
    
    # === 返回融合后的配置 ===
    # 返回解析后的命令行参数和完整的配置字典
    # config包含了所有训练所需的配置信息，已经过参数覆盖和默认值设置
    return args, config



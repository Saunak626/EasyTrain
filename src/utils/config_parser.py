"""配置解析器 - 处理 YAML 配置和命令行参数

设计思路：
1. 统一配置管理：将YAML配置文件和命令行参数统一处理，实现配置的灵活性和可扩展性
2. 分层覆盖机制：YAML配置作为基础，命令行参数可以覆盖配置文件中的值
3. 嵌套参数支持：支持点号分隔的嵌套参数（如optimizer.name），便于精确控制配置
4. 模式适配：支持网格搜索和单实验两种模式，满足不同的训练需求
5. GPU环境管理：智能处理GPU设备分配，避免与分布式训练框架冲突

参数优先级机制（从高到低）：
1. 命令行参数（最高优先级）
   - 直接参数：--learning_rate, --batch_size, --epochs 等
   - 嵌套参数：--model.type, --optimizer.name 等
2. YAML配置文件（基础优先级）
   - 配置文件中的 hp, model, optimizer 等节点
3. 代码默认值（最低优先级）
   - 各模块中定义的默认参数值

模型参数命名规范：
- 统一使用 model.type 作为模型类型参数名
- 兼容 --model_name 命令行参数（映射到 model.type）
- 配置文件中统一使用 model.type 字段

核心功能：
- 解析命令行参数和YAML配置文件
- 处理嵌套配置参数的覆盖逻辑
- 为网格搜索和单实验提供统一的配置入口
- 管理GPU设备分配和环境变量
"""
import argparse
import yaml
import os
from typing import Dict, List, Tuple, Any, Optional

# ============================================================================
# 模块级常量：参数映射配置
# ============================================================================

# 基础超参数映射：(命令行参数名, HP配置键名)
BASIC_PARAM_MAPPINGS: List[Tuple[str, str]] = [
    ("learning_rate", "learning_rate"),
    ("batch_size", "batch_size"),
    ("epochs", "epochs"),
    ("dropout", "dropout"),
    ("data_percentage", "data_percentage"),
]

# ============================================================================
# 辅助函数：配置处理工具
# ============================================================================

def set_nested_value(config_dict: Dict[str, Any], key_path: str, value: Any) -> None:
    """设置嵌套字典的值，支持点号分隔的路径

    Args:
        config_dict: 目标配置字典
        key_path: 点号分隔的键路径，如 'optimizer.params.weight_decay'
        value: 要设置的值

    Example:
        set_nested_value(config, 'optimizer.name', 'adam')
        # 等价于 config['optimizer']['name'] = 'adam'
    """
    keys = key_path.split('.')
    current = config_dict
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def ensure_hp_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """确保配置中存在hp节点

    Args:
        config: 配置字典

    Returns:
        hp节点的引用
    """
    if "hp" not in config:
        config["hp"] = {}
    return config["hp"]


def setup_gpu_config(config: Optional[Dict[str, Any]]) -> None:
    """GPU环境配置管理函数

    设计思路：
    - 智能检测分布式训练环境，避免与Accelerate/torchrun框架冲突
    - 仅在主进程中设置GPU设备，子进程由框架自动管理
    - 通过环境变量CUDA_VISIBLE_DEVICES控制GPU可见性

    Args:
        config: 包含GPU配置的字典，格式为 {"gpu": {"device_ids": "0,1,2"}}

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
        
# ============================================================================
# 参数处理函数：职责分离的配置处理逻辑
# ============================================================================


def apply_command_line_overrides(args: argparse.Namespace, hp: Dict[str, Any]) -> None:
    """应用命令行参数覆盖到hp节点

    将命令行中指定的基础参数值覆盖到hp配置中。
    这些参数具有最高优先级，会覆盖配置文件和网格默认值。

    Args:
        args: 解析后的命令行参数
        hp: hp节点的引用
    """
    for arg_name, hp_key in BASIC_PARAM_MAPPINGS:
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            hp[hp_key] = arg_value


def apply_nested_parameter_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """处理嵌套参数覆盖（点号分隔的参数）

    处理形如 --optimizer.name, --model.type 等嵌套参数，
    将其正确设置到配置字典的对应位置。

    Args:
        config: 完整配置字典
        args: 解析后的命令行参数
    """
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and '.' in arg_name:
            set_nested_value(config, arg_name, arg_value)


def apply_fps_sampling_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """处理FPS采样简化参数

    将--fps和--random_sampling参数转换为相应的配置设置，
    提供比--override更简洁的FPS采样控制方式。

    Args:
        config: 完整配置字典
        args: 解析后的命令行参数
    """
    # 确保data.params节点存在
    if "data" not in config:
        config["data"] = {}
    if "params" not in config["data"]:
        config["data"]["params"] = {}

    data_params = config["data"]["params"]

    # 处理--fps参数
    if args.fps is not None:
        if args.fps <= 0:
            raise ValueError(f"--fps参数必须大于0，当前值: {args.fps}")

        data_params["sampling_mode"] = "fps"
        data_params["target_fps"] = args.fps

        # 如果没有设置original_fps，使用默认值
        if "original_fps" not in data_params:
            data_params["original_fps"] = 16

        print(f"🎯 FPS采样模式已启用: target_fps={args.fps}, original_fps={data_params['original_fps']}")

    # 处理--random_sampling参数
    elif args.random_sampling:
        data_params["sampling_mode"] = "random"
        # 清除可能存在的FPS相关参数
        data_params.pop("target_fps", None)
        print(f"🎯 随机采样模式已启用")

    # 如果两个参数都没有指定，保持配置文件中的设置


def apply_single_experiment_configs(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """应用单实验模式特有的配置

    处理只在单实验模式下需要的配置，如实验名称和模型类型。

    Args:
        config: 完整配置字典
        args: 解析后的命令行参数
    """
    # 处理实验名称
    if args.exp_name is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["exp_name"] = args.exp_name

    # 处理模型配置 - 支持两种参数名
    model_type = getattr(args, 'model.type', None) or args.model_name
    if model_type is not None:
        if "model" not in config:
            config["model"] = {}
        config["model"]["type"] = model_type


def create_base_parser(description: str) -> argparse.ArgumentParser:
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
        description: 解析器描述信息，用于帮助文档显示

    Returns:
        配置好的参数解析器，包含所有必要的参数定义
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
    parser.add_argument("--results_file", type=str, default=None, help="结果文件名（默认使用时间戳）")
    parser.add_argument("--top_n", type=int, default=10, help="显示前n名实验结果")
    
    # === 核心超参数覆盖 ===
    # 用于网格搜索中单个实验的参数覆盖，这些参数可以覆盖配置文件中的值
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--batch_size", type=int, help="批大小")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--dropout", type=float, help="Dropout率")
    parser.add_argument("--model_name", type=str, help="模型名称")
    parser.add_argument("--exp_name", type=str, help="实验名称")
    parser.add_argument("--data_percentage", type=float, default=None, help="使用数据的百分比 (0.0-1.0)")
    parser.add_argument("--result_file", type=str, help="结果文件路径（用于网格搜索）")

    # === FPS采样简化参数 ===
    # 简化的FPS采样控制参数，替代复杂的--override方式
    parser.add_argument("--fps", type=float, default=None, help="启用FPS采样模式并设置目标帧率 (例如: --fps 8)")
    parser.add_argument("--random_sampling", action="store_true", help="强制使用随机采样模式（覆盖配置文件设置）")
    
    # === 嵌套配置参数 ===
    # 支持点号分隔的深层配置覆盖，实现精确的配置控制
    parser.add_argument("--model.type", type=str, help="模型类型")
    parser.add_argument("--optimizer.name", type=str, help="优化器名称")
    parser.add_argument("--optimizer.params.weight_decay", type=float, help="权重衰减")
    parser.add_argument("--weight_decay", type=float, help="权重衰减")  # 兼容性参数
    parser.add_argument("--scheduler.name", type=str, help="调度器名称")
    parser.add_argument("--scheduler.params.warmup_epochs", type=int, help="Warmup期长度")
    parser.add_argument("--scheduler.params.warmup_start_factor", type=float, help="Warmup起始学习率比例")
    parser.add_argument("--scheduler.params.eta_min_factor", type=float, help="最小学习率比例")
    parser.add_argument("--scheduler.params.step_size", type=int, help="Step调度器的步长")
    parser.add_argument("--scheduler.params.gamma", type=float, help="Step调度器的衰减因子")
    parser.add_argument("--loss", type=str, help="损失函数类型")
    parser.add_argument("--loss.name", type=str, help="损失函数名称")
    parser.add_argument("--loss.params.pos_weight", type=float, help="多标签BCE损失的正样本权重")

    # === 数据集参数 ===
    # 数据集相关的嵌套参数
    parser.add_argument("--data.params.sampling_mode", type=str, help="采样模式: random 或 fps")
    parser.add_argument("--data.params.target_fps", type=float, help="目标采样帧率")
    parser.add_argument("--data.params.original_fps", type=float, help="原始视频帧率")
    parser.add_argument("--data.params.clip_len", type=int, help="每个视频片段的帧数")

    # === 超参数命名空间 ===
    # 使用hp前缀的参数，与配置文件中的hp节点对应
    parser.add_argument("--hp.learning_rate", type=float, help="学习率")
    parser.add_argument("--hp.batch_size", type=int, help="批大小")
    parser.add_argument("--hp.epochs", type=int, help="训练轮数")
    
    return parser


def parse_arguments(mode: str = "grid_search") -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """解析命令行参数和YAML配置文件，支持参数覆盖

    支持网格搜索和单实验两种模式，将命令行参数与配置文件融合。
    使用分离的函数处理不同类型的参数覆盖，提高代码可维护性。

    Args:
        mode: 运行模式，'grid_search' 或 'single_experiment'

    Returns:
        (args, config) 命令行参数和融合后的配置字典
    """
    # 创建参数解析器
    if mode == "single_experiment":
        parser = create_base_parser("单个实验训练")
    else:  # grid_search
        parser = create_base_parser("网格搜索训练")

    # 解析命令行参数
    args = parser.parse_args()

    # 加载YAML配置文件
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 应用参数覆盖逻辑（按优先级顺序）
    config = apply_parameter_overrides(config, args, mode)

    # 配置GPU环境
    setup_gpu_config(config)

    return args, config


def apply_parameter_overrides(config: Dict[str, Any], args: argparse.Namespace, mode: str) -> Dict[str, Any]:
    """应用命令行参数覆盖配置文件设置

    使用分离的函数处理不同类型的参数覆盖，提高代码的可读性和可维护性。
    处理顺序：确保hp节点 -> 命令行覆盖 -> 嵌套参数 -> FPS采样简化参数 -> 模式特定配置

    Args:
        config: 原始配置字典
        args: 解析后的命令行参数
        mode: 运行模式（'grid_search' 或 'single_experiment'）

    Returns:
        处理后的配置字典
    """
    # 1. 确保hp节点存在
    hp = ensure_hp_section(config)

    # 2. 应用命令行参数覆盖到hp节点
    apply_command_line_overrides(args, hp)

    # 3. 处理嵌套参数（点号分隔）
    apply_nested_parameter_overrides(config, args)

    # 4. 处理FPS采样简化参数
    apply_fps_sampling_overrides(config, args)

    # 5. 应用模式特定的配置
    if mode == "single_experiment":
        apply_single_experiment_configs(config, args)

    return config



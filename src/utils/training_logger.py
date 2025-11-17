"""训练日志管理器

统一管理训练过程中的所有终端输出和日志记录。
"""

from typing import Dict, Any, Optional
from accelerate import Accelerator
from tqdm import tqdm


class TrainingLogger:
    """统一的训练日志管理器
    
    负责管理训练过程中的所有终端输出，支持日志级别控制和简洁/详细模式切换。
    
    特性：
    - 支持简洁模式和详细模式
    - 自动处理 accelerator.is_main_process 检查
    - 统一的日志格式
    """
    
    def __init__(self, accelerator: Accelerator, verbose: bool = False):
        """初始化日志管理器
        
        Args:
            accelerator: Accelerator实例
            verbose: 是否启用详细模式（默认False，使用简洁模式）
        """
        self.accelerator = accelerator
        self.verbose = verbose
    
    def info(self, message: str, force: bool = False):
        """打印INFO级别日志（始终显示）
        
        Args:
            message: 日志消息
            force: 是否强制打印（忽略主进程检查）
        """
        if force or self.accelerator.is_main_process:
            tqdm.write(message)
    
    def debug(self, message: str):
        """打印DEBUG级别日志（只在详细模式下显示）
        
        Args:
            message: 日志消息
        """
        if self.verbose and self.accelerator.is_main_process:
            tqdm.write(message)
    
    def print_experiment_config(self, config: Dict[str, Any]):
        """打印实验配置（根据模式选择详细程度）
        
        Args:
            config: 实验配置字典
        """
        if self.verbose:
            self._print_detailed_config(config)
        else:
            self._print_compact_config(config)
    
    def _print_compact_config(self, config: Dict[str, Any]):
        """打印简洁的实验配置（3-4 行）
        
        Args:
            config: 实验配置字典
        """
        if not self.accelerator.is_main_process:
            return
        
        # 第1行：实验名称、模型、数据集
        model_params = config.get('model_params_m', 0)
        model_size_mb = config.get('model_size_mb', 0)
        self.info(f"🚀 实验: {config['exp_name']} | "
                  f"模型: {config['model_name']} ({model_params:.1f}M, {model_size_mb:.1f}MB) | "
                  f"数据: {config['dataset_type']}")
        
        # 第2行：训练配置
        train_size = config.get('train_size', 0)
        test_size = config.get('test_size', 0)
        data_pct = config.get('data_percentage', 1.0)
        self.info(f"📊 数据: 训练{train_size:,} | 测试{test_size:,} | 使用{data_pct:.0%} | "
                  f"配置: {config['epochs']}ep×bs{config['batch_size']}×lr{config['learning_rate']}")
        
        # 第3行：优化器和调度器
        scheduler_info = config.get('scheduler_info', 'default')
        optimizer_name = config.get('optimizer_name', 'adam')
        weight_decay = config.get('weight_decay', 0)
        self.info(f"⚙️  优化: {optimizer_name}(wd={weight_decay}) | 调度: {scheduler_info} | "
                  f"多卡: {'是' if self.accelerator.num_processes > 1 else '否'}")
        
        self.info("═" * 80)
    
    def _print_detailed_config(self, config: Dict[str, Any]):
        """打印详细的实验配置（原有的完整打印）
        
        Args:
            config: 实验配置字典
        """
        if not self.accelerator.is_main_process:
            return
        
        self.info(f"🚀 ========== 训练实验开始 ==========")
        self.info(f"📋 实验配置:")
        self.info(f"  └─ 实验名称: {config['exp_name']}")
        self.info(f"  └─ 任务类型: {config.get('task_description', 'Unknown')} ({config['dataset_type'].upper()})")
        
        # 模型信息
        model_params = config.get('model_params_m', 0)
        model_size_mb = config.get('model_size_mb', 0)
        self.info(f"  └─ 模型架构: {config['model_name']} ({model_params:.1f}M参数, {model_size_mb:.1f}MB)")
        
        # 数据配置
        train_size = config.get('train_size', 0)
        test_size = config.get('test_size', 0)
        data_pct = config.get('data_percentage', 1.0)
        self.info(f"  └─ 数据配置: 训练集 {train_size:,} | 测试集 {test_size:,} | 使用比例 {data_pct:.0%}")
        
        # 训练配置
        self.info(f"  └─ 训练配置: {config['epochs']} epochs | batch_size {config['batch_size']} | 初始LR {config['learning_rate']}")
        
        # 调度器信息
        scheduler_info = config.get('scheduler_info', 'default')
        self.info(f"  └─ 调度策略: {scheduler_info}")
        
        # 优化器信息
        optimizer_name = config.get('optimizer_name', 'adam')
        weight_decay = config.get('weight_decay', 0)
        self.info(f"  └─ 优化器配置: {optimizer_name} (weight_decay={weight_decay})")
        self.info(f"  └─ 多卡训练: {'是' if self.accelerator.num_processes > 1 else '否'}")
        
        self.info("═" * 63)
    
    def print_pos_weight_summary(self, total_samples: int, num_classes: int):
        """打印pos_weight计算摘要
        
        Args:
            total_samples: 总样本数
            num_classes: 类别数
        """
        if self.accelerator.is_main_process:
            self.info(f"✅ pos_weight已计算 (基于{total_samples:,}个样本，{num_classes}个类别)")
    
    def print_pos_weight_details(self, pos_weight, pos_counts, neg_counts, 
                                 raw_ratio, scale_factor, class_names=None):
        """打印pos_weight详细信息（DEBUG级别）
        
        Args:
            pos_weight: pos_weight张量
            pos_counts: 正样本计数
            neg_counts: 负样本计数
            raw_ratio: 原始比例
            scale_factor: 缩放因子
            class_names: 类别名称列表
        """
        if not self.verbose or not self.accelerator.is_main_process:
            return
        
        num_classes = len(pos_weight)
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f"类别{i}"
            scale = scale_factor[i].item() if hasattr(scale_factor, 'item') else scale_factor
            self.debug(f"   {class_name}: pos={int(pos_counts[i])}, neg={int(neg_counts[i])}, "
                      f"ratio={raw_ratio[i]:.2f}, scale={scale:.1f}, pos_weight={pos_weight[i]:.2f}")


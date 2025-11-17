"""训练日志管理器

统一管理训练过程中的所有终端输出和日志记录。
"""

from typing import Dict, Any, Optional
from accelerate import Accelerator
from tqdm import tqdm


class TrainingLogger:
    """统一的训练日志管理器

    负责管理训练过程中的所有终端输出。

    特性：
    - 自动处理 accelerator.is_main_process 检查
    - 统一的日志格式
    - 简洁清晰的输出
    """

    def __init__(self, accelerator: Accelerator):
        """初始化日志管理器

        Args:
            accelerator: Accelerator实例
        """
        self.accelerator = accelerator
    
    def info(self, message: str, force: bool = False):
        """打印日志消息

        Args:
            message: 日志消息
            force: 是否强制打印（忽略主进程检查）
        """
        if force or self.accelerator.is_main_process:
            tqdm.write(message)
    
    def print_experiment_config(self, config: Dict[str, Any]):
        """打印实验配置（简洁模式）

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
    
    def print_pos_weight_summary(self, total_samples: int, num_classes: int):
        """打印pos_weight计算摘要

        Args:
            total_samples: 总样本数
            num_classes: 类别数
        """
        if self.accelerator.is_main_process:
            self.info(f"✅ pos_weight已计算 (基于{total_samples:,}个样本，{num_classes}个类别)")

    def print_learning_rate_info(self, lr_info: Dict[str, Any], epoch: int,
                                 total_epochs: int, phase: str = "开始"):
        """打印学习率信息

        Args:
            lr_info: 学习率信息字典
            epoch: 当前epoch
            total_epochs: 总epoch数
            phase: 阶段描述（"开始" 或 "结束"）
        """
        if self.accelerator.is_main_process:
            self.info(f"📊 Epoch {epoch}/{total_epochs} {phase} | "
                     f"调度策略: {lr_info['scheduler_name']} | "
                     f"初始LR: {lr_info['initial_lr']:.6f} | "
                     f"当前LR: {lr_info['current_lr']:.6f}")

    def print_experiment_info_full(self, config: Dict[str, Any], exp_name: str,
                                   task_info: Dict[str, Any], dataset_info: Dict[str, Any],
                                   model, train_dataloader, test_dataloader):
        """打印完整的实验配置信息

        负责打印完整的实验配置信息，包括模型、数据、训练配置等。

        Args:
            config: 完整配置字典
            exp_name: 实验名称
            task_info: 任务信息字典
            dataset_info: 数据集信息字典
            model: 已创建的模型
            train_dataloader: 训练数据加载器
            test_dataloader: 测试数据加载器
        """
        if not self.accelerator.is_main_process:
            return

        hyperparams = config['hp']
        data_config = config.get('data', {})
        model_config = config.get('model', {})
        dataset_type = data_config.get('type', 'cifar10')
        model_name = model_config.get('type', model_config.get('name', task_info['default_model']))

        # 获取模型参数信息
        total_params = sum(p.numel() for p in model.parameters())
        model_size_bytes_per_param = 4  # float32
        bytes_to_mb = 1024 * 1024
        model_size_mb = total_params * model_size_bytes_per_param / bytes_to_mb

        # 调度器信息
        scheduler_config = config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'default')
        scheduler_params = []
        if scheduler_name == 'warmup_cosine':
            warmup_epochs = scheduler_config.get('params', {}).get('warmup_epochs', 1)
            eta_min_factor = scheduler_config.get('params', {}).get('eta_min_factor', 0.01)
            scheduler_params.append(f"warmup_epochs={warmup_epochs}")
            scheduler_params.append(f"eta_min_factor={eta_min_factor}")

        scheduler_info = f"{scheduler_name}"
        if scheduler_params:
            scheduler_info += f" ({', '.join(scheduler_params)})"

        # 优化器信息
        optimizer_name = config.get('optimizer', {}).get('name', 'adam')
        weight_decay = config.get('optimizer', {}).get('params', {}).get('weight_decay', 0)

        # 构建配置字典
        config_dict = {
            'exp_name': exp_name,
            'model_name': model_name,
            'dataset_type': dataset_type,
            'task_description': task_info['description'],
            'model_params_m': total_params / 1e6,
            'model_size_mb': model_size_mb,
            'train_size': len(train_dataloader.dataset),
            'test_size': len(test_dataloader.dataset),
            'data_percentage': hyperparams.get('data_percentage', 1.0),
            'epochs': hyperparams['epochs'],
            'batch_size': hyperparams['batch_size'],
            'learning_rate': hyperparams['learning_rate'],
            'scheduler_info': scheduler_info,
            'optimizer_name': optimizer_name,
            'weight_decay': weight_decay
        }

        # 使用日志管理器打印（根据模式选择详细程度）
        self.print_experiment_config(config_dict)


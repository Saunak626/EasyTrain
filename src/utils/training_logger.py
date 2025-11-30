"""训练日志管理器

统一管理训练过程中的终端输出。
"""

from typing import Dict, Any
from accelerate import Accelerator
from tqdm import tqdm


class TrainingLogger:
    """训练日志管理器"""

    def __init__(self, accelerator: Accelerator):
        self.accelerator = accelerator

    def info(self, message: str, force: bool = False):
        """打印日志消息"""
        if force or self.accelerator.is_main_process:
            tqdm.write(message)

    def print_experiment_info(self, config: Dict[str, Any], exp_name: str,
                              task_info: Dict[str, Any], dataset_info: Dict[str, Any],
                              model, train_dataloader, test_dataloader):
        """打印实验信息"""
        if not self.accelerator.is_main_process:
            return

        hyperparams = config['hp']
        data_config = config.get('data', {})
        model_config = config.get('model', {})
        dataset_type = data_config.get('type', 'cifar10')
        model_name = model_config.get('type', model_config.get('name', task_info['default_model']))

        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)

        train_size = len(train_dataloader.dataset)
        test_size = len(test_dataloader.dataset)

        self.info(f"实验: {exp_name} | 模型: {model_name} ({total_params/1e6:.1f}M, {model_size_mb:.1f}MB)")
        self.info(f"数据: {dataset_type} | 训练:{train_size:,} 测试:{test_size:,}")
        self.info(f"配置: {hyperparams['epochs']}ep x bs{hyperparams['batch_size']} x lr{hyperparams['learning_rate']}")
        self.info("-" * 60)

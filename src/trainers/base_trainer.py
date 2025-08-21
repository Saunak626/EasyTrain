"""
基础训练器模块

提供统一的训练接口，支持图像和视频分类任务。
集成Accelerate库实现多GPU训练和SwanLab实验追踪。
"""

import os
import sys
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from tqdm import tqdm
from accelerate import Accelerator

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入项目内部模块
from src.models.image_net import get_model                     # 图像模型工厂函数
from src.models.video_net import get_video_model               # 视频模型工厂函数
from src.losses.loss_factory import get_loss_function         # 损失函数工厂函数
from src.optimizers.optimizer_factory import get_optimizer    # 优化器工厂函数
from src.schedules.scheduler_factory import get_scheduler     # 学习率调度器工厂函数
from src.datasets import create_dataloaders, get_dataset_info  # 统一数据加载器工厂
from src.utils.data_utils import set_seed

# ============================================================================
# 模块级常量配置
# ============================================================================

# 训练相关常量
TRAINING_CONSTANTS = {
    'default_seed': 42,
    'default_num_workers': 8,
    'progress_update_interval': 10,
    'model_size_bytes_per_param': 4,  # float32
    'bytes_to_mb': 1024 * 1024
}

# 支持的任务类型配置
SUPPORTED_TASKS = {
    'image_classification': {
        'description': '图像分类任务',
        'supported_datasets': ['cifar10', 'custom'],
        'model_factory': 'get_model',
        'default_model': 'resnet18'
    },
    'video_classification': {
        'description': '视频分类任务',
        'supported_datasets': ['ucf101', 'ucf101_video', 'neonatal_multilabel'],
        'model_factory': 'get_video_model',
        'default_model': 'r3d_18'
    }
}

# ============================================================================
# 进度条管理类
# ============================================================================

class ProgressBarManager:
    """统一的进度条管理器

    负责创建和管理训练、测试阶段的进度条，避免重复的进度条创建逻辑。
    """

    def __init__(self, accelerator: Accelerator):
        """初始化进度条管理器

        Args:
            accelerator: Accelerator实例，用于检查是否为主进程
        """
        self.accelerator = accelerator

    def create_training_progress_bar(self, dataloader, epoch: int) -> Optional[tqdm]:
        """创建训练进度条

        Args:
            dataloader: 训练数据加载器
            epoch: 当前epoch编号

        Returns:
            进度条实例，如果不是主进程则返回None
        """
        if self.accelerator.is_main_process:
            return tqdm(
                total=len(dataloader),
                desc=f"Epoch {epoch} Training",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
        return None

    def create_testing_progress_bar(self, dataloader, epoch: int) -> Optional[tqdm]:
        """创建测试进度条

        Args:
            dataloader: 测试数据加载器
            epoch: 当前epoch编号

        Returns:
            进度条实例，如果不是主进程则返回None
        """
        if self.accelerator.is_main_process:
            return tqdm(
                total=len(dataloader),
                desc=f"Epoch {epoch} Testing",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
        return None


# ============================================================================
# 辅助函数
# ============================================================================

def is_main_process() -> bool:
    """检查是否为主进程（用于避免重复输出）"""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def get_learning_rate_info(optimizer, lr_scheduler, scheduler_config, initial_lr):
    """获取学习率监控信息

    Args:
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        scheduler_config: 调度器配置
        initial_lr: 初始学习率

    Returns:
        dict: 包含学习率信息的字典
    """
    current_lr = optimizer.param_groups[0]['lr']
    scheduler_name = scheduler_config.get('name', 'default')

    return {
        'initial_lr': initial_lr,
        'current_lr': current_lr,
        'scheduler_name': scheduler_name
    }


def print_learning_rate_info(lr_info, epoch, total_epochs, phase="开始"):
    """打印学习率信息

    Args:
        lr_info: 学习率信息字典
        epoch: 当前epoch
        total_epochs: 总epoch数
        phase: 阶段描述（"开始" 或 "结束"）
    """
    print(f"📊 Epoch {epoch}/{total_epochs} {phase} | "
          f"调度策略: {lr_info['scheduler_name']} | "
          f"初始LR: {lr_info['initial_lr']:.6f} | "
          f"当前LR: {lr_info['current_lr']:.6f}")


def train_epoch(dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch):
    """执行单个训练轮次

    Args:
        dataloader: 训练数据加载器
        model: 神经网络模型
        loss_fn: 损失函数
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        accelerator: Accelerator实例
        epoch: 当前轮次编号

    Returns:
        float: 平均训练损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # 使用统一的进度条管理器
    progress_manager = ProgressBarManager(accelerator)
    progress_bar = progress_manager.create_training_progress_bar(dataloader, epoch)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        accelerator.log({"train/loss": loss.item(), "epoch_num": epoch})

        # 更新进度条
        if progress_bar and batch_idx % TRAINING_CONSTANTS['progress_update_interval'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{current_lr:.2e}"
            )

        if progress_bar:
            progress_bar.update(1)

    # 关闭进度条
    if progress_bar:
        progress_bar.close()

    # 返回平均训练损失
    avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_train_loss


def test_epoch(dataloader, model, loss_fn, accelerator, epoch, train_batches=None):
    """
    执行单个测试轮次

    该函数在测试集上评估模型性能，计算平均损失和准确率。
    支持多GPU环境下的指标聚合，确保结果的准确性。

    Args:
        dataloader (torch.utils.data.DataLoader): 测试数据加载器，提供测试批次数据
        model (torch.nn.Module): 神经网络模型
        loss_fn (torch.nn.Module): 损失函数，用于计算测试损失
        accelerator (accelerate.Accelerator): Accelerator实例，处理多GPU指标聚合
        epoch (int): 当前测试轮次编号

    Returns:
        tuple: (平均损失, 准确率百分比) 或 (None, None) 如果不是主进程
    """
    # 设置模型为评估模式，禁用dropout和batch normalization的训练行为
    model.eval()
    device = accelerator.device

    # 初始化累计指标张量，用于跨GPU聚合
    local_loss_sum = torch.tensor(0.0, device=device)  # 当前GPU的总损失
    local_correct = torch.tensor(0, device=device)     # 当前GPU的正确预测数
    local_samples = torch.tensor(0, device=device)     # 当前GPU的样本总数

    # 使用统一的进度条管理器
    progress_manager = ProgressBarManager(accelerator)
    progress_bar = progress_manager.create_testing_progress_bar(dataloader, epoch)

    # 禁用梯度计算以节省内存和加速推理
    with torch.no_grad():
        for inputs, targets in dataloader:
            # 前向传播获取预测结果
            outputs = model(inputs)
            # 计算当前批次的损失
            loss = loss_fn(outputs, targets)

            # 计算当前批次的统计信息
            batch_size = targets.size(0)

            # 检查是否为多标签分类（标签维度大于1且包含浮点数）
            is_multilabel = len(targets.shape) > 1 and targets.shape[1] > 1 and targets.dtype == torch.float32

            if is_multilabel:
                # 多标签分类：使用sigmoid + 阈值
                predictions = torch.sigmoid(outputs) > 0.5
                # 计算完全匹配的样本数（所有标签都正确）
                correct = (predictions == targets.bool()).all(dim=1).sum()
            else:
                # 单标签分类：使用argmax
                correct = outputs.argmax(dim=1).eq(targets).sum()

            # 累加到本地统计量（考虑批次大小权重）
            local_loss_sum += loss * batch_size
            local_correct += correct
            local_samples += batch_size

            # 更新进度条
            if progress_bar:
                progress_bar.update(1)

    # 关闭进度条
    if progress_bar:
        progress_bar.close()

    # 跨所有GPU进程聚合统计指标
    total_loss = accelerator.reduce(local_loss_sum, reduction="sum")
    total_correct = accelerator.reduce(local_correct, reduction="sum")
    total_samples = accelerator.reduce(local_samples, reduction="sum")

    # 只在主进程计算最终指标并记录
    if accelerator.is_main_process:
        # 计算平均损失和准确率
        avg_loss = (total_loss / total_samples).item()
        accuracy = 100. * total_correct.item() / total_samples.item()

        # 使用tqdm.write()输出摘要，不破坏进度条显示
        log_msg = f'Epoch {epoch:03d} | val_loss={avg_loss:.4f} | val_acc={accuracy:.2f}%'
        if train_batches is not None:
            log_msg += f' | train_batches={train_batches}'
        tqdm.write(log_msg)

        # 记录测试指标到实验追踪系统
        accelerator.log({"test/loss": avg_loss, "test/accuracy": accuracy}, step=epoch)

        # GPU监控功能已移除

        return avg_loss, accuracy

    # 非主进程返回None
    return None, None


# ============================================================================
# 训练流程拆分函数
# ============================================================================

def setup_experiment(config: Dict[str, Any], exp_name: Optional[str] = None) -> Tuple[str, Dict[str, Any], str, Dict[str, Any], Accelerator]:
    """实验环境初始化

    负责设置随机种子、解析任务配置、验证数据集兼容性，并初始化Accelerator和SwanLab追踪。

    Args:
        config: 包含所有训练配置的字典
        exp_name: 实验名称，用于追踪和日志记录

    Returns:
        Tuple[实验名称, 任务信息, 任务标签, 数据配置, Accelerator实例]

    Raises:
        ValueError: 当任务类型不支持或数据集不兼容时
    """
    # 设置随机种子确保实验可重现性
    set_seed(TRAINING_CONSTANTS['default_seed'])

    # 实验名称，优先使用传入函数的参数
    if exp_name is None:
        exp_name = config['training']['exp_name']

    # 解析任务配置
    task_config = config.get('task', {})
    task_tag = task_config.get('tag')

    # 验证任务类型必须明确指定
    if not task_tag:
        raise ValueError(f"必须在配置文件中明确指定task.tag。支持的任务类型: {list(SUPPORTED_TASKS.keys())}")

    if task_tag not in SUPPORTED_TASKS:
        raise ValueError(f"不支持的任务类型: {task_tag}。支持的任务: {list(SUPPORTED_TASKS.keys())}")

    task_info = SUPPORTED_TASKS[task_tag]

    # 解析和验证数据配置
    data_config = config.get('data', {})
    dataset_type = data_config.get('type', 'cifar10')

    # 验证数据集与任务的兼容性
    if dataset_type not in task_info['supported_datasets']:
        raise ValueError(f"任务 '{task_tag}' 不支持数据集 '{dataset_type}'。"
                        f"支持的数据集: {task_info['supported_datasets']}")

    # 初始化Accelerator，指定swanlab为日志记录工具
    accelerator = Accelerator(log_with="swanlab")

    # 记录到SwanLab的超参数
    hyperparams = config['hp']
    tracker_config = {**hyperparams, "exp_name": exp_name, "task_tag": task_tag}

    # 初始化SwanLab实验追踪器
    accelerator.init_trackers(
        project_name=config['swanlab']['project_name'],  # SwanLab UI中项目名称
        config=tracker_config,    # 要记录的超参数
        init_kwargs={             # 额外初始化参数
            "swanlab": {
                "exp_name": exp_name,
                "description": config['swanlab']['description']
            }
        }
    )

    return exp_name, task_info, task_tag, data_config, accelerator


def setup_data_and_model(config: Dict[str, Any], task_info: Dict[str, Any], data_config: Dict[str, Any], accelerator: Accelerator) -> Tuple:
    """数据和模型初始化

    负责创建数据加载器、获取数据集信息、创建模型。

    Args:
        config: 完整配置字典
        task_info: 任务信息字典
        data_config: 数据配置字典
        accelerator: Accelerator实例

    Returns:
        Tuple[训练数据加载器, 测试数据加载器, 模型, 数据集信息]
    """
    # 获取超参数和模型配置
    hyperparams = config['hp']
    model_config = config.get('model', {})
    dataset_type = data_config.get('type', 'cifar10')
    model_name = model_config.get('type', model_config.get('name', task_info['default_model']))

    # 使用简化的数据加载器创建函数
    train_dataloader, test_dataloader, num_classes = create_dataloaders(
        dataset_name=dataset_type,
        data_dir=data_config.get('root', './data'),
        batch_size=hyperparams['batch_size'],
        num_workers=data_config.get('num_workers', TRAINING_CONSTANTS['default_num_workers']),
        model_type=model_name,  # 传递模型类型用于动态transforms
        data_percentage=hyperparams.get('data_percentage', 1.0),
        **data_config.get('params', {})
    )

    # 获取数据集信息
    dataset_info = get_dataset_info(dataset_type)
    dataset_info['num_classes'] = num_classes or dataset_info['num_classes']

    # 基于任务类型创建模型
    model_factory_name = task_info['model_factory']
    model_factory = globals()[model_factory_name]

    # 统一的模型创建逻辑
    model_params = model_config.get('params', {}).copy()
    model_params['num_classes'] = dataset_info['num_classes']

    model = model_factory(
        model_type=model_name,
        **model_params
    )

    return train_dataloader, test_dataloader, model, dataset_info


def setup_training_components(config: Dict[str, Any], model, train_dataloader, accelerator: Accelerator) -> Tuple:
    """优化器、调度器、损失函数初始化

    负责创建损失函数、优化器和学习率调度器，并使用Accelerator包装所有组件。

    Args:
        config: 完整配置字典
        model: 已创建的模型
        train_dataloader: 训练数据加载器
        accelerator: Accelerator实例

    Returns:
        Tuple[损失函数, 优化器, 学习率调度器]
    """
    hyperparams = config['hp']

    # 创建损失函数 - 使用工厂函数
    loss_fn = get_loss_function(config.get('loss', {}))

    # 创建优化器 - 使用工厂函数
    optimizer = get_optimizer(model, config.get('optimizer', {}), hyperparams['learning_rate'])

    # 创建学习率调度器 - 使用工厂函数
    # 需要传递steps_per_epoch给调度器
    scheduler_config = config.get('scheduler', {}).copy()
    if 'steps_per_epoch' not in scheduler_config:
        scheduler_config['steps_per_epoch'] = len(train_dataloader)

    lr_scheduler = get_scheduler(optimizer, scheduler_config, hyperparams)

    return loss_fn, optimizer, lr_scheduler


def print_experiment_info(config: Dict[str, Any], exp_name: str, task_info: Dict[str, Any],
                         dataset_info: Dict[str, Any], model, train_dataloader, test_dataloader,
                         accelerator: Accelerator) -> None:
    """实验信息打印

    负责打印完整的实验配置信息，包括模型、数据、训练配置等。

    Args:
        config: 完整配置字典
        exp_name: 实验名称
        task_info: 任务信息字典
        dataset_info: 数据集信息字典
        model: 已创建的模型
        train_dataloader: 训练数据加载器
        test_dataloader: 测试数据加载器
        accelerator: Accelerator实例
    """
    if not (accelerator.is_main_process and is_main_process()):
        return

    hyperparams = config['hp']
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    dataset_type = data_config.get('type', 'cifar10')
    model_name = model_config.get('type', model_config.get('name', task_info['default_model']))

    print(f"🚀 ========== 训练实验开始 ==========")
    print(f"📋 实验配置:")
    print(f"  └─ 实验名称: {exp_name}")
    print(f"  └─ 任务类型: {task_info['description']} ({dataset_type.upper()})")

    # 获取模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * TRAINING_CONSTANTS['model_size_bytes_per_param'] / TRAINING_CONSTANTS['bytes_to_mb']

    print(f"  └─ 模型架构: {model_name} ({total_params/1e6:.1f}M参数, {model_size_mb:.1f}MB)")
    print(f"  └─ 数据配置: 训练集 {len(train_dataloader.dataset):,} | 测试集 {len(test_dataloader.dataset):,} | 使用比例 {hyperparams.get('data_percentage', 1.0):.0%}")
    print(f"  └─ 训练配置: {hyperparams['epochs']} epochs | batch_size {hyperparams['batch_size']} | 初始LR {hyperparams['learning_rate']}")

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
    print(f"  └─ 调度策略: {scheduler_info}")

    # 优化器信息
    optimizer_name = config.get('optimizer', {}).get('name', 'adam')
    weight_decay = config.get('optimizer', {}).get('params', {}).get('weight_decay', 0)
    print(f"  └─ 优化器配置: {optimizer_name} (weight_decay={weight_decay})")
    print(f"  └─ 多卡训练: {'是' if accelerator.num_processes > 1 else '否'}")

    print("═" * 63)


def run_training_loop(config: Dict[str, Any], model, optimizer, lr_scheduler, loss_fn,
                     train_dataloader, test_dataloader, accelerator: Accelerator) -> Tuple[float, float, int]:
    """主训练循环

    负责执行完整的训练循环，包括训练和测试阶段。

    Args:
        config: 完整配置字典
        model: 已准备的模型
        optimizer: 已准备的优化器
        lr_scheduler: 已准备的学习率调度器
        loss_fn: 损失函数
        train_dataloader: 已准备的训练数据加载器
        test_dataloader: 已准备的测试数据加载器
        accelerator: Accelerator实例

    Returns:
        Tuple[最佳准确率, 最终准确率, 训练轮数]
    """
    hyperparams = config['hp']
    scheduler_config = config.get('scheduler', {})
    initial_lr = hyperparams['learning_rate']

    # 初始化最佳准确率追踪
    best_accuracy = 0.0
    trained_epochs = 0
    val_accuracy = 0.0

    # 主训练循环：执行指定轮数的训练
    for epoch in range(1, hyperparams['epochs'] + 1):
        if accelerator.is_main_process:
            # 打印epoch开始时的学习率信息
            lr_info = get_learning_rate_info(optimizer, lr_scheduler, scheduler_config, initial_lr)
            print_learning_rate_info(lr_info, epoch, hyperparams['epochs'], "开始")

        # 训练epoch
        train_epoch(train_dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch)
        # 测试epoch
        _, val_accuracy = test_epoch(test_dataloader, model, loss_fn, accelerator, epoch, train_batches=len(train_dataloader))

        # 打印epoch结束时的学习率信息
        if accelerator.is_main_process:
            lr_info = get_learning_rate_info(optimizer, lr_scheduler, scheduler_config, initial_lr)
            print_learning_rate_info(lr_info, epoch, hyperparams['epochs'], "结束")

        # 更新并记录最佳准确率
        if accelerator.is_main_process and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            tqdm.write(f"新最佳准确率: {best_accuracy:.2f}%")

        # 记录完成的训练轮数
        trained_epochs = epoch

    return best_accuracy, val_accuracy, trained_epochs


def cleanup_and_return(accelerator: Accelerator, exp_name: str, best_accuracy: float,
                      val_accuracy: float, trained_epochs: int, tracker_config: Dict[str, Any]) -> Dict[str, Any]:
    """清理和结果返回

    负责结束实验追踪、清理GPU缓存并返回训练结果。

    Args:
        accelerator: Accelerator实例
        exp_name: 实验名称
        best_accuracy: 最佳准确率
        val_accuracy: 最终准确率
        trained_epochs: 训练轮数
        tracker_config: 追踪配置

    Returns:
        训练结果字典
    """
    # 结束实验追踪，保存日志和结果
    accelerator.end_training()

    # 输出训练完成信息
    if accelerator.is_main_process:
        tqdm.write(f"训练完成! 最佳准确率: {best_accuracy:.2f}%")

    # 清理GPU缓存，为下一个实验释放资源
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 返回训练结果摘要
    return {
        "success": True,                       # 训练成功标志
        "exp_name": exp_name,                  # 实验名称
        "best_accuracy": best_accuracy,        # 最佳测试准确率
        "final_accuracy": val_accuracy,        # 最终准确率
        "trained_epochs": trained_epochs,      # 实际训练轮数
        "config": tracker_config               # 完整的训练配置
    }


def run_training(config: Dict[str, Any], exp_name: Optional[str] = None) -> Dict[str, Any]:
    """
    训练的主入口函数，负责整个训练过程的协调，包括：
    - 环境初始化（随机种子、实验追踪）
    - 数据加载器创建
    - 模型、损失函数、优化器初始化
    - 多GPU环境配置
    - 训练循环执行
    - 结果记录和返回

    Args:
        config: 包含所有训练配置的字典，包括模型、数据、超参数等设置
        exp_name: 实验名称，用于追踪和日志记录

    Returns:
        训练结果字典，包含实验名称、最佳准确率和配置信息
    """
    # 第1步：实验环境初始化
    exp_name, task_info, task_tag, data_config, accelerator = setup_experiment(config, exp_name)

    # 第2步：数据和模型初始化
    train_dataloader, test_dataloader, model, dataset_info = setup_data_and_model(config, task_info, data_config, accelerator)

    # 第3步：训练组件初始化
    loss_fn, optimizer, lr_scheduler = setup_training_components(config, model, train_dataloader, accelerator)

    # 清理GPU缓存，释放未使用的内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 使用Accelerator包装训练组件，自动处理分布式训练
    model, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )

    # 第4步：打印实验信息
    print_experiment_info(config, exp_name, task_info, dataset_info, model, train_dataloader, test_dataloader, accelerator)

    # 第5步：执行训练循环
    best_accuracy, val_accuracy, trained_epochs = run_training_loop(
        config, model, optimizer, lr_scheduler, loss_fn, train_dataloader, test_dataloader, accelerator
    )

    # 第6步：清理和返回结果
    hyperparams = config['hp']
    tracker_config = {**hyperparams, "exp_name": exp_name, "task_tag": task_tag}

    return cleanup_and_return(accelerator, exp_name, best_accuracy, val_accuracy, trained_epochs, tracker_config)
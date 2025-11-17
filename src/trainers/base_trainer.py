"""
基础训练器模块

提供统一的训练接口，支持图像和视频分类任务。
集成Accelerate库实现多GPU训练和SwanLab实验追踪。
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

from tqdm import tqdm
from accelerate import Accelerator

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入项目内部模块
from src.models.image_net import get_model                     # 图像模型工厂函数
from src.models.video_net import get_video_model               # 视频模型工厂函数
from src.losses.loss_factory import get_loss_function          # 损失函数工厂函数
from src.optimizers.optimizer_factory import get_optimizer     # 优化器工厂函数
from src.schedules.scheduler_factory import get_scheduler      # 学习率调度器工厂函数
from src.datasets import create_dataloaders, get_dataset_info  # 统一数据加载器工厂
from src.utils.data_utils import set_seed
from src.utils.training_logger import TrainingLogger           # 训练日志管理器
from src.utils.dataset_utils import unwrap_subset_dataset, get_dataset_metadata  # 数据集工具函数
from src.utils.training_utils import log_multilabel_metrics_to_swanlab, get_learning_rate_info  # 训练工具函数

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
# 核心训练函数
# ============================================================================

def train_epoch(dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch,
                metrics_calculator=None, scheduler_step_interval='batch'):
    """执行单个训练轮次

    Args:
        dataloader: 训练数据加载器
        model: 神经网络模型
        loss_fn: 损失函数
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        accelerator: Accelerator实例
        epoch: 当前轮次编号
        metrics_calculator: 多标签指标计算器（可选）

    Returns:
        tuple: (平均训练损失, 训练准确率)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # 🔧 新增：收集训练数据用于指标计算
    collected_outputs = []
    collected_targets = []
    is_multilabel = metrics_calculator is not None

    # 使用统一的进度条管理器
    progress_manager = ProgressBarManager(accelerator)
    progress_bar = progress_manager.create_training_progress_bar(dataloader, epoch)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None and scheduler_step_interval == 'batch':
            lr_scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # 🔧 新增：收集预测和目标数据（用于训练集指标计算）
        if is_multilabel:
            collected_outputs.append(outputs.detach())
            collected_targets.append(targets.detach())

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

    # 计算平均训练损失
    avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if lr_scheduler is not None and scheduler_step_interval == 'epoch':
        lr_scheduler.step()

    # 🔧 新增：计算训练集指标（如果是多标签任务）
    train_accuracy = 0.0
    if is_multilabel and collected_outputs:
        stacked_outputs = torch.cat(collected_outputs, dim=0)
        stacked_targets = torch.cat(collected_targets, dim=0)

        gathered_outputs = accelerator.gather_for_metrics(stacked_outputs)
        gathered_targets = accelerator.gather_for_metrics(stacked_targets)

        if accelerator.is_main_process:
            probs = torch.sigmoid(gathered_outputs).cpu().numpy()
            targets_np = gathered_targets.cpu().numpy()

            # 计算训练集详细指标
            train_metrics = metrics_calculator.calculate_detailed_metrics(
                probs, targets_np, threshold=0.5
            )

            # 保存训练集指标到单独的CSV文件
            metrics_calculator.save_train_metrics(train_metrics, epoch, avg_train_loss)

            # 记录训练集指标到SwanLab（使用统一的辅助函数）
            log_multilabel_metrics_to_swanlab(accelerator, train_metrics, 'train', epoch)

            train_accuracy = train_metrics['macro_avg']['accuracy']

    return avg_train_loss, train_accuracy


def test_epoch(dataloader, model, loss_fn, accelerator, epoch, train_batches=None,
               metrics_calculator=None):
    """
    执行单个测试轮次

    该函数在测试集上评估模型性能，计算平均损失和准确率。
    支持多GPU环境下的指标聚合，确保结果的准确性。
    支持详细的多标签分类指标计算和报告。

    Args:
        dataloader (torch.utils.data.DataLoader): 测试数据加载器，提供测试批次数据
        model (torch.nn.Module): 神经网络模型
        loss_fn (torch.nn.Module): 损失函数，用于计算测试损失
        accelerator (accelerate.Accelerator): Accelerator实例，处理多GPU指标聚合
        epoch (int): 当前测试轮次编号
        train_batches (int, optional): 训练批次数，用于日志显示
        metrics_calculator (MultilabelMetricsCalculator, optional): 多标签指标计算器

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

    # 用于详细多标签评估的数据收集
    all_predictions = []
    all_targets = []
    is_multilabel = False

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
                # 多标签分类：使用每类别平均准确率
                sigmoid_outputs = torch.sigmoid(outputs)
                predictions = sigmoid_outputs > 0.5
                targets_bool = targets.bool()

                # 收集预测和目标数据用于详细评估
                if metrics_calculator is not None:
                    # 收集sigmoid概率和真实标签
                    all_predictions.append(sigmoid_outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                # 计算每个类别的准确率，然后平均（宏平均）
                class_accuracies = (predictions == targets_bool).float().mean(dim=0)
                macro_accuracy = class_accuracies.mean()
                # 转换为正确样本数（用于兼容现有统计逻辑），确保类型为Long
                correct = (macro_accuracy * batch_size).long()
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

        # 如果有多标签指标计算器且收集了数据，进行详细评估
        if metrics_calculator is not None and all_predictions and is_multilabel:
            # 合并所有批次的预测和目标
            all_pred_array = np.concatenate(all_predictions, axis=0)
            all_target_array = np.concatenate(all_targets, axis=0)

            # 计算详细指标
            detailed_metrics = metrics_calculator.calculate_detailed_metrics(
                all_pred_array, all_target_array, threshold=0.5
            )

            # 更新最佳指标（传递预测和目标数组用于视频级别报告）
            is_best = metrics_calculator.update_best_metrics(
                detailed_metrics, epoch,
                predictions=all_pred_array,
                targets=all_target_array
            )

            # 保存指标（保持原有功能）
            metrics_calculator.save_metrics(detailed_metrics, epoch, avg_loss, is_best)

            # 🔧 新增：保存测试集指标到单独的CSV文件
            metrics_calculator.save_test_metrics(detailed_metrics, epoch, avg_loss)

            # 显示详细指标
            detailed_display = metrics_calculator.format_metrics_display(
                detailed_metrics, epoch, avg_loss, train_batches or 0
            )
            tqdm.write(detailed_display)

            if is_best:
                tqdm.write(f"🏆 新最佳宏平均F1分数: {detailed_metrics['macro_avg']['f1']:.4f}")

            # 记录详细指标到实验追踪系统（使用统一的辅助函数）
            accelerator.log({"test/loss": avg_loss}, step=epoch)
            log_multilabel_metrics_to_swanlab(accelerator, detailed_metrics, 'test', epoch)
        else:
            # 标准输出（单标签或无详细评估）
            log_msg = f'Epoch {epoch:03d} | val_loss={avg_loss:.4f} | val_acc={accuracy:.2f}%'
            if train_batches is not None:
                log_msg += f' | train_batches={train_batches}'
            tqdm.write(log_msg)

            # 记录测试指标到实验追踪系统
            accelerator.log({"test/loss": avg_loss, "test/accuracy": accuracy}, step=epoch)

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

    # 🔧 使用统一的元数据获取函数（支持Subset包装的数据集）
    # 从实际数据集实例获取类别名称、类别数量和多标签标志
    metadata = get_dataset_metadata(train_dataloader.dataset, dataset_type)

    # 更新 dataset_info（优先使用从数据集获取的元数据）
    if metadata['num_classes'] is not None:
        dataset_info['num_classes'] = metadata['num_classes']
    if metadata['classes'] is not None:
        dataset_info['classes'] = metadata['classes']
    dataset_info['is_multilabel'] = metadata['is_multilabel']

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


def setup_training_components(config: Dict[str, Any], model, train_dataloader,
                             accelerator: Accelerator, logger: TrainingLogger,
                             dataset_info: Dict[str, Any]) -> Tuple:
    """优化器、调度器、损失函数初始化

    负责创建损失函数、优化器和学习率调度器，并使用Accelerator包装所有组件。

    Args:
        config: 完整配置字典
        model: 已创建的模型
        train_dataloader: 训练数据加载器
        accelerator: Accelerator实例
        logger: 训练日志管理器
        dataset_info: 数据集信息字典（包含 num_classes, classes, is_multilabel 等）

    Returns:
        Tuple[损失函数, 优化器, 学习率调度器]
    """
    hyperparams = config['hp']

    # 创建损失函数 - 使用工厂函数，传递类别数量信息
    loss_config = config.get('loss', {}).copy()

    # 🔧 修复：为所有多标签损失函数添加类别数量信息
    multilabel_loss_types = [
        'multilabel_bce',
        'focal_multilabel_bce',
        'focal_multilabel_balanced',
        'multilabel_focal_balanced'
    ]

    loss_name = loss_config.get('name') or loss_config.get('type')
    if loss_name in multilabel_loss_types:
        # 🔧 使用 dataset_info 中的类别数量（避免重复获取）
        num_classes = dataset_info.get('num_classes')

        # 向后兼容：如果 dataset_info 中没有类别数量，从模型配置获取
        if num_classes is None:
            num_classes = config.get('model', {}).get('params', {}).get('num_classes', 24)

        if 'params' not in loss_config:
            loss_config['params'] = {}
        loss_config['params']['num_classes'] = num_classes

        # 🔧 新增：动态计算pos_weight（高优先级修复）
        # 如果配置中指定了pos_weight但是标量值，则根据训练集统计动态计算
        config_pos_weight = loss_config.get('params', {}).get('pos_weight', None)
        if config_pos_weight is not None and isinstance(config_pos_weight, (int, float)):
            #  使用辅助函数解包 Subset 数据集
            dataset = unwrap_subset_dataset(train_dataloader.dataset)

            # 🔧 优化：直接从数据集的samples属性读取标签，避免加载图像数据
            # 收集所有训练样本的标签
            all_labels = []

            # 检查数据集是否有samples属性（NeonatalMultilabelDataset有）
            if hasattr(dataset, 'samples'):
                # 直接从samples读取标签，避免加载图像
                for sample in dataset.samples:
                    labels = sample['labels']
                    if isinstance(labels, list):
                        labels = torch.tensor(labels, dtype=torch.float32)
                    elif not isinstance(labels, torch.Tensor):
                        labels = torch.tensor(labels, dtype=torch.float32)
                    all_labels.append(labels)
            else:
                # 降级方案：遍历DataLoader（较慢）
                for batch_idx, (_, targets) in enumerate(train_dataloader):
                    all_labels.append(targets.cpu())
                    # 只采样部分数据以加快计算（最多1000个batch）
                    if batch_idx >= 1000:
                        break

            if all_labels:
                all_labels = torch.stack(all_labels) if isinstance(all_labels[0], torch.Tensor) and all_labels[0].dim() == 1 else torch.cat(all_labels, dim=0)
                pos_counts = all_labels.sum(dim=0)  # 每个类别的正样本数
                total_samples = all_labels.shape[0]
                neg_counts = total_samples - pos_counts  # 每个类别的负样本数

                # 🔧 优化：使用自适应缩放计算pos_weight，避免极端值
                # 原始公式: pos_weight = neg_samples / pos_samples
                # 问题: 对于极度稀有的类别会产生极大的权重(如51.36)，导致模型过度预测正类
                #
                # 新公式: 自适应缩放策略
                # - 对于轻度不平衡(ratio < 5): pos_weight = sqrt(ratio) * 0.8
                # - 对于中度不平衡(5 <= ratio < 20): pos_weight = sqrt(ratio) * 0.6
                # - 对于极度不平衡(ratio >= 20): pos_weight = sqrt(ratio) * 0.4
                #
                # 优点:
                #   1. 对于极度稀有的类别使用更激进的降权，避免过度预测
                #   2. 对于轻度不平衡的类别保持较高权重，确保学习效果
                #   3. 例如: 发脾气(ratio=51.36) -> sqrt(51.36) * 0.4 ≈ 2.87
                raw_ratio = neg_counts / (pos_counts + 1e-6)

                # 自适应缩放因子
                scale_factor = torch.where(
                    raw_ratio < 5.0,
                    torch.tensor(0.8, device=raw_ratio.device),  # 轻度不平衡
                    torch.where(
                        raw_ratio < 20.0,
                        torch.tensor(0.6, device=raw_ratio.device),  # 中度不平衡
                        torch.tensor(0.4, device=raw_ratio.device)   # 极度不平衡
                    )
                )

                pos_weight = torch.sqrt(raw_ratio) * scale_factor

                # 限制pos_weight的范围，避免极端值
                pos_weight = torch.clamp(pos_weight, min=1.0, max=5.0)

                loss_config['params']['pos_weight'] = pos_weight

                # 打印摘要信息
                logger.print_pos_weight_summary(total_samples, num_classes)

    loss_fn = get_loss_function(loss_config)

    # 创建优化器 - 使用工厂函数
    optimizer = get_optimizer(model, config.get('optimizer', {}), hyperparams['learning_rate'])

    # 创建学习率调度器 - 使用工厂函数
    # 需要传递steps_per_epoch给调度器
    scheduler_config = config.get('scheduler', {}).copy()
    if 'steps_per_epoch' not in scheduler_config:
        scheduler_config['steps_per_epoch'] = len(train_dataloader)

    lr_scheduler = get_scheduler(optimizer, scheduler_config, hyperparams)

    scheduler_name = (scheduler_config.get('name') or scheduler_config.get('type') or '').lower()
    scheduler_step_interval = scheduler_config.get('step_interval')
    if scheduler_step_interval is None:
        scheduler_step_interval = 'batch' if scheduler_name in ['onecycle'] else 'epoch'

    return loss_fn, optimizer, lr_scheduler, scheduler_step_interval


def get_task_output_dir(task_tag: str, dataset_type: str) -> str:
    """根据任务类型获取输出目录

    Args:
        task_tag: 任务标签
        dataset_type: 数据集类型

    Returns:
        任务对应的输出目录路径
    """
    # 基础输出目录
    base_dir = "runs"

    # 根据任务类型确定子目录名
    if 'multilabel' in task_tag.lower() or 'multilabel' in dataset_type.lower():
        if 'neonatal' in dataset_type.lower():
            task_subdir = "neonatal_multilabel"
        else:
            task_subdir = "multilabel_classification"
    elif 'video' in task_tag.lower():
        task_subdir = "video_classification"
    elif 'image' in task_tag.lower():
        task_subdir = "image_classification"
    else:
        # 默认使用数据集类型作为子目录名
        task_subdir = dataset_type.replace('_', '_').lower() or "general"

    output_dir = os.path.join(base_dir, task_subdir)

    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def run_training_loop(config: Dict[str, Any], model, optimizer, lr_scheduler, loss_fn,
                     train_dataloader, test_dataloader, accelerator: Accelerator, logger: TrainingLogger,
                     metrics_calculator=None, scheduler_step_interval='batch') -> Tuple[float, float, int]:
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
        logger: 训练日志管理器
        metrics_calculator: 多标签指标计算器（可选）
        scheduler_step_interval: 调度器步进间隔（'batch' 或 'epoch'）

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

    # metrics_calculator 现在作为参数传入

    # 主训练循环：执行指定轮数的训练
    for epoch in range(1, hyperparams['epochs'] + 1):
        if accelerator.is_main_process:
            # 打印epoch开始时的学习率信息
            lr_info = get_learning_rate_info(optimizer, lr_scheduler, scheduler_config, initial_lr)
            logger.print_learning_rate_info(lr_info, epoch, hyperparams['epochs'], "开始")

        # 训练epoch（传递metrics_calculator用于训练集指标计算）
        train_loss, train_accuracy = train_epoch(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            lr_scheduler,
            accelerator,
            epoch,
            metrics_calculator,
            scheduler_step_interval
        )
        # 测试epoch
        _, val_accuracy = test_epoch(test_dataloader, model, loss_fn, accelerator, epoch,
                                   train_batches=len(train_dataloader),
                                   metrics_calculator=metrics_calculator)

        # 打印epoch结束时的学习率信息
        if accelerator.is_main_process:
            lr_info = get_learning_rate_info(optimizer, lr_scheduler, scheduler_config, initial_lr)
            logger.print_learning_rate_info(lr_info, epoch, hyperparams['epochs'], "结束")

        # 更新并记录最佳准确率
        if accelerator.is_main_process and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            tqdm.write(f"新最佳准确率: {best_accuracy:.2f}%")

        # 记录完成的训练轮数
        trained_epochs = epoch

    # 训练结束后显示总结报告
    if accelerator.is_main_process and metrics_calculator is not None:
        summary_report = metrics_calculator.get_summary_report()
        tqdm.write(summary_report)

    return best_accuracy, val_accuracy, trained_epochs


def cleanup_and_return(accelerator: Accelerator, exp_name: str, best_accuracy: float,
                      val_accuracy: float, trained_epochs: int, tracker_config: Dict[str, Any],
                      metrics_calculator=None) -> Dict[str, Any]:
    """清理和结果返回

    负责结束实验追踪、清理GPU缓存并返回训练结果。

    Args:
        accelerator: Accelerator实例
        exp_name: 实验名称
        best_accuracy: 最佳准确率
        val_accuracy: 最终准确率
        trained_epochs: 训练轮数
        tracker_config: 追踪配置
        metrics_calculator: 多标签指标计算器

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

    # 准备返回结果
    result = {
        "success": True,                       # 训练成功标志
        "exp_name": exp_name,                  # 实验名称
        "best_accuracy": best_accuracy,        # 最佳测试准确率
        "final_accuracy": val_accuracy,        # 最终准确率
        "trained_epochs": trained_epochs,      # 实际训练轮数
        "config": tracker_config               # 完整的训练配置
    }

    # 如果有多标签指标计算器，添加详细的多标签指标
    if metrics_calculator is not None:
        best_metrics = metrics_calculator.best_metrics

        # 检查是否有有效的指标数据（通过检查macro_avg是否为空字典）
        has_valid_metrics = (
            best_metrics.get("macro_avg") and
            isinstance(best_metrics.get("macro_avg"), dict) and
            len(best_metrics.get("macro_avg", {})) > 0
        )

        if has_valid_metrics:
            # 获取最新的指标（最后一次评估的结果）
            latest_metrics = None
            if metrics_calculator.metrics_history:
                latest_metrics = metrics_calculator.metrics_history[-1]

            multilabel_metrics = {
                "best": {
                    "macro_accuracy": best_metrics.get("macro_avg", {}).get("accuracy"),
                    "micro_accuracy": best_metrics.get("micro_avg", {}).get("accuracy"),
                    "weighted_accuracy": best_metrics.get("weighted_avg", {}).get("accuracy"),
                    "macro_f1": best_metrics.get("macro_avg_f1"),
                    "micro_f1": best_metrics.get("micro_avg", {}).get("f1"),
                    "weighted_f1": best_metrics.get("weighted_avg", {}).get("f1"),
                    "macro_precision": best_metrics.get("macro_avg", {}).get("precision"),
                    "macro_recall": best_metrics.get("macro_avg", {}).get("recall"),
                    "epoch": best_metrics.get("epoch")
                }
            }

            # 添加最终指标
            if latest_metrics:
                multilabel_metrics["final"] = {
                    "macro_accuracy": latest_metrics.get("macro_avg", {}).get("accuracy"),
                    "micro_accuracy": latest_metrics.get("micro_avg", {}).get("accuracy"),
                    "weighted_accuracy": latest_metrics.get("weighted_avg", {}).get("accuracy"),
                    "macro_f1": latest_metrics.get("macro_avg", {}).get("f1"),
                    "micro_f1": latest_metrics.get("micro_avg", {}).get("f1"),
                    "weighted_f1": latest_metrics.get("weighted_avg", {}).get("f1"),
                }

            result["multilabel_metrics"] = multilabel_metrics

            # 为网格搜索详情表添加完整的详细指标
            result["detailed_metrics"] = best_metrics
        else:
            # 没有有效指标数据时的警告
            if accelerator.is_main_process:
                tqdm.write("⚠️ 多标签指标计算器未收集到有效数据，可能是训练未正常执行或数据集为空")

    return result


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

    # 创建训练日志管理器
    logger = TrainingLogger(accelerator)

    # 第2步：数据和模型初始化
    train_dataloader, test_dataloader, model, dataset_info = setup_data_and_model(config, task_info, data_config, accelerator)

    # 第3步：训练组件初始化
    loss_fn, optimizer, lr_scheduler, scheduler_step_interval = setup_training_components(
        config, model, train_dataloader, accelerator, logger, dataset_info
    )

    # 清理GPU缓存，释放未使用的内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 使用Accelerator包装训练组件，自动处理分布式训练
    model, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )

    # 第4步：创建多标签指标计算器（如果是多标签任务）
    metrics_calculator = None
    task_config = config.get('task', {})
    task_tag = task_config.get('tag', '')
    dataset_type = config.get('data', {}).get('type', '')

    # 检测多标签任务：通过dataset_type（主要）或task_tag
    is_multilabel_task = ('multilabel' in dataset_type.lower() or
                         'multilabel' in task_tag.lower())

    if is_multilabel_task:
        from src.evaluation import MultilabelMetricsCalculator

        # 从setup_data_and_model返回的dataset_info获取类别名称
        class_names = dataset_info.get('classes', [])

        if class_names:
            # 根据任务类型创建对应的输出目录
            task_dir = get_task_output_dir(task_tag, dataset_type)

            # 优先使用grid_search_dir（如果存在），否则使用默认的task_dir
            output_dir = config.get('grid_search_dir', task_dir)

            # 获取测试数据集（用于视频级别报告）
            test_dataset = test_dataloader.dataset

            # 提取model_type：优先从model.type获取，其次回退到hp.model_type
            model_type = config.get('model', {}).get(
                'type',
                config.get('hp', {}).get('model_type', 'unknown')
            )

            metrics_calculator = MultilabelMetricsCalculator(
                class_names=class_names,
                output_dir=output_dir,
                dataset=test_dataset,
                model_type=model_type,
                exp_name=exp_name
            )
        else:
            if accelerator.is_main_process:
                tqdm.write(f"⚠️ 多标签任务检测成功，但未获取到类别名称")

    # 第5步：打印实验信息
    logger.print_experiment_info_full(config, exp_name, task_info, dataset_info, model, train_dataloader, test_dataloader)

    # 第6步：执行训练循环
    best_accuracy, val_accuracy, trained_epochs = run_training_loop(
        config, model, optimizer, lr_scheduler, loss_fn, train_dataloader, test_dataloader, accelerator, logger, metrics_calculator, scheduler_step_interval
    )

    # 第7步：清理和返回结果
    hyperparams = config['hp']
    tracker_config = {**hyperparams, "exp_name": exp_name, "task_tag": task_tag}

    return cleanup_and_return(accelerator, exp_name, best_accuracy, val_accuracy, trained_epochs, tracker_config, metrics_calculator)

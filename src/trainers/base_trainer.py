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
# 工厂函数内部处理配置解析


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
        'supported_datasets': ['ucf101', 'ucf101_video'],
        'model_factory': 'get_video_model',
        'default_model': 'r3d_18'
    }
}


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

    if accelerator.is_main_process:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Epoch {epoch} Training",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

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
        if accelerator.is_main_process and batch_idx % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{current_lr:.2e}"
            )

        if accelerator.is_main_process:
            progress_bar.update(1)

    # 关闭进度条
    if accelerator.is_main_process:
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

    # 只在主进程显示进度条
    if accelerator.is_main_process:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Epoch {epoch} Testing",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

    # 禁用梯度计算以节省内存和加速推理
    with torch.no_grad():
        for inputs, targets in dataloader:
            # 前向传播获取预测结果
            outputs = model(inputs)
            # 计算当前批次的损失
            loss = loss_fn(outputs, targets)

            # 计算当前批次的统计信息
            batch_size = targets.size(0)
            # 获取预测类别并计算正确预测数量
            correct = outputs.argmax(dim=1).eq(targets).sum()

            # 累加到本地统计量（考虑批次大小权重）
            local_loss_sum += loss * batch_size
            local_correct += correct
            local_samples += batch_size

            # 更新进度条
            if accelerator.is_main_process:
                progress_bar.update(1)

    # 关闭进度条
    if accelerator.is_main_process:
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
        return avg_loss, accuracy

    # 非主进程返回None
    return None, None


def run_training(config, exp_name=None):
    """
    训练的主入口函数，负责整个训练过程的协调，包括：
    - 环境初始化（随机种子、实验追踪）
    - 数据加载器创建
    - 模型、损失函数、优化器初始化
    - 多GPU环境配置
    - 训练循环执行
    - 结果记录和返回

    Args:
        config (dict): 包含所有训练配置的字典，包括模型、数据、超参数等设置
        exp_name (str, optional): 实验名称，用于追踪和日志记录

    Returns:
        dict: 训练结果字典，包含实验名称、最佳准确率和配置信息
    """
    # 设置随机种子确保实验可重现性
    set_seed(42)

    # 实验名称，优先使用传入函数的参数
    if exp_name is None:
        exp_name = config['training']['exp_name']

    # === 第1步：解析任务配置 ===
    task_config = config.get('task', {})
    task_tag = task_config.get('tag')

    # 验证任务类型必须明确指定
    if not task_tag:
        raise ValueError(f"必须在配置文件中明确指定task.tag。支持的任务类型: {list(SUPPORTED_TASKS.keys())}")

    if task_tag not in SUPPORTED_TASKS:
        raise ValueError(f"不支持的任务类型: {task_tag}。支持的任务: {list(SUPPORTED_TASKS.keys())}")

    task_info = SUPPORTED_TASKS[task_tag]

    # === 第2步：解析和验证数据配置 ===
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
        project_name=config['swanlab']['project_name'], # SwanLab UI中项目名称
        config=tracker_config,    # 要记录的超参数
        init_kwargs={             # 额外初始化参数
            "swanlab": {
                "exp_name": exp_name,
                "description": config['swanlab']['description']
            }
        }
    )

    # === 第3步：获取模型配置（用于数据预处理） ===
    model_config = config.get('model', {})
    model_name = model_config.get('type',
                                 model_config.get('name', task_info['default_model']))

    # 使用简化的数据加载器创建函数
    train_dataloader, test_dataloader, num_classes = create_dataloaders(
        dataset_name=dataset_type,
        data_dir=data_config.get('root', './data'),
        batch_size=hyperparams['batch_size'],
        num_workers=data_config.get('num_workers', 8),
        model_type=model_name,  # 传递模型类型用于动态transforms
        data_percentage=hyperparams.get('data_percentage', 1.0),
        **data_config.get('params', {})
    )

    # === 第4步：获取数据集信息 ===
    dataset_info = get_dataset_info(dataset_type)
    dataset_info['num_classes'] = num_classes or dataset_info['num_classes']

    # === 第5步：基于任务类型创建模型 ===

    # 使用任务驱动的模型工厂选择
    model_factory_name = task_info['model_factory']
    model_factory = globals()[model_factory_name]

    # 统一的模型创建逻辑
    model_params = model_config.get('params', {}).copy()
    model_params['num_classes'] = dataset_info['num_classes']

    model = model_factory(
        model_type=model_name,
        **model_params
    )

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

    # 使用Accelerator包装所有训练组件，自动处理多GPU分布式训练
    
    # # 清理GPU缓存，释放未使用的内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 使用Accelerator包装训练组件，自动处理分布式训练
    model, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )

    # 打印训练配置信息（仅在主进程）
    if accelerator.is_main_process:
        print(f"========== 训练实验: {exp_name} ==========")
        print(f"  任务类型: {task_tag} ({task_info['description']})")
        print(f"  数据集: {dataset_type}")
        print(f"  模型: {model_name}")
        print(f"  参数: {hyperparams}")
        print("=" * 80)

    # 设置结果目录
    result_dir = os.path.join("runs", exp_name) if exp_name else None

    # 初始化最佳准确率追踪
    best_accuracy = 0.0

    # 主训练循环：执行指定轮数的训练
    for epoch in range(1, hyperparams['epochs'] + 1):
        if accelerator.is_main_process:
            tqdm.write(f"Epoch {epoch}/{hyperparams['epochs']}")

        # 训练epoch
        train_loss = train_epoch(train_dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch)
        # 测试epoch
        val_loss, val_accuracy = test_epoch(test_dataloader, model, loss_fn, accelerator, epoch, train_batches=len(train_dataloader))

        # 更新并记录最佳准确率
        if accelerator.is_main_process and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            tqdm.write(f"新最佳准确率: {best_accuracy:.2f}%")

    # 结束实验追踪，保存日志和结果
    accelerator.end_training()

    # 输出训练完成信息
    if accelerator.is_main_process:
        tqdm.write(f"训练完成! 最佳准确率: {best_accuracy:.2f}%")

    # 清理GPU缓存，为下一个实验释放资源
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 返回训练结果摘要（直接返回，不写入文件）
    return {
        "success": True,                       # 训练成功标志
        "exp_name": exp_name,                  # 实验名称
        "best_accuracy": best_accuracy,        # 最佳测试准确率
        "final_accuracy": val_accuracy,        # 最终准确率
        "config": tracker_config               # 完整的训练配置
    }
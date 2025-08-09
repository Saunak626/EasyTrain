"""
基础训练器模块
核心设计原则包括：
- 统一接口：为图像和视频模型提供统一的训练接口
- 模块解耦：将模型、损失函数、优化器、调度器等组件解耦，便于扩展和维护
- 分布式支持：基于Accelerate库实现多GPU和分布式训练
- 实验追踪：集成SwanLab等实验管理工具，便于实验监控和结果分析
- 配置驱动：通过配置文件控制训练行为，支持灵活的实验设置

核心功能：
- train_epoch: 单个训练轮次的执行，包括前向传播、损失计算、反向传播
- test_epoch: 单个测试轮次的执行，用于模型评估和验证
- run_training: 完整训练流程的主函数，协调各个组件完成训练任务
- write_epoch_metrics/write_final_result: 实验结果记录和持久化

支持特性：
- 多GPU训练和分布式训练
- 图像分类和视频分类任务
- 多种优化器、调度器和损失函数
- 实验追踪和结果可视化
- 灵活的数据集配置和加载
- 自动混合精度训练
"""

import os
import sys
import json
import torch
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入项目内部模块
from src.models.image_net import get_model                     # 图像模型工厂函数
from src.models.video_net import get_video_model               # 视频模型工厂函数
from src.losses.image_loss import get_loss_function            # 损失函数工厂函数
from src.optimizers.optim import get_optimizer                 # 优化器工厂函数
from src.schedules.scheduler import get_scheduler              # 学习率调度器工厂函数
from src.datasets import create_dataloaders, get_dataset_info  # 统一数据加载器工厂
from src.utils.data_utils import set_seed


# JSON文件写入函数已删除，改为直接返回训练结果


def train_epoch(dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch):
    """
    执行单个训练轮次

    设计思路：
    - 标准的深度学习训练循环：前向传播 -> 损失计算 -> 反向传播 -> 参数更新
    - 使用Accelerator库实现多GPU和混合精度训练的透明支持
    - 集成进度条显示，提供训练过程的实时反馈
    - 支持学习率调度，实现动态学习率调整策略
    - 统计训练指标，便于监控训练效果

    训练流程：
    1. 设置模型为训练模式
    2. 初始化训练指标和进度条
    3. 遍历数据批次，执行训练步骤
    4. 更新学习率调度器
    5. 返回平均损失

    Args:
        dataloader (torch.utils.data.DataLoader): 训练数据加载器，提供批次数据
        model (torch.nn.Module): 神经网络模型
        loss_fn (torch.nn.Module): 损失函数，用于计算预测与真实标签的差异
        optimizer (torch.optim.Optimizer): 优化器，用于更新模型参数
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器，动态调整学习率
        accelerator (accelerate.Accelerator): Accelerator实例，处理多GPU和混合精度训练
        epoch (int): 当前训练轮次编号

    Returns:
        float: 平均训练损失
    """
    # 设置模型为训练模式，启用dropout和batch normalization的训练行为
    model.train()

    # 初始化训练指标
    total_loss = 0.0
    num_batches = 0

    # 只在主进程显示进度条，避免多GPU时重复显示
    if accelerator.is_main_process:
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Epoch {epoch} Training",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

    # 遍历训练数据的每个批次
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # 前向传播：将输入数据通过模型得到预测结果
        outputs = model(inputs)
        # 计算损失：比较预测结果与真实标签
        loss = loss_fn(outputs, targets)

        # 反向传播：使用accelerator处理梯度计算，支持混合精度和多GPU
        accelerator.backward(loss)
        # 更新模型参数：根据计算的梯度调整权重
        optimizer.step()
        # 清零梯度：为下一次迭代准备
        optimizer.zero_grad()
        # 更新学习率：根据调度策略调整学习率
        lr_scheduler.step()

        # 累计损失统计
        total_loss += loss.item()
        num_batches += 1

        # 记录训练指标到实验追踪系统（如SwanLab）
        accelerator.log({"train/loss": loss.item(), "epoch_num": epoch})

        # 定期更新进度条显示，避免过于频繁的更新影响性能
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

    # 初始化Accelerator，指定swanlab为日志记录工具
    accelerator = Accelerator(log_with="swanlab")

    # 记录到SwanLab的超参数
    hyperparams = config['hp']
    tracker_config = {**hyperparams, "exp_name": exp_name}

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

    # 解析数据配置
    data_config = config.get('data', {})
    dataset_type = data_config.get('type', 'cifar10')

    # 使用简化的数据加载器创建函数
    train_dataloader, test_dataloader, num_classes = create_dataloaders(
        dataset_name=dataset_type,
        data_dir=data_config.get('root', './data'),
        batch_size=hyperparams['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        data_percentage=hyperparams.get('data_percentage', 1.0),
        **data_config.get('params', {})
    )
    
    # 获取数据集信息
    dataset_info = get_dataset_info(dataset_type)
    dataset_info['num_classes'] = num_classes or dataset_info['num_classes']

    # 解析模型配置
    model_config = config.get('model', {})
    model_name = model_config.get('type', model_config.get('name', 'resnet18'))

    # 根据模型类型选择对应的工厂函数
    video_model_prefixes = ['r3d_', 'mc3_', 'r2plus1d_', 's3d']
    is_video_model = any(model_name.startswith(prefix) for prefix in video_model_prefixes)
    
    if is_video_model:
        # 使用视频模型工厂函数
        video_params = model_config.get('params', {}).copy()
        # 确保使用数据集的实际类别数
        video_params['num_classes'] = dataset_info['num_classes']
        model = get_video_model(
            model_name=model_name,
            **video_params
        )
    else:
        # 使用图像模型工厂函数
        model = get_model(
            model_name=model_name,
            num_classes=dataset_info['num_classes'],
            **model_config.get('params', {})
        )

    # 创建损失函数
    loss_config = config.get('loss', {})
    loss_fn = get_loss_function(
        loss_config.get('name', 'crossentropy'),
        **loss_config.get('params', {})
    )

    # 创建优化器
    optimizer_config = config.get('optimizer', {})
    optimizer = get_optimizer(
        model,
        optimizer_config.get('name', 'adam'),
        hyperparams['learning_rate'],
        **optimizer_config.get('params', {})
    )

    # 创建学习率调度器
    scheduler_config = config.get('scheduler', {})
    scheduler_name = scheduler_config.get('name', 'onecycle')
    scheduler_params = scheduler_config.get('params', {})
    
    # 根据调度器类型设置默认参数
    if scheduler_name == 'onecycle':
        scheduler_params.setdefault('max_lr', 5 * hyperparams['learning_rate'])
        scheduler_params.setdefault('epochs', hyperparams['epochs'])
        scheduler_params.setdefault('steps_per_epoch', len(train_dataloader))
    elif scheduler_name == 'cosine':
        scheduler_params.setdefault('T_max', hyperparams['epochs'])
    
    lr_scheduler = get_scheduler(
        optimizer,
        scheduler_name,
        **scheduler_params
    )

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

        # JSON文件写入已删除，只保留SwanLab记录

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
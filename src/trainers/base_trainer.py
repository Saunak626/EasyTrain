"""基础训练器模块

这个模块提供了深度学习训练的核心功能，包括：
- train_epoch: 单个训练轮次的执行
- test_epoch: 单个测试轮次的执行  
- run_training: 完整训练流程的主函数

支持多GPU训练、实验追踪和各种数据集类型
"""

import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
import sys
import os
import json
from datetime import datetime

# 添加项目根目录到路径，确保可以正确导入项目内的模块
# 这是一种常见的做法，用于解决Python模块导入路径问题
# 通过os.path.dirname的三层嵌套调用，获取到项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入项目内部模块
from src.models.image_net import get_model                     # 图像模型工厂函数
from src.models.video_net import get_video_model               # 视频模型工厂函数
from src.losses.image_loss import get_loss_function            # 损失函数工厂函数
from src.optimizers.optim import get_optimizer                 # 优化器工厂函数
from src.schedules.scheduler import get_scheduler              # 学习率调度器工厂函数
from src.datasets import create_dataloaders, get_dataset_info  # 统一数据加载器工厂
from src.utils.data_utils import set_seed                      # 随机种子设置工具


def write_epoch_metrics(result_dir, epoch_data, accelerator):
    """写入epoch级别的指标数据到JSONL文件"""
    if not accelerator.is_main_process:
        return

    os.makedirs(result_dir, exist_ok=True)
    jsonl_path = os.path.join(result_dir, "metrics.jsonl")

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(epoch_data, ensure_ascii=False) + "\n")


def write_final_result(result_dir, result_data, accelerator):
    """写入最终训练结果到JSON文件"""
    if not accelerator.is_main_process:
        return

    os.makedirs(result_dir, exist_ok=True)
    final_path = os.path.join(result_dir, "result.json")

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)


def train_epoch(dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch):
    """
    执行单个训练轮次

    该函数负责一个完整epoch的训练过程，包括前向传播、损失计算、
    反向传播、参数更新和学习率调整。支持多GPU分布式训练。

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


def test_epoch(dataloader, model, loss_fn, accelerator, epoch):
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
        tqdm.write(f'Epoch {epoch:03d} | val_loss={avg_loss:.4f} | val_acc={accuracy:.2f}%')

        # 记录测试指标到实验追踪系统
        accelerator.log({"test/loss": avg_loss, "test/accuracy": accuracy}, step=epoch)
        return avg_loss, accuracy

    # 非主进程返回None
    return None, None


def run_training(config, experiment_name=None):
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
        experiment_name (str, optional): 实验名称，用于追踪和日志记录

    Returns:
        dict: 训练结果字典，包含实验名称、最佳准确率和配置信息
    """
    # 设置随机种子确保实验可重现性
    set_seed(42)

    # 确定实验名称，优先使用传入参数
    if experiment_name is None:
        experiment_name = config['training']['experiment_name']

    # 初始化Accelerator，自动处理多GPU和混合精度训练
    accelerator = Accelerator(log_with="swanlab")

    # 准备实验追踪配置
    hyperparams = config['hyperparameters']
    tracker_config = {**hyperparams, "experiment_name": experiment_name}

    # 初始化SwanLab实验追踪器
    accelerator.init_trackers(
        project_name=config['swanlab']['project_name'],
        config=tracker_config,
        init_kwargs={"swanlab": {
            "experiment_name": experiment_name,
            "description": config['swanlab']['description']
        }}
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
    lr_scheduler = get_scheduler(
        optimizer,
        scheduler_config.get('name', 'onecycle'),
        max_lr=5 * hyperparams['learning_rate'],
        epochs=hyperparams['epochs'],
        steps_per_epoch=len(train_dataloader),
        **scheduler_config.get('params', {})
    )

    # 使用Accelerator包装所有训练组件，自动处理多GPU分布式训练
    try:
        # 清理GPU缓存，释放未使用的内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 使用Accelerator包装训练组件，自动处理分布式训练
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, test_dataloader
        )

    except RuntimeError as e:
        # 处理常见的GPU内存不足错误
        if "out of memory" in str(e):
            print(f"❌ GPU内存不足: {e}")
            print("💡 建议解决方案:")
            print("  1. 减少batch_size")
            print("  2. 使用更小的模型")
            print("  3. 使用CPU训练: --use_cpu")
            raise e
        else:
            raise e

    # 打印训练配置信息（仅在主进程）
    if accelerator.is_main_process:
        print(f"\n=== 训练实验: {experiment_name} ===")
        print(f"数据集: {dataset_type}")
        print(f"模型: {model_name}")
        print(f"参数: {hyperparams}")
        print("=" * 50)

    # 设置结果目录
    result_dir = os.path.join("runs", experiment_name) if experiment_name else None

    # 初始化最佳准确率追踪
    best_accuracy = 0.0

    # 主训练循环：执行指定轮数的训练
    for epoch in range(1, hyperparams['epochs'] + 1):
        if accelerator.is_main_process:
            tqdm.write(f"\nEpoch {epoch}/{hyperparams['epochs']}")

        # 执行一轮训练和测试
        train_loss = train_epoch(train_dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch)
        
        val_loss, val_accuracy = test_epoch(test_dataloader, model, loss_fn, accelerator, epoch)

        # 更新并记录最佳准确率
        if accelerator.is_main_process and val_accuracy and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            tqdm.write(f"新最佳准确率: {best_accuracy:.2f}%")

        # 写入epoch级别的结构化数据
        if accelerator.is_main_process and result_dir and val_accuracy is not None:
            epoch_data = {
                "event": "epoch_end",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_accuracy,
                "best_acc": best_accuracy,
                "timestamp": datetime.now().isoformat()
            }
            write_epoch_metrics(result_dir, epoch_data, accelerator)

    # 结束实验追踪，保存日志和结果
    accelerator.end_training()

    # 写入最终结果
    if accelerator.is_main_process:
        tqdm.write(f"\n训练完成! 最佳准确率: {best_accuracy:.2f}%")

        # 输出机器可读的结果行
        result_json = {"best_accuracy": best_accuracy, "final_accuracy": best_accuracy}
        print("##RESULT## " + json.dumps(result_json))

        # 写入最终结果文件
        if result_dir:
            final_result = {
                "experiment_name": experiment_name,
                "best_accuracy": best_accuracy,
                "final_accuracy": best_accuracy,
                "total_epochs": hyperparams['epochs'],
                "config": tracker_config,
                "timestamp": datetime.now().isoformat()
            }
            write_final_result(result_dir, final_result, accelerator)

    # 返回训练结果摘要
    return {
        "experiment_name": experiment_name,    # 实验名称
        "best_accuracy": best_accuracy,        # 最佳测试准确率
        "config": tracker_config               # 完整的训练配置
    }
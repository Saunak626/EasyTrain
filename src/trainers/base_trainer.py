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

# 添加项目根目录到路径，确保可以正确导入项目内的模块
# 这是一种常见的做法，用于解决Python模块导入路径问题
# 通过os.path.dirname的三层嵌套调用，获取到项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入项目内部模块
from src.models.image_net import get_model                      # 模型工厂函数
from src.losses.image_loss import get_loss_function            # 损失函数工厂函数
from src.optimizers.optim import get_optimizer                 # 优化器工厂函数
from src.schedules.scheduler import get_scheduler              # 学习率调度器工厂函数
from src.data_preprocessing.dataloader_factory import create_dataloaders, get_dataset_info  # 统一数据加载器工厂
from src.utils.data_utils import set_seed                      # 随机种子设置工具


def train_epoch(dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch):
    """
    执行单个训练轮次
    
    该函数负责一个完整epoch的训练过程，包括前向传播、损失计算、
    反向传播、参数更新和学习率调整。支持多GPU分布式训练。

    Args:
        dataloader: 训练数据加载器，提供批次数据
        model: 神经网络模型
        loss_fn: 损失函数，用于计算预测与真实标签的差异
        optimizer: 优化器，用于更新模型参数
        lr_scheduler: 学习率调度器，动态调整学习率
        accelerator: Accelerator实例，处理多GPU和混合精度训练
        epoch: 当前训练轮次编号
    """
    # 设置模型为训练模式，启用dropout和batch normalization的训练行为
    model.train()

    # 只在主进程显示进度条，避免多GPU时重复显示
    if accelerator.is_main_process:
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch} Training", unit="batch")

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

        # 记录训练指标到实验追踪系统（如SwanLab）
        accelerator.log({"train/loss": loss.item(), "epoch_num": epoch})

        # 定期更新进度条显示，避免过于频繁的更新影响性能
        if accelerator.is_main_process and batch_idx % 50 == 0:
            progress_bar.update(50)
            progress_bar.set_postfix_str(f"Loss: {loss.item():.4f}")

    # 关闭进度条
    if accelerator.is_main_process:
        progress_bar.close()


def test_epoch(dataloader, model, loss_fn, accelerator, epoch):
    """
    执行单个测试轮次
    
    该函数在测试集上评估模型性能，计算平均损失和准确率。
    支持多GPU环境下的指标聚合，确保结果的准确性。

    Args:
        dataloader: 测试数据加载器，提供测试批次数据
        model: 神经网络模型
        loss_fn: 损失函数，用于计算测试损失
        accelerator: Accelerator实例，处理多GPU指标聚合
        epoch: 当前测试轮次编号

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
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch} Testing", unit="batch")

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
        print(f'Epoch {epoch} - Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # 记录测试指标到实验追踪系统
        accelerator.log({"test/loss": avg_loss, "test/accuracy": accuracy}, step=epoch)
        return avg_loss, accuracy

    # 非主进程返回None
    return None, None


def run_training(config, experiment_name=None):
    """
    执行完整的训练流程
    
    这是训练的主入口函数，负责整个训练过程的协调，包括：
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

    # 使用工厂函数创建数据加载器
    train_dataloader, test_dataloader = create_dataloaders(
        data_config=data_config,
        batch_size=hyperparams['batch_size'],
        num_workers=data_config.get('num_workers', 4)
    )
    
    # 获取数据集信息
    dataset_info = get_dataset_info(data_config)

    # 解析模型配置
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'simple_cnn')

    # 配置模型参数
    model_params = {
        'num_classes': dataset_info['num_classes'],
        'input_size': dataset_info['input_size'],
        'dropout': hyperparams['dropout'],
        'freeze_backbone': model_config.get('freeze_backbone', False)
    }
    
    # 为自定义数据集添加额外参数
    if dataset_type == 'custom':
        model_params['input_features'] = data_config.get('input_features', 20)

    # 合并用户在配置文件中指定的额外模型参数
    model_params.update(model_config.get('params', {}))

    # 使用工厂函数创建模型实例
    model = get_model(model_type, **model_params)

    # 创建损失函数
    loss_config = config.get('loss', {})
    loss_fn = get_loss_function(
        loss_config.get('type', 'cross_entropy'),
        **loss_config.get('params', {})
    )

    # 创建优化器
    optimizer_config = config.get('optimizer', {})
    optimizer = get_optimizer(
        model,
        optimizer_config.get('type', 'adam'),
        hyperparams['learning_rate'],
        **optimizer_config.get('params', {})
    )

    # 创建学习率调度器
    scheduler_config = config.get('scheduler', {})
    lr_scheduler = get_scheduler(
        optimizer,
        scheduler_config.get('type', 'onecycle'),
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
        print(f"模型: {model_type}")
        print(f"参数: {hyperparams}")
        print("=" * 50)

    # 初始化最佳准确率追踪
    best_accuracy = 0.0

    # 主训练循环：执行指定轮数的训练
    for epoch in range(1, hyperparams['epochs'] + 1):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch}/{hyperparams['epochs']}")

        # 执行一轮训练和测试
        train_epoch(train_dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch)
        _, test_accuracy = test_epoch(test_dataloader, model, loss_fn, accelerator, epoch)

        # 更新并记录最佳准确率
        if accelerator.is_main_process and test_accuracy and test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"新最佳准确率: {best_accuracy:.2f}%")

    # 结束实验追踪，保存日志和结果
    accelerator.end_training()

    # 打印训练完成信息
    if accelerator.is_main_process:
        print(f"\n训练完成! 最佳准确率: {best_accuracy:.2f}%")

    # 返回训练结果摘要
    return {
        "experiment_name": experiment_name,    # 实验名称
        "best_accuracy": best_accuracy,        # 最佳测试准确率
        "config": tracker_config               # 完整的训练配置
    }
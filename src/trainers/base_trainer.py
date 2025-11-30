"""基础训练器模块

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.image_net import get_model
from src.models.video_net import get_video_model
from src.losses.loss_factory import get_loss_function
from src.optimizers.optimizer_factory import get_optimizer
from src.schedules.scheduler_factory import get_scheduler
from src.datasets import create_dataloaders, get_dataset_info
from src.utils.data_utils import set_seed
from src.utils.training_logger import TrainingLogger
from src.utils.dataset_utils import unwrap_subset_dataset, get_dataset_metadata
from src.utils.training_utils import log_multilabel_metrics_to_swanlab

# 常量配置
TRAINING_CONSTANTS = {
    'default_seed': 42,
    'default_num_workers': 8,
    'progress_update_interval': 10,
}

SUPPORTED_TASKS = {
    'image_classification': {
        'description': '图像分类任务',
        'supported_datasets': ['cifar10', 'custom'],
        'model_factory': 'get_model',
        'default_model': 'resnet18'
    },
    'video_classification': {
        'description': '视频分类任务',
        'supported_datasets': ['ucf101', 'ucf101_video', 'neonatal_multilabel', 'neonatal_multilabel_simple'],
        'model_factory': 'get_video_model',
        'default_model': 'r3d_18'
    }
}


def train_epoch(dataloader, model, loss_fn, optimizer, lr_scheduler, accelerator, epoch,
                metrics_calculator=None, scheduler_step_interval='batch'):
    """执行单个训练轮次"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    collected_outputs = []
    collected_targets = []
    is_multilabel = metrics_calculator is not None

    progress_bar = None
    if accelerator.is_main_process:
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch} Train", 
                           unit="batch", dynamic_ncols=True, leave=False)

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

        if is_multilabel:
            collected_outputs.append(outputs.detach())
            collected_targets.append(targets.detach())

        accelerator.log({"train/loss": loss.item(), "epoch_num": epoch})

        if progress_bar and batch_idx % TRAINING_CONSTANTS['progress_update_interval'] == 0:
            progress_bar.set_postfix(loss=f"{total_loss/num_batches:.4f}", 
                                    lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if progress_bar:
            progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if lr_scheduler is not None and scheduler_step_interval == 'epoch':
        lr_scheduler.step()

    train_accuracy = 0.0
    if is_multilabel and collected_outputs:
        stacked_outputs = torch.cat(collected_outputs, dim=0)
        stacked_targets = torch.cat(collected_targets, dim=0)
        gathered_outputs = accelerator.gather_for_metrics(stacked_outputs)
        gathered_targets = accelerator.gather_for_metrics(stacked_targets)

        if accelerator.is_main_process:
            probs = torch.sigmoid(gathered_outputs).cpu().numpy()
            targets_np = gathered_targets.cpu().numpy()
            train_metrics = metrics_calculator.calculate_detailed_metrics(probs, targets_np, threshold=0.5)
            metrics_calculator.save_train_metrics(train_metrics, epoch, avg_train_loss)
            log_multilabel_metrics_to_swanlab(accelerator, train_metrics, 'train', epoch)
            train_accuracy = train_metrics['macro_avg']['accuracy']

    return avg_train_loss, train_accuracy


def test_epoch(dataloader, model, loss_fn, accelerator, epoch, train_batches=None, metrics_calculator=None):
    """执行单个测试轮次"""
    model.eval()
    device = accelerator.device

    local_loss_sum = torch.tensor(0.0, device=device)
    local_correct = torch.tensor(0, device=device)
    local_samples = torch.tensor(0, device=device)

    all_predictions = []
    all_targets = []
    is_multilabel = False

    progress_bar = None
    if accelerator.is_main_process:
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch} Test",
                           unit="batch", dynamic_ncols=True, leave=False)

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            batch_size = targets.size(0)

            is_multilabel = len(targets.shape) > 1 and targets.shape[1] > 1 and targets.dtype == torch.float32

            if is_multilabel:
                sigmoid_outputs = torch.sigmoid(outputs)
                predictions = sigmoid_outputs > 0.5
                targets_bool = targets.bool()

                if metrics_calculator is not None:
                    all_predictions.append(sigmoid_outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                class_accuracies = (predictions == targets_bool).float().mean(dim=0)
                macro_accuracy = class_accuracies.mean()
                correct = (macro_accuracy * batch_size).long()
            else:
                correct = outputs.argmax(dim=1).eq(targets).sum()

            local_loss_sum += loss * batch_size
            local_correct += correct
            local_samples += batch_size

            if progress_bar:
                progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    total_loss = accelerator.reduce(local_loss_sum, reduction="sum")
    total_correct = accelerator.reduce(local_correct, reduction="sum")
    total_samples = accelerator.reduce(local_samples, reduction="sum")

    if accelerator.is_main_process:
        avg_loss = (total_loss / total_samples).item()
        accuracy = 100. * total_correct.item() / total_samples.item()

        if metrics_calculator is not None and all_predictions and is_multilabel:
            all_pred_array = np.concatenate(all_predictions, axis=0)
            all_target_array = np.concatenate(all_targets, axis=0)

            detailed_metrics = metrics_calculator.calculate_detailed_metrics(
                all_pred_array, all_target_array, threshold=0.5)

            is_best = metrics_calculator.update_best_metrics(
                detailed_metrics, epoch, predictions=all_pred_array, targets=all_target_array)

            metrics_calculator.save_metrics(detailed_metrics, epoch, avg_loss, is_best)
            metrics_calculator.save_test_metrics(detailed_metrics, epoch, avg_loss)

            display = metrics_calculator.format_metrics_display(
                detailed_metrics, epoch, avg_loss, train_batches or 0)
            tqdm.write(display)

            if is_best:
                tqdm.write(f"新最佳F1: {detailed_metrics['macro_avg']['f1']:.4f}")

            accelerator.log({"test/loss": avg_loss}, step=epoch)
            log_multilabel_metrics_to_swanlab(accelerator, detailed_metrics, 'test', epoch)
        else:
            tqdm.write(f'Epoch {epoch:03d} | val_loss={avg_loss:.4f} | val_acc={accuracy:.2f}%')
            accelerator.log({"test/loss": avg_loss, "test/accuracy": accuracy}, step=epoch)

        return avg_loss, accuracy

    return None, None


def setup_experiment(config: Dict[str, Any], exp_name: Optional[str] = None) -> Tuple:
    """实验环境初始化"""
    set_seed(TRAINING_CONSTANTS['default_seed'])

    if exp_name is None:
        exp_name = config['training']['exp_name']

    task_config = config.get('task', {})
    task_tag = task_config.get('tag')

    if not task_tag:
        raise ValueError(f"必须指定task.tag。支持: {list(SUPPORTED_TASKS.keys())}")
    if task_tag not in SUPPORTED_TASKS:
        raise ValueError(f"不支持的任务类型: {task_tag}")

    task_info = SUPPORTED_TASKS[task_tag]
    data_config = config.get('data', {})
    dataset_type = data_config.get('type', 'cifar10')

    if dataset_type not in task_info['supported_datasets']:
        raise ValueError(f"任务 '{task_tag}' 不支持数据集 '{dataset_type}'")

    accelerator = Accelerator(log_with="swanlab")
    hyperparams = config['hp']
    tracker_config = {**hyperparams, "exp_name": exp_name, "task_tag": task_tag}

    accelerator.init_trackers(
        project_name=config['swanlab']['project_name'],
        config=tracker_config,
        init_kwargs={"swanlab": {"exp_name": exp_name, "description": config['swanlab']['description']}}
    )

    return exp_name, task_info, task_tag, data_config, accelerator


def setup_data_and_model(config: Dict[str, Any], task_info: Dict[str, Any], 
                         data_config: Dict[str, Any], accelerator: Accelerator) -> Tuple:
    """数据和模型初始化"""
    hyperparams = config['hp']
    model_config = config.get('model', {})
    dataset_type = data_config.get('type', 'cifar10')
    model_name = model_config.get('type', model_config.get('name', task_info['default_model']))

    train_dataloader, test_dataloader, num_classes = create_dataloaders(
        dataset_name=dataset_type,
        data_dir=data_config.get('root', './data'),
        batch_size=hyperparams['batch_size'],
        num_workers=data_config.get('num_workers', TRAINING_CONSTANTS['default_num_workers']),
        model_type=model_name,
        data_percentage=hyperparams.get('data_percentage', 1.0),
        **data_config.get('params', {})
    )

    dataset_info = get_dataset_info(dataset_type)
    dataset_info['num_classes'] = num_classes or dataset_info['num_classes']

    metadata = get_dataset_metadata(train_dataloader.dataset, dataset_type)
    if metadata['num_classes'] is not None:
        dataset_info['num_classes'] = metadata['num_classes']
    if metadata['classes'] is not None:
        dataset_info['classes'] = metadata['classes']
    dataset_info['is_multilabel'] = metadata['is_multilabel']

    model_factory = globals()[task_info['model_factory']]
    model_params = model_config.get('params', {}).copy()
    model_params['num_classes'] = dataset_info['num_classes']

    model = model_factory(model_type=model_name, **model_params)

    return train_dataloader, test_dataloader, model, dataset_info


def setup_training_components(config: Dict[str, Any], model, train_dataloader,
                              accelerator: Accelerator, logger: TrainingLogger,
                              dataset_info: Dict[str, Any]) -> Tuple:
    """优化器、调度器、损失函数初始化"""
    hyperparams = config['hp']
    loss_config = config.get('loss', {}).copy()

    multilabel_loss_types = ['multilabel_bce', 'focal_multilabel_bce', 
                             'focal_multilabel_balanced', 'multilabel_focal_balanced']
    loss_name = loss_config.get('name') or loss_config.get('type')
    
    if loss_name in multilabel_loss_types:
        num_classes = dataset_info.get('num_classes')
        if num_classes is None:
            num_classes = config.get('model', {}).get('params', {}).get('num_classes', 24)

        if 'params' not in loss_config:
            loss_config['params'] = {}
        loss_config['params']['num_classes'] = num_classes

        config_pos_weight = loss_config.get('params', {}).get('pos_weight', None)
        if config_pos_weight is not None and isinstance(config_pos_weight, (int, float)):
            dataset = unwrap_subset_dataset(train_dataloader.dataset)
            all_labels = []

            if hasattr(dataset, 'samples'):
                for sample in dataset.samples:
                    labels = sample['labels']
                    if isinstance(labels, list):
                        labels = torch.tensor(labels, dtype=torch.float32)
                    elif not isinstance(labels, torch.Tensor):
                        labels = torch.tensor(labels, dtype=torch.float32)
                    all_labels.append(labels)
            else:
                for batch_idx, (_, targets) in enumerate(train_dataloader):
                    all_labels.append(targets.cpu())
                    if batch_idx >= 1000:
                        break

            if all_labels:
                all_labels = torch.stack(all_labels) if isinstance(all_labels[0], torch.Tensor) and all_labels[0].dim() == 1 else torch.cat(all_labels, dim=0)
                pos_counts = all_labels.sum(dim=0)
                total_samples = all_labels.shape[0]
                neg_counts = total_samples - pos_counts

                raw_ratio = neg_counts / (pos_counts + 1e-6)
                scale_factor = torch.where(
                    raw_ratio < 5.0, torch.tensor(0.8, device=raw_ratio.device),
                    torch.where(raw_ratio < 20.0, torch.tensor(0.6, device=raw_ratio.device),
                               torch.tensor(0.4, device=raw_ratio.device)))

                pos_weight = torch.sqrt(raw_ratio) * scale_factor
                pos_weight = torch.clamp(pos_weight, min=1.0, max=5.0)
                loss_config['params']['pos_weight'] = pos_weight

    loss_fn = get_loss_function(loss_config)
    optimizer = get_optimizer(model, config.get('optimizer', {}), hyperparams['learning_rate'])

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
    """获取任务输出目录"""
    base_dir = "runs"

    if 'multilabel' in task_tag.lower() or 'multilabel' in dataset_type.lower():
        task_subdir = "neonatal_multilabel" if 'neonatal' in dataset_type.lower() else "multilabel_classification"
    elif 'video' in task_tag.lower():
        task_subdir = "video_classification"
    elif 'image' in task_tag.lower():
        task_subdir = "image_classification"
    else:
        task_subdir = dataset_type.lower() or "general"

    output_dir = os.path.join(base_dir, task_subdir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_training_loop(config: Dict[str, Any], model, optimizer, lr_scheduler, loss_fn,
                      train_dataloader, test_dataloader, accelerator: Accelerator, 
                      logger: TrainingLogger, metrics_calculator=None, 
                      scheduler_step_interval='batch') -> Tuple[float, float, int]:
    """主训练循环"""
    hyperparams = config['hp']
    best_accuracy = 0.0
    trained_epochs = 0
    val_accuracy = 0.0

    for epoch in range(1, hyperparams['epochs'] + 1):
        train_loss, _ = train_epoch(
            train_dataloader, model, loss_fn, optimizer, lr_scheduler,
            accelerator, epoch, metrics_calculator, scheduler_step_interval)

        _, val_accuracy = test_epoch(
            test_dataloader, model, loss_fn, accelerator, epoch,
            train_batches=len(train_dataloader), metrics_calculator=metrics_calculator)

        if accelerator.is_main_process and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            tqdm.write(f"新最佳准确率: {best_accuracy:.2f}%")

        trained_epochs = epoch

    if accelerator.is_main_process and metrics_calculator is not None:
        summary_report = metrics_calculator.get_summary_report()
        tqdm.write(summary_report)

    return best_accuracy, val_accuracy, trained_epochs


def cleanup_and_return(accelerator: Accelerator, exp_name: str, best_accuracy: float,
                       val_accuracy: float, trained_epochs: int, tracker_config: Dict[str, Any],
                       metrics_calculator=None) -> Dict[str, Any]:
    """清理和结果返回"""
    accelerator.end_training()

    if accelerator.is_main_process:
        tqdm.write(f"训练完成! 最佳准确率: {best_accuracy:.2f}%")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "success": True,
        "exp_name": exp_name,
        "best_accuracy": best_accuracy,
        "final_accuracy": val_accuracy,
        "trained_epochs": trained_epochs,
        "config": tracker_config
    }

    if metrics_calculator is not None:
        best_metrics = metrics_calculator.best_metrics
        has_valid_metrics = (best_metrics.get("macro_avg") and 
                            isinstance(best_metrics.get("macro_avg"), dict) and 
                            len(best_metrics.get("macro_avg", {})) > 0)

        if has_valid_metrics:
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
            result["detailed_metrics"] = best_metrics

    return result


def run_training(config: Dict[str, Any], exp_name: Optional[str] = None) -> Dict[str, Any]:
    """训练主入口函数"""
    # 第1步：实验环境初始化
    exp_name, task_info, task_tag, data_config, accelerator = setup_experiment(config, exp_name)
    logger = TrainingLogger(accelerator)

    # 第2步：数据和模型初始化
    train_dataloader, test_dataloader, model, dataset_info = setup_data_and_model(
        config, task_info, data_config, accelerator)

    # 第3步：训练组件初始化
    loss_fn, optimizer, lr_scheduler, scheduler_step_interval = setup_training_components(
        config, model, train_dataloader, accelerator, logger, dataset_info)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Accelerator包装
    model, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader)

    # 第4步：创建多标签指标计算器
    metrics_calculator = None
    task_config = config.get('task', {})
    task_tag = task_config.get('tag', '')
    dataset_type = config.get('data', {}).get('type', '')

    is_multilabel_task = ('multilabel' in dataset_type.lower() or 'multilabel' in task_tag.lower())

    if is_multilabel_task:
        from src.evaluation import MultilabelMetricsCalculator
        class_names = dataset_info.get('classes', [])

        if class_names:
            task_dir = get_task_output_dir(task_tag, dataset_type)
            output_dir = config.get('grid_search_dir', task_dir)
            test_dataset = test_dataloader.dataset
            model_type = config.get('model', {}).get('type', 
                         config.get('hp', {}).get('model_type', 'unknown'))

            metrics_calculator = MultilabelMetricsCalculator(
                class_names=class_names, output_dir=output_dir,
                dataset=test_dataset, model_type=model_type, exp_name=exp_name)

    # 第5步：打印实验信息
    logger.print_experiment_info(config, exp_name, task_info, dataset_info, 
                                 model, train_dataloader, test_dataloader)

    # 第6步：执行训练循环
    best_accuracy, val_accuracy, trained_epochs = run_training_loop(
        config, model, optimizer, lr_scheduler, loss_fn, 
        train_dataloader, test_dataloader, accelerator, logger, 
        metrics_calculator, scheduler_step_interval)

    # 第7步：清理和返回结果
    hyperparams = config['hp']
    tracker_config = {**hyperparams, "exp_name": exp_name, "task_tag": task_tag}

    return cleanup_and_return(accelerator, exp_name, best_accuracy, val_accuracy, 
                              trained_epochs, tracker_config, metrics_calculator)

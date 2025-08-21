import os
import sys
import json
import time
import yaml
import torch
import timeit
import argparse
import numpy as np
import pandas as pd
import random 

from tqdm import tqdm
from pathlib import Path
from itertools import islice
from torch import nn, optim
from datetime import datetime
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

# 导入自定义模块
from neonatal_dataset import NeonatalVideoDataset
import C3D_model

# 导入工具函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.random_utils import set_random_seeds


class MultiLabelC3D(nn.Module):
    """多标签行为识别模型，基于C3D架构"""

    def __init__(self, num_classes, pretrained=False):
        super(MultiLabelC3D, self).__init__()

        # 加载C3D模型
        base_model = C3D_model.C3D(101, pretrained=pretrained)  # 默认不使用预训练模型

        # 提取所有层，除了最后的分类层
        self.features = nn.Sequential(
            # 卷积块 1
            base_model.conv1,
            nn.ReLU(),
            base_model.pool1,

            # 卷积块 2
            base_model.conv2,
            nn.ReLU(),
            base_model.pool2,

            # 卷积块 3
            base_model.conv3a,
            nn.ReLU(),
            base_model.conv3b,
            nn.ReLU(),
            base_model.pool3,

            # 卷积块 4
            base_model.conv4a,
            nn.ReLU(),
            base_model.conv4b,
            nn.ReLU(),
            base_model.pool4,

            # 卷积块 5
            base_model.conv5a,
            nn.ReLU(),
            base_model.conv5b,
            nn.ReLU(),
            base_model.pool5,
        )

        # 全连接层
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        # 多标签分类输出层
        self.fc8 = nn.Linear(4096, num_classes)

        # 辅助层
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        # 初始化新添加的层
        self._init_weights()

        # 如果使用预训练权重，复制预训练权重
        if pretrained:
            self._copy_weights(base_model)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 8192)

        x = self.relu(self.fc6(x))
        x = self.dropout(x)

        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        x = self.fc8(x)  # 输出多标签logits

        return x

    def _init_weights(self):
        # 初始化FC层权重
        for m in [self.fc6, self.fc7, self.fc8]:
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def _copy_weights(self, base_model):
        try:
            # 复制预训练的FC6和FC7权重
            self.fc6.weight.data = base_model.fc6.weight.data
            self.fc6.bias.data = base_model.fc6.bias.data

            self.fc7.weight.data = base_model.fc7.weight.data
            self.fc7.bias.data = base_model.fc7.bias.data
            print("成功复制预训练全连接层权重")
        except Exception as e:
            print(f"复制预训练权重时出错: {e}")
            print("使用随机初始化全连接层")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="训练新生儿行为多标签识别模型")
    parser.add_argument('--gpu', type=int, default=2,
                        help='使用的GPU ID')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--clip_len', type=int, default=16,
                        help='视频片段长度 (帧数)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器的工作线程数')
    parser.add_argument('--prefetch_videos', action='store_true',
                        help='是否预加载视频到内存')
    parser.add_argument('--max_prefetch', type=int, default=500,
                        help='最多预加载的视频数量')
    parser.add_argument('--pin_memory', action='store_true',
                        help='是否在数据加载时使用pin_memory')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='是否保持数据加载工作进程持续运行')
    parser.add_argument('--use_pretrained', action='store_true', default=True,
                        help='是否使用预训练权重')
    parser.add_argument('--pretrained_path', type=str,
                        default='ucf101-caffe.pth',
                        help='预训练权重路径')
    parser.add_argument('--transform_in_worker', action='store_true', default=True,
                        help='是否在预加载线程中进行视频变换处理')
    parser.add_argument('--cache_warm', action='store_true',
                        help='在训练前预热缓存，先执行一个epoch的批处理')
    parser.add_argument('--load_mode', type=str, default='video',
                        choices=['video', 'frames'],
                        help='数据加载模式: video(直接从视频文件加载) 或 frames(从预处理帧目录加载)')
    parser.add_argument('--frames_dir', type=str, default=None,
                        help='预处理帧目录路径 (默认为data/frames)')
    parser.add_argument('--model_save_dir', type=str, default='src/recognition/model_result',
                        help='模型保存目录')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据根目录')
    return parser.parse_args()


def process_batch(model, inputs, targets, criterion, device, optimizer=None, scaler=None):
    """处理一个批次的数据：前向传播、计算损失和可选的反向传播
    
    Args:
        model: 模型
        inputs: 输入数据
        targets: 目标标签
        criterion: 损失函数
        device: 设备
        optimizer: 优化器（如果为None则不执行反向传播）
        scaler: 混合精度训练的scaler
    
    Returns:
        loss: 批次损失
        preds: 预测结果
    """
    inputs, targets = inputs.to(device), targets.to(device)
    
    with torch.amp.autocast('cuda'):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # 如果提供了优化器，执行反向传播（训练模式）
    if optimizer is not None and scaler is not None:
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # 计算预测结果
    with torch.no_grad():
        preds = torch.sigmoid(outputs) > 0.5
    
    return loss.item() * inputs.size(0), preds.cpu().numpy(), targets.cpu().numpy()


def evaluate_model(model, dataloader, criterion, device, mode='val', optimizer=None, scaler=None):
    """评估模型性能
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        mode: 模式 ('train', 'val', 'test')
        optimizer: 优化器（仅训练模式需要）
        scaler: 混合精度训练的scaler（仅训练模式需要）
    
    Returns:
        metrics: 包含损失和各项指标的字典
    """
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    # 设置进度条描述
    desc = f"{'训练' if mode == 'train' else '验证' if mode == 'val' else '测试'}"
    
    # 使用torch.no_grad()包装非训练模式的评估
    if mode != 'train':
        eval_context = torch.no_grad()
    else:
        eval_context = nullcontext()
    
    with eval_context:
        for inputs, targets in tqdm(dataloader, desc=desc):
            # 处理批次
            batch_loss, batch_preds, batch_targets = process_batch(
                model, inputs, targets, criterion, device, 
                optimizer if mode == 'train' else None,
                scaler if mode == 'train' else None
            )
            
            total_loss += batch_loss
            all_preds.append(batch_preds)
            all_targets.append(batch_targets)
    
    # 计算性能指标
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    total_loss /= len(dataloader.dataset)
    
    # 计算整体指标
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # 返回结果
    return {
        'loss': total_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'targets': all_targets
    }


def warmup_cuda(model, dataloader, device):
    """执行CUDA预热以减少第一次推理时的延迟
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    """
    print("执行CUDA预热...")
    with torch.no_grad():
        for inputs, _ in islice(dataloader, 1):
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                model(inputs.to(device))
    print("CUDA预热完成")


def train_model(num_epochs, num_classes, lr, device, save_dir, train_dataloader, val_dataloader, test_dataloader, use_pretrained=True, pretrained_path='ucf101-caffe.pth'):
    """训练多标签行为识别模型"""
    print("\n=== 初始化模型 ===")
    # 加载模型
    if use_pretrained and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), pretrained_path)):
        C3D_model.weight_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), pretrained_path)
        model = MultiLabelC3D(num_classes, pretrained=True)
        print(f"成功加载预训练权重: {pretrained_path}")
    else:
        model = MultiLabelC3D(num_classes, pretrained=False)
        print("使用随机初始化模型")
    
    model.to(device)
    
    # 使用混合精度训练
    scaler = torch.amp.GradScaler()

    # 计算类别权重
    print("计算类别权重...")
    pos_counts = np.zeros(num_classes)
    sample_count = len(train_dataloader.dataset)
    
    # 统计正样本数量
    for sample in train_dataloader.dataset.samples:
        labels = sample['labels']
        for i in range(num_classes):
            if labels[i] > 0:
                pos_counts[i] += 1
    
    # 计算并限制权重范围
    pos_weights = np.zeros(num_classes)
    for i in range(num_classes):
        if pos_counts[i] > 0:
            pos_weights[i] = (sample_count - pos_counts[i]) / pos_counts[i]
        else:
            pos_weights[i] = 1.0
    pos_weights = np.clip(pos_weights, 0.1, 10.0)
    pos_weight = torch.FloatTensor(pos_weights).to(device)
    
    # 打印类别统计信息
    label_names = train_dataloader.dataset.idx_to_label if hasattr(train_dataloader.dataset, 'idx_to_label') else [f"类别{i}" for i in range(num_classes)]
    print("\n类别分布统计:")
    for i in range(num_classes):
        print(f"{label_names[i]}: 正样本数={pos_counts[i]}/{sample_count} ({pos_counts[i]/sample_count*100:.2f}%), 权重={pos_weights[i]:.2f}")
    
    # 损失函数、优化器和学习率调度器
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 创建实验目录
    current_time = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    experiment_dir = os.path.join(save_dir, f"{current_time}_C3D_neonatal_multilabel")
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    
    # 设置TensorBoard和保存标签映射
    tb_writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tensorboard'))
    label_map = train_dataloader.dataset.label_to_idx
    idx_to_label = {v: k for k, v in label_map.items()}
    
    # 保存标签映射
    with open(os.path.join(experiment_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump({'label_map': label_map, 'idx_to_label': idx_to_label}, f, indent=4, ensure_ascii=False)
    
    # 训练配置信息
    print("\n=== 训练配置 ===")
    print(f"批次大小: {train_dataloader.batch_size}, 学习率: {lr}, 类别数量: {num_classes}")
    print(f"训练样本数: {len(train_dataloader.dataset)}, 验证样本数: {len(val_dataloader.dataset)}, 测试样本数: {len(test_dataloader.dataset)}")
    print(f"模型保存路径: {experiment_dir}")
    print("=================\n")

    # 记录训练指标
    metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_f1', 'val_loss', 'val_f1', 'learning_rate'])
    
    # 初始化训练
    best_val_f1 = 0.0
    torch.cuda.empty_cache()
    
    # 预热CUDA
    warmup_cuda(model, train_dataloader, device)
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # 训练阶段
        train_metrics = evaluate_model(model, train_dataloader, criterion, device, 
                                      mode='train', optimizer=optimizer, scaler=scaler)
        
        # 验证阶段
        val_metrics = evaluate_model(model, val_dataloader, criterion, device, mode='val')
        
        # 更新学习率
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录指标到TensorBoard
        tb_writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        tb_writer.add_scalar('F1/train', train_metrics['f1'], epoch)
        tb_writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        tb_writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        
        # 记录每个类别的验证指标
        for i in range(num_classes):
            class_name = idx_to_label[i]
            class_f1 = f1_score(val_metrics['targets'][:, i], val_metrics['predictions'][:, i], zero_division=0)
            tb_writer.add_scalar(f'Class_F1/{class_name}', class_f1, epoch)
        
        # 保存当前指标
        new_row = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_f1': train_metrics['f1'],
            'val_loss': val_metrics['loss'],
            'val_f1': val_metrics['f1'],
            'learning_rate': current_lr
        }
        
        # 使用loc方法添加行，避免concat产生的警告
        metrics_df.loc[len(metrics_df)] = new_row
        metrics_df.to_csv(os.path.join(experiment_dir, 'training_metrics.csv'), index=False)
        
        # 打印当前结果
        print(f"训练: Loss={train_metrics['loss']:.4f}, F1={train_metrics['f1']:.4f} | "
              f"验证: Loss={val_metrics['loss']:.4f}, F1={val_metrics['f1']:.4f} | "
              f"学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'val_loss': val_metrics['loss'],
            }, os.path.join(experiment_dir, 'checkpoints', 'best_model.pth'))
            print(f"已保存最佳模型 (val_f1: {val_metrics['f1']:.4f})")
        
        torch.cuda.empty_cache()

    # 关闭TensorBoard
    tb_writer.close()

    # 测试阶段
    print("\n开始测试最佳模型...")
    best_model_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    
    # 测试最佳模型
    test_metrics = evaluate_model(model, test_dataloader, criterion, device, mode='test')
    
    # 计算并保存每个类别的测试指标
    class_metrics = {}
    for i, label in idx_to_label.items():
        pos_samples = np.sum(test_metrics['targets'][:, i] == 1)
        neg_samples = np.sum(test_metrics['targets'][:, i] == 0)
        
        class_precision = precision_score(test_metrics['targets'][:, i], test_metrics['predictions'][:, i], zero_division=0)
        class_recall = recall_score(test_metrics['targets'][:, i], test_metrics['predictions'][:, i], zero_division=0)
        class_f1 = f1_score(test_metrics['targets'][:, i], test_metrics['predictions'][:, i], zero_division=0)
        
        class_metrics[label] = {
            'f1': float(class_f1),
            'precision': float(class_precision),
            'recall': float(class_recall),
            'pos_samples': int(pos_samples),
            'neg_samples': int(neg_samples)
        }

    # 保存测试结果
    test_results = {
        'test_loss': float(test_metrics['loss']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1']),
        'class_metrics': class_metrics
    }
    
    with open(os.path.join(experiment_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=4, ensure_ascii=False)

    # 打印测试结果
    print("\n测试结果:")
    print(f"损失: {test_metrics['loss']:.4f}, F1: {test_metrics['f1']:.4f}, "
          f"精确率: {test_metrics['precision']:.4f}, 召回率: {test_metrics['recall']:.4f}")
    print("\n各类别指标:")
    print(f"{'类别':<15} {'F1':<8} {'精确率':<8} {'召回率':<8} {'正样本':<8} {'负样本':<8}")
    print("-" * 60)
    
    for label, metrics in class_metrics.items():
        print(f"{label:<15} {metrics['f1']:.4f}    {metrics['precision']:.4f}    {metrics['recall']:.4f}    "
              f"{metrics['pos_samples']:<8} {metrics['neg_samples']:<8}")
              
    return model, test_results


def main():
    args = parse_args()
    
    # 设置随机种子
    generator, seed_worker = set_random_seeds(seed=42)
    print('Random seed set to: 42')
    print('Deterministic mode enabled for reproducibility')

    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        print(f"使用GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("使用CPU训练")

    # 设置数据路径
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    data_path = data_path.replace("\\", "/")
    label_file = os.path.join(data_path, "neonatal_behavior_labels.json")
    label_file = label_file.replace("\\", "/")

    # 打印数据加载配置
    print("\n=== 数据加载配置 ===")
    print(f"GPU: {args.gpu}, 批次大小: {args.batch_size}, 学习率: {args.lr}")
    print(f"数据加载模式: {args.load_mode}, 工作线程数: {args.num_workers}")
    print(f"使用预训练权重: {args.use_pretrained}, 路径: {args.pretrained_path}")
    if args.frames_dir:
        print(f"预处理帧目录: {args.frames_dir}")
    print("===================\n")

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 创建数据集
    print("加载数据集...")
    datasets = {}
    for mode in ['train', 'val', 'test']:
        datasets[mode] = NeonatalVideoDataset(
            data_path, label_file, clip_len=args.clip_len, mode=mode,
            prefetch_videos=args.prefetch_videos, max_prefetch=args.max_prefetch,
            transform_in_worker=args.transform_in_worker, load_mode=args.load_mode,
            frames_dir=args.frames_dir)
        print(f"{mode}数据集大小: {len(datasets[mode])}")

    # 创建数据加载器
    dataloaders = {}
    for mode, dataset in datasets.items():
        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(mode == 'train'),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=2,
            drop_last=(mode == 'train'),
            generator=generator,
            worker_init_fn=seed_worker
        )

    # 开始训练
    train_model(
        num_epochs=args.epochs,
        num_classes=datasets['train'].num_classes,
        lr=args.lr,
        device=device,
        save_dir=args.model_save_dir,
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders['val'],
        test_dataloader=dataloaders['test'],
        use_pretrained=args.use_pretrained,
        pretrained_path=args.pretrained_path
    )

if __name__ == "__main__":
    main()

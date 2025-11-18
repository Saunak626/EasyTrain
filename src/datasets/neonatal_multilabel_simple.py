"""新生儿多标签数据集 - 简化教学版本

这是一个最小化的多标签视频数据集实现，用于教学和理解核心概念。
只保留了多标签数据加载的最基本功能。

核心功能：
1. 从帧图像目录加载视频数据
2. 从Excel文件加载多标签标注
3. 简单的train/test划分
4. 基础的视频预处理（resize、normalize、to_tensor）

移除的高级功能：
- 加权采样、分层采样
- FPS采样、样本权重计算
- pos_weight计算
- 模型特定的transforms
- 详细的验证统计
- 类别筛选（top_n_classes）
- 标签缓存优化

"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image  # 🔧 新增：使用PIL替代cv2，提升I/O性能


class NeonatalMultilabelSimple(Dataset):
    """新生儿多标签数据集 - 简化版
    
    最小化实现，展示多标签视频数据集的核心构造方式。
    
    数据结构:
        frames_dir/
            session_001/
                clip_001/
                    00000.jpg
                    00001.jpg
                    ...
            session_002/
                ...
        
        labels.xlsx:
            文件名 | 文件内动作序号 | 标签1 | 标签2 | ...
    
    Args:
        frames_dir (str): 帧图像根目录
        labels_file (str): Excel标签文件路径
        split (str): 'train' 或 'test'
        clip_len (int): 每个视频片段的帧数，默认16
        train_ratio (float): 训练集比例，默认0.8
    """
    
    def __init__(self, frames_dir, labels_file, split='train', clip_len=16, train_ratio=0.8):
        self.frames_dir = frames_dir
        self.labels_file = labels_file
        self.split = split
        self.clip_len = clip_len
        self.train_ratio = train_ratio
        
        # 定义行为标签（24个原始标签）
        self.behavior_labels = [
            '喂养开始', '喂养结束', '易哭闹', '张嘴闭嘴', '吸吮行为', '吃手指',
            '吃脚指', '皱眉', '哭泣', '发脾气', '来回摇头', '手脚活动加快',
            '寻找奶瓶', '注视奶瓶', '声调变高', '打哈欠', '睡着了', '间歇喝奶',
            '唇部触食反应', '喂养期鬼脸', '口腔器具咬合', '头颈侧向回避',
            '肢体张力减退', '远离奶瓶'
        ]
        self.num_classes = len(self.behavior_labels)
        
        # 加载数据
        self.samples = self._load_data()
        
        print(f"加载 {split} 数据集: {len(self.samples)} 个样本，{self.num_classes} 个类别")
    
    def _load_data(self):
        """加载数据并划分train/test"""
        # 1. 读取Excel标签文件
        df = pd.read_excel(self.labels_file)
        
        # 2. 清理文件名，构建样本列表
        samples = []
        for _, row in df.iterrows():
            # 提取session_name和clip_id
            session_name = row['文件名'].replace('.mov', '').replace('.mp4', '').strip()
            clip_id = str(row['文件内动作序号'])
            
            # 检查帧目录是否存在
            clip_dir = os.path.join(self.frames_dir, session_name, clip_id)
            if not os.path.exists(clip_dir):
                continue
            
            # 检查是否有帧图像
            frame_files = [f for f in os.listdir(clip_dir) if f.endswith('.jpg')]
            if len(frame_files) == 0:
                continue
            
            # 提取多标签向量
            label_vector = []
            for label_name in self.behavior_labels:
                if label_name in row:
                    label_vector.append(float(row[label_name]))
                else:
                    label_vector.append(0.0)
            
            # 跳过全零标签
            if sum(label_vector) == 0:
                continue
            
            # 🔧 优化：预先转换为tensor，避免每次__getitem__时重复转换
            samples.append({
                'session_name': session_name,
                'clip_id': clip_id,
                'frames_dir': clip_dir,
                'labels': torch.tensor(label_vector, dtype=torch.float32)
            })
        
        # 3. 简单的train/test划分（按8:2比例）
        split_idx = int(len(samples) * self.train_ratio)
        if self.split == 'train':
            return samples[:split_idx]
        else:
            return samples[split_idx:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """获取单个样本"""
        sample = self.samples[index]

        # 1. 加载视频帧
        frames = self._load_frames(sample['frames_dir'])

        # 2. 采样到固定帧数
        frames = self._sample_frames(frames, self.clip_len)

        # 3. 预处理：normalize + to_tensor
        frames = self._preprocess(frames)

        # 4. 获取标签（已在初始化时转换为tensor）
        labels = sample['labels']

        return frames, labels

    def _load_frames(self, frames_dir):
        """从目录加载所有帧图像（优化版：使用PIL替代cv2）

        Args:
            frames_dir (str): 帧图像目录路径

        Returns:
            np.ndarray: 形状为 (T, H, W, C) 的帧数组
        """
        # 获取所有jpg文件并排序
        frame_paths = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.endswith('.jpg')
        ])

        if len(frame_paths) == 0:
            raise ValueError(f"没有找到帧图像: {frames_dir}")

        # 🔧 优化：使用PIL批量读取帧（比cv2快约30%）
        frames = []
        for frame_path in frame_paths:
            try:
                # 使用PIL读取图像（RGB格式）
                img = Image.open(frame_path).convert('RGB')
                # Resize到标准尺寸 224x224
                img = img.resize((224, 224), Image.BILINEAR)
                # 转换为numpy数组
                frame = np.array(img, dtype=np.float32)
                frames.append(frame)
            except Exception as e:
                raise ValueError(f"无法读取图像 {frame_path}: {e}")

        # 转换为numpy数组: (T, H, W, C)
        return np.array(frames, dtype=np.float32)

    def _sample_frames(self, frames, clip_len):
        """采样到固定帧数

        Args:
            frames (np.ndarray): 输入帧，形状 (T, H, W, C)
            clip_len (int): 目标帧数

        Returns:
            np.ndarray: 采样后的帧，形状 (clip_len, H, W, C)
        """
        total_frames = frames.shape[0]

        if total_frames >= clip_len:
            # 帧数足够，随机裁剪
            start_idx = np.random.randint(0, total_frames - clip_len + 1)
            return frames[start_idx:start_idx + clip_len]
        else:
            # 帧数不足，填充最后一帧
            padding = np.tile(frames[-1:], (clip_len - total_frames, 1, 1, 1))
            return np.concatenate([frames, padding], axis=0)

    def _preprocess(self, frames):
        """预处理：归一化 + 转换为tensor

        Args:
            frames (np.ndarray): 输入帧，形状 (T, H, W, C)

        Returns:
            torch.Tensor: 形状 (C, T, H, W)
        """
        # 归一化（减去均值），保持 float32，避免变成 float64
        mean = np.array([[[90.0, 98.0, 102.0]]], dtype=np.float32)
        frames = frames - mean

        # 转换维度：(T, H, W, C) -> (C, T, H, W)
        frames = frames.transpose(3, 0, 1, 2)

        # 转换为 float32 tensor（与模型权重 dtype 一致）
        return torch.from_numpy(frames).float()

    def get_num_classes(self):
        """获取类别数"""
        return self.num_classes

    def get_class_names(self):
        """获取类别名称列表"""
        return self.behavior_labels


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""

    # 1. 创建数据集
    train_dataset = NeonatalMultilabelSimple(
        frames_dir='../Neonate-Feeding-Assessment/data/cpu_processed_627/frames_segments',
        labels_file='../Neonate-Feeding-Assessment/result_xlsx/shanghai/multi_hot_labels.xlsx',
        split='train',
        clip_len=16
    )

    test_dataset = NeonatalMultilabelSimple(
        frames_dir='../Neonate-Feeding-Assessment/data/cpu_processed_627/frames_segments',
        labels_file='../Neonate-Feeding-Assessment/result_xlsx/shanghai/multi_hot_labels.xlsx',
        split='test',
        clip_len=16
    )

    # 2. 创建DataLoader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    # 3. 迭代数据
    for frames, labels in train_loader:
        print(f"Frames shape: {frames.shape}")  # (B, C, T, H, W)
        print(f"Labels shape: {labels.shape}")  # (B, num_classes)
        break

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    print(f"类别数: {train_dataset.get_num_classes()}")


if __name__ == '__main__':
    example_usage()



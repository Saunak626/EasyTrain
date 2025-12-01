"""视频数据集 - UCF-101帧序列实现

简化版实现，参考 tmp/dataset_simple.py 的 UCF101FrameDataset。
使用简单的预处理：resize + normalize([0,1]) + permute。
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """UCF-101帧序列数据集
    
    从预处理的帧图像目录加载数据。
    结构: root/split/类别名/视频名/帧图片.jpg
    
    Args:
        dataset_path: 数据集根目录
        images_path: 子目录名 ('train', 'val', 'test')
        clip_len: 每个视频片段的帧数
        img_size: 输出图像尺寸 (H, W)
    """
    
    def __init__(self, dataset_path, images_path, clip_len=16, img_size=(112, 112), **kwargs):
        self.dataset_path = dataset_path
        self.split = images_path
        self.clip_len = clip_len
        self.img_size = img_size
        
        # 扫描类别和样本
        folder = os.path.join(self.dataset_path, images_path)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"数据目录不存在: {folder}")
            
        self.class_names = sorted([d for d in os.listdir(folder) 
                                   if os.path.isdir(os.path.join(folder, d))])
        self.label_map = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        # 收集所有样本
        self.samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(folder, class_name)
            class_idx = self.label_map[class_name]
            
            for video_name in sorted(os.listdir(class_dir)):
                video_dir = os.path.join(class_dir, video_name)
                if os.path.isdir(video_dir):
                    self.samples.append((video_dir, class_idx))
        
        if not self.samples:
            raise ValueError(f"未找到有效样本: {folder}")
        
        print(f"[VideoDataset] {images_path}: {len(self.samples)} 样本, {self.num_classes} 类别")
    
    def __len__(self):
        return len(self.samples)
    
    def _get_sample_indices(self, total_frames):
        """计算均匀采样的帧索引"""
        return np.linspace(0, total_frames - 1, self.clip_len, dtype=int)
    
    def _load_frames(self, video_dir):
        """加载视频帧
        
        预处理流程：resize(112x112) + normalize([0,1]) + permute
        
        Returns:
            frames: (C, T, H, W) 归一化到 [0,1] 的张量，适配3D CNN
        """
        # 获取所有帧文件
        frame_files = sorted([f for f in os.listdir(video_dir) 
                             if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        if not frame_files:
            raise ValueError(f"目录无帧: {video_dir}")
        
        total_frames = len(frame_files)
        sample_indices = self._get_sample_indices(total_frames)
        
        frames = []
        for idx in sample_indices:
            frame_path = os.path.join(video_dir, frame_files[idx])
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]))
                frames.append(frame)
            else:
                # 读取失败时使用前一帧或零帧
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8))
        
        # 转换为 numpy 数组
        frames = np.stack(frames)  # (T, H, W, C)
        
        # 归一化到 [0, 1] 并转换维度
        frames = frames.astype(np.float32) / 255.0
        frames = frames.transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        
        # 转为 (C, T, H, W) 格式，适配 3D CNN
        frames = torch.from_numpy(frames).permute(1, 0, 2, 3)  # (C, T, H, W)
        
        return frames
    
    def __getitem__(self, index):
        video_dir, label = self.samples[index]
        frames = self._load_frames(video_dir)
        return frames, torch.tensor(label, dtype=torch.long)
    
    def get_class_names(self):
        return self.class_names
    
    def get_sample_id(self, index):
        """获取样本ID"""
        video_dir, _ = self.samples[index]
        rel_path = os.path.relpath(video_dir, self.dataset_path)
        return rel_path.replace('\\', '/')


class CombinedVideoDataset(Dataset):
    """合并 val 和 test 数据集"""
    
    def __init__(self, dataset_path, clip_len=16, img_size=(112, 112), **kwargs):
        self.val_dataset = VideoDataset(dataset_path, 'val', clip_len, img_size)
        self.test_dataset = VideoDataset(dataset_path, 'test', clip_len, img_size)
        
        self.val_len = len(self.val_dataset)
        self.total_len = self.val_len + len(self.test_dataset)
        
        # 继承属性
        self.class_names = self.val_dataset.class_names
        self.num_classes = self.val_dataset.num_classes
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, index):
        if index < self.val_len:
            return self.val_dataset[index]
        else:
            return self.test_dataset[index - self.val_len]
    
    def get_class_names(self):
        return self.class_names
    
    def get_sample_id(self, index):
        if index < self.val_len:
            return self.val_dataset.get_sample_id(index)
        else:
            return self.test_dataset.get_sample_id(index - self.val_len)


if __name__ == "__main__":
    # 测试
    train_data = VideoDataset(dataset_path='data/ucf101', images_path='train', clip_len=16)
    print(f"训练集: {len(train_data)} 样本")
    
    # 测试数据加载
    frames, label = train_data[0]
    print(f"帧形状: {frames.shape}, 标签: {label}")
    print(f"数值范围: [{frames.min():.3f}, {frames.max():.3f}]")

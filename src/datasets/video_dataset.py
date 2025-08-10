"""视频数据集基类和UCF-101实现

该模块提供了视频数据集的基础接口和UCF-101的具体实现。
支持从预处理帧图像加载数据，提供统一的视频数据集接口。

Classes:
    BaseVideoDataset: 视频数据集基类
    VideoDataset: UCF-101视频帧数据集实现
    CombinedVideoDataset: 合并多个数据集的包装器
"""

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class BaseVideoDataset(Dataset, ABC):
    """视频数据集基类

    定义了视频数据集的通用接口和基础功能。
    所有视频数据集实现都应该继承此类。

    Attributes:
        num_classes (int): 数据集的类别数
        clip_len (int): 每个视频片段的帧数
        class_names (list): 类别名称列表
    """

    def __init__(self, clip_len=16):
        """初始化基础视频数据集

        Args:
            clip_len (int): 每个视频片段的帧数，默认16
        """
        self.clip_len = clip_len
        self.num_classes = None
        self.class_names = []

    @abstractmethod
    def __len__(self):
        """返回数据集大小"""
        pass

    @abstractmethod
    def __getitem__(self, index):
        """获取数据项

        Args:
            index (int): 数据索引

        Returns:
            tuple: (video_tensor, label) 其中video_tensor形状为(C, T, H, W)
        """
        pass

    def get_num_classes(self):
        """获取类别数"""
        return self.num_classes

    def get_class_names(self):
        """获取类别名称列表"""
        return self.class_names


class VideoDataset(BaseVideoDataset):
    """UCF-101视频帧数据集类
    
    从预处理的帧图像中加载UCF-101数据集，支持train/val/test目录结构。
    
    Args:
        dataset_path (str): 数据集根目录路径
        images_path (str): 子目录名称，如'train'、'val'、'test'
        clip_len (int): 每个视频片段的帧数
    """
    
    def __init__(self, dataset_path, images_path, clip_len=16):
        super().__init__(clip_len)
        self.dataset_path = dataset_path  # 数据集的地址
        self.split = images_path  # 训练集，测试集，验证集的名字

        # 后续数据预处理的值
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        # 直接从目录结构创建标签映射
        folder = os.path.join(self.dataset_path, images_path)
        class_names = sorted(os.listdir(folder))

        # 创建类别名到索引的映射
        self.label_map = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # 收集所有样本文件路径和标签
        self.fnames, labels = [], []
        for class_name in class_names:
            class_dir = os.path.join(folder, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for fname in os.listdir(class_dir):
                sample_dir = os.path.join(class_dir, fname)
                if os.path.isdir(sample_dir):  # 确保是目录（包含帧图像）
                    self.fnames.append(sample_dir)
                    labels.append(self.label_map[class_name])

        if not self.fnames:
            raise ValueError("没有找到有效的训练数据！请检查数据集路径和标签名称。")

        self.label_array = np.array(labels, dtype=int)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # 加载对应类别的动作数据集，并转化为（帧数， 高， 宽， 3通道）
        buffer = self.load_frames(self.fnames[index])

        # 原始视频分辨率约为3:4 (128*171)
        # 在数据的深度，高度，宽度方向进行随机裁剪，将数据数据转化为（clip_len， 112, 112， 3）
        buffer = self.crop(buffer, self.clip_len, self.crop_size)

        buffer = self.normalize(buffer)  # 对模型进行归一化处理
        buffer = self.to_tensor(buffer)  # 对维度进行转化

        # 获取对应视频的标签数据
        labels = np.array(self.label_array[index])

        # 返回torch格式的特征和标签
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def load_frames(self, file_dir):
        """从目录中加载视频帧"""
        frame_paths = sorted([os.path.join(file_dir, img)
                             for img in os.listdir(file_dir)])
        frame_count = len(frame_paths)
        # print(f"Video {file_dir} has {frame_count} frames")  # 添加这行来调试

        # 生成一个空的 (frame_count, resize_height, resize_width, 3) 维度的数据
        buffer = np.empty((frame_count, self.resize_height,
                          self.resize_width, 3), np.dtype('float32'))
        # 遍历循环获取动作的路径
        for i, frame_name in enumerate(frame_paths):
            # 利用cv去读取图片数据，并转化为np.array格式
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            # 不断遍历循环赋值给buffer
            buffer[i] = frame
        return buffer

    def crop(self, buffer, clip_len, crop_size):
        """对视频帧进行时间和空间裁剪"""
        # 添加帧数不足时的处理逻辑
        if buffer.shape[0] <= clip_len:
            # 方案1：重复最后一帧直到达到所需帧数
            last_frame = buffer[-1]
            pad_size = clip_len - buffer.shape[0]
            padding = np.tile(last_frame, (pad_size, 1, 1, 1))
            buffer = np.concatenate([buffer, padding], axis=0)
            time_index = 0
        else:
            time_index = np.random.randint(buffer.shape[0] - clip_len)

        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        buffer = buffer[time_index:time_index + clip_len,
                        height_index:height_index + crop_size,
                        width_index:width_index + crop_size, :]

        return buffer

    def normalize(self, buffer):
        """对视频帧进行归一化处理"""
        # 进行归一化
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def to_tensor(self, buffer):
        """将numpy数组转换为tensor格式"""
        # 进行维度的转化，将最后的一个维调转到第一维
        return buffer.transpose((3, 0, 1, 2))

    def get_sample_id(self, index):
        """获取样本ID"""
        # 返回相对路径作为样本ID
        full_path = self.fnames[index]
        rel_path = os.path.relpath(full_path, self.dataset_path)
        # 使用正斜杠替换反斜杠，保持路径格式一致
        return rel_path.replace('\\', '/')


class CombinedVideoDataset(Dataset):
    """合并val和test数据集的包装类"""
    
    def __init__(self, dataset_path, clip_len):
        """初始化合并数据集
        
        Args:
            dataset_path (str): 数据集根目录路径
            clip_len (int): 每个视频片段的帧数
        """
        # 创建val和test数据集
        self.val_dataset = VideoDataset(dataset_path, 'val', clip_len)
        self.test_dataset = VideoDataset(dataset_path, 'test', clip_len)
        
        # 计算总长度
        self.val_len = len(self.val_dataset)
        self.test_len = len(self.test_dataset)
        self.total_len = self.val_len + self.test_len
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, index):
        if index < self.val_len:
            return self.val_dataset[index]
        else:
            return self.test_dataset[index - self.val_len]
    
    def get_sample_id(self, index):
        if index < self.val_len:
            return self.val_dataset.get_sample_id(index)
        else:
            return self.test_dataset.get_sample_id(index - self.val_len)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # 测试单独的数据集
    train_data = VideoDataset(
        dataset_path='data/ucf101', images_path='train', clip_len=16)
    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=0)

    val_data = VideoDataset(dataset_path='data/ucf101',
                            images_path='val', clip_len=16)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=0)

    test_data = VideoDataset(dataset_path='data/ucf101',
                             images_path='test', clip_len=16)
    test_loader = DataLoader(test_data, batch_size=64,
                            shuffle=True, num_workers=0)
    
    # 测试合并数据集
    combined_test_data = CombinedVideoDataset(
        dataset_path='data/ucf101', clip_len=16)
    combined_test_loader = DataLoader(
        combined_test_data, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")
    print(f"合并测试集大小: {len(combined_test_data)}")
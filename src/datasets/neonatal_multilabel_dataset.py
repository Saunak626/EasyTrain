"""新生儿多标签行为识别数据集

该模块实现了基于CPU处理帧图像和Excel标签文件的新生儿多标签行为识别数据集。
参考UCF101视频数据集的实现，支持从预处理帧图像加载数据。

Classes:
    NeonatalMultilabelDataset: 新生儿多标签行为识别数据集实现
"""

import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging
from .label_cache import LabelCache, OptimizedLabelProcessor

logger = logging.getLogger(__name__)


class NeonatalMultilabelDataset(Dataset):
    """新生儿多标签行为识别数据集
    
    从预处理的帧图像和Excel标签文件中加载新生儿行为数据，支持多标签分类。
    
    数据结构:
    - 帧图像: frames_segments/session_name/clip_id/00000.jpg
    - 标签: multi_hot_labels.xlsx (24维多标签向量)
    
    Args:
        frames_dir (str): 帧图像根目录路径
        labels_file (str): Excel标签文件路径
        split (str): 数据集分割 ('train', 'test')
        clip_len (int): 每个视频片段的帧数
        model_type (str): 模型类型，用于获取对应的transforms
    """
    
    def __init__(self, frames_dir, labels_file, split='train', clip_len=16, model_type=None):
        """初始化新生儿多标签数据集
        
        Args:
            frames_dir (str): 帧图像根目录路径
            labels_file (str): Excel标签文件路径
            split (str): 数据集分割，'train'或'test'
            clip_len (int): 每个视频片段的帧数，默认16
            model_type (str): 模型类型，用于获取对应的transforms
        """
        self.frames_dir = frames_dir
        self.labels_file = labels_file
        self.split = split
        self.clip_len = clip_len
        self.model_type = model_type
        
        # 获取模型特定的transforms
        self.model_transforms = self._get_model_transforms()
        
        # 优化的预处理参数
        self.resize_height = 224
        self.resize_width = 224
        self.crop_size = 112
        
        # 定义24个行为标签
        self.behavior_labels = [
            '喂养开始', '喂养结束', '易哭闹', '张嘴闭嘴', '吸吮行为', '吃手指', 
            '吃脚指', '皱眉', '哭泣', '发脾气', '来回摇头', '手脚活动加快', 
            '寻找奶瓶', '注视奶瓶', '声调变高', '打哈欠', '睡着了', '间歇喝奶', 
            '唇部触食反应', '喂养期鬼脸', '口腔器具咬合', '头颈侧向回避', 
            '肢体张力减退', '远离奶瓶'
        ]
        
        self.num_classes = len(self.behavior_labels)
        self.class_names = self.behavior_labels
        
        # 使用优化的标签处理器
        self.label_processor = OptimizedLabelProcessor(self.labels_file, self.behavior_labels)

        # 加载和处理数据（使用缓存）
        self.samples = self._load_samples_optimized()

        # 数据分割
        self.samples = self._split_data(self.samples, split)
        
        logger.info(f"加载 {split} 数据集: {len(self.samples)} 个样本")
    
    def _get_model_transforms(self):
        """获取模型特定的transforms（参考UCF101实现）"""
        if self.model_type:
            try:
                from ..models.model_registry import get_video_model_transforms, validate_model_transforms_compatibility

                # 获取transforms
                transforms = get_video_model_transforms(self.model_type)
                if transforms is None:
                    return None

                # 验证兼容性
                is_compatible, message = validate_model_transforms_compatibility(self.model_type, verbose=False)
                if not is_compatible:
                    logger.warning(f"{self.model_type} transforms不兼容: {message}，回退到传统预处理")
                    return None

                return transforms

            except ImportError:
                # 如果导入失败，使用传统方式
                pass
        return None
    
    def _load_samples(self):
        """加载样本数据"""
        logger.info(f"从 {self.labels_file} 加载标签数据...")
        
        # 读取标签文件
        df = pd.read_excel(self.labels_file)
        logger.info(f"标签文件包含 {len(df)} 行数据")
        
        samples = []
        available_sessions = set(os.listdir(self.frames_dir))
        
        for idx, row in df.iterrows():
            # 提取文件名和clip序号
            filename = str(row['文件名']).strip()
            clip_id = str(int(row['文件内动作序号']))
            
            # 清理文件名（移除.mov等扩展名）
            session_name = filename.replace('.mov', '').replace('.mp4', '')
            
            # 检查对应的帧图像目录是否存在
            session_dir = os.path.join(self.frames_dir, session_name)
            clip_dir = os.path.join(session_dir, clip_id)
            
            if not os.path.exists(clip_dir):
                continue
            
            # 检查是否有帧图像
            frame_files = [f for f in os.listdir(clip_dir) if f.endswith('.jpg')]
            if len(frame_files) == 0:
                continue
            
            # 提取多标签向量
            label_vector = []
            for label in self.behavior_labels:
                if label in row:
                    label_vector.append(float(row[label]))
                else:
                    label_vector.append(0.0)
            
            # 跳过全零标签（可选，根据需求调整）
            if sum(label_vector) == 0:
                continue
            
            sample = {
                'session_name': session_name,
                'clip_id': clip_id,
                'frames_dir': clip_dir,
                'labels': label_vector,
                'start_time': row.get('开始时间(秒)', 0),
                'end_time': row.get('结束时间(秒)', 0),
                'duration': row.get('时长(秒)', 0)
            }
            samples.append(sample)
        
        logger.info(f"成功加载 {len(samples)} 个有效样本")
        return samples

    def _load_samples_optimized(self):
        """使用优化的标签处理器加载样本数据"""
        logger.info(f"使用优化缓存加载样本数据...")

        # 使用优化的标签处理器获取所有有效样本
        samples = self.label_processor.get_all_valid_samples(self.frames_dir)

        logger.info(f"优化加载完成: {len(samples)} 个有效样本")
        return samples
    
    def _split_data(self, samples, split):
        """数据分割（8:2分割）"""
        if len(samples) <= 2:
            return samples
        
        # 8:2分割，但确保测试集至少有1个样本
        split_idx = max(1, int(len(samples) * 0.8))
        
        if split == 'train':
            return samples[:split_idx]
        else:
            return samples[split_idx:]
    
    def set_model_type(self, model_type):
        """设置模型类型并更新transforms（用于网格搜索）"""
        self.model_type = model_type
        self.model_transforms = self._get_model_transforms()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """获取单个样本"""
        sample = self.samples[index]
        
        # 加载视频帧
        buffer = self.load_frames(sample['frames_dir'])
        
        # 如果有模型特定的transforms，使用官方transforms
        if self.model_transforms is not None:
            # 简单的时间维度采样，保持原始空间尺寸
            if buffer.shape[0] > self.clip_len:
                # 随机选择起始帧
                start_idx = np.random.randint(0, buffer.shape[0] - self.clip_len + 1)
                buffer = buffer[start_idx:start_idx + self.clip_len]
            elif buffer.shape[0] < self.clip_len:
                # 重复最后一帧
                last_frame = buffer[-1]
                pad_size = self.clip_len - buffer.shape[0]
                padding = np.tile(last_frame[np.newaxis], (pad_size, 1, 1, 1))
                buffer = np.concatenate([buffer, padding], axis=0)

            # 转换为torch tensor格式: (T, H, W, C) -> (T, C, H, W)
            buffer = torch.from_numpy(buffer).float() / 255.0
            buffer = buffer.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

            # 应用模型特定的官方transforms (输入: T, C, H, W -> 输出: C, T, H, W)
            buffer = self.model_transforms(buffer)
        else:
            # 使用传统的预处理方式（向后兼容）
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
            buffer = self.normalize(buffer)  # 对模型进行归一化处理
            buffer = self.to_tensor(buffer)  # 对维度进行转化
            buffer = torch.from_numpy(buffer)
        
        # 获取多标签向量
        labels = torch.tensor(sample['labels'], dtype=torch.float32)
        
        # 返回torch格式的特征和标签
        return buffer, labels
    
    def load_frames(self, frames_dir):
        """从目录中加载视频帧（优化版本，减少内存占用和加载时间）"""
        frame_paths = sorted([os.path.join(frames_dir, img)
                             for img in os.listdir(frames_dir) if img.endswith('.jpg')])
        frame_count = len(frame_paths)

        if frame_count == 0:
            raise ValueError(f"没有找到帧图像文件: {frames_dir}")

        # 限制最大帧数，避免内存过载
        max_frames = 60  # 限制最大帧数
        if frame_count > max_frames:
            # 均匀采样，而非截断
            indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
            frame_count = max_frames

        # 读取第一帧以获取实际尺寸
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            raise ValueError(f"无法读取第一帧图像: {frame_paths[0]}")

        actual_height, actual_width = first_frame.shape[:2]

        # 预处理尺寸，减少内存占用
        target_height, target_width = 224, 224  # 标准尺寸

        # 根据目标尺寸创建缓冲区
        buffer = np.empty((frame_count, target_height, target_width, 3), dtype=np.float32)

        # 处理第一帧
        if (actual_height, actual_width) != (target_height, target_width):
            first_frame = cv2.resize(first_frame, (target_width, target_height))
        buffer[0] = first_frame.astype(np.float32)  # 直接使用float32

        # 批量加载剩余帧
        for i, frame_name in enumerate(frame_paths[1:], 1):
            frame = cv2.imread(frame_name)
            if frame is None:
                raise ValueError(f"无法读取图像文件: {frame_name}")

            # 统一resize到目标尺寸
            if frame.shape[:2] != (target_height, target_width):
                frame = cv2.resize(frame, (target_width, target_height))

            buffer[i] = frame.astype(np.float32)

        return buffer
    
    def crop(self, buffer, clip_len, crop_size):
        """对视频帧进行时间和空间裁剪（适应不同输入尺寸）"""
        # 处理时间维度
        if buffer.shape[0] <= clip_len:
            # 重复最后一帧直到达到所需帧数
            if buffer.shape[0] == 0:
                # 如果没有帧，创建黑色帧
                buffer = np.zeros((1, buffer.shape[1], buffer.shape[2], 3), dtype=buffer.dtype)

            last_frame = buffer[-1]
            pad_size = clip_len - buffer.shape[0]
            padding = np.tile(last_frame[np.newaxis], (pad_size, 1, 1, 1))
            buffer = np.concatenate([buffer, padding], axis=0)
            time_index = 0
        else:
            time_index = np.random.randint(buffer.shape[0] - clip_len)

        buffer = buffer[time_index:time_index + clip_len, :, :, :]

        # 处理空间维度 - 如果输入尺寸小于crop_size，则resize；否则随机裁剪
        target_shape = (clip_len, crop_size, crop_size, 3)

        if buffer.shape[1] < crop_size or buffer.shape[2] < crop_size:
            # 输入尺寸小于目标尺寸，进行resize
            resized_buffer = np.zeros(target_shape, dtype=buffer.dtype)
            for i in range(clip_len):
                resized_buffer[i] = cv2.resize(buffer[i], (crop_size, crop_size))
            return resized_buffer
        else:
            # 输入尺寸大于等于目标尺寸，进行随机裁剪
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)

            buffer = buffer[:,
                            height_index:height_index + crop_size,
                            width_index:width_index + crop_size, :]
            return buffer

    def normalize(self, buffer):
        """对视频帧进行归一化处理（参考UCF101实现）"""
        # 进行归一化
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def to_tensor(self, buffer):
        """将numpy数组转换为tensor格式（参考UCF101实现）"""
        # 进行维度的转化，将最后的一个维调转到第一维
        return buffer.transpose((3, 0, 1, 2))
    
    def get_num_classes(self):
        """获取类别数"""
        return self.num_classes
    
    def get_class_names(self):
        """获取类别名称列表"""
        return self.class_names
    
    def get_sample_id(self, index):
        """获取样本ID"""
        sample = self.samples[index]
        return f"{sample['session_name']}/{sample['clip_id']}"

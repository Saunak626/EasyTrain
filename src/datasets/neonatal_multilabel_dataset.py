"""新生儿多标签行为识别数据集

该模块实现了基于CPU处理帧图像和Excel标签文件的新生儿多标签行为识别数据集。
参考UCF101视频数据集的实现，支持从预处理帧图像加载数据。

Classes:
    NeonatalMultilabelDataset: 新生儿多标签行为识别数据集实现
"""

import os
import cv2
import torch
import logging
import numpy as np
import pandas as pd


from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .label_cache import LabelCache, OptimizedLabelProcessor

logger = logging.getLogger(__name__)

# 尝试导入iterative-stratification库（用于真正的多标签分层抽样）
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    ITERSTRAT_AVAILABLE = True
except ImportError:
    ITERSTRAT_AVAILABLE = False
    logger.warning(
        "未安装iterative-stratification库，将使用简化的分层抽样策略。\n"
        "建议安装以获得更好的多标签分层效果: pip install iterative-stratification"
    )


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
    
    def __init__(self, frames_dir, labels_file, split='train', clip_len=16, model_type=None,
                 top_n_classes=None, stratified_split=True, min_samples_per_class=10,
                 sampling_mode='random', target_fps=None, original_fps=16):
        """初始化新生儿多标签数据集

        Args:
            frames_dir (str): 帧图像根目录路径
            labels_file (str): Excel标签文件路径
            split (str): 数据集分割，'train'或'test'
            clip_len (int): 每个视频片段的帧数，默认16
            model_type (str): 模型类型，用于获取对应的transforms
            top_n_classes (int, optional): 只使用样本数量前N多的类别，None表示使用全部类别
            stratified_split (bool): 是否使用分层抽样进行数据划分，默认True
            min_samples_per_class (int): 类别最小样本数阈值，默认10
            sampling_mode (str): 采样模式，'random'或'fps'，默认'random'
            target_fps (float, optional): 目标采样帧率，仅在sampling_mode='fps'时使用
            original_fps (float): 原始视频帧率，默认16fps
        """
        self.frames_dir = frames_dir
        self.labels_file = labels_file
        self.split = split
        self.clip_len = clip_len
        self.model_type = model_type
        self.top_n_classes = top_n_classes
        self.stratified_split = stratified_split
        self.min_samples_per_class = min_samples_per_class

        # FPS采样相关参数
        self.sampling_mode = sampling_mode
        self.target_fps = target_fps
        self.original_fps = original_fps

        # 参数验证
        self._validate_sampling_params()

        # 获取模型特定的transforms
        self.model_transforms = self._get_model_transforms()

        # 优化的预处理参数
        self.resize_height = 224
        self.resize_width = 224
        self.crop_size = 112

        # 定义24个原始行为标签
        self.original_behavior_labels = [
            '喂养开始', '喂养结束', '易哭闹', '张嘴闭嘴', '吸吮行为', '吃手指',
            '吃脚指', '皱眉', '哭泣', '发脾气', '来回摇头', '手脚活动加快',
            '寻找奶瓶', '注视奶瓶', '声调变高', '打哈欠', '睡着了', '间歇喝奶',
            '唇部触食反应', '喂养期鬼脸', '口腔器具咬合', '头颈侧向回避',
            '肢体张力减退', '远离奶瓶'
        ]

        # 使用优化的标签处理器（使用原始标签）
        self.label_processor = OptimizedLabelProcessor(self.labels_file, self.original_behavior_labels)

        # 加载和处理数据（使用缓存）
        all_samples = self._load_samples_optimized()

        # 分析类别分布并筛选类别
        self.selected_classes, self.class_mapping = self._select_top_classes(all_samples)
        self.behavior_labels = self.selected_classes
        self.num_classes = len(self.behavior_labels)
        self.class_names = self.behavior_labels

        # 更新样本标签（只保留选定的类别）
        all_samples = self._update_sample_labels(all_samples)

        # 数据分割（使用分层抽样或简单分割）
        self.samples = self._split_data(all_samples, split)

        logger.info(f"加载 {split} 数据集: {len(self.samples)} 个样本，使用 {self.num_classes} 个类别")
        if self.sampling_mode == 'fps':
            logger.info(f"使用FPS采样模式: target_fps={self.target_fps}, original_fps={self.original_fps}")
        if self.top_n_classes is not None:
            logger.info(f"选定的类别: {self.behavior_labels}")

    def _validate_sampling_params(self):
        """验证采样参数的有效性"""
        if self.sampling_mode not in ['random', 'fps']:
            raise ValueError(f"不支持的采样模式: {self.sampling_mode}。支持的模式: 'random', 'fps'")

        if self.sampling_mode == 'fps':
            if self.target_fps is None:
                raise ValueError("使用FPS采样模式时，必须指定target_fps参数")
            if self.target_fps <= 0:
                raise ValueError(f"target_fps必须大于0，当前值: {self.target_fps}")
            if self.original_fps <= 0:
                raise ValueError(f"original_fps必须大于0，当前值: {self.original_fps}")

    def fps_sampling_neonatal(self, buffer, clip_len, target_fps, original_fps=16, min_fps=4):
        """针对新生儿数据集优化的FPS采样方法

        专门针对5秒左右的短视频进行优化，确保采样质量不低于4fps。

        Args:
            buffer (np.ndarray): 输入视频帧缓冲区，形状为(T, H, W, C)
            clip_len (int): 目标输出帧数
            target_fps (float): 目标采样帧率
            original_fps (float): 原始视频帧率，默认16fps
            min_fps (float): 最低采样帧率限制，默认4fps

        Returns:
            np.ndarray: 采样后的帧缓冲区，形状为(clip_len, H, W, C)
        """
        total_frames = buffer.shape[0]

        # 应用最低FPS限制（针对新生儿数据集的特殊要求）
        effective_fps = max(target_fps, min_fps)

        # 计算采样间隔
        interval = original_fps / effective_fps

        # 生成采样索引
        indices = []
        current_idx = 0.0

        while len(indices) < clip_len and int(current_idx) < total_frames:
            indices.append(int(current_idx))
            current_idx += interval

        # 针对短视频的帧数不足处理策略
        if len(indices) < clip_len:
            # 策略1: 从视频开始位置进行更密集的采样
            remaining_frames = clip_len - len(indices)

            # 计算更密集的间隔，确保能够采样到足够的帧
            if total_frames >= clip_len:
                # 如果总帧数足够，使用均匀分布采样
                dense_interval = (total_frames - 1) / (clip_len - 1)
                indices = [int(i * dense_interval) for i in range(clip_len)]
            else:
                # 如果总帧数不足，先均匀采样所有帧，然后重复关键帧
                indices = list(range(total_frames))

                # 重复关键帧填充到clip_len
                while len(indices) < clip_len:
                    # 优先重复中间帧和最后帧
                    if total_frames > 1:
                        mid_frame = total_frames // 2
                        last_frame = total_frames - 1
                        indices.extend([mid_frame, last_frame])
                    else:
                        # 只有一帧的情况，重复该帧
                        indices.append(0)

                # 截断到所需长度
                indices = indices[:clip_len]

        # 如果采样帧数超过clip_len，截断
        indices = indices[:clip_len]

        # 确保索引不超出范围
        indices = [min(idx, total_frames - 1) for idx in indices]

        # 采样帧
        sampled_buffer = buffer[indices]

        return sampled_buffer

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
                is_compatible, message = validate_model_transforms_compatibility(self.model_type)
                if not is_compatible:
                    logger.warning(f"{self.model_type} transforms不兼容: {message}，回退到传统预处理")
                    return None

                return transforms

            except ImportError:
                # 如果导入失败，使用传统方式
                pass
        return None
    def _load_samples_optimized(self):
        """使用优化的标签处理器加载样本数据"""
        logger.info(f"使用优化缓存加载样本数据...")

        # 使用优化的标签处理器获取所有有效样本
        samples = self.label_processor.get_all_valid_samples(self.frames_dir)

        logger.info(f"优化加载完成: {len(samples)} 个有效样本")
        return samples

    def _select_top_classes(self, samples):
        """根据样本数量选择前N个类别

        Args:
            samples (list): 所有样本数据

        Returns:
            tuple: (selected_classes, class_mapping) 选定的类别列表和映射关系
        """
        if self.top_n_classes is None:
            selected_classes = self.original_behavior_labels.copy()
            class_mapping = {i: i for i in range(len(selected_classes))}
            logger.info(f"使用全部 {len(selected_classes)} 个类别")
            return selected_classes, class_mapping

        # 统计每个类别的样本数量
        class_counts = Counter()
        for sample in samples:
            labels = sample['labels']
            for i, label_value in enumerate(labels):
                if label_value > 0:  # 正样本
                    class_counts[i] += 1

        # 按样本数量排序，选择前N个类别
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

        # 过滤掉样本数量少于阈值的类别
        filtered_classes = [(class_idx, count) for class_idx, count in sorted_classes
                           if count >= self.min_samples_per_class]

        # 选择前top_n_classes个类别
        selected_class_indices = [class_idx for class_idx, count in filtered_classes[:self.top_n_classes]]
        selected_classes = [self.original_behavior_labels[i] for i in selected_class_indices]

        # 创建新旧类别索引的映射关系
        class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_class_indices)}

        return selected_classes, class_mapping

    def _update_sample_labels(self, samples):
        """更新样本标签，只保留选定的类别

        Args:
            samples (list): 原始样本数据

        Returns:
            list: 更新后的样本数据
        """
        if self.top_n_classes is None:
            return samples  # 不需要更新

        updated_samples = []
        for sample in samples:
            old_labels = sample['labels']
            new_labels = []

            # 根据class_mapping重新构建标签向量
            for new_idx in range(len(self.class_mapping)):
                # 找到对应的原始类别索引
                old_idx = None
                for old_i, new_i in self.class_mapping.items():
                    if new_i == new_idx:
                        old_idx = old_i
                        break

                if old_idx is not None:
                    new_labels.append(old_labels[old_idx])
                else:
                    new_labels.append(0.0)

            # 跳过全零标签的样本
            if sum(new_labels) == 0:
                continue

            # 创建新的样本
            updated_sample = sample.copy()
            updated_sample['labels'] = new_labels
            updated_samples.append(updated_sample)

        logger.info(f"标签更新完成: {len(samples)} -> {len(updated_samples)} 个有效样本")
        return updated_samples
    
    def _split_data(self, samples, split):
        """数据分割（支持真正的多标签分层抽样）

        使用iterative stratification算法确保每个类别在训练集和测试集中的分布比例一致。
        如果iterative-stratification库未安装，回退到简化的分层策略。

        Args:
            samples (list): 所有样本数据
            split (str): 'train' 或 'test'

        Returns:
            list: 划分后的样本数据
        """
        if len(samples) <= 2:
            return samples

        if not self.stratified_split:
            # 使用简单的8:2分割
            split_idx = max(1, int(len(samples) * 0.8))
            if split == 'train':
                return samples[:split_idx]
            else:
                return samples[split_idx:]

        # 🔧 新增：使用真正的多标签分层抽样（Iterative Stratification）
        if ITERSTRAT_AVAILABLE:
            try:
                # 构建多标签矩阵 (n_samples, n_classes)
                y = np.array([sample['labels'] for sample in samples])

                # 检查是否有足够的样本进行分层
                if len(samples) < 10:
                    logger.warning(f"样本数量太少({len(samples)})，回退到简单分割")
                    split_idx = max(1, int(len(samples) * 0.8))
                    return samples[:split_idx] if split == 'train' else samples[split_idx:]

                # 使用MultilabelStratifiedShuffleSplit进行分层分割
                msss = MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=0.2,
                    random_state=42
                )

                # 执行分层分割
                for train_indices, test_indices in msss.split(X=np.zeros(len(samples)), y=y):
                    if split == 'train':
                        result_samples = [samples[i] for i in train_indices]
                    else:
                        result_samples = [samples[i] for i in test_indices]

                    # 验证分层效果
                    self._validate_stratified_split(samples, train_indices, test_indices,
                                                    method="Iterative Stratification")

                    return result_samples

            except Exception as e:
                logger.warning(f"Iterative Stratification失败: {e}，回退到简化分层策略")
                # 继续执行下面的简化分层策略

        # 🔧 降级方案：使用简化的分层抽样（只考虑第一个正标签）
        try:
            # 为每个样本创建分层标识
            # 使用样本的第一个正标签作为分层依据
            stratify_labels = []
            for sample in samples:
                labels = sample['labels']
                if sum(labels) == 0:
                    stratify_labels.append(-1)  # 无标签样本
                else:
                    # 找到第一个正标签作为分层依据
                    main_class = next(i for i, label in enumerate(labels) if label > 0)
                    stratify_labels.append(main_class)

            # 检查每个类别是否有足够的样本进行分层
            class_counts = Counter(stratify_labels)
            min_count = min(count for label, count in class_counts.items() if label != -1)

            if min_count < 2:
                # 如果某些类别样本太少，回退到简单分割
                logger.warning(f"某些类别样本数少于2个，回退到简单分割")
                split_idx = max(1, int(len(samples) * 0.8))
                if split == 'train':
                    return samples[:split_idx]
                else:
                    return samples[split_idx:]

            # 执行分层分割
            train_indices, test_indices = train_test_split(
                range(len(samples)),
                test_size=0.2,
                stratify=stratify_labels,
                random_state=42
            )

            if split == 'train':
                result_samples = [samples[i] for i in train_indices]
            else:
                result_samples = [samples[i] for i in test_indices]

            # 验证分层效果
            self._validate_stratified_split(samples, train_indices, test_indices,
                                           method="Simplified Stratification (first label)")

            return result_samples

        except Exception as e:
            logger.warning(f"分层抽样失败: {e}，回退到简单分割")
            # 回退到简单分割
            split_idx = max(1, int(len(samples) * 0.8))
            if split == 'train':
                return samples[:split_idx]
            else:
                return samples[split_idx:]

    def _validate_stratified_split(self, samples, train_indices, test_indices, method="Stratification"):
        """验证分层分割的效果

        Args:
            samples (list): 所有样本数据
            train_indices (list): 训练集索引
            test_indices (list): 测试集索引
            method (str): 分层方法名称，用于日志输出
        """
        train_class_counts = Counter()
        test_class_counts = Counter()

        # 统计训练集类别分布
        for idx in train_indices:
            labels = samples[idx]['labels']
            for i, label_value in enumerate(labels):
                if label_value > 0:
                    train_class_counts[i] += 1

        # 统计测试集类别分布
        for idx in test_indices:
            labels = samples[idx]['labels']
            for i, label_value in enumerate(labels):
                if label_value > 0:
                    test_class_counts[i] += 1

        # 计算分布差异的统计指标（用于验证分层效果）
        ratios_diff = []
        for i in range(len(self.behavior_labels)):
            train_count = train_class_counts.get(i, 0)
            test_count = test_class_counts.get(i, 0)
            total_count = train_count + test_count

            if total_count > 0:
                train_ratio = train_count / total_count
                ratio_diff = abs(train_ratio - 0.8)  # 理想情况下训练集应该占80%
                ratios_diff.append(ratio_diff)

        return True
    
    def set_model_type(self, model_type):
        """设置模型类型并更新transforms（用于网格搜索）"""
        self.model_type = model_type
        self.model_transforms = self._get_model_transforms()
    
    def __len__(self):
        return len(self.samples)
    
    def _sample_frames(self, buffer):
        """统一的帧采样方法

        Args:
            buffer: 输入帧缓冲区，形状为 (T, H, W, C)

        Returns:
            采样后的帧缓冲区，形状为 (clip_len, H, W, C)
        """
        if self.sampling_mode == 'fps' and self.target_fps is not None:
            # FPS 采样
            return self.fps_sampling_neonatal(buffer, self.clip_len, self.target_fps, self.original_fps)
        else:
            # 随机采样
            return self._temporal_crop(buffer, self.clip_len)

    def __getitem__(self, index):
        """获取单个样本"""
        sample = self.samples[index]

        # 加载视频帧
        buffer = self.load_frames(sample['frames_dir'])

        # 统一的帧采样
        buffer = self._sample_frames(buffer)

        # 如果有模型特定的transforms，使用官方transforms
        if self.model_transforms is not None:
            # 转换为torch tensor格式: (T, H, W, C) -> (T, C, H, W)
            buffer = torch.from_numpy(buffer).float() / 255.0
            buffer = buffer.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

            # 应用模型特定的官方transforms (输入: T, C, H, W -> 输出: C, T, H, W)
            buffer = self.model_transforms(buffer)
        else:
            # 使用传统的预处理方式（向后兼容）
            buffer = self.crop(buffer, crop_size=self.crop_size, temporal_crop=False)
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
    
    def _temporal_crop(self, buffer, clip_len):
        """时间维度裁剪

        Args:
            buffer: 输入帧缓冲区，形状为 (T, H, W, C)
            clip_len: 目标帧数

        Returns:
            裁剪后的帧缓冲区，形状为 (clip_len, H, W, C)
        """
        if buffer.shape[0] <= clip_len:
            # 帧数不足，填充
            if buffer.shape[0] == 0:
                buffer = np.zeros((1, buffer.shape[1], buffer.shape[2], 3), dtype=buffer.dtype)
            last_frame = buffer[-1]
            pad_size = clip_len - buffer.shape[0]
            padding = np.tile(last_frame[np.newaxis], (pad_size, 1, 1, 1))
            return np.concatenate([buffer, padding], axis=0)
        else:
            # 随机裁剪
            time_index = np.random.randint(buffer.shape[0] - clip_len)
            return buffer[time_index:time_index + clip_len, :, :, :]

    def _spatial_crop(self, buffer, crop_size):
        """空间维度裁剪（resize 或随机裁剪）

        Args:
            buffer: 输入帧缓冲区，形状为 (T, H, W, C)
            crop_size: 裁剪尺寸

        Returns:
            裁剪后的帧缓冲区，形状为 (T, crop_size, crop_size, C)
        """
        clip_len = buffer.shape[0]
        target_shape = (clip_len, crop_size, crop_size, 3)

        if buffer.shape[1] < crop_size or buffer.shape[2] < crop_size:
            # 输入尺寸小于目标尺寸，进行 resize
            resized_buffer = np.zeros(target_shape, dtype=buffer.dtype)
            for i in range(clip_len):
                resized_buffer[i] = cv2.resize(buffer[i], (crop_size, crop_size))
            return resized_buffer
        else:
            # 输入尺寸大于等于目标尺寸，进行随机裁剪
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
            return buffer[:, height_index:height_index + crop_size,
                          width_index:width_index + crop_size, :]

    def crop(self, buffer, clip_len=None, crop_size=None, temporal_crop=True):
        """统一的裁剪方法

        Args:
            buffer: 输入帧缓冲区，形状为 (T, H, W, C)
            clip_len: 目标帧数（None 表示不进行时间裁剪）
            crop_size: 裁剪尺寸（None 表示不进行空间裁剪）
            temporal_crop: 是否进行时间维度裁剪

        Returns:
            裁剪后的帧缓冲区
        """
        # 时间维度裁剪
        if temporal_crop and clip_len is not None:
            buffer = self._temporal_crop(buffer, clip_len)

        # 空间维度裁剪
        if crop_size is not None:
            buffer = self._spatial_crop(buffer, crop_size)

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

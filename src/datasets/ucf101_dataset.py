import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import numpy as np
from pathlib import Path


class UCF101Dataset(Dataset):
    """
    UCF-101视频动作识别数据集
    直接处理UCF-101数据结构，支持视频片段提取和动作分类
    """
    
    def __init__(self, root, annotation_path, frames_per_clip=16, 
                 step_between_clips=4, fold=1, train=True, transform=None):
        """
        初始化UCF-101数据集
        
        Args:
            root: 视频文件根目录路径
            annotation_path: 标注文件目录路径
            frames_per_clip: 每个视频片段的帧数
            step_between_clips: 片段间的帧间隔
            fold: 使用的数据折（1-3）
            train: 是否为训练集
            transform: 数据变换
        """
        self.root = root
        self.annotation_path = annotation_path
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.fold = fold
        self.train = train
        
        # 获取类别列表（排除ucfTrainTestlist目录）
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 加载视频文件列表
        self.video_clips = self._load_video_clips()
        
        # 设置变换
        self.transform = transform
    
    def _get_classes(self):
        """获取UCF-101动作类别列表，排除非动作目录"""
        classes = []
        root_path = Path(self.root)
        
        for item in root_path.iterdir():
            if item.is_dir() and item.name != 'ucfTrainTestlist':
                classes.append(item.name)
        
        return sorted(classes)
    
    def _load_video_clips(self):
        """根据标注文件加载视频片段列表"""
        split_name = "train" if self.train else "test"
        split_file = f"{split_name}list{self.fold:02d}.txt"
        split_path = os.path.join(self.annotation_path, split_file)
        
        video_clips = []
        
        with open(split_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                video_path = parts[0]
                class_name = video_path.split('/')[0]
                
                if class_name in self.class_to_idx:
                    full_video_path = os.path.join(self.root, video_path)
                    if os.path.exists(full_video_path):
                        # 获取视频的所有可能片段
                        clips = self._get_video_clips(full_video_path, class_name)
                        video_clips.extend(clips)
        
        return video_clips
    
    def _get_video_clips(self, video_path, class_name):
        """从单个视频文件中提取所有可能的片段"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        clips = []
        start_frame = 0
        
        while start_frame + self.frames_per_clip <= total_frames:
            clips.append({
                'video_path': video_path,
                'start_frame': start_frame,
                'class_name': class_name,
                'label': self.class_to_idx[class_name]
            })
            start_frame += self.step_between_clips
        
        return clips
    
    def _load_video_frames(self, video_path, start_frame, num_frames):
        """从视频文件中加载指定帧数的片段"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # 转换BGR到RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # 如果帧数不足，重复最后一帧
        while len(frames) < num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                # 如果没有帧，创建黑色帧
                frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
        
        return frames
        
    def __len__(self):
        return len(self.video_clips)
    
    def __getitem__(self, idx):
        clip_info = self.video_clips[idx]
        
        # 加载视频帧
        frames = self._load_video_frames(
            clip_info['video_path'], 
            clip_info['start_frame'], 
            self.frames_per_clip
        )
        
        # 将帧列表转换为numpy数组: (T, H, W, C)
        video_array = np.stack(frames, axis=0)
        
        # 处理视频数据
        # 转换为tensor: (T, H, W, C) -> (C, T, H, W)
        video_tensor = torch.from_numpy(video_array).float() / 255.0
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        
        # 应用基本的resize和normalize
        # Resize到112x112
        video_tensor = torch.nn.functional.interpolate(
            video_tensor, size=(112, 112), mode='bilinear', align_corners=False
        )
        
        # 标准化 (使用Kinetics数据集的均值和标准差)
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        return video_tensor, clip_info['label']
    
    @property
    def num_classes(self):
        """返回UCF-101的类别数"""
        return len(self.classes)
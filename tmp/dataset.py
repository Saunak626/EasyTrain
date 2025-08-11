"""
# 1. 数据预处理模块
# 功能：处理视频数据，包括帧提取、转换和增强
# 处理顺序：视频加载 -> 帧提取 -> 预处理 -> 数据增强 -> 张量转换
"""

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from torch.utils.data import DataLoader,random_split
import matplotlib.pyplot as plt
from torchvision import transforms

# 导入预训练模型权重
from torchvision.models.video import (mc3_18,MC3_18_Weights,
                                      r3d_18, R3D_18_Weights,
                                      mvit_v1_b,MViT_V1_B_Weights,
                                      mvit_v2_s,MViT_V2_S_Weights,
                                      s3d,S3D_Weights,
                                      r2plus1d_18,R2Plus1D_18_Weights,
                                      swin3d_b,Swin3D_B_Weights,
                                      swin3d_s,Swin3D_S_Weights,
                                      swin3d_t,Swin3D_T_Weights)

class VideoDataset(Dataset):
    """
    # 2. 视频数据集类
    # 功能：处理视频数据的加载和预处理
    # 输入：data_dir - 数据目录，args - 配置参数
    """
    def __init__(self, data_dir, args):
        # 2.1 初始化基本参数
        self.data_dir = data_dir
        self.transform = args.transform
        self.rotate_flag = args.rotate_flag
        self.crop_flag = args.crop_flag
        self.num_frames = args.num_frames
        self.mode = args.mode
        self.step = args.step

        # 2.2 根据模型类型设置对应的数据转换
        if args.model_type == "mc3_18":
            self.transform = MC3_18_Weights.DEFAULT.transforms()
        if args.model_type == "resnet":
            self.transform = R3D_18_Weights.DEFAULT.transforms()
        if args.model_type == "mvit_v1":
            self.transform = MViT_V1_B_Weights.DEFAULT.transforms()
        if args.model_type == "mvit_v2":
            self.transform = MViT_V2_S_Weights.DEFAULT.transforms()
        if args.model_type == "r2+":
            self.transform = R2Plus1D_18_Weights.DEFAULT.transforms()
        if args.model_type == "s3d":
            self.transform = S3D_Weights.DEFAULT.transforms()
        if args.model_type == "swin_b":
            self.transform = Swin3D_B_Weights.DEFAULT.transforms()
        if args.model_type == "swin_s":
            self.transform = Swin3D_S_Weights.DEFAULT.transforms()
        if args.model_type == "swin_t":
            self.transform = Swin3D_T_Weights.DEFAULT.transforms()

        # 2.3 加载数据集标签和路径
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.video_list = []

        # 2.4 根据不同的处理模式加载视频
        for cls in self.classes:
            class_dir = os.path.join(data_dir, cls)
            for video in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video)
                if self.mode == 'fixed':
                    self.video_list.append((video_path, self.class_to_idx[cls]))
                elif self.mode == 'complete':
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    self.video_list.append((video_path, total_frames, self.class_to_idx[cls]))
                elif self.mode == 'sliding':
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    starts = list(range(0, total_frames - self.num_frames + 1, self.step))
                    if starts:
                        for start in starts:
                            self.video_list.append((video_path, start, self.class_to_idx[cls]))
                    else:
                        self.video_list.append((video_path, 0, self.class_to_idx[cls]))
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        """
        # 3. 数据获取方法
        # 功能：根据索引获取处理后的视频帧和标签
        # 处理流程：加载视频 -> 提取帧 -> 预处理 -> 转换为张量
        """
        if self.mode == 'fixed':
            video_path, label = self.video_list[idx]
            frames = self.load_fixed_frames(video_path)
        elif self.mode == 'complete':
            video_path, total_frames, label = self.video_list[idx]
            frames = self.load_complete_video(video_path, total_frames)
        elif self.mode == 'sliding':
            video_path, start_frame, label = self.video_list[idx]
            frames = self.load_sliding_frames(video_path, start_frame)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # 3.1 数据预处理和转换
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
        frames = torch.from_numpy(frames).float()/255.0
        frames = self.transform(frames)
        label = torch.tensor(label).long()
        return frames, label

    def transform_frame(self, frame):
        """
        # 4. 帧转换方法
        # 功能：对单帧图像进行预处理
        # 处理步骤：调整大小 -> 中心裁剪 -> 标准化
        """
        frame = cv2.resize(frame, (256, 256))
        h, w, _ = frame.shape
        start_h = (h - 224) // 2
        start_w = (w - 224) // 2
        frame = frame[start_h:start_h + 112, start_w:start_w + 112, :]
        frame = torch.from_numpy(frame).float()/ 255.0
        mean_tensor = torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 3)
        std_tensor = torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 3)
        frame = (frame - mean_tensor)/std_tensor
        return frame

    def rotate_image(self, image, angle=-90):
        """
        # 5. 图像旋转方法
        # 功能：对图像进行旋转处理
        # 处理条件：当图像宽度小于高度时进行旋转
        """
        (h, w) = image.shape[:2]
        if w < h:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            return rotated
        return image

    def random_crop(self, image, crop_ratio=0.8):
        """
        # 6. 随机裁剪方法
        # 功能：对图像进行随机裁剪以保留核心区域
        # 用途：数据增强
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        y = np.random.randint(0, h - new_h)
        x = np.random.randint(0, w - new_w)
        return image[y:y + new_h, x:x + new_w]

    def load_fixed_frames(self, video_path):
        """
        # 7. 固定帧加载方法
        # 功能：从视频中提取固定数量的帧
        # 处理步骤：打开视频 -> 均匀采样帧 -> 预处理 -> 填充不足帧
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if cnt in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.rotate_image(frame)
                frames.append(frame)
                if len(frames) == self.num_frames:
                    break
            cnt += 1
        cap.release()
        if len(frames) < self.num_frames:
            frames.extend([frames[-1]] * (self.num_frames - len(frames)))
        frames = np.stack(frames, axis=0)
        return frames

    def load_complete_video(self, video_path, total_frames):
        """
        # 8. 完整视频加载方法
        # 功能：加载视频的所有帧
        # 处理步骤：读取所有帧 -> 预处理 -> 填充不足帧
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.rotate_image(frame)
            frames.append(frame)
        cap.release()
        if len(frames) < total_frames:
            frames.extend([frames[-1]] * (total_frames - len(frames)))
        frames = np.stack(frames, axis=0)
        return frames

    def load_sliding_frames(self, video_path, start_frame, isrotate=False):
        """
        # 9. 滑动窗口帧加载方法
        # 功能：使用滑动窗口方式加载视频帧
        # 处理步骤：设置起始帧 -> 读取连续帧 -> 预处理 -> 处理不足帧
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.rotate_image(frame)
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                    frames.append(blank_frame)
        cap.release()
        frames = np.stack(frames, axis=0)
        return frames

def show_image(videopath):
    """
    # 10. 图像显示方法
    # 功能：显示视频帧用于调试和可视化
    # 处理步骤：加载视频 -> 采样帧 -> 预处理 -> 显示
    """
    cap = cv2.VideoCapture(videopath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cnt in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2,0,1))
            frame = torch.from_numpy(frame).float() / 255.0
            frame = torch.unsqueeze(frame,dim=0)
            frame = MC3_18_Weights.DEFAULT.transforms()(frame)
            frame = np.transpose(frame.squeeze(), (1,2,0))
            plt.imshow(frame)
            plt.axis('off')
            plt.show()
            cap.release()

if __name__ == "__main__":
    videopaths = glob.glob("../data/*/*.mp4")
    show_image(videopaths[0])
    dataset = VideoDataset("../data", num_frames=100, mode='sliding', step=10)
    #
    # dataset = VideoDataset("../data", num_frames=100, mode='fixed', step=10)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    # for batch in dataloader:
    #     break
    # inputs, labels = batch
    # print(torch.mean(inputs[0],dim=(1,2,3)),torch.std(inputs[0],dim=(1,2,3)))

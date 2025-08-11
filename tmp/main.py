"""
# 1. 主训练模块
# 功能：模型训练和评估的主要流程
# 处理顺序：数据准备 -> 模型训练 -> 模型评估 -> 结果保存
"""

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import VideoNet3D
from utils.dataset import VideoDataset
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import pandas as pd
import datetime
from tqdm import tqdm
from utils.experimental_setting import args
from trainers import model_train, model_test

# 设置随机种子
torch.manual_seed(3569)
formatted_date = datetime.datetime.now().strftime("%Y-%m-%d")

if __name__ == '__main__':
    """
    # 2. 主训练流程
    # 功能：执行模型训练和评估
    # 处理步骤：
    #   1. 设置模型类型和训练参数
    #   2. 加载和预处理数据
    #   3. 训练模型
    #   4. 评估模型
    #   5. 保存结果
    """
    # 2.1 设置模型类型和训练参数
    model_types = ["r2+"]  # 可扩展支持其他模型类型
    result = []
    args.device = "cuda:1"
    test_dir = './dataset/test'
    args.trainflag = True

    # 2.2 遍历不同模型类型进行训练
    for model_type in model_types:
        args.model_type = model_type
        model = VideoNet3D(args)
        model = model.to(args.device)

        # 2.3 处理不同的训练模式
        for exp in ["no_pretrained"]:  # 可扩展支持其他训练模式
            if exp == "no_pretrained": 
                args.pretrained = False
                args.frozen = False
            if exp == "frozen": 
                args.pretrained = True
                args.frozen = True
            if exp == "main": 
                args.pretrained = True
                args.frozen = False

            # 2.4 训练模型
            if args.trainflag:
                val_acc, best_val_loss, epoch_loss = model_train(model, args, data_dir="./dataset")
                best_train_loss = np.round(np.min(epoch_loss), 2)

            # 2.5 加载最佳模型
            model = VideoNet3D(args)
            if args.pretrained == False:
                model.load_state_dict(
                    torch.load(f'./saved_models/best_{args.model_type}_no_pretrained.pth', 
                             map_location=args.device))
            else:
                if args.frozen == True:
                    model.load_state_dict(
                        torch.load(f'./saved_models/best_{args.model_type}_frozen.pth', 
                                 map_location=args.device))
                else:
                    model.load_state_dict(
                        torch.load(f'./saved_models/best_{args.model_type}.pth', 
                                 map_location=args.device))

            # 2.6 评估模型
            test_df = model_test(model, args, test_dir)

            # 2.7 保存结果
            if args.pretrained == False:
                test_df.to_csv(f"./results/{model_type}_{formatted_date}_no_pretrained.csv")
            else:
                if args.frozen == True:
                    test_df.to_csv(f"./results/{model_type}_{formatted_date}_frozen.csv")
                else:
                    test_df.to_csv(f"./results/{model_type}_{formatted_date}.csv")

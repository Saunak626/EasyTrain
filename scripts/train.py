"""
单次训练启动脚本
解析参数并调用核心训练模块

该脚本是训练流程的入口点，主要功能包括：
1. 解析命令行参数和配置文件
2. 根据参数决定是否使用多GPU训练
3. 启动训练过程
4. 显示训练结果
"""

import sys
import os
import subprocess

# 添加项目根目录到路径，确保可以正确导入项目内的模块
# 通过os.path.dirname的两层嵌套调用，获取到项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments  # 参数解析器
from src.trainers.base_trainer import run_training                   # 核心训练函数





def print_training_info(args, config):
    """打印训练信息"""
    experiment_name = config['training']['experiment_name']
    hp = config['hyperparameters']
    print(f"\n🚀 开始训练: {experiment_name}")
    print(f"📦 批大小: {hp.get('batch_size', 'N/A')}, 轮数: {hp.get('epochs', 'N/A')}, 学习率: {hp.get('learning_rate', 'N/A')}")
    print("-" * 50)


def main():
    """主函数，处理单次实验训练"""
    # 解析命令行参数和配置文件
    args, config = parse_arguments(mode='train')
    
    # 打印训练信息
    print_training_info(args, config)
    
    # 获取实验名称并启动训练
    experiment_name = config['training']['experiment_name']
    result = run_training(config, experiment_name, args.is_grid_search)
    
    return 0


# 程序入口点
if __name__ == "__main__":
    # 执行主函数并获取退出码
    exit_code = main()
    # 退出程序
    sys.exit(exit_code)
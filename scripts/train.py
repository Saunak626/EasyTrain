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

from src.utils.config_parser import parse_single_training_arguments  # 参数解析器
from src.trainers.base_trainer import run_training                   # 核心训练函数


def launch_with_accelerate(original_args):
    """
    使用accelerate launch重新启动脚本以支持多GPU训练
    
    当用户指定--multi_gpu参数但脚本未在分布式环境中运行时，
    需要使用accelerate launch重新启动脚本以正确初始化多GPU环境。

    Args:
        original_args: 原始命令行参数对象

    Returns:
        int: 子进程返回码
    """
    # 构建accelerate launch命令
    cmd = ["accelerate", "launch"]
    
    # 如果有额外的accelerate参数，则添加到命令中
    if hasattr(original_args, 'accelerate_args') and original_args.accelerate_args:
        cmd.extend(original_args.accelerate_args.split())
    
    # 添加当前脚本路径和过滤掉--multi_gpu的参数
    cmd.append(__file__)
    cmd.extend([arg for arg in sys.argv[1:] if arg != "--multi_gpu"])
    
    # 执行命令并返回结果
    result = subprocess.run(cmd)
    return result.returncode


def print_training_info(args, config):
    """
    打印训练完成后的信息摘要
    
    展示实验名称、超参数配置和训练环境等关键信息，
    帮助用户快速了解训练结果和配置。
    只在主进程中打印，避免多卡训练时的重复输出。

    Args:
        args: 命令行参数对象
        config: 训练配置字典
    """
    # 在分布式训练中，只让主进程打印信息
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank != 0:
        return
        
    print("\n" + "="*60)
    print("🎯 训练完成信息")
    print("="*60)
    
    # 实验信息
    experiment_name = config['training']['experiment_name']
    print(f"📋 实验名称: {experiment_name}")
    
    # 超参数信息
    hp = config['hyperparameters']
    print(f"📊 超参数:")
    print(f"   学习率: {hp.get('learning_rate', 'N/A')}")
    print(f"   批大小: {hp.get('batch_size', 'N/A')}")
    print(f"   训练轮数: {hp.get('epochs', 'N/A')}")
    print(f"   Dropout: {hp.get('dropout', 'N/A')}")
    
    # GPU环境信息
    if args.use_cpu:
        # 使用CPU训练
        print(f"💻 训练环境: CPU")
    else:
        # 使用GPU训练，获取可见的GPU设备
        gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '所有可用GPU')
        
        # 检查是否在分布式环境中运行
        is_distributed = (os.environ.get('LOCAL_RANK') is not None or 
                         os.environ.get('RANK') is not None)
        
        if is_distributed:
            # 分布式训练环境
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            training_mode = "多卡训练" if world_size > 1 else "单卡训练"
            local_rank = os.environ.get('LOCAL_RANK', 'N/A')
            print(f"🖥️  训练环境: {training_mode}")
            print(f"🔧 GPU设备: {gpu_ids}")
            print(f"🌐 分布式信息: LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
        else:
            # 非分布式训练环境
            training_mode = "多卡训练" if args.multi_gpu else "单卡训练"
            print(f"🖥️  训练环境: {training_mode}")
            print(f"🔧 GPU设备: {gpu_ids}")
    
    print("="*60)


def main():
    """
    主函数，处理单次实验训练
    
    整个训练流程的控制中心，协调参数解析、环境检查、
    训练启动和结果展示等各个环节。
    
    Returns:
        int: 程序退出码，0表示成功
    """
    # 解析命令行参数和配置文件
    args, config = parse_single_training_arguments()
    
    # 检查当前是否已在分布式环境中运行
    # 通过检查accelerate设置的环境变量来判断
    is_distributed = (os.environ.get('LOCAL_RANK') is not None or 
                     os.environ.get('RANK') is not None)
    
    # 如果用户指定多GPU训练但当前不在分布式环境中，使用accelerate重新启动
    if args.multi_gpu and not is_distributed:
        return launch_with_accelerate(args)
    
    # 获取实验名称并启动训练
    experiment_name = config['training']['experiment_name']
    result = run_training(config, experiment_name)
    
    # 打印训练完成信息
    print_training_info(args, config)
    
    return 0


# 程序入口点
if __name__ == "__main__":
    # 执行主函数并获取退出码
    exit_code = main()
    # 退出程序
    sys.exit(exit_code)
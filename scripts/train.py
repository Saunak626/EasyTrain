"""训练流程的入口点（统一网格搜索模式），主要功能包括：
1. 解析命令行参数和配置文件
2. 根据参数决定是否使用多GPU训练
3. 启动网格搜索训练过程
4. 显示训练结果
"""
import sys
import os
import subprocess

# 添加项目根目录到路径，确保可以正确导入项目内的模块
# 通过os.path.dirname的两层嵌套调用，获取到项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments  # 参数解析器
from src.trainers.base_trainer import run_training   # 核心训练函数

# 导入网格搜索模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from grid_search import run_grid_search

def launch_with_accelerate():
    """使用accelerate launch重新启动当前脚本"""
    # 获取当前脚本的所有参数，但移除--multi_gpu
    current_args = [arg for arg in sys.argv[1:] if arg != '--multi_gpu']
    
    # 构建accelerate launch命令
    cmd = ['accelerate', 'launch', sys.argv[0]] + current_args
    
    print(f"🚀 启动多卡训练: {' '.join(cmd)}")
    print("-" * 50)
    
    # 执行accelerate launch命令
    result = subprocess.run(cmd)
    return result.returncode

def print_training_info(args, config):
    """打印训练信息"""
    experiment_name = config['training']['experiment_name']
    hp = config['hyperparameters']
    print(f"\n🚀 开始训练: {experiment_name}")
    print(f"📦 批大小: {hp.get('batch_size', 'N/A')}, 轮数: {hp.get('epochs', 'N/A')}, 学习率: {hp.get('learning_rate', 'N/A')}")
    print("-" * 50)


def main():
    """主函数，处理网格搜索训练（统一模式）"""
    # 检查是否是单个实验调用（来自网格搜索）
    if '--experiment_name' in sys.argv:
        # 这是网格搜索中的单个实验
        args, config = parse_arguments(mode='single_experiment')
        
        # 打印训练信息
        print_training_info(args, config)
        
        # 获取实验名称并启动训练
        experiment_name = config['training']['experiment_name']
        result = run_training(config, experiment_name)
        
        return 0
    else:
        # 这是主网格搜索调用
        args, config = parse_arguments(mode='grid_search')
        
        print("🚀 启动网格搜索训练模式")
        print(f"📊 配置文件: {args.config}")
        print("-" * 50)
        
        # 调用网格搜索
        return run_grid_search(args)


# 程序入口点
if __name__ == "__main__":
    # 执行主函数并获取退出码
    exit_code = main()
    # 退出程序
    sys.exit(exit_code)
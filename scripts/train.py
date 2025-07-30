import sys
import os

# 添加项目根目录到路径，确保可以正确导入项目内的模块
# 通过os.path.dirname的两层嵌套调用，获取到项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments  # 参数解析器
from src.trainers.base_trainer import run_training   # 核心训练函数

def print_training_info(args, config):
    """打印训练信息"""
    experiment_name = config['training']['experiment_name']
    hp = config['hyperparameters']
    print(f"\n🚀 开始训练: {experiment_name}")
    print(f"📦 批大小: {hp.get('batch_size', 'N/A')}, 轮数: {hp.get('epochs', 'N/A')}, 学习率: {hp.get('learning_rate', 'N/A')}")
    print("-" * 50)


def main():
    """主函数，专门处理单个实验的训练"""
    # 解析参数并配置
    args, config = parse_arguments(mode='single_experiment')
    
    # 打印训练信息
    print_training_info(args, config)
    
    # 获取实验名称并启动训练
    experiment_name = config['training']['experiment_name']
    result = run_training(config, experiment_name)
    
    return 0


# 程序入口点
if __name__ == "__main__":
    # 执行主函数并获取退出码
    exit_code = main()
    # 退出程序
    sys.exit(exit_code)
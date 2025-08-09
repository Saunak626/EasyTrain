"""
单实验训练启动脚本：
1. 单独运行单个训练实验。通过统一的配置解析器获取训练参数，支持命令行和配置文件
2. 作为网格搜索的子进程被调用

与grid_search.py的关系：
- train.py: 执行器，负责单个实验的具体执行
- grid_search.py: 调度器，负责多实验的协调和管理
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments  # 参数解析器
from src.trainers.base_trainer import run_training   # 核心训练函数

def main():
    """
    1. parse_arguments解析命令行参数和配置文件
    2. 使用'single_experiment'模式解析参数，区别于网格搜索模式
    3. 实验名称用于结果文件命名和实验跟踪
    4. 返回成功状态码
    
    Returns:
        int: 退出码，0表示成功
    """
    
    # 解析参数并配置（单实验模式）
    args, config = parse_arguments(mode='single_experiment')
    
    # 获取实验名称并启动训练
    exp_name = config['training']['exp_name']
    result = run_training(config, exp_name)
    
    return 0

if __name__ == "__main__":
    # 执行主函数并获取退出码
    exit_code = main()
    # 使用标准退出码退出程序，便于脚本集成
    sys.exit(exit_code)
    
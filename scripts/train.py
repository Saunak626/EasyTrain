"""
单实验训练启动脚本：
- 单一职责：专注于单个实验的训练执行，不处理网格搜索等复杂逻辑
- 配置驱动：通过统一的配置解析器获取训练参数，支持命令行和配置文件
- 信息透明：清晰展示训练关键信息，便于实验跟踪和调试
- 错误处理：规范的退出码处理，便于脚本集成和自动化

使用场景：
1. 单独运行单个训练实验
2. 作为网格搜索的子进程被调用
3. 集成到自动化训练流水线中
4. 开发调试时的快速训练入口

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
    主函数，专门处理单个实验的训练

    核心设计原则包括：
    - 线性流程：按顺序执行参数解析、信息展示、训练执行
    - 异常透明：不捕获异常，让错误信息直接传播到调用方
    - 状态返回：返回标准退出码，便于脚本集成
    - 配置传递：将解析后的配置直接传递给训练器
    
    执行流程：
    1. 调用parse_arguments解析命令行参数和配置文件
    2. 展示训练关键信息，提供用户反馈
    3. 提取实验名称，启动核心训练流程
    4. 返回成功状态码
    
    Returns:
        int: 退出码，0表示成功
    
    注意：
        - 使用'single_experiment'模式解析参数，区别于网格搜索模式
        - 实验名称用于结果文件命名和实验跟踪
        - 训练过程中的异常会直接抛出，由调用方处理
    """
    # 解析参数并配置（单实验模式）
    args, config = parse_arguments(mode='single_experiment')
    
    # 打印训练信息，提供用户反馈
    # print_training_info(args, config)
    
    # 获取实验名称并启动训练
    exp_name = config['training']['exp_name']
    result = run_training(config, exp_name)
    
    return 0


# 程序入口点
if __name__ == "__main__":
    """程序入口点
    
    设计思路：
    采用标准的Python程序入口模式，确保脚本既可以独立运行，
    也可以作为模块被其他脚本导入而不会自动执行。
    
    执行流程：
    1. 调用main()函数执行核心逻辑
    2. 获取返回的退出码
    3. 使用sys.exit()规范退出，便于shell脚本和进程管理
    
    退出码含义：
    - 0: 训练成功完成
    - 非0: 训练过程中出现错误（由异常处理机制决定）
    """
    # 执行主函数并获取退出码
    exit_code = main()
    # 使用标准退出码退出程序，便于脚本集成
    sys.exit(exit_code)
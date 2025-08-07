"""单实验训练启动脚本

设计思路：
本脚本是EasyTrain框架的单实验训练入口，采用简洁明了的设计原则。
核心设计理念包括：
- 单一职责：专注于单个实验的训练执行，不处理网格搜索等复杂逻辑
- 配置驱动：通过统一的配置解析器获取训练参数，支持命令行和配置文件
- 信息透明：清晰展示训练关键信息，便于实验跟踪和调试
- 错误处理：规范的退出码处理，便于脚本集成和自动化

核心功能：
- main: 主控制流程，协调参数解析、信息展示和训练执行
- print_training_info: 训练信息展示，提供实验可视化反馈

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

# 添加项目根目录到路径，确保可以正确导入项目内的模块
# 通过os.path.dirname的两层嵌套调用，获取到项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import parse_arguments  # 参数解析器
from src.trainers.base_trainer import run_training   # 核心训练函数

def print_training_info(args, config):
    """打印训练信息
    
    设计思路：
    提供清晰、友好的训练信息展示，帮助用户快速了解当前实验的关键参数。
    采用emoji和格式化输出，提升用户体验和信息可读性。
    
    Args:
        args: 命令行参数对象（当前版本未使用，预留扩展）
        config (dict): 完整的训练配置字典
            - training.experiment_name: 实验名称
            - hp.batch_size: 批大小
            - hp.epochs: 训练轮数
            - hp.learning_rate: 学习率
    
    功能：
        - 展示实验名称，便于实验跟踪
        - 显示核心超参数，便于参数确认
        - 使用分隔线，提升输出格式美观性
    """
    experiment_name = config['training']['experiment_name']
    hp = config['hp']
    print(f"\n🚀 开始训练: {experiment_name}")
    print(f"📦 批大小: {hp.get('batch_size', 'N/A')}, 轮数: {hp.get('epochs', 'N/A')}, 学习率: {hp.get('learning_rate', 'N/A')}")
    print("-" * 50)


def main():
    """主函数，专门处理单个实验的训练
    
    设计思路：
    采用标准的主函数设计模式，实现清晰的控制流程。
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
    print_training_info(args, config)
    
    # 获取实验名称并启动训练
    experiment_name = config['training']['experiment_name']
    result = run_training(config, experiment_name)
    
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
import yaml
import os
from pprint import pprint # 使用 pprint 可以让字典打印得更美观

def load_and_parse_yaml(file_path):
    """
    加载并解析指定的 YAML 文件。

    Args:
        file_path (str): YAML 文件的路径。

    Returns:
        dict: 解析后得到的 Python 字典。如果文件不存在则返回 None。
    """
    
    # 使用 'with open' 安全地打开文件
    # 'r' 表示读取模式, 'encoding="utf-8"' 确保能正确处理中文字符
    with open(file_path, "r", encoding="utf-8") as f:
        # 使用 yaml.safe_load() 将文件内容解析成 Python 对象
        # YAML 的顶层结构是一个映射，所以这里会得到一个字典
        config = yaml.safe_load(f)
        return config

if __name__ == "__main__":
    # 定义配置文件的路径
    # 请确保这个路径是正确的，或者在同级目录下创建一个 config/ucf101_video_grid.yaml 文件
    config_file = "config/ucf101_video_grid.yaml"
    
    # 调用函数加载配置
    config_data = load_and_parse_yaml(config_file)

    # 如果成功加载和解析
    if config_data:
        print("--- 配置文件已成功解析为 Python 字典：---")
        # 使用 pprint 打印整个配置字典，结构更清晰
        pprint(config_data, sort_dicts=False)
        
        # 1. 访问顶层键
        exp_name = config_data['training']['experiment_name']
        print(f"实验名称 (training -> experiment_name): {exp_name}")
        
        # 2. 访问嵌套的键
        num_classes = config_data['model']['params']['num_classes']
        print(f"模型分类数 (model -> params -> num_classes): {num_classes}")
        
        # 3. 访问列表中的值
        learning_rates_to_try = config_data['grid_search']['grid']['hp.learning_rate']
        print(f"网格搜索的学习率选项: {learning_rates_to_try}")
        print(f"第一个备选学习率是: {learning_rates_to_try[0]}")

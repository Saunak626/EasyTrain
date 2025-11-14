"""统一数据加载器工厂

该模块提供统一的数据加载器创建接口，使用src/datasets中定义的数据集类。
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from collections import Counter
from .cifar10_dataset import CIFAR10Dataset
from .custom_dataset import CustomDatasetWrapper
from .video_dataset import VideoDataset, CombinedVideoDataset
from .neonatal_multilabel_dataset import NeonatalMultilabelDataset


def is_main_process():
    """检查是否为主进程（用于避免重复输出）"""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def calculate_sample_weights(dataset, mode='inverse_frequency', verbose=True):
    """为多标签数据集计算样本权重

    Args:
        dataset: 数据集对象（支持Subset包装）
        mode (str): 权重计算模式
            - 'inverse_frequency': 基于类别逆频率的权重
            - 'label_combination': 基于标签组合稀有性的权重
        verbose (bool): 是否打印详细信息

    Returns:
        torch.Tensor: 每个样本的权重向量
    """
    # 处理Subset包装的情况
    actual_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))

    # 检查数据集是否支持多标签
    if not hasattr(actual_dataset, 'get_num_classes'):
        raise ValueError("数据集不支持多标签权重计算")

    num_classes = actual_dataset.get_num_classes()
    class_names = actual_dataset.get_class_names() if hasattr(actual_dataset, 'get_class_names') else None

    # 🔧 优化：直接从数据集的samples属性读取标签，避免加载图像数据
    # 收集所有标签
    all_labels = []

    # 检查数据集是否有samples属性（NeonatalMultilabelDataset有）
    if hasattr(actual_dataset, 'samples'):
        # 直接从samples读取标签，避免加载图像
        for idx in indices:
            sample = actual_dataset.samples[idx]
            labels = sample['labels']
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            all_labels.append(labels)
    else:
        # 降级方案：通过__getitem__获取标签（会加载图像，较慢）
        if verbose and is_main_process():
            print(f"   ⚠️  数据集没有samples属性，使用__getitem__方法（较慢）...")
        for idx in indices:
            _, labels = actual_dataset[idx]
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            all_labels.append(labels)

    all_labels = np.array(all_labels)  # (n_samples, n_classes)

    if mode == 'inverse_frequency':
        # 🔧 模式1: 基于类别逆频率的权重
        # 统计每个类别的正样本数
        class_counts = all_labels.sum(axis=0)  # (n_classes,)

        # 计算每个类别的逆频率权重
        # 使用平滑因子避免除零和极端值
        class_weights = 1.0 / (class_counts + 1.0)
        class_weights = class_weights / class_weights.sum() * num_classes  # 归一化

        # 计算每个样本的权重（所有正标签权重的平均）
        sample_weights = []
        for labels in all_labels:
            if labels.sum() > 0:
                # 样本权重 = 其所有正标签权重的平均值
                weight = (class_weights * labels).sum() / labels.sum()
            else:
                # 无标签样本使用平均权重
                weight = 1.0
            sample_weights.append(weight)

        sample_weights = np.array(sample_weights)

        if verbose and is_main_process():
            print(f"\n📊 加权采样统计 (模式: {mode}):")
            print(f"   类别权重:")
            for i in range(num_classes):
                class_name = class_names[i] if class_names else f"类别{i}"
                print(f"     {class_name}: 样本数={int(class_counts[i])}, 权重={class_weights[i]:.4f}")

    elif mode == 'label_combination':
        # 🔧 模式2: 基于标签组合稀有性的权重
        # 将每个标签组合转换为字符串作为键
        label_combinations = Counter()
        label_to_indices = {}

        for idx, labels in enumerate(all_labels):
            label_key = tuple(labels)
            label_combinations[label_key] += 1
            if label_key not in label_to_indices:
                label_to_indices[label_key] = []
            label_to_indices[label_key].append(idx)

        # 计算每个标签组合的权重（逆频率）
        total_samples = len(all_labels)
        combination_weights = {
            label_key: total_samples / count
            for label_key, count in label_combinations.items()
        }

        # 归一化权重
        max_weight = max(combination_weights.values())
        combination_weights = {
            label_key: weight / max_weight
            for label_key, weight in combination_weights.items()
        }

        # 为每个样本分配权重
        sample_weights = np.array([
            combination_weights[tuple(labels)]
            for labels in all_labels
        ])

        if verbose and is_main_process():
            print(f"\n📊 加权采样统计 (模式: {mode}):")
            print(f"   标签组合数量: {len(label_combinations)}")
            print(f"   前10个最稀有的标签组合:")
            sorted_combinations = sorted(
                label_combinations.items(),
                key=lambda x: x[1]
            )[:10]
            for label_key, count in sorted_combinations:
                label_str = ','.join([
                    class_names[i] if class_names else str(i)
                    for i, val in enumerate(label_key) if val > 0
                ])
                weight = combination_weights[label_key]
                print(f"     [{label_str}]: 样本数={count}, 权重={weight:.4f}")

    else:
        raise ValueError(f"不支持的权重计算模式: {mode}")

    # 输出权重统计
    if verbose and is_main_process():
        print(f"   样本权重统计:")
        print(f"     最小值: {sample_weights.min():.4f}")
        print(f"     最大值: {sample_weights.max():.4f}")
        print(f"     平均值: {sample_weights.mean():.4f}")
        print(f"     中位数: {np.median(sample_weights):.4f}")
        print(f"     标准差: {sample_weights.std():.4f}")

    return torch.from_numpy(sample_weights).float()


def create_dataloaders(dataset_name, data_dir, batch_size, num_workers=4, model_type=None, **kwargs):
    """
    统一的数据加载器创建函数

    Args:
        dataset_name (str): 数据集名称，支持'cifar10'、'custom'或'ucf101'
        data_dir (str): 数据存储根目录路径
        batch_size (int): 批大小
        num_workers (int, optional): 数据加载的工作进程数，默认为4
        model_type (str, optional): 模型类型，用于视频数据集的动态transforms
        **kwargs: 其他数据集特定参数，如augment, download, csv_file等

    Returns:
        tuple: (train_loader, test_loader, num_classes) 训练和测试数据加载器及类别数

    Raises:
        ValueError: 当指定的数据集名称不支持时
    """
    dataset_name = dataset_name.lower()
    # 数据子采样比例（0-1），1.0表示使用全部数据
    data_percentage = float(kwargs.get('data_percentage', 1.0))

    if dataset_name == "cifar10":
        # 创建CIFAR-10数据集
        cifar10_dataset = CIFAR10Dataset(
            data_dir=data_dir,
            augment=kwargs.get('augment', True),
            download=kwargs.get('download', True)
        )
        
        train_dataset, test_dataset = cifar10_dataset.get_datasets()
        num_classes = cifar10_dataset.num_classes

    elif dataset_name == "custom":
        # 创建自定义数据集
        custom_dataset = CustomDatasetWrapper(
            data_dir=data_dir,
            csv_file=kwargs.get('csv_file', None),
            image_size=kwargs.get('image_size', 224),
            augment=kwargs.get('augment', True),
            train_split=kwargs.get('train_split', 0.8)
        )
        
        train_dataset, test_dataset = custom_dataset.get_datasets()
        num_classes = custom_dataset.num_classes

    elif dataset_name in ["ucf101", "ucf101_video"]:
        # 统一使用VideoDataset处理UCF-101视频数据（从预处理帧图像加载）
        clip_len = kwargs.get('clip_len', kwargs.get('frames_per_clip', 16))  # 兼容两种参数名

        # FPS采样相关参数
        sampling_mode = kwargs.get('sampling_mode', 'random')
        target_fps = kwargs.get('target_fps', None)
        original_fps = kwargs.get('original_fps', 16)

        train_dataset = VideoDataset(
            dataset_path=data_dir,
            images_path='train',
            clip_len=clip_len,
            model_type=model_type,  # 传递模型类型用于动态transforms
            sampling_mode=sampling_mode,
            target_fps=target_fps,
            original_fps=original_fps
        )

        # 将val和test合并作为测试集
        test_dataset = CombinedVideoDataset(
            dataset_path=data_dir,
            clip_len=clip_len,
            model_type=model_type,  # 传递模型类型用于动态transforms
            sampling_mode=sampling_mode,
            target_fps=target_fps,
            original_fps=original_fps
        )

        num_classes = 101  # UCF-101固定为101个类别

    elif dataset_name == "neonatal_multilabel":
        # 新生儿多标签行为识别数据集
        clip_len = kwargs.get('clip_len', kwargs.get('frames_per_clip', 16))
        top_n_classes = kwargs.get('top_n_classes', None)
        stratified_split = kwargs.get('stratified_split', True)
        min_samples_per_class = kwargs.get('min_samples_per_class', 10)

        # FPS采样相关参数
        sampling_mode = kwargs.get('sampling_mode', 'random')
        target_fps = kwargs.get('target_fps', None)
        original_fps = kwargs.get('original_fps', 16)

        # 数据路径：优先使用config中的root/params，未提供时回退到默认路径（相对路径）
        import os
        # 获取项目根目录（假设dataloader_factory.py在src/datasets/下）
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_frames = os.path.join(project_root, "../Neonate-Feeding-Assessment/data/cpu_processed_627/frames_segments")
        default_labels = os.path.join(project_root, "../Neonate-Feeding-Assessment/result_xlsx/shanghai/multi_hot_labels.xlsx")

        # 如果data_dir是相对路径，则相对于项目根目录解析
        if data_dir:
            if not os.path.isabs(data_dir):
                frames_dir = os.path.join(project_root, data_dir)
            else:
                frames_dir = data_dir
        else:
            frames_dir = default_frames

        # 标签文件路径处理
        labels_file_param = (
            kwargs.get('labels_file') or
            kwargs.get('label_file') or
            kwargs.get('labels_path')
        )
        if labels_file_param:
            if not os.path.isabs(labels_file_param):
                labels_file = os.path.join(project_root, labels_file_param)
            else:
                labels_file = labels_file_param
        else:
            labels_file = default_labels

        train_dataset = NeonatalMultilabelDataset(
            frames_dir=frames_dir,
            labels_file=labels_file,
            split='train',
            clip_len=clip_len,
            model_type=model_type,
            top_n_classes=top_n_classes,
            stratified_split=stratified_split,
            min_samples_per_class=min_samples_per_class,
            sampling_mode=sampling_mode,
            target_fps=target_fps,
            original_fps=original_fps
        )

        test_dataset = NeonatalMultilabelDataset(
            frames_dir=frames_dir,
            labels_file=labels_file,
            split='test',
            clip_len=clip_len,
            model_type=model_type,
            top_n_classes=top_n_classes,
            stratified_split=stratified_split,
            min_samples_per_class=min_samples_per_class,
            sampling_mode=sampling_mode,
            target_fps=target_fps,
            original_fps=original_fps
        )

        num_classes = train_dataset.get_num_classes()

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}。支持的数据集: cifar10, custom, ucf101, ucf101_video, neonatal_multilabel")

    # 按比例随机抽样数据子集（支持快速实验）
    if 0 < data_percentage < 1.0:
        def _sample_subset(dataset, split_name):
            total = len(dataset)
            sample_size = max(1, int(total * data_percentage))
            indices = torch.randperm(total)[:sample_size]
            # 数据子采样信息将在训练器中统一显示
            # if is_main_process():
            #     print(f"📊 数据子采样 - {split_name}: {total} -> {sample_size} 样本 (比例: {data_percentage:.1%})")
            return Subset(dataset, indices)
        
        original_train_size = len(train_dataset)
        original_test_size = len(test_dataset)
        
        train_dataset = _sample_subset(train_dataset, "训练集")
        test_dataset = _sample_subset(test_dataset, "测试集")
        
        # 数据采样信息将在训练器中统一显示
        # if is_main_process():
        #     print(f"🎯 数据采样完成 - 训练集: {original_train_size} -> {len(train_dataset)}, 测试集: {original_test_size} -> {len(test_dataset)}")
    # else:
        # if is_main_process():
        #     print(f"📊 使用完整数据集 - 训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")

    # 检查数据集是否为空
    train_size = len(train_dataset)
    test_size = len(test_dataset)

    if train_size == 0:
        raise ValueError(
            f"训练集为空！请检查以下可能的原因:\n"
            f"  1. data_percentage参数设置过小 (当前: {data_percentage:.1%})\n"
            f"  2. 数据集路径不正确: {data_dir}\n"
            f"  3. 数据过滤条件过于严格\n"
            f"  建议: 增大data_percentage或检查数据集配置"
        )

    if test_size == 0:
        raise ValueError(
            f"测试集为空！请检查以下可能的原因:\n"
            f"  1. data_percentage参数设置过小 (当前: {data_percentage:.1%})\n"
            f"  2. 数据集路径不正确: {data_dir}\n"
            f"  3. 数据划分比例不合理\n"
            f"  建议: 增大data_percentage或检查数据集配置"
        )

    # 🔧 新增：支持加权随机采样（仅用于多标签数据集的训练集）
    use_weighted_sampling = kwargs.get('use_weighted_sampling', False)
    weighted_sampling_mode = kwargs.get('weighted_sampling_mode', 'inverse_frequency')

    train_sampler = None
    train_shuffle = True

    if use_weighted_sampling and dataset_name == "neonatal_multilabel":
        if is_main_process():
            print(f"\n🎯 启用加权随机采样 (模式: {weighted_sampling_mode})")

        try:
            # 计算样本权重
            sample_weights = calculate_sample_weights(
                train_dataset,
                mode=weighted_sampling_mode,
                verbose=True
            )

            # 创建加权采样器
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True  # 允许重复采样
            )

            # 使用sampler时不能同时使用shuffle
            train_shuffle = False

            if is_main_process():
                print(f"✅ 加权采样器创建成功")

        except Exception as e:
            if is_main_process():
                print(f"⚠️  加权采样器创建失败: {e}")
                print(f"   回退到普通随机采样")
            train_sampler = None
            train_shuffle = True

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, num_classes


def get_dataset_info(dataset_name):
    """
    获取数据集基本信息
    
    Args:
        dataset_name (str): 数据集名称，支持'cifar10'、'custom'或'ucf101'
        
    Returns:
        dict: 包含数据集名称、类别数、输入尺寸和类别列表的字典
        
    Raises:
        ValueError: 当指定的数据集名称不支持时
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "cifar10":
        return {
            "name": "CIFAR-10",
            "num_classes": 10,
            "input_size": (3, 32, 32),
            "classes": ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        }
    elif dataset_name == "custom":
        return {
            "name": "Custom Dataset",
            "num_classes": None,  # 需要运行时确定
            "input_size": (3, 224, 224),  # 默认大小
            "classes": None  # 需要运行时确定
        }
    elif dataset_name in ["ucf101", "ucf101_video"]:
        return {
            "name": "UCF-101 Video",
            "num_classes": 101,
            "input_size": (3, 16, 112, 112),  # (C, T, H, W)
            "classes": None  # 需要运行时确定
        }
    elif dataset_name == "neonatal_multilabel":
        # 注意：实际的类别数量和类别名称需要在运行时从数据集实例获取
        return {
            "name": "Neonatal Multilabel Behavior Recognition",
            "num_classes": None,  # 需要运行时确定（取决于top_n_classes参数）
            "input_size": (3, 16, 112, 112),  # (C, T, H, W)
            "classes": None  # 需要运行时确定（取决于类别筛选结果）
        }
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

"""数据预处理模块

该模块负责处理原始数据(data/raw)并输出处理后的数据(data/processed)。
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional


class DataProcessor:
    """数据预处理器"""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # 确保目录存在
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def process_image_classification_dataset(self, 
                                            dataset_name: str,
                                            min_samples_per_class: int = 10,
                                            max_image_size: Tuple[int, int] = (1024, 1024),
                                            valid_extensions: List[str] = None) -> str:
        """
        处理图像分类数据集
        
        Args:
            dataset_name: 数据集名称
            min_samples_per_class: 每个类别的最小样本数
            max_image_size: 最大图像尺寸
            valid_extensions: 有效的图像扩展名
        
        Returns:
            str: 处理后的数据集路径
        """
        if valid_extensions is None:
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        raw_dataset_path = self.raw_data_dir / dataset_name
        processed_dataset_path = self.processed_data_dir / dataset_name
        
        if not raw_dataset_path.exists():
            raise FileNotFoundError(f"原始数据集路径不存在: {raw_dataset_path}")
        
        # 清理并创建处理后的数据集目录
        if processed_dataset_path.exists():
            shutil.rmtree(processed_dataset_path)
        processed_dataset_path.mkdir(parents=True)
        
        # 统计信息
        total_images = 0
        processed_images = 0
        class_stats = {}
        
        print(f"开始处理数据集: {dataset_name}")
        print(f"原始路径: {raw_dataset_path}")
        print(f"处理后路径: {processed_dataset_path}")
        
        # 遍历类别目录
        for class_dir in raw_dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            processed_class_dir = processed_dataset_path / class_name
            processed_class_dir.mkdir(exist_ok=True)
            
            # 收集该类别的所有有效图像
            valid_images = []
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in valid_extensions:
                    valid_images.append(img_file)
                    total_images += 1
            
            # 检查最小样本数要求
            if len(valid_images) < min_samples_per_class:
                print(f"警告: 类别 '{class_name}' 只有 {len(valid_images)} 个样本，少于最小要求 {min_samples_per_class}")
                continue
            
            # 处理图像
            class_processed = 0
            for img_file in valid_images:
                try:
                    # 处理单个图像
                    if self._process_single_image(img_file, processed_class_dir, max_image_size):
                        class_processed += 1
                        processed_images += 1
                except Exception as e:
                    print(f"处理图像失败 {img_file}: {e}")
            
            class_stats[class_name] = {
                'original': len(valid_images),
                'processed': class_processed
            }
            
            print(f"类别 '{class_name}': {class_processed}/{len(valid_images)} 图像处理成功")
        
        # 生成数据集信息文件
        self._generate_dataset_info(processed_dataset_path, class_stats, total_images, processed_images)
        
        print(f"\n数据集处理完成!")
        print(f"总图像数: {total_images}")
        print(f"成功处理: {processed_images}")
        print(f"处理后路径: {processed_dataset_path}")
        
        return str(processed_dataset_path)
    
    def _process_single_image(self, img_path: Path, output_dir: Path, max_size: Tuple[int, int]) -> bool:
        """
        处理单个图像
        
        Args:
            img_path: 输入图像路径
            output_dir: 输出目录
            max_size: 最大尺寸
        
        Returns:
            bool: 是否处理成功
        """
        try:
            # 打开图像
            with Image.open(img_path) as img:
                # 转换为RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整尺寸（如果需要）
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 保存处理后的图像
                output_path = output_dir / f"{img_path.stem}.jpg"
                img.save(output_path, 'JPEG', quality=95)
                
                return True
        except Exception as e:
            print(f"处理图像失败 {img_path}: {e}")
            return False
    
    def _generate_dataset_info(self, dataset_path: Path, class_stats: dict, total_original: int, total_processed: int):
        """
        生成数据集信息文件
        
        Args:
            dataset_path: 数据集路径
            class_stats: 类别统计信息
            total_original: 原始图像总数
            total_processed: 处理后图像总数
        """
        info = {
            'dataset_name': dataset_path.name,
            'total_classes': len(class_stats),
            'total_original_images': total_original,
            'total_processed_images': total_processed,
            'class_statistics': class_stats,
            'processed_path': str(dataset_path)
        }
        
        # 保存为JSON文件
        import json
        info_file = dataset_path / 'dataset_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        # 生成CSV文件（用于数据加载器）
        csv_data = []
        for class_name, stats in class_stats.items():
            class_dir = dataset_path / class_name
            class_idx = list(class_stats.keys()).index(class_name)
            
            for img_file in class_dir.glob('*.jpg'):
                csv_data.append([str(img_file.relative_to(dataset_path)), class_idx])
        
        if csv_data:
            df = pd.DataFrame(csv_data, columns=['image_path', 'label'])
            csv_file = dataset_path / 'dataset.csv'
            df.to_csv(csv_file, index=False)


def process_dataset(dataset_name: str, **kwargs) -> str:
    """
    便捷函数：处理数据集
    
    Args:
        dataset_name: 数据集名称
        **kwargs: 其他参数
    
    Returns:
        str: 处理后的数据集路径
    """
    processor = DataProcessor()
    return processor.process_image_classification_dataset(dataset_name, **kwargs)
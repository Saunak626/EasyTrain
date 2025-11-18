"""新生儿多标签数据集简化版单元测试

测试目标：
1. 验证简化版数据集的基本功能
2. 验证输出格式与完整版一致
3. 验证与DataLoader的集成
4. 对比简化版和完整版的性能

运行方式：
    python tests/test_neonatal_simple.py
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.neonatal_multilabel_simple import NeonatalMultilabelSimple
from src.datasets.neonatal_multilabel_dataset import NeonatalMultilabelDataset


def test_basic_functionality():
    """测试1：基本功能测试"""
    print("=" * 80)
    print("测试1：基本功能测试")
    print("=" * 80)
    
    # 数据路径
    frames_dir = '../Neonate-Feeding-Assessment/data/cpu_processed_627/frames_segments'
    labels_file = '../Neonate-Feeding-Assessment/result_xlsx/shanghai/multi_hot_labels.xlsx'
    
    # 检查路径是否存在
    if not os.path.exists(frames_dir):
        print(f"❌ 数据路径不存在: {frames_dir}")
        print("⚠️  跳过测试（需要实际数据）")
        return False
    
    try:
        # 创建简化版数据集
        dataset = NeonatalMultilabelSimple(
            frames_dir=frames_dir,
            labels_file=labels_file,
            split='train',
            clip_len=16
        )
        
        # 测试 __len__()
        length = len(dataset)
        print(f"✅ __len__() 返回: {length}")
        assert length > 0, "数据集长度应该大于0"
        
        # 测试 get_num_classes()
        num_classes = dataset.get_num_classes()
        print(f"✅ get_num_classes() 返回: {num_classes}")
        assert num_classes == 24, "类别数应该是24"
        
        # 测试 get_class_names()
        class_names = dataset.get_class_names()
        print(f"✅ get_class_names() 返回: {len(class_names)} 个类别")
        assert len(class_names) == 24, "类别名称列表长度应该是24"
        
        # 测试 __getitem__()
        frames, labels = dataset[0]
        print(f"✅ __getitem__(0) 返回:")
        print(f"   - frames shape: {frames.shape}")
        print(f"   - labels shape: {labels.shape}")
        print(f"   - frames dtype: {frames.dtype}")
        print(f"   - labels dtype: {labels.dtype}")
        
        # 验证shape
        assert frames.shape[0] == 3, "frames第一维应该是3（RGB通道）"
        assert frames.shape[1] == 16, "frames第二维应该是16（clip_len）"
        assert frames.shape[2] == 224, "frames第三维应该是224（高度）"
        assert frames.shape[3] == 224, "frames第四维应该是224（宽度）"
        assert labels.shape[0] == 24, "labels维度应该是24（类别数）"
        
        # 验证dtype
        assert frames.dtype == torch.float32, "frames dtype应该是float32"
        assert labels.dtype == torch.float32, "labels dtype应该是float32"
        
        print("\n✅ 测试1通过：基本功能正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_integration():
    """测试2：DataLoader集成测试"""
    print("\n" + "=" * 80)
    print("测试2：DataLoader集成测试")
    print("=" * 80)
    
    frames_dir = '../Neonate-Feeding-Assessment/data/cpu_processed_627/frames_segments'
    labels_file = '../Neonate-Feeding-Assessment/result_xlsx/shanghai/multi_hot_labels.xlsx'
    
    if not os.path.exists(frames_dir):
        print("⚠️  跳过测试（需要实际数据）")
        return False
    
    try:
        # 创建数据集
        dataset = NeonatalMultilabelSimple(
            frames_dir=frames_dir,
            labels_file=labels_file,
            split='train',
            clip_len=16
        )
        
        # 创建DataLoader
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # 使用0避免多进程问题
        )
        
        print(f"✅ DataLoader创建成功")
        print(f"   - batch_size: 4")
        print(f"   - 总batch数: {len(loader)}")
        
        # 迭代一个batch
        for batch_frames, batch_labels in loader:
            print(f"✅ 成功迭代一个batch:")
            print(f"   - batch_frames shape: {batch_frames.shape}")
            print(f"   - batch_labels shape: {batch_labels.shape}")
            
            # 验证batch shape
            assert batch_frames.shape[0] <= 4, "batch size应该<=4"
            assert batch_frames.shape[1] == 3, "通道数应该是3"
            assert batch_frames.shape[2] == 16, "帧数应该是16"
            assert batch_labels.shape[1] == 24, "类别数应该是24"
            
            break  # 只测试一个batch
        
        print("\n✅ 测试2通过：DataLoader集成正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_with_full_version():
    """测试3：与完整版对比测试"""
    print("\n" + "=" * 80)
    print("测试3：与完整版对比测试")
    print("=" * 80)
    
    frames_dir = '../Neonate-Feeding-Assessment/data/cpu_processed_627/frames_segments'
    labels_file = '../Neonate-Feeding-Assessment/result_xlsx/shanghai/multi_hot_labels.xlsx'
    
    if not os.path.exists(frames_dir):
        print("⚠️  跳过测试（需要实际数据）")
        return False
    
    try:
        # 创建简化版
        print("创建简化版数据集...")
        start_time = time.time()
        simple_dataset = NeonatalMultilabelSimple(
            frames_dir=frames_dir,
            labels_file=labels_file,
            split='train',
            clip_len=16
        )
        simple_time = time.time() - start_time
        
        # 创建完整版（使用默认参数，不筛选类别）
        print("\n创建完整版数据集...")
        start_time = time.time()
        full_dataset = NeonatalMultilabelDataset(
            frames_dir=frames_dir,
            labels_file=labels_file,
            split='train',
            clip_len=16,
            top_n_classes=None,  # 不筛选类别
            stratified_split=False  # 使用简单划分
        )
        full_time = time.time() - start_time
        
        # 对比加载时间
        print(f"\n⏱️  加载时间对比:")
        print(f"   - 简化版: {simple_time:.2f}秒")
        print(f"   - 完整版: {full_time:.2f}秒")
        print(f"   - 差异: {abs(simple_time - full_time):.2f}秒")
        
        # 对比数据集大小
        print(f"\n📊 数据集大小对比:")
        print(f"   - 简化版: {len(simple_dataset)} 样本")
        print(f"   - 完整版: {len(full_dataset)} 样本")
        
        # 对比类别数
        print(f"\n🏷️  类别数对比:")
        print(f"   - 简化版: {simple_dataset.get_num_classes()} 类")
        print(f"   - 完整版: {full_dataset.get_num_classes()} 类")
        
        # 对比输出格式（取第一个样本）
        if len(simple_dataset) > 0 and len(full_dataset) > 0:
            simple_frames, simple_labels = simple_dataset[0]
            full_frames, full_labels = full_dataset[0]
            
            print(f"\n📐 输出格式对比:")
            print(f"   - 简化版 frames shape: {simple_frames.shape}")
            print(f"   - 完整版 frames shape: {full_frames.shape}")
            print(f"   - 简化版 labels shape: {simple_labels.shape}")
            print(f"   - 完整版 labels shape: {full_labels.shape}")
            
            # 验证shape一致性
            assert simple_frames.shape[0] == full_frames.shape[0], "通道数应该一致"
            assert simple_frames.shape[1] == full_frames.shape[1], "帧数应该一致"
            # 注意：完整版可能有crop，所以H和W可能不同
            assert simple_labels.shape == full_labels.shape, "标签shape应该一致"
        
        print("\n✅ 测试3通过：与完整版对比正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("新生儿多标签数据集简化版单元测试")
    print("=" * 80)
    
    results = []
    
    # 运行测试
    results.append(("基本功能测试", test_basic_functionality()))
    results.append(("DataLoader集成测试", test_dataloader_integration()))
    results.append(("与完整版对比测试", test_comparison_with_full_version()))
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        return 1


if __name__ == '__main__':
    exit(main())


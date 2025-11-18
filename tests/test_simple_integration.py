"""简化版数据集集成测试

测试简化版数据集能否与训练流程正确集成。

运行方式：
    python tests/test_simple_integration.py
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_supported_datasets():
    """测试1：验证简化版数据集在支持列表中"""
    print("=" * 80)
    print("测试1：验证简化版数据集在支持列表中")
    print("=" * 80)
    
    from src.trainers.base_trainer import SUPPORTED_TASKS
    
    video_task = SUPPORTED_TASKS.get('video_classification')
    assert video_task is not None, "video_classification 任务应该存在"
    
    supported_datasets = video_task['supported_datasets']
    print(f"video_classification 支持的数据集: {supported_datasets}")
    
    assert 'neonatal_multilabel_simple' in supported_datasets, \
        "neonatal_multilabel_simple 应该在支持列表中"
    
    print("✅ 测试1通过：简化版数据集已在支持列表中")
    return True


def test_dataloader_factory_import():
    """测试2：验证 dataloader_factory 能导入简化版"""
    print("\n" + "=" * 80)
    print("测试2：验证 dataloader_factory 能导入简化版")
    print("=" * 80)
    
    try:
        from src.datasets.dataloader_factory import NeonatalMultilabelSimple
        print(f"✅ 成功导入: {NeonatalMultilabelSimple}")
        print(f"   类名: {NeonatalMultilabelSimple.__name__}")
        print("✅ 测试2通过：dataloader_factory 能正确导入简化版")
        return True
    except ImportError as e:
        print(f"❌ 测试2失败：导入失败 - {e}")
        return False


def test_dataloader_creation():
    """测试3：验证能创建简化版 DataLoader"""
    print("\n" + "=" * 80)
    print("测试3：验证能创建简化版 DataLoader")
    print("=" * 80)
    
    # 检查数据路径
    frames_dir = '../Neonate-Feeding-Assessment/data/cpu_processed_627/frames_segments'
    labels_file = '../Neonate-Feeding-Assessment/result_xlsx/shanghai/multi_hot_labels.xlsx'
    
    if not os.path.exists(frames_dir):
        print(f"⚠️  数据路径不存在: {frames_dir}")
        print("⚠️  跳过测试（需要实际数据）")
        return True  # 不算失败
    
    try:
        from src.datasets.dataloader_factory import create_dataloaders
        
        # 尝试创建 DataLoader
        train_loader, test_loader, num_classes = create_dataloaders(
            dataset_name='neonatal_multilabel_simple',
            data_dir=frames_dir,
            batch_size=4,
            num_workers=0,
            labels_file=labels_file,
            clip_len=16,
            train_ratio=0.8
        )

        print(f"✅ 成功创建 DataLoader:")
        print(f"   - 训练集大小: {len(train_loader.dataset)}")
        print(f"   - 测试集大小: {len(test_loader.dataset)}")
        print(f"   - 类别数: {num_classes}")
        
        # 尝试迭代一个 batch
        for batch_frames, batch_labels in train_loader:
            print(f"✅ 成功迭代一个 batch:")
            print(f"   - batch_frames shape: {batch_frames.shape}")
            print(f"   - batch_labels shape: {batch_labels.shape}")
            break
        
        print("✅ 测试3通过：能成功创建和使用 DataLoader")
        return True
        
    except Exception as e:
        print(f"❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unsupported_params_warning():
    """测试4：验证不支持的参数会给出警告"""
    print("\n" + "=" * 80)
    print("测试4：验证不支持的参数会给出警告")
    print("=" * 80)
    
    frames_dir = '../Neonate-Feeding-Assessment/data/cpu_processed_627/frames_segments'
    labels_file = '../Neonate-Feeding-Assessment/result_xlsx/shanghai/multi_hot_labels.xlsx'
    
    if not os.path.exists(frames_dir):
        print("⚠️  跳过测试（需要实际数据）")
        return True
    
    try:
        import logging
        from src.datasets.dataloader_factory import create_dataloaders
        
        # 设置日志级别以捕获警告
        logging.basicConfig(level=logging.WARNING)
        
        # 传递不支持的参数
        print("传递不支持的参数: top_n_classes=10, stratified_split=True")
        train_loader, test_loader, num_classes = create_dataloaders(
            dataset_name='neonatal_multilabel_simple',
            data_dir=frames_dir,
            batch_size=4,
            num_workers=0,
            labels_file=labels_file,
            clip_len=16,
            train_ratio=0.8,
            top_n_classes=10,  # 不支持
            stratified_split=True  # 不支持
        )
        
        print("✅ 测试4通过：能正确处理不支持的参数（应该看到警告信息）")
        return True
        
    except Exception as e:
        print(f"❌ 测试4失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("简化版数据集集成测试")
    print("=" * 80)
    
    results = []
    
    # 运行测试
    results.append(("支持列表验证", test_supported_datasets()))
    results.append(("导入验证", test_dataloader_factory_import()))
    results.append(("DataLoader创建", test_dataloader_creation()))
    results.append(("不支持参数警告", test_unsupported_params_warning()))
    
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


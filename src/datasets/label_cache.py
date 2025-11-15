"""
标签数据缓存模块
提供高效的标签数据缓存和访问机制，避免重复读取Excel文件
"""

import os
import pandas as pd
import pickle
import logging
from typing import Optional, Dict, Any
import hashlib

try:
    import fcntl  # POSIX文件锁
except ImportError:  # pragma: no cover - 非POSIX系统回退
    fcntl = None

logger = logging.getLogger(__name__)


class LabelCache:
    """标签数据缓存类
    
    使用单例模式缓存标签数据，避免重复读取Excel文件。
    支持文件变更检测和自动更新缓存。
    """
    
    _instance = None
    _cache_data: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LabelCache, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_labels_data(cls, labels_file: str, force_reload: bool = False) -> pd.DataFrame:
        """获取标签数据(支持磁盘持久化缓存,跨进程共享)

        Args:
            labels_file (str): 标签文件路径
            force_reload (bool): 是否强制重新加载

        Returns:
            pd.DataFrame: 标签数据
        """
        cache_key = os.path.abspath(labels_file)

        # 生成持久化缓存文件路径
        cache_file = labels_file.replace('.xlsx', '_cache.pkl').replace('.xls', '_cache.pkl')
        lock_file = cache_file + '.lock'

        # 检查是否需要重新加载
        if force_reload or not cls._should_use_cache(cache_key, labels_file):
            lock_handle = cls._acquire_lock(lock_file)
            try:
                # 加锁后再次尝试从磁盘缓存读取（避免竞争）
                if (not force_reload and
                        cls._is_disk_cache_valid(cache_file, labels_file)):
                    df = cls._load_from_disk_cache(cache_file)
                    cls._update_memory_cache(cache_key, labels_file, df)
                    return df

                # 从Excel文件读取
                logger.info(f"加载标签文件: {labels_file}")
                start_time = pd.Timestamp.now()
                df = pd.read_excel(labels_file)
                load_time = (pd.Timestamp.now() - start_time).total_seconds()
                logger.info(f"标签文件加载完成: {len(df)} 行, 耗时 {load_time:.2f}秒")

                # 保存到磁盘缓存
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(df, f)
                    logger.info(f"标签数据已缓存到磁盘: {cache_file}")
                except Exception as e:
                    logger.warning(f"保存磁盘缓存失败: {e}")

                cls._update_memory_cache(cache_key, labels_file, df)
            finally:
                cls._release_lock(lock_handle)
        else:
            logger.debug(f"使用内存缓存的标签数据: {cache_key}")

        return cls._cache_data[cache_key]['data']

    @staticmethod
    def _is_disk_cache_valid(cache_file: str, labels_file: str) -> bool:
        return os.path.exists(cache_file) and os.path.getmtime(cache_file) >= os.path.getmtime(labels_file)

    @classmethod
    def _load_from_disk_cache(cls, cache_file: str) -> pd.DataFrame:
        logger.info(f"从磁盘缓存加载标签数据: {cache_file}")
        start_time = pd.Timestamp.now()
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        load_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"磁盘缓存加载完成: {len(df)} 行, 耗时 {load_time:.2f}秒")
        return df

    @classmethod
    def _update_memory_cache(cls, cache_key: str, labels_file: str, df: pd.DataFrame):
        cls._cache_data[cache_key] = {
            'data': df,
            'file_mtime': os.path.getmtime(labels_file),
            'file_size': os.path.getsize(labels_file),
            'load_time': pd.Timestamp.now()
        }

    @staticmethod
    def _acquire_lock(lock_file: str):
        if fcntl is None:
            return None
        lock_dir = os.path.dirname(lock_file) or '.'
        os.makedirs(lock_dir, exist_ok=True)
        handle = open(lock_file, 'w')
        fcntl.flock(handle, fcntl.LOCK_EX)
        return handle

    @staticmethod
    def _release_lock(handle):
        if handle is None:
            return
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()
    
    @classmethod
    def _should_use_cache(cls, cache_key: str, labels_file: str) -> bool:
        """检查是否应该使用缓存
        
        Args:
            cache_key (str): 缓存键
            labels_file (str): 标签文件路径
            
        Returns:
            bool: 是否使用缓存
        """
        if cache_key not in cls._cache_data:
            return False
        
        # 检查文件是否存在
        if not os.path.exists(labels_file):
            return False
        
        cached_info = cls._cache_data[cache_key]
        current_mtime = os.path.getmtime(labels_file)
        current_size = os.path.getsize(labels_file)
        
        # 检查文件是否被修改
        if (cached_info['file_mtime'] != current_mtime or 
            cached_info['file_size'] != current_size):
            logger.info(f"检测到文件变更，将重新加载: {labels_file}")
            return False
        
        return True
    
    @classmethod
    def clear_cache(cls, labels_file: Optional[str] = None):
        """清除缓存
        
        Args:
            labels_file (str, optional): 特定文件的缓存，None表示清除所有缓存
        """
        if labels_file is None:
            cls._cache_data.clear()
            logger.info("已清除所有标签缓存")
        else:
            cache_key = os.path.abspath(labels_file)
            if cache_key in cls._cache_data:
                del cls._cache_data[cache_key]
                logger.info(f"已清除标签缓存: {labels_file}")
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """获取缓存信息
        
        Returns:
            Dict[str, Any]: 缓存统计信息
        """
        info = {
            'cached_files': len(cls._cache_data),
            'files': []
        }
        
        for cache_key, cached_info in cls._cache_data.items():
            file_info = {
                'file_path': cache_key,
                'rows': len(cached_info['data']),
                'columns': len(cached_info['data'].columns),
                'load_time': cached_info['load_time'],
                'file_size_mb': cached_info['file_size'] / 1024 / 1024
            }
            info['files'].append(file_info)
        
        return info


class OptimizedLabelProcessor:
    """优化的标签处理器
    
    提供高效的标签数据处理和样本构建功能
    """
    
    def __init__(self, labels_file: str, behavior_labels: list):
        """初始化标签处理器
        
        Args:
            labels_file (str): 标签文件路径
            behavior_labels (list): 行为标签列表
        """
        self.labels_file = labels_file
        self.behavior_labels = behavior_labels
        self.cache = LabelCache()
        
        # 加载标签数据
        self.df = self.cache.get_labels_data(labels_file)
        
        # 预处理标签数据
        self._preprocess_labels()
    
    def _preprocess_labels(self):
        """预处理标签数据"""
        logger.info("预处理标签数据...")
        
        # 清理文件名
        self.df['session_name'] = self.df['文件名'].str.replace('.mov', '').str.replace('.mp4', '').str.strip()
        self.df['clip_id'] = self.df['文件内动作序号'].astype(str)
        
        # 预计算标签向量
        label_vectors = []
        for _, row in self.df.iterrows():
            label_vector = []
            for label in self.behavior_labels:
                if label in row:
                    label_vector.append(float(row[label]))
                else:
                    label_vector.append(0.0)
            label_vectors.append(label_vector)
        
        self.df['label_vector'] = label_vectors
        
        # 创建快速查找索引
        self.df['lookup_key'] = self.df['session_name'] + '/' + self.df['clip_id']
        self.lookup_dict = dict(zip(self.df['lookup_key'], self.df.index))
        
        logger.info(f"标签预处理完成: {len(self.df)} 条记录")
    
    def get_label_vector(self, session_name: str, clip_id: str) -> Optional[list]:
        """获取标签向量
        
        Args:
            session_name (str): session名称
            clip_id (str): clip ID
            
        Returns:
            Optional[list]: 标签向量，如果不存在返回None
        """
        lookup_key = f"{session_name}/{clip_id}"
        
        if lookup_key in self.lookup_dict:
            idx = self.lookup_dict[lookup_key]
            return self.df.iloc[idx]['label_vector']
        
        return None
    
    def get_sample_info(self, session_name: str, clip_id: str) -> Optional[Dict[str, Any]]:
        """获取样本完整信息
        
        Args:
            session_name (str): session名称
            clip_id (str): clip ID
            
        Returns:
            Optional[Dict[str, Any]]: 样本信息，如果不存在返回None
        """
        lookup_key = f"{session_name}/{clip_id}"
        
        if lookup_key in self.lookup_dict:
            idx = self.lookup_dict[lookup_key]
            row = self.df.iloc[idx]
            
            return {
                'session_name': session_name,
                'clip_id': clip_id,
                'labels': row['label_vector'],
                'start_time': row.get('开始时间(秒)', 0),
                'end_time': row.get('结束时间(秒)', 0),
                'duration': row.get('时长(秒)', 0)
            }
        
        return None
    
    def get_all_valid_samples(self, frames_dir: str) -> list:
        """获取所有有效样本
        
        Args:
            frames_dir (str): 帧图像根目录
            
        Returns:
            list: 有效样本列表
        """
        logger.info("构建有效样本列表...")
        
        valid_samples = []
        available_sessions = set(os.listdir(frames_dir)) if os.path.exists(frames_dir) else set()
        
        for _, row in self.df.iterrows():
            session_name = row['session_name']
            clip_id = row['clip_id']
            
            # 检查帧图像目录是否存在
            if session_name not in available_sessions:
                continue
            
            clip_dir = os.path.join(frames_dir, session_name, clip_id)
            if not os.path.exists(clip_dir):
                continue
            
            # 检查是否有帧图像
            frame_files = [f for f in os.listdir(clip_dir) if f.endswith('.jpg')]
            if len(frame_files) == 0:
                continue
            
            # 跳过全零标签（可选）
            if sum(row['label_vector']) == 0:
                continue
            
            sample_info = {
                'session_name': session_name,
                'clip_id': clip_id,
                'frames_dir': clip_dir,
                'labels': row['label_vector'],
                'start_time': row.get('开始时间(秒)', 0),
                'end_time': row.get('结束时间(秒)', 0),
                'duration': row.get('时长(秒)', 0)
            }
            valid_samples.append(sample_info)
        
        logger.info(f"找到 {len(valid_samples)} 个有效样本")
        return valid_samples

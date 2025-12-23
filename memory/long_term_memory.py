"""
长期记忆管理器 (Long-Term Memory Manager)

职责：
- 存储和检索知识经验
- 管理错误类型和纠正策略
- 维护主题相关的思考角度
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from core.schemas import (
    LongTermMemoryEntry,
    ErrorExample,
)


class LongTermMemoryManager:
    """
    长期记忆管理器
    
    提供长期记忆的存储、检索、更新功能
    支持基于主题和错误类型的索引
    """
    
    def __init__(self, storage_dir: str = "memory/storage/long_term"):
        """
        初始化长期记忆管理器
        
        Args:
            storage_dir: 存储目录路径
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 内存索引
        self._entries: Dict[str, LongTermMemoryEntry] = {}
        self._topic_index: Dict[str, List[str]] = {}  # topic -> entry_ids
        self._error_type_index: Dict[str, List[str]] = {}  # error_type -> entry_ids
        
        # 加载已有数据
        self._load_all()
    
    def add_entry(self, entry: LongTermMemoryEntry) -> str:
        """
        添加长期记忆条目
        
        Args:
            entry: 长期记忆条目
            
        Returns:
            条目ID
        """
        if not entry.entry_id:
            entry.entry_id = self._generate_entry_id()
        
        entry.update_timestamp = datetime.now()
        self._entries[entry.entry_id] = entry
        
        # 更新索引
        self._update_index(entry)
        
        # 持久化
        self._save_entry(entry)
        
        return entry.entry_id
    
    def get_entry(self, entry_id: str) -> Optional[LongTermMemoryEntry]:
        """
        获取长期记忆条目
        
        Args:
            entry_id: 条目ID
            
        Returns:
            LongTermMemoryEntry实例，不存在则返回None
        """
        return self._entries.get(entry_id)
    
    def find_by_topic(self, topic_category: str) -> List[LongTermMemoryEntry]:
        """
        按主题查找条目
        
        Args:
            topic_category: 主题类别
            
        Returns:
            匹配的条目列表
        """
        entry_ids = self._topic_index.get(topic_category, [])
        return [self._entries[eid] for eid in entry_ids if eid in self._entries]
    
    def find_by_error_type(self, error_type: str) -> List[LongTermMemoryEntry]:
        """
        按错误类型查找条目
        
        Args:
            error_type: 错误类型
            
        Returns:
            匹配的条目列表
        """
        entry_ids = self._error_type_index.get(error_type, [])
        return [self._entries[eid] for eid in entry_ids if eid in self._entries]
    
    def search(
        self,
        topic_category: Optional[str] = None,
        error_type: Optional[str] = None,
        min_success_rate: float = 0.0
    ) -> List[LongTermMemoryEntry]:
        """
        搜索长期记忆条目
        
        Args:
            topic_category: 主题类别（可选）
            error_type: 错误类型（可选）
            min_success_rate: 最小成功率
            
        Returns:
            匹配的条目列表
        """
        results = list(self._entries.values())
        
        if topic_category:
            results = [e for e in results if e.topic_category == topic_category]
        
        if error_type:
            results = [e for e in results if e.error_type == error_type]
        
        results = [e for e in results if e.success_rate >= min_success_rate]
        
        # 按成功率降序排序
        results.sort(key=lambda x: x.success_rate, reverse=True)
        
        return results
    
    def add_error_example(
        self,
        entry_id: str,
        question: str,
        wrong_answer: str,
        correct_answer: str,
        error_analysis: str
    ) -> None:
        """
        添加错误案例到指定条目
        
        Args:
            entry_id: 条目ID
            question: 问题
            wrong_answer: 错误答案
            correct_answer: 正确答案
            error_analysis: 错误分析
        """
        entry = self.get_entry(entry_id)
        if entry is None:
            raise ValueError(f"条目不存在: {entry_id}")
        
        example = ErrorExample(
            example_id=f"ex_{len(entry.error_examples) + 1}",
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            error_analysis=error_analysis,
            created_at=datetime.now()
        )
        
        entry.add_error_example(example)
        self._save_entry(entry)
    
    def add_correction_strategy(self, entry_id: str, strategy: str) -> None:
        """
        添加纠正策略
        
        Args:
            entry_id: 条目ID
            strategy: 纠正策略描述
        """
        entry = self.get_entry(entry_id)
        if entry is None:
            raise ValueError(f"条目不存在: {entry_id}")
        
        if strategy not in entry.correction_strategies:
            entry.correction_strategies.append(strategy)
            entry.update_timestamp = datetime.now()
            self._save_entry(entry)
    
    def add_thinking_angle(self, entry_id: str, angle: str) -> None:
        """
        添加思考角度
        
        Args:
            entry_id: 条目ID
            angle: 思考角度描述
        """
        entry = self.get_entry(entry_id)
        if entry is None:
            raise ValueError(f"条目不存在: {entry_id}")
        
        if angle not in entry.thinking_angles:
            entry.thinking_angles.append(angle)
            entry.update_timestamp = datetime.now()
            self._save_entry(entry)
    
    def update_success_rate(self, entry_id: str, success: bool) -> None:
        """
        更新成功率
        
        Args:
            entry_id: 条目ID
            success: 是否成功
        """
        entry = self.get_entry(entry_id)
        if entry is None:
            raise ValueError(f"条目不存在: {entry_id}")
        
        entry.update_success_rate(success)
        self._save_entry(entry)
    
    def get_correction_strategies(
        self,
        topic_category: Optional[str] = None,
        error_type: Optional[str] = None
    ) -> List[str]:
        """
        获取相关的纠正策略
        
        Args:
            topic_category: 主题类别（可选）
            error_type: 错误类型（可选）
            
        Returns:
            纠正策略列表
        """
        entries = self.search(topic_category=topic_category, error_type=error_type)
        
        strategies = []
        for entry in entries:
            for strategy in entry.correction_strategies:
                if strategy not in strategies:
                    strategies.append(strategy)
        
        return strategies
    
    def get_thinking_angles(
        self,
        topic_category: Optional[str] = None,
        error_type: Optional[str] = None
    ) -> List[str]:
        """
        获取相关的思考角度
        
        Args:
            topic_category: 主题类别（可选）
            error_type: 错误类型（可选）
            
        Returns:
            思考角度列表
        """
        entries = self.search(topic_category=topic_category, error_type=error_type)
        
        angles = []
        for entry in entries:
            for angle in entry.thinking_angles:
                if angle not in angles:
                    angles.append(angle)
        
        return angles
    
    def _generate_entry_id(self) -> str:
        """生成唯一的条目ID"""
        import uuid
        return f"ltm_{uuid.uuid4().hex[:8]}"
    
    def _update_index(self, entry: LongTermMemoryEntry) -> None:
        """更新索引"""
        # 更新主题索引
        if entry.topic_category:
            if entry.topic_category not in self._topic_index:
                self._topic_index[entry.topic_category] = []
            if entry.entry_id not in self._topic_index[entry.topic_category]:
                self._topic_index[entry.topic_category].append(entry.entry_id)
        
        # 更新错误类型索引
        if entry.error_type:
            if entry.error_type not in self._error_type_index:
                self._error_type_index[entry.error_type] = []
            if entry.entry_id not in self._error_type_index[entry.error_type]:
                self._error_type_index[entry.error_type].append(entry.entry_id)
    
    def _save_entry(self, entry: LongTermMemoryEntry) -> None:
        """保存条目到存储"""
        filepath = self.storage_dir / f"{entry.entry_id}.json"
        data = self._serialize(entry)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def _load_all(self) -> None:
        """加载所有已存储的条目"""
        if not self.storage_dir.exists():
            return
        
        for filepath in self.storage_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                entry = self._deserialize(data)
                self._entries[entry.entry_id] = entry
                self._update_index(entry)
            except Exception as e:
                print(f"加载长期记忆条目失败 {filepath}: {e}")
    
    def _serialize(self, entry: LongTermMemoryEntry) -> dict:
        """序列化条目"""
        return {
            "entry_id": entry.entry_id,
            "topic_category": entry.topic_category,
            "content_pattern": entry.content_pattern,
            "error_type": entry.error_type,
            "error_examples": [
                {
                    "example_id": ex.example_id,
                    "question": ex.question,
                    "wrong_answer": ex.wrong_answer,
                    "correct_answer": ex.correct_answer,
                    "error_analysis": ex.error_analysis,
                    "created_at": ex.created_at.isoformat()
                }
                for ex in entry.error_examples
            ],
            "correction_strategies": entry.correction_strategies,
            "thinking_angles": entry.thinking_angles,
            "success_rate": entry.success_rate,
            "usage_count": entry.usage_count,
            "created_at": entry.created_at.isoformat(),
            "update_timestamp": entry.update_timestamp.isoformat()
        }
    
    def _deserialize(self, data: dict) -> LongTermMemoryEntry:
        """反序列化条目"""
        error_examples = [
            ErrorExample(
                example_id=ex["example_id"],
                question=ex["question"],
                wrong_answer=ex["wrong_answer"],
                correct_answer=ex["correct_answer"],
                error_analysis=ex["error_analysis"],
                created_at=datetime.fromisoformat(ex["created_at"])
            )
            for ex in data.get("error_examples", [])
        ]
        
        return LongTermMemoryEntry(
            entry_id=data["entry_id"],
            topic_category=data.get("topic_category", ""),
            content_pattern=data.get("content_pattern", ""),
            error_type=data.get("error_type", ""),
            error_examples=error_examples,
            correction_strategies=data.get("correction_strategies", []),
            thinking_angles=data.get("thinking_angles", []),
            success_rate=data.get("success_rate", 0.0),
            usage_count=data.get("usage_count", 0),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            update_timestamp=datetime.fromisoformat(data.get("update_timestamp", datetime.now().isoformat()))
        )


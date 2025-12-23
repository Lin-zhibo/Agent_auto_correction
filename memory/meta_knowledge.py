"""
元知识管理器 (Meta-Knowledge Manager)

职责：
- 管理Agent表现统计
- 计算循环次数和停止条件
- 提供Agent推荐
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from core.schemas import (
    MetaKnowledge,
    AgentPerformanceRecord,
    AgentType,
)


class MetaKnowledgeManager:
    """
    元知识管理器
    
    提供元知识的存储、更新、决策支持功能
    核心功能：
    1. Agent表现追踪
    2. 循环次数计算
    3. Agent推荐
    """
    
    def __init__(self, storage_dir: str = "memory/storage/meta"):
        """
        初始化元知识管理器
        
        Args:
            storage_dir: 存储目录路径
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 按问题类型存储的元知识
        self._knowledge: Dict[str, MetaKnowledge] = {}
        
        # 全局Agent表现统计
        self._global_agent_performance: Dict[str, AgentPerformanceRecord] = {}
        
        # 加载已有数据
        self._load_all()
    
    def get_or_create(self, question_type: str) -> MetaKnowledge:
        """
        获取或创建指定问题类型的元知识
        
        Args:
            question_type: 问题类型
            
        Returns:
            MetaKnowledge实例
        """
        if question_type not in self._knowledge:
            self._knowledge[question_type] = MetaKnowledge(
                question_type=question_type,
                last_updated=datetime.now()
            )
        return self._knowledge[question_type]
    
    def update_agent_performance(
        self,
        question_type: str,
        agent_id: str,
        agent_type: AgentType,
        success: bool,
        confidence: float
    ) -> None:
        """
        更新Agent表现记录
        
        Args:
            question_type: 问题类型
            agent_id: Agent ID
            agent_type: Agent类型
            success: 是否成功
            confidence: 置信度
        """
        # 更新特定问题类型的元知识
        mk = self.get_or_create(question_type)
        mk.update_agent_performance(agent_id, agent_type, success, confidence)
        
        # 更新全局Agent表现
        if agent_id not in self._global_agent_performance:
            self._global_agent_performance[agent_id] = AgentPerformanceRecord(
                agent_id=agent_id,
                agent_type=agent_type
            )
        self._global_agent_performance[agent_id].update_performance(success, confidence)
        
        # 持久化
        self._save(question_type)
        self._save_global_performance()
    
    def update_error_rate(self, question_type: str, error_rate: float) -> None:
        """
        更新错误率
        
        Args:
            question_type: 问题类型
            error_rate: 错误率
        """
        mk = self.get_or_create(question_type)
        mk.error_rate = error_rate
        mk.last_updated = datetime.now()
        self._save(question_type)
    
    def update_avg_confidence(self, question_type: str, avg_confidence: float) -> None:
        """
        更新平均置信度
        
        Args:
            question_type: 问题类型
            avg_confidence: 平均置信度
        """
        mk = self.get_or_create(question_type)
        mk.avg_confidence = avg_confidence
        mk.last_updated = datetime.now()
        self._save(question_type)
    
    def calculate_max_rounds(
        self,
        question_type: str,
        current_confidence: float,
        verbose: bool = False
    ) -> int:
        """
        计算推荐的最大循环次数
        
        基于公式: N = min(N_max, ceil(α × E × (1-C)^β × N_max))
        
        Args:
            question_type: 问题类型
            current_confidence: 当前置信度
            verbose: 是否输出调试信息
            
        Returns:
            推荐的循环次数
        """
        mk = self.get_or_create(question_type)
        return mk.calculate_max_rounds(current_confidence, verbose=verbose)
    
    def should_stop(
        self,
        question_type: str,
        current_confidence: float,
        current_round: int
    ) -> bool:
        """
        判断是否应该停止循环
        
        停止条件:
        1. 当前置信度 >= 阈值
        2. 已达到最大循环次数
        
        Args:
            question_type: 问题类型
            current_confidence: 当前置信度
            current_round: 当前轮次
            
        Returns:
            是否应该停止
        """
        mk = self.get_or_create(question_type)
        return mk.should_stop(current_confidence, current_round)
    
    def get_recommended_agents(
        self,
        question_type: str,
        top_k: int = 3
    ) -> List[str]:
        """
        获取推荐的Agent列表
        
        使用ε-greedy策略:
        - 以exploration_prob概率随机选择（探索）
        - 以1-exploration_prob概率选择历史表现最好的（利用）
        
        Args:
            question_type: 问题类型
            top_k: 返回前k个Agent
            
        Returns:
            推荐的Agent ID列表
        """
        mk = self.get_or_create(question_type)
        
        # 如果该问题类型没有足够的历史数据，使用全局表现
        if len(mk.agent_performance_log) < top_k:
            return self._get_global_recommended_agents(top_k)
        
        return mk.get_recommended_agents(top_k)
    
    def _get_global_recommended_agents(self, top_k: int = 3) -> List[str]:
        """获取全局推荐的Agent列表"""
        import random
        
        if not self._global_agent_performance:
            # 返回默认Agent列表
            return []
        
        # 按正确率排序
        sorted_agents = sorted(
            self._global_agent_performance.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        )
        
        return [agent_id for agent_id, _ in sorted_agents[:top_k]]
    
    def set_loop_parameters(
        self,
        question_type: str,
        max_rounds: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        exploration_prob: Optional[float] = None
    ) -> None:
        """
        设置循环控制参数
        
        Args:
            question_type: 问题类型
            max_rounds: 最大循环次数
            confidence_threshold: 置信度阈值
            alpha: 循环次数计算参数α
            beta: 循环次数计算参数β
            exploration_prob: 探索概率
        """
        mk = self.get_or_create(question_type)
        
        if max_rounds is not None:
            mk.max_rounds = max_rounds
        if confidence_threshold is not None:
            mk.confidence_threshold = confidence_threshold
        if alpha is not None:
            mk.alpha = alpha
        if beta is not None:
            mk.beta = beta
        if exploration_prob is not None:
            mk.exploration_prob = exploration_prob
        
        mk.last_updated = datetime.now()
        self._save(question_type)
    
    def get_agent_performance(
        self,
        agent_id: str,
        question_type: Optional[str] = None
    ) -> Optional[AgentPerformanceRecord]:
        """
        获取Agent表现记录
        
        Args:
            agent_id: Agent ID
            question_type: 问题类型（可选，不指定则返回全局表现）
            
        Returns:
            AgentPerformanceRecord实例
        """
        if question_type:
            mk = self._knowledge.get(question_type)
            if mk and agent_id in mk.agent_performance_log:
                return mk.agent_performance_log[agent_id]
        
        return self._global_agent_performance.get(agent_id)
    
    def _save(self, question_type: str) -> None:
        """保存指定问题类型的元知识"""
        mk = self._knowledge.get(question_type)
        if mk is None:
            return
        
        filepath = self.storage_dir / f"{question_type}.json"
        data = self._serialize(mk)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def _save_global_performance(self) -> None:
        """保存全局Agent表现"""
        filepath = self.storage_dir / "_global_performance.json"
        data = {
            agent_id: {
                "agent_id": record.agent_id,
                "agent_type": record.agent_type.value,
                "accuracy": record.accuracy,
                "confidence_variance": record.confidence_variance,
                "total_tasks": record.total_tasks,
                "successful_tasks": record.successful_tasks,
                "last_updated": record.last_updated.isoformat()
            }
            for agent_id, record in self._global_agent_performance.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_all(self) -> None:
        """加载所有已存储的元知识"""
        if not self.storage_dir.exists():
            return
        
        # 加载问题类型相关的元知识
        for filepath in self.storage_dir.glob("*.json"):
            if filepath.name == "_global_performance.json":
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                mk = self._deserialize(data)
                self._knowledge[mk.question_type] = mk
            except Exception as e:
                print(f"加载元知识失败 {filepath}: {e}")
        
        # 加载全局Agent表现
        global_filepath = self.storage_dir / "_global_performance.json"
        if global_filepath.exists():
            try:
                with open(global_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for agent_id, record_data in data.items():
                    self._global_agent_performance[agent_id] = AgentPerformanceRecord(
                        agent_id=record_data["agent_id"],
                        agent_type=AgentType(record_data["agent_type"]),
                        accuracy=record_data["accuracy"],
                        confidence_variance=record_data["confidence_variance"],
                        total_tasks=record_data["total_tasks"],
                        successful_tasks=record_data["successful_tasks"],
                        last_updated=datetime.fromisoformat(record_data["last_updated"])
                    )
            except Exception as e:
                print(f"加载全局Agent表现失败: {e}")
    
    def _serialize(self, mk: MetaKnowledge) -> dict:
        """序列化MetaKnowledge"""
        return {
            "question_type": mk.question_type,
            "error_rate": mk.error_rate,
            "avg_confidence": mk.avg_confidence,
            "agent_performance_log": {
                agent_id: {
                    "agent_id": record.agent_id,
                    "agent_type": record.agent_type.value,
                    "accuracy": record.accuracy,
                    "confidence_variance": record.confidence_variance,
                    "total_tasks": record.total_tasks,
                    "successful_tasks": record.successful_tasks,
                    "last_updated": record.last_updated.isoformat()
                }
                for agent_id, record in mk.agent_performance_log.items()
            },
            "agent_recommendation_list": mk.agent_recommendation_list,
            "exploration_prob": mk.exploration_prob,
            "max_rounds": mk.max_rounds,
            "confidence_threshold": mk.confidence_threshold,
            "alpha": mk.alpha,
            "beta": mk.beta,
            "last_updated": mk.last_updated.isoformat()
        }
    
    def _deserialize(self, data: dict) -> MetaKnowledge:
        """反序列化MetaKnowledge"""
        agent_performance_log = {}
        for agent_id, record_data in data.get("agent_performance_log", {}).items():
            agent_performance_log[agent_id] = AgentPerformanceRecord(
                agent_id=record_data["agent_id"],
                agent_type=AgentType(record_data["agent_type"]),
                accuracy=record_data["accuracy"],
                confidence_variance=record_data["confidence_variance"],
                total_tasks=record_data["total_tasks"],
                successful_tasks=record_data["successful_tasks"],
                last_updated=datetime.fromisoformat(record_data["last_updated"])
            )
        
        return MetaKnowledge(
            question_type=data["question_type"],
            error_rate=data.get("error_rate", 0.0),
            avg_confidence=data.get("avg_confidence", 0.0),
            agent_performance_log=agent_performance_log,
            agent_recommendation_list=data.get("agent_recommendation_list", []),
            exploration_prob=data.get("exploration_prob", 0.1),
            max_rounds=data.get("max_rounds", 10),
            confidence_threshold=data.get("confidence_threshold", 0.8),
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))
        )


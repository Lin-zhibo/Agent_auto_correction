"""
工作记忆管理器 (Working Memory Manager)

职责：
- 管理当前任务的临时状态
- 存储各Agent的输出结果
- 追踪循环轮次和冲突点
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from core.schemas import (
    WorkingMemory,
    AgentOutput,
    AgentType,
)


class WorkingMemoryManager:
    """
    工作记忆管理器
    
    提供工作记忆的创建、更新、持久化等功能
    """
    
    def __init__(self, storage_dir: str = "memory/storage/working"):
        """
        初始化工作记忆管理器
        
        Args:
            storage_dir: 存储目录路径
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._active_memories: Dict[str, WorkingMemory] = {}
    
    def create(self, task_id: str, question_text: str) -> WorkingMemory:
        """
        创建新的工作记忆
        
        Args:
            task_id: 任务ID
            question_text: 问题文本
            
        Returns:
            新创建的WorkingMemory实例
        """
        memory = WorkingMemory(
            task_id=task_id,
            question_text=question_text,
            round_index=1,
            current_phase="init",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self._active_memories[task_id] = memory
        return memory
    
    def get(self, task_id: str) -> Optional[WorkingMemory]:
        """
        获取工作记忆
        
        Args:
            task_id: 任务ID
            
        Returns:
            WorkingMemory实例，不存在则返回None
        """
        # 先从内存中查找
        if task_id in self._active_memories:
            return self._active_memories[task_id]
        
        # 尝试从存储中加载
        return self._load_from_storage(task_id)
    
    def update(self, memory: WorkingMemory) -> None:
        """
        更新工作记忆
        
        Args:
            memory: 要更新的WorkingMemory实例
        """
        memory.updated_at = datetime.now()
        self._active_memories[memory.task_id] = memory
    
    def add_agent_output(
        self,
        task_id: str,
        agent_id: str,
        agent_type: AgentType,
        response: str,
        confidence: float,
        reasoning: str = ""
    ) -> None:
        """
        添加Agent输出到工作记忆
        
        Args:
            task_id: 任务ID
            agent_id: Agent ID
            agent_type: Agent类型
            response: 回答内容
            confidence: 置信度
            reasoning: 推理过程
        """
        memory = self.get(task_id)
        if memory is None:
            raise ValueError(f"工作记忆不存在: {task_id}")
        
        output = AgentOutput(
            agent_id=agent_id,
            agent_type=agent_type,
            response=response,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
        memory.add_agent_output(output)
        self.update(memory)
    
    def update_student_response(
        self,
        task_id: str,
        response: str,
        confidence: float
    ) -> None:
        """
        更新Student Agent的回答
        
        Args:
            task_id: 任务ID
            response: 回答内容
            confidence: 置信度
        """
        memory = self.get(task_id)
        if memory is None:
            raise ValueError(f"工作记忆不存在: {task_id}")
        
        memory.student_response = response
        memory.student_confidence = confidence
        self.update(memory)
    
    def set_conflicts(self, task_id: str, conflicts: List[str]) -> None:
        """
        设置检测到的冲突点
        
        Args:
            task_id: 任务ID
            conflicts: 冲突描述列表
        """
        memory = self.get(task_id)
        if memory is None:
            raise ValueError(f"工作记忆不存在: {task_id}")
        
        memory.conflicts_detected = conflicts
        self.update(memory)
    
    def set_complementary_points(self, task_id: str, points: List[str]) -> None:
        """
        设置互补观点
        
        Args:
            task_id: 任务ID
            points: 互补观点列表
        """
        memory = self.get(task_id)
        if memory is None:
            raise ValueError(f"工作记忆不存在: {task_id}")
        
        memory.complementary_points = points
        self.update(memory)
    
    def set_summary_prompt(self, task_id: str, prompt: str) -> None:
        """
        设置汇总提示词
        
        Args:
            task_id: 任务ID
            prompt: 提示词内容
        """
        memory = self.get(task_id)
        if memory is None:
            raise ValueError(f"工作记忆不存在: {task_id}")
        
        memory.summary_prompt = prompt
        self.update(memory)
    
    def advance_round(self, task_id: str) -> int:
        """
        进入下一轮循环
        
        Args:
            task_id: 任务ID
            
        Returns:
            新的轮次索引
        """
        memory = self.get(task_id)
        if memory is None:
            raise ValueError(f"工作记忆不存在: {task_id}")
        
        memory.advance_round()
        self.update(memory)
        return memory.round_index
    
    def set_phase(self, task_id: str, phase: str) -> None:
        """
        设置当前处理阶段
        
        Args:
            task_id: 任务ID
            phase: 阶段名称 (init/student/multi_agent/insight/complete)
        """
        valid_phases = ["init", "student", "multi_agent", "insight", "complete"]
        if phase not in valid_phases:
            raise ValueError(f"无效的阶段: {phase}，有效值: {valid_phases}")
        
        memory = self.get(task_id)
        if memory is None:
            raise ValueError(f"工作记忆不存在: {task_id}")
        
        memory.current_phase = phase
        self.update(memory)
    
    def save(self, task_id: str) -> None:
        """
        持久化工作记忆到存储
        
        Args:
            task_id: 任务ID
        """
        memory = self.get(task_id)
        if memory is None:
            raise ValueError(f"工作记忆不存在: {task_id}")
        
        filepath = self.storage_dir / f"{task_id}.json"
        data = self._serialize(memory)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def delete(self, task_id: str) -> None:
        """
        删除工作记忆
        
        Args:
            task_id: 任务ID
        """
        if task_id in self._active_memories:
            del self._active_memories[task_id]
        
        filepath = self.storage_dir / f"{task_id}.json"
        if filepath.exists():
            filepath.unlink()
    
    def _load_from_storage(self, task_id: str) -> Optional[WorkingMemory]:
        """从存储加载工作记忆"""
        filepath = self.storage_dir / f"{task_id}.json"
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self._deserialize(data)
    
    def _serialize(self, memory: WorkingMemory) -> dict:
        """序列化WorkingMemory为字典"""
        return {
            "task_id": memory.task_id,
            "question_text": memory.question_text,
            "round_index": memory.round_index,
            "current_phase": memory.current_phase,
            "agent_outputs": [
                {
                    "agent_id": o.agent_id,
                    "agent_type": o.agent_type.value,
                    "response": o.response,
                    "confidence": o.confidence,
                    "reasoning": o.reasoning,
                    "timestamp": o.timestamp.isoformat(),
                    "metadata": o.metadata
                }
                for o in memory.agent_outputs
            ],
            "conflicts_detected": memory.conflicts_detected,
            "complementary_points": memory.complementary_points,
            "summary_prompt": memory.summary_prompt,
            "student_response": memory.student_response,
            "student_confidence": memory.student_confidence,
            "round_history": memory.round_history,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat()
        }
    
    def _deserialize(self, data: dict) -> WorkingMemory:
        """反序列化字典为WorkingMemory"""
        agent_outputs = [
            AgentOutput(
                agent_id=o["agent_id"],
                agent_type=AgentType(o["agent_type"]),
                response=o["response"],
                confidence=o["confidence"],
                reasoning=o.get("reasoning", ""),
                timestamp=datetime.fromisoformat(o["timestamp"]),
                metadata=o.get("metadata", {})
            )
            for o in data.get("agent_outputs", [])
        ]
        
        return WorkingMemory(
            task_id=data["task_id"],
            question_text=data["question_text"],
            round_index=data.get("round_index", 1),
            current_phase=data.get("current_phase", "init"),
            agent_outputs=agent_outputs,
            conflicts_detected=data.get("conflicts_detected", []),
            complementary_points=data.get("complementary_points", []),
            summary_prompt=data.get("summary_prompt", ""),
            student_response=data.get("student_response", ""),
            student_confidence=data.get("student_confidence", 0.0),
            round_history=data.get("round_history", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )


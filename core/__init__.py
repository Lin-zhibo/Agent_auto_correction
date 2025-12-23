"""
核心模块

提供系统核心功能：
- Orchestrator: 流程编排器，协调各Agent工作
- LoopController: 循环控制器，管理反思循环
- MemoryCoordinator: 记忆协调器，管理三种记忆的交互
- schemas: 数据结构定义
"""

from core.orchestrator import Orchestrator
from core.loop_controller import LoopController
from core.memory_coordinator import MemoryCoordinator
from core.schemas import (
    AgentType,
    TaskStatus,
    AgentOutput,
    ErrorExample,
    AgentPerformanceRecord,
    WorkingMemory,
    LongTermMemoryEntry,
    MetaKnowledge,
    AgentConfig,
    Task,
    TaskResult,
)

__all__ = [
    "Orchestrator",
    "LoopController",
    "MemoryCoordinator",
    # 数据结构
    "AgentType",
    "TaskStatus",
    "AgentOutput",
    "ErrorExample",
    "AgentPerformanceRecord",
    "WorkingMemory",
    "LongTermMemoryEntry",
    "MetaKnowledge",
    "AgentConfig",
    "Task",
    "TaskResult",
]


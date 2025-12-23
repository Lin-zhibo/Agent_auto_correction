"""
Agent模块

提供多Agent系统的核心Agent实现：
- BaseAgent: Agent基类
- StudentAgent: 学生Agent（主答题者）
- InsightAgent: 洞察Agent（汇总反馈）
- 多Agent子模块: 包含各种角色Agent
"""

from agent.base_agent import BaseAgent
from agent.student_agent import StudentAgent
from agent.insight_agent import InsightAgent

__all__ = [
    "BaseAgent",
    "StudentAgent",
    "InsightAgent",
]


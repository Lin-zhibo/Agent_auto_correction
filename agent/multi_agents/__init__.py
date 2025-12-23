"""
多Agent子模块

包含多种角色的专家Agent：
- AuthorityAgent: 权威者 - 提供权威观点
- QuestionerAgent: 质疑者 - 挑战现有答案
- LogicAnalystAgent: 逻辑分析员 - 逻辑推理分析
- ListenerAgent: 倾听者 - 综合意见
- CompanionAgent: 同伴 - 协作支持
- HeuristicSolverAgent: 启发式解决者
- ExplanationGeneratorAgent: 解释生成者
- ConceptAnalogistAgent: 概念类比者
"""

from agent.multi_agents.authority_agent import AuthorityAgent
from agent.multi_agents.questioner_agent import QuestionerAgent
from agent.multi_agents.logic_analyst_agent import LogicAnalystAgent
from agent.multi_agents.listener_agent import ListenerAgent
from agent.multi_agents.companion_agent import CompanionAgent
from agent.multi_agents.heuristic_solver_agent import HeuristicSolverAgent
from agent.multi_agents.explanation_generator_agent import ExplanationGeneratorAgent
from agent.multi_agents.concept_analogist_agent import ConceptAnalogistAgent

__all__ = [
    "AuthorityAgent",
    "QuestionerAgent",
    "LogicAnalystAgent",
    "ListenerAgent",
    "CompanionAgent",
    "HeuristicSolverAgent",
    "ExplanationGeneratorAgent",
    "ConceptAnalogistAgent",
]


# -*- coding: utf-8 -*-
"""Agents：Student、Meta-Knowledge、多角色 Agent、Insight。"""

from agents.student_agent import StudentAgent
from agents.meta_knowledge import MetaKnowledge
from agents.multi_agents import (
    BaseAgent,
    Questioner,
    LogicAnalyzer,
    AuthorityChecker,
    Explainer,
    EconomicsExpert,
    MathExpert,
    PhilosophyExpert,
    ScienceExpert,
    Critic,
    Supporter,
    ClarityChecker,
    CompletenessChecker,
    EvidenceChecker,
    BrevityAdvisor,
    AudienceAdvisor,
    AgentFactory,
)
from agents.insight_agent import InsightAgent

__all__ = [
    "StudentAgent",
    "MetaKnowledge",
    "BaseAgent",
    "Questioner",
    "LogicAnalyzer",
    "AuthorityChecker",
    "Explainer",
    "EconomicsExpert",
    "MathExpert",
    "PhilosophyExpert",
    "ScienceExpert",
    "Critic",
    "Supporter",
    "ClarityChecker",
    "CompletenessChecker",
    "EvidenceChecker",
    "BrevityAdvisor",
    "AudienceAdvisor",
    "AgentFactory",
    "InsightAgent",
]

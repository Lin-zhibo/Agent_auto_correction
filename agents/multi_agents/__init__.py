# -*- coding: utf-8 -*-
"""多 Agent 阶段：从不同角度评估/优化回答。"""

from agents.multi_agents.base import BaseAgent, parse_comment_score, parse_agent_output, AGENT_OUTPUT_JSON_SCHEMA
from agents.multi_agents.questioner import Questioner
from agents.multi_agents.logic_analyzer import LogicAnalyzer
from agents.multi_agents.authority_checker import AuthorityChecker
from agents.multi_agents.explainer import Explainer
from agents.multi_agents.economics_expert import EconomicsExpert
from agents.multi_agents.math_expert import MathExpert
from agents.multi_agents.philosophy_expert import PhilosophyExpert
from agents.multi_agents.science_expert import ScienceExpert
from agents.multi_agents.critic import Critic
from agents.multi_agents.supporter import Supporter
from agents.multi_agents.clarity_checker import ClarityChecker
from agents.multi_agents.completeness_checker import CompletenessChecker
from agents.multi_agents.evidence_checker import EvidenceChecker
from agents.multi_agents.brevity_advisor import BrevityAdvisor
from agents.multi_agents.audience_advisor import AudienceAdvisor
from agents.multi_agents.factory import AgentFactory

__all__ = [
    "BaseAgent",
    "parse_comment_score",
    "parse_agent_output",
    "AGENT_OUTPUT_JSON_SCHEMA",
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
]

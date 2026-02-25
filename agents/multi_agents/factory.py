# -*- coding: utf-8 -*-
"""多 Agent 工厂：按名称返回对应 Agent 实例。"""

from agents.multi_agents.base import BaseAgent
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


class AgentFactory:
    """多 Agent 工厂，按名称返回对应 Agent 实例。"""

    agents = {
        "questioner": Questioner(),
        "logic_analyzer": LogicAnalyzer(),
        "authority_checker": AuthorityChecker(),
        "explainer": Explainer(),
        "economics_expert": EconomicsExpert(),
        "math_expert": MathExpert(),
        "philosophy_expert": PhilosophyExpert(),
        "science_expert": ScienceExpert(),
        "critic": Critic(),
        "supporter": Supporter(),
        "clarity_checker": ClarityChecker(),
        "completeness_checker": CompletenessChecker(),
        "evidence_checker": EvidenceChecker(),
        "brevity_advisor": BrevityAdvisor(),
        "audience_advisor": AudienceAdvisor(),
    }

    @staticmethod
    def create(agent_name: str) -> BaseAgent:
        if agent_name not in AgentFactory.agents:
            return Questioner()
        return AgentFactory.agents[agent_name]

# -*- coding: utf-8 -*-
"""经济学专家 Agent：从经济学视角审视回答。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output
from utils.llm import llm_call


class EconomicsExpert(BaseAgent):
    """经济学专家：从成本、收益、激励、市场等经济学角度审视回答。"""

    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = '''
你是一位“经济学专家（Economics Expert）”。请从经济学视角评估以下回答：
原始问题：
{question}

待审查回答：
{answer}

请审视：
1. 回答是否涵盖成本、收益、风险、激励机制和市场行为分析？
2. 是否有不符合经济学基本假设的表述？
3. 给出从经济学原理出发的补充或修正建议。

{json_schema}
'''.format(question=question, answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.75)
        return {
            "agent": "economics_expert",
            "comment": comment_text or "可从经济学角度补充成本收益或激励分析。",
            "score": score,
        }

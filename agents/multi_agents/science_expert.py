# -*- coding: utf-8 -*-
"""科学专家 Agent：从科学方法与实证角度审视回答。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output
from utils.llm import llm_call


class ScienceExpert(BaseAgent):
    """科学专家：检查可验证性、因果推断、实验与证据是否恰当。"""

    def review(self, answer: str) -> dict[str, Any]:
        prompt = '''
你是一位“科学专家（Science Expert）”。请从科学方法与实证角度分析以下回答：
{answer}

请检查：
1. 回答是否具备可验证性与科学逻辑？
2. 证据与因果推断是否可靠？
3. 是否符合科学表述规范，并给出强化科学性的建议。

{json_schema}
'''.format(answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.8)
        return {
            "agent": "science_expert",
            "comment": comment_text or "可从可验证性与证据链角度加强科学表述。",
            "score": score,
        }

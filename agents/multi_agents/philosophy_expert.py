# -*- coding: utf-8 -*-
"""哲学专家 Agent：从哲学与概念分析角度审视回答。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output
from utils.llm import llm_call


class PhilosophyExpert(BaseAgent):
    """哲学专家：检查概念界定、前提假设、论证结构及哲学意涵。"""

    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = '''
你是一位“哲学专家（Philosophy Expert）”。请从哲学与概念分析角度审视以下回答：
原始问题：
{question}

待审查回答：
{answer}

请审查：
1. 概念界定是否清晰、有无前提假设不明？
2. 论证结构是否严密，有无隐藏的逻辑漏洞？
3. 提出澄清或深化哲学意涵的建议，使论述更具思想深度。

{json_schema}
'''.format(question=question, answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.78)
        return {
            "agent": "philosophy_expert",
            "comment": comment_text or "可从概念界定与论证结构上做哲学层面的澄清。",
            "score": score,
        }

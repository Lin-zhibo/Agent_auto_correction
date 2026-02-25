# -*- coding: utf-8 -*-
"""批判者 Agent：严格批判回答中的漏洞与薄弱环节。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output
from utils.llm import llm_call


class Critic(BaseAgent):
    """批判者：严格批判回答中的漏洞、薄弱环节与可反驳之处。"""

    def review(self, answer: str) -> dict[str, Any]:
        prompt = '''
你是一位“批判性审稿者（Critical Reviewer）”。请对以下回答进行严格、专业的批评性分析：
{answer}

请指出：
1. 回答中的逻辑漏洞、薄弱环节或可反驳之处。
2. 过度断言或缺乏支撑的结论。
3. 提供具体改进方法，使论点更有力、更合理。

{json_schema}
'''.format(answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.7)
        return {
            "agent": "critic",
            "comment": comment_text or "请指出回答中的漏洞与薄弱环节并改进。",
            "score": score,
        }

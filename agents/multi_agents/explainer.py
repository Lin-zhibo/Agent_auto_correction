# -*- coding: utf-8 -*-
"""解释者 Agent：建议用实例或类比帮助理解。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output
from utils.llm import llm_call


class Explainer(BaseAgent):
    """解释者：建议用实例或类比帮助理解。"""

    def review(self, answer: str) -> dict[str, Any]:
        prompt = '''
你是一位“解释者（Explainer）”。请对以下回答进行分析：
{answer}

请建议：
1. 可以加入哪些实例、比喻或场景化说明帮助读者理解？
2. 是否存在抽象难懂的部分需具体化？
3. 提供一个示例或比喻提升表达的可理解性。

{json_schema}
'''.format(answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.9)
        return {
            "agent": "explainer",
            "comment": comment_text or "用实例或类比来帮助理解。",
            "score": score,
        }

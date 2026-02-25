# -*- coding: utf-8 -*-
"""逻辑分析者 Agent：检查因果与逻辑自洽。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output
from utils.llm import llm_call


class LogicAnalyzer(BaseAgent):
    """逻辑分析者：检查因果与逻辑自洽。"""

    def review(self, answer: str) -> dict[str, Any]:
        prompt = '''
你是一位“逻辑分析专家（Logic Analyzer）”。请对以下回答进行严谨的逻辑评估：
{answer}

请检查：
1. 因果关系与推理链是否自洽？
2. 是否出现逻辑跳跃、矛盾或循环论证？
3. 给出改进建议，提高论证的逻辑一致性与严密性。

{json_schema}
'''.format(answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.8)
        return {
            "agent": "logic_analyzer",
            "comment": comment_text or "检查逻辑因果是否自洽。",
            "score": score,
        }

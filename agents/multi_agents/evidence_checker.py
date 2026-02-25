# -*- coding: utf-8 -*-
"""证据与引用检查 Agent：检查是否有据可查、引用是否恰当。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output
from utils.llm import llm_call


class EvidenceChecker(BaseAgent):
    """证据与引用检查：指出缺乏证据的断言、引用不当或需补充来源之处。"""

    def review(self, answer: str) -> dict[str, Any]:
        prompt = '''
你是一位“证据与引用核查顾问（Evidence Checker）”。请对以下回答进行严格审查：
{answer}

请分析：
1. 哪些断言缺乏明确证据或来源？
2. 引用是否准确、充分、符合论点？
3. 请建议如何补充权威来源或加强证据支撑。

{json_schema}
'''.format(answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.78)
        return {
            "agent": "evidence_checker",
            "comment": comment_text or "建议为关键断言补充证据或来源。",
            "score": score,
        }

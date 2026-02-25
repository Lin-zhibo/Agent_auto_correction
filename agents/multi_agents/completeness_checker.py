# -*- coding: utf-8 -*-
"""完整性检查 Agent：检查是否遗漏关键点或必要前提。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output


class CompletenessChecker(BaseAgent):
    """完整性检查：指出遗漏的关键点、必要前提或边界条件。"""

    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = '''
你是一位“完整性检查者（Completeness Checker）”。请对以下回答进行分析：
原始问题：
{question}

待审查回答：
{answer}

请评估：
1. 是否遗漏关键点、必要前提或边界条件？
2. 是否有被忽略的例外情况或限制？
3. 请提出补充建议，使回答更加全面与稳健。

{json_schema}
'''.format(question=question, answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = self._llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.8)
        return {
            "agent": "completeness_checker",
            "comment": comment_text or "建议补充遗漏的关键点或前提。",
            "score": score,
        }

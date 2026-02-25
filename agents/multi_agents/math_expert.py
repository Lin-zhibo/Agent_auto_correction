# -*- coding: utf-8 -*-
"""数学专家 Agent：从数学严谨性与形式化角度审视回答。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output


class MathExpert(BaseAgent):
    """数学专家：检查定义是否严谨、推理是否可形式化、有无数学谬误。"""

    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = '''
你是一位“数学专家（Mathematics Expert）”。请从数学与形式化角度审视以下回答：
原始问题：
{question}

待审查回答：
{answer}

请分析：
1. 术语和定义是否严谨、无歧义？
2. 推理是否可形式化、是否具有逻辑一致性？
3. 若存在数学表述错误或非标准表达，请指出并提出改进建议。

{json_schema}
'''.format(question=question, answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = self._llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.8)
        return {
            "agent": "math_expert",
            "comment": comment_text or "可从数学严谨性与形式化角度改进表述。",
            "score": score,
        }

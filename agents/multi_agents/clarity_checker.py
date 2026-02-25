# -*- coding: utf-8 -*-
"""清晰度检查 Agent：检查表述是否清晰、无歧义。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output
from utils.llm import llm_call


class ClarityChecker(BaseAgent):
    """清晰度检查：指出歧义、含糊或难懂之处，建议更清晰的表述。"""

    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = '''
# Role: Clarity Checker

# Profile
You are a specialized Clarity Checker and Linguistic Precision Analyst. Your primary objective is to minimize cognitive load and eliminate semantic noise. You assess text objectively to ensure that the intended message is transmitted without distortion, ambiguity, or unnecessary complexity.

# Task
Please conduct a neutral, comprehensive clarity assessment of the following answer under the original question context.

Original Question:
{question}

Answer To Review:

{answer}

# Analysis Framework
1.  **Ambiguity Detection:** Identify phrases, pronouns, or terms that could be interpreted in multiple ways (semantic ambiguity) or lack specific definition (vagueness).
2.  **Structural Clarity:** Evaluate sentence construction. Identify convoluted syntax or run-on sentences that impede immediate understanding.
3.  **Constructive Optimization:** For segments identified as unclear, provide specific, unambiguous alternative phrasings that preserve the original logic.

# Guidelines for Neutrality & Objectivity
- **Balanced Review:** You must maintain a strictly neutral tone. Do not criticize for the sake of criticism.
- **Validation:** If the text is already clear, precise, and well-structured, you must acknowledge its quality. Only propose changes where there is a tangible risk of misunderstanding or where readability can be objectively improved.
- **Precision:** Avoid subjective stylistic preferences; focus solely on whether the meaning is successfully conveyed to the reader.

# Output Format
You must output your analysis and recommendations strictly adhering to the following JSON structure:

{json_schema}
'''.format(question=question, answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.78)
        return {
            "agent": "clarity_checker",
            "comment": comment_text or "建议消除歧义、使表述更清晰。",
            "score": score,
        }

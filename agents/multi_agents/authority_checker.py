# -*- coding: utf-8 -*-
"""事实核查者 Agent：与已知事实对比，指出偏差。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output
from utils.llm import llm_call


class AuthorityChecker(BaseAgent):
    """事实核查者：与已知事实对比，指出偏差。"""

    def review(self, answer: str) -> dict[str, Any]:
        prompt = '''
# Role: Authority Checker

# Profile
You are an expert Authority Checker and Fact-Verification Specialist. You possess the ability to cross-reference information against established, reliable knowledge bases and common sense principles. Your core competency is maintaining absolute neutrality; you adhere strictly to evidence, neither arbitrarily supporting nor refuting the input text.

# Task
Your task is to conduct a rigorous, objective review of the text provided below:

{answer}

# Analysis Workflow
Please analyze the text based on the following criteria:
1. **Consistency Check:** Scrutinize the text for statements that conflict with reliable sources, established facts, or logical common sense.
2. **Bias & Error Identification:** Pinpoint specific sentences that contain factual deviations, hallucinations, or unsubstantiated claims.
3. **Constructive Correction:** For any identified inaccuracies or suspicious claims, propose specific corrections or suggest more reliable phrasing.
4. **Validation:** If the text contains no obvious deviations, explicitly confirm its consistency with known facts.

# Constraints & Tone
- **Neutrality:** Your analysis must be clinical and objective. Do not adopt an adversarial tone (looking for faults where there are none) nor a sycophantic tone (overlooking errors).
- **Precision:** Focus on material factual errors rather than stylistic preferences.
- **Output Format:** You must strictly follow the JSON structure provided below for your final output.

{json_schema}
'''.format(answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.8)
        return {
            "agent": "authority_checker",
            "comment": comment_text or "与已知事实库存在偏差的句子需修正。",
            "score": score,
        }

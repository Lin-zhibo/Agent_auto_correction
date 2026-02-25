# -*- coding: utf-8 -*-
"""简洁性建议 Agent：检查是否冗长、可精简。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output
from utils.llm import llm_call


class BrevityAdvisor(BaseAgent):
    """简洁性建议：指出冗长或重复之处，建议在不失信息的前提下精简表述。"""

    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = '''
# Role: Brevity Advisor

# Profile
You are an expert Brevity Advisor and Editorial Strategist. Your core competency is "Linguistic Economy"—the ability to convey the maximum amount of information with the minimum number of words, without sacrificing clarity, nuance, or professional tone. You focus on high signal-to-noise ratio in communication.

# Task
Please conduct a rigorous efficiency audit of the following answer under the original question context.

Original Question:
{question}

Answer To Review:

{answer}

# Analysis Objectives
1.  **Redundancy Identification:** Detect and isolate tautologies, repetitive phrasing, and unnecessary modifiers (e.g., "absolutely essential").
2.  **Information Distillation:** Identify "filler" content that adds no semantic value. Determine which information can be condensed without losing the core message or key details.
3.  **Syntactic Optimization:** Analyze sentence structures. Look for opportunities to convert passive voice to active voice, reduce prepositional phrases, and simplify complex constructions for better readability.

# Action Required
- Provide a clear breakdown of inefficiencies.
- Offer optimized, concise rewriting examples that improve flow and impact.
- **Crucial:** Ensure the original meaning and intent are 100% preserved in your suggested revisions.

# Output Format
You must output your analysis and suggestions strictly according to the following JSON structure:

{json_schema}
'''.format(question=question, answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.75)
        return {
            "agent": "brevity_advisor",
            "comment": comment_text or "建议精简冗长或重复表述。",
            "score": score,
        }

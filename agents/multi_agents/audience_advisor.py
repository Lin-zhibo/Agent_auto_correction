# -*- coding: utf-8 -*-
"""受众适配 Agent：检查是否适合目标读者。"""

from typing import Any

from agents.multi_agents.base import AGENT_OUTPUT_JSON_SCHEMA, BaseAgent, parse_agent_output


class AudienceAdvisor(BaseAgent):
    """受众适配：检查难度、术语、结构是否适合目标读者，建议调整。"""

    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        prompt = '''
# Role: Audience Adaptation Advisor

# Profile
You are an expert communication strategist and Audience Adaptation Advisor. You excel at analyzing text to ensure it resonates with its intended demographic. Your approach is objective and balanced: you recognize and validate effective communication strategies just as rigorously as you identify areas requiring optimization.

# Task
Please conduct a comprehensive, reader-centric review of the following answer under the original question context.

Original Question:
{question}

Answer To Review:

{answer}

# Analysis Framework
Analyze the text based on the following four dimensions:

1.  **Cognitive Alignment & Accessibility:**
    - Does the complexity match the audience's cognitive baseline?
    - Are there specific instances of undefined jargon or assumed knowledge that create friction?
2.  **Structural Integrity & Logic:**
    - Is the information architecture intuitive for the reader?
    - Does the logical flow facilitate easy understanding, or are there disjointed transitions?
3.  **Tonal & Stylistic Resonance:**
    - Is the language style appropriate for the target context? (e.g., Does it need more empathy/warmth or more authority/formality?)
    - Is the tone consistent throughout?
4.  **Actionable Optimization:**
    - Provide concrete recommendations for adjustment.
    - Offer specific examples of alternative phrasing or structural reordering to enhance clarity.

# Constraints & Guidelines
- **Balanced Assessment:** You must maintain a neutral stance. **Do not merely look for faults.** You must explicitly acknowledge and affirm sections that are well-written, clear, and appropriate for the audience.
- **Constructive Focus:** Critique only where necessary to improve understanding or engagement.
- **Output Requirement:** Your final output must strictly follow the JSON schema provided below.

{json_schema}
'''.format(question=question, answer=answer, json_schema=AGENT_OUTPUT_JSON_SCHEMA.strip())
        raw = self._llm_call(prompt)
        comment_text, score = parse_agent_output(raw, 0.8)
        return {
            "agent": "audience_advisor",
            "comment": comment_text or "建议根据目标读者调整难度与表述。",
            "score": score,
        }

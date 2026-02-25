# -*- coding: utf-8 -*-
"""
Insight Agent：收集多 Agent 阶段输出，调用 LLM 综合问题、Student 回答与各 Agent 反馈，
生成下一轮应关注与修改的总结，指导 Student 改进回答。
"""

from typing import Any

from config import INSIGHT_MODEL_KEY, INSIGHT_MODEL_NAME, INSIGHT_MODEL_URL
from utils.llm import llm_call
from utils.logger import get_logger
from utils.parse_llm_json import parse_llm_output_to_dict

logger = get_logger(__name__)


class InsightAgent:
    """
    综合多 Agent 反馈与当前回答，通过 LLM 生成下一轮改进指导。
    输出：下一轮 Student 应关注的问题、注意事项与修改建议的总结。
    """

    def integrate_feedback(
        self,
        question: str,
        base_answer: str,
        feedbacks: list[dict[str, Any]],
    ) -> str:
        """
        收集多 Agent 阶段的输出，结合问题与 Student 当前回答，
        调用 LLM 做综合总结，明确下一轮应关注什么、注意什么、需要修改什么。
        返回给 Student 作为下一轮 revise 的指导文本。
        """
        # 1. 收集多 Agent 阶段的输出
        agent_lines = []
        for f in feedbacks:
            agent = f.get("agent", "unknown")
            comment = (f.get("comment") or "").strip()
            if comment:
                agent_lines.append(f"【{agent}】\n{comment}")
        raw_feedback = "\n\n".join(agent_lines) if agent_lines else "（各 Agent 暂无具体文字反馈。）"
        logger.info("Insight 收集到 %s 条 Agent 反馈", len(agent_lines))

        # 2. 若无有效反馈，返回默认提示，不调用 LLM
        if not agent_lines:
            return "各 Agent 暂无具体反馈意见，请在表述严谨性、完整性与逻辑清晰度上稍作检查与优化。"

        # 3. 调用 LLM 综合问题、当前回答与多 Agent 反馈，生成下一轮改进总结（JSON）
        prompt = '''You are an expert "Reflection & Improvement Advisor" specialized in analyzing LLM-generated answers. Your sole purpose is to help improve the quality, accuracy, reasoning depth, completeness, safety, and user-value of the next iteration of the answer.
你应当确认你的每一条输出建议对于得出正确答案都是必要的

---

You will be given exactly three pieces of information:

【Question】
{question}

【Current Student Answer】
{base_answer}

【Multi-Agent Raw Feedback】
{raw_feedback}

Your task is to deeply analyze the above three parts together, then distill the **most impactful and highest-priority improvement directions** for the next version of the answer.

Follow these strict analysis guidelines:

• Prioritize issues that affect factual correctness, logical coherence, major omissions, or severe misalignments with the question most highly.
• Give strong weight to feedback that appears repeatedly or comes from multiple agents.
• Identify any safety, bias, toxicity, overconfidence, hallucination, or misleading statement risks.
• Consider clarity, conciseness, structure, professional tone, and usefulness to the end user.
• Distinguish between "must-fix" problems and "nice-to-have" polish.
• Think about what specific evidence or reasoning is missing that could strengthen the answer.

Output **exclusively** a valid JSON object with **exactly** the following three keys. Do not include any other text, comments, markdown, explanations, apologies or code fences before/after the JSON.

{{
  "focus_points": [
    "short, clear, high-priority improvement areas (use imperative form, 1 sentence each)",
    "example: Correct factual error about X mentioned in feedback"
  ],
  "cautions": [
    "critical things the next answer MUST avoid",
    "example: Do not repeat the hallucinated statistic about Y"
  ],
  "revision_suggestions": [
    "concrete, actionable improvement methods or content directions",
    "example: Add a step-by-step reasoning chain to demonstrate how conclusion is reached"
  ]
}}

• Each array should contain 2–6 items (aim for precision, not quantity).
• Use concise, professional English.
• Items in each list should be independent, specific and non-redundant.
• Order the items in each array from most important → least important.

Output only the JSON.
'''.format(
            question=question.strip(),
            base_answer=(base_answer or "").strip(),
            raw_feedback=raw_feedback,
        )
        try:
            raw = llm_call(
                prompt,
                model=INSIGHT_MODEL_NAME or None,
                api_key=INSIGHT_MODEL_KEY or None,
                base_url=INSIGHT_MODEL_URL or None,
            ).strip()
            obj = parse_llm_output_to_dict(raw)
            if obj is not None:
                guidance = _format_improvement_guidance(obj)
                if guidance:
                    logger.info("Insight LLM 综合完成，指导长度=%s 字 (全文): %s", len(guidance), guidance)
                    return guidance
            return raw if raw else raw_feedback
        except Exception:
            return raw_feedback


def _format_improvement_guidance(obj: dict[str, Any]) -> str:
    """将 LLM 返回的 JSON 格式化为供 Student 使用的改进指导文本。"""
    lines = []
    focus = obj.get("focus_points")
    if isinstance(focus, list) and focus:
        lines.append("【下一轮应优先关注】")
        for i, item in enumerate(focus, 1):
            lines.append(f"{i}. {item}")
    cautions = obj.get("cautions")
    if isinstance(cautions, list) and cautions:
        lines.append("\n【需避免或特别注意】")
        for i, item in enumerate(cautions, 1):
            lines.append(f"{i}. {item}")
    suggestions = obj.get("revision_suggestions")
    if isinstance(suggestions, list) and suggestions:
        lines.append("\n【具体修改建议】")
        for i, item in enumerate(suggestions, 1):
            lines.append(f"{i}. {item}")
    if not lines:
        return ""
    return "\n".join(lines).strip()

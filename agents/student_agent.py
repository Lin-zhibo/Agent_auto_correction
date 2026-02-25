# -*- coding: utf-8 -*-
"""
Student Agent：按顺序完成
  1) 在 LTM 的 RAG 中按问题逐层检索（LTM 内部由 LLM 根据问题选节点）
  2) 根据检索到的 LTM 信息作答
  返回答案与问题类型（问题类型由 LLM 输出或 MK 规则兜底，供主流程选 Agent 与策略）。
"""

from typing import Any

from memory.mk_memory import infer_question_type
from utils.llm import llm_call, semantic_similarity
from utils.logger import get_logger
from utils.parse_llm_json import parse_llm_output_to_dict

logger = get_logger(__name__)


def _parse_answer_json(raw: str) -> str:
    """
    从 LLM 回复中解析出 answer 字段。
    使用 utils 的 parse_llm_output_to_dict 转成字典后 .get("answer") 提取。
    """
    obj = parse_llm_output_to_dict(raw or "")
    if obj is not None:
        return (obj.get("answer") or "").strip()
    return (raw or "").strip()


def _parse_answer_and_question_type(
    raw: str, question: str, mk: dict[str, Any] | None
) -> tuple[str, str]:
    """
    从 LLM 回复中解析出 answer 与 question_type。
    使用 parse_llm_output_to_dict 转成字典后 .get("answer")、.get("question_type") 提取；
    若 question_type 不在 MK 的 question_types 中则用 infer_question_type 兜底。
    """
    from config import DEFAULT_QUESTION_TYPE

    default_type = DEFAULT_QUESTION_TYPE if not mk else (mk.get("default_type") or DEFAULT_QUESTION_TYPE)
    if not (raw and raw.strip()):
        return "", infer_question_type(question, mk) if (question and mk) else default_type
    obj = parse_llm_output_to_dict(raw)
    if obj is None:
        return _parse_answer_json(raw), infer_question_type(question, mk) if (question and mk) else default_type
    answer = (obj.get("answer") or "").strip() or _parse_answer_json(raw)
    qt = (obj.get("question_type") or "").strip()
    types = (mk or {}).get("question_types", {})
    if qt and qt in types:
        return answer, qt
    return answer, infer_question_type(question, mk) if (question and mk) else (qt or default_type)


class StudentAgent:
    """负责：LTM RAG 检索（按问题逐层检索）→ 基于 LTM 生成初答；并评估回答的置信度与一致性。"""

    def __init__(self, rag_search):
        self.rag_search = rag_search

    def answer(self, question: str, ltm: dict[str, Any], mk: dict[str, Any]) -> tuple[str, str]:
        """
        按步骤生成初答：
        1. 在 LTM 的 RAG 中按问题逐层检索（RAG 内部由 LLM 根据问题选节点，不依赖问题类型）
        2. 根据检索到的 LTM 信息进行回答
        返回 (答案, 问题类型)；问题类型由 LLM 在作答时一并输出（须为 MK 中类型之一），供主流程选 Agent 与策略。
        """
        # Step 1: 在 LTM 的 RAG 中按问题检索（传入问题，LTM 内部 LLM 逐层选节点）
        retrieved_knowledge = self.rag_search.search(question, ltm)
        logger.info("Student RAG 检索完成，检索内容长度=%s 字", len(retrieved_knowledge or ""))
        logger.info("Student RAG 检索内容(全文): %s", retrieved_knowledge)

        # Step 2: 根据 LTM 信息进行回答（LLM 同时输出答案与问题类型）
        types = mk.get("question_types", {})
        type_descriptions = "\n".join(
            f"- {k}: {v.get('description', k)}" for k, v in types.items()
        ) if types else "- general: 通用问题"
        prompt = '''
以下是从长期记忆中检索到的相关知识：
{retrieved_knowledge}
请严格基于上述知识回答问题。若知识中未涵盖，请根据自己的理解回答，不必再参考上述知识。
只输出你的答案正文，不要输出「根据以上知识…」等前缀或多余解释。
问题：{question}
请输出的格式严格按照下面（只输出该 JSON，不要其他文字）：
{{
    "answer": "问题的答案",
    "question_type": "问题类型"
}}
其中 question_type 必须从下面选一个最贴切的：
{type_descriptions}

'''.format(retrieved_knowledge=retrieved_knowledge, question=question, type_descriptions=type_descriptions)
        raw = self.generate_response(prompt)
        answer, question_type = _parse_answer_and_question_type(raw, question, mk)
        logger.info("Student 初答完成 question_type=%s 答案长度=%s", question_type, len(answer or ""))
        return answer, question_type

    def generate_response(self, text: str) -> str:
        """调用大模型生成回复。"""
        return llm_call(text)

    def revise_answer(self, question: str, current_answer: str, organized_feedback: str) -> str:
        """
        根据 Insight 整理后的反馈意见，结合当前答案再次回答，生成改进后的答案。
        供反思循环中使用（不参考 LTM，仅基于当前答案与多 Agent 反馈）。
        """
        prompt = '''
You are a high-performance, strict assistant whose job is to revise or optimize an existing answer given user feedback. Follow these rules exactly.

INPUT VARIABLES (do not change these names, counts, or braces):

* {question}
* {current_answer}
* {organized_feedback}

HARD CONSTRAINTS (mandatory):

1. Preserve the three variables exactly as named above. You must not add, remove, rename, or reorder any of them. Treat them as immutable input placeholders.
2. Default output language: **English**. Only use a different language if the {question} explicitly contains a directive in natural language of the form `Output language: <language>` (for example, `Output language: Chinese`). If that directive appears, obey it; otherwise produce English output.
3. Output **only** raw JSON text and nothing else. The output must be a single JSON object with exactly this shape (no extra keys, no surrounding text, no code fences, no Markdown):

{{
"answer": "<full rewritten/optimized answer text>"
}}

4. Do not include line breaks, commentary, diagnostics, or metadata outside the JSON object. The value of `"answer"` may contain multiple paragraphs and examples, but the overall output must still be a single valid JSON object.
5. If you cannot comply with any rule, still return the single JSON object; set `"answer"` to a short explanatory message describing why you cannot comply (keep message concise, factual, and in the chosen language).

BEHAVIORAL RULES (what to do):

1. Primary goal: **Rewrite or optimize** `{current_answer}` **in-place** so it (a) preserves the original meaning, (b) reads more rigorously, (c) is logically clearer, and (d) adds short, concrete examples or clarifications **only when** they materially improve comprehension.
2. Treat `{organized_feedback}` as advisory — integrate applicable points into the revision. However, you are allowed to retain your own expert judgment (i.e., you may **not** be forced to follow feedback verbatim). If you diverge from any item in `{organized_feedback}`, explicitly incorporate a brief parenthetical note into the rewritten answer saying you chose a different approach and why.
3. Do not introduce new top-level sections or modify the three input variable names. You may reorder sentences inside the answer, add short examples, add numbered steps, and improve precision of terms, but you must not append new metadata blocks or change the template.
4. Keep style: professional, precise, and direct. Prefer numbered lists for steps, short definitions for technical terms at first use, and concise examples illustrating edge cases or typical inputs/outputs where helpful.
5. Length target: produce an answer that is as short as possible while fully addressing the feedback — prefer clarity over verbosity. If a longer explanation is necessary, provide a short summary first (1–2 sentences), then the expanded content.

TECHNICAL RULES FOR JSON VALIDITY:

1. The JSON must be valid UTF-8. Escape only characters required by JSON rules.
2. Do not output trailing commas. Use double quotes for strings.
3. If the rewritten answer contains quotation marks or line breaks, ensure they are properly encoded inside the JSON string.

EXAMPLES OF ACCEPTABLE `"answer"` CONTENT (for your internal guidance only — do not output these examples):

* A concise rewritten paragraph with 1–2 short bullet examples embedded.
* A short numbered procedure (1., 2., 3.) that resolves ambiguity in the original answer.
* If diverging from `{organized_feedback}`, append a parenthetical: `(Note: I retained X because...)`.

ERROR/REFUSAL HANDLING:

* If the inputs are malformed (e.g., missing one of the three variables), return:
  {{
  "answer": "Cannot comply: input missing required variable(s): list them."
  }}
* If a security or policy constraint prevents fulfilling the request, return a compliant JSON with a concise reason.

FINAL ACTION (what you must output now):

* Produce the rewritten/optimized answer **only** in the required JSON format above.
* Do not include any additional commentary, analysis, or metadata outside the JSON.

'''.format(question=question, current_answer=current_answer, organized_feedback=organized_feedback)
        logger.info("Student revise_answer 输入指导长度=%s 字", len(organized_feedback or ""))
        raw = self.generate_response(prompt)
        obj = parse_llm_output_to_dict(raw)
        if obj is not None and obj.get("answer") is not None:
            out = (obj.get("answer") or "").strip()
            logger.info("Student revise_answer 输出长度=%s (全文): %s", len(out), out)
            return out
        return _parse_answer_json(raw) if raw else ""

    def evaluate_answer(
        self,
        answer: str,
        ltm: dict[str, Any],
        *,
        initial_answer: str | None = None,
        previous_answer: str | None = None,
    ) -> tuple[float, float]:
        """
        评估回答的置信度与一致性。
        - 首次循环（未传 initial_answer / previous_answer）：仅参考 LTM 计算一致性，用于选 Agent。
        - 后续循环：不参考 LTM，与「首答」和「上一轮回答」做一致性；一致性高表示改动小。
        :return: (confidence, consistency_score)
        """
        if initial_answer is None and previous_answer is None:
            # 首次循环：只参考 LTM，不做“与前面回答”的一致性判断
            consistency_score = self._consistency_with_ltm(answer, ltm)
        else:
            # 后续循环：与前面回答做一致性（与上一轮回答的相似度，高则表示改动小）
            ref = previous_answer if previous_answer else initial_answer
            consistency_score = semantic_similarity(answer, ref) if ref else 0.5
        confidence = 0.5 + 0.5 * consistency_score
        return confidence, consistency_score

    def _consistency_with_ltm(self, answer: str, ltm: dict[str, Any]) -> float:
        """计算回答与 LTM 知识的一致性（语义相似度），仅用于首次循环。"""
        def extract_all_content(tree: dict[str, Any]) -> str:
            contents = []
            if isinstance(tree, dict):
                content = tree.get("content", "").strip()
                if content:
                    contents.append(content)
                children = tree.get("children", {})
                if isinstance(children, dict):
                    for child in children.values():
                        contents.append(extract_all_content(child))
            return " ".join(contents)

        tree = ltm.get("tree", {})
        if not tree:
            return 0.5
        all_knowledge = extract_all_content(tree)
        if not all_knowledge.strip():
            return 0.5
        return semantic_similarity(answer, all_knowledge)

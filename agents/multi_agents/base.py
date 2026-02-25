# -*- coding: utf-8 -*-
"""多角色 Agent 基类与公共解析逻辑。"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any

# 各 Agent review 输出统一使用此 JSON 结构，便于后续解析与处理（仅 comment，不要求 score）
AGENT_OUTPUT_JSON_SCHEMA = """
请务必只输出一个合法的 JSON 对象，不要输出任何其他文字、解释或 Markdown 标记。
JSON 格式固定为（仅此一个字段）：
{{
    "comment": "简短评论内容，指出问题与改进建议（1-3 句）"
}}
不要输出 ```json 等代码块标记，只输出纯 JSON。
"""


def parse_agent_output(raw: str, default_score: float) -> tuple[str, float]:
    """
    从 LLM 回复中解析出 comment 与 score。
    优先尝试 JSON 解析；若失败则退回原有「最后一行分数」的文本解析。
    """
    if not (raw and raw.strip()):
        return "", default_score
    raw = raw.strip()
    # 去掉 markdown 代码块包裹
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        comment = (obj.get("comment") or "").strip()
        return comment or "", default_score
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # 尝试从文本中提取第一个完整 JSON 对象
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw[start : i + 1])
                        comment = (obj.get("comment") or "").strip()
                        return comment or "", default_score
                    except (json.JSONDecodeError, TypeError, ValueError):
                        break
    return _parse_comment_score_fallback(raw, default_score)


def _parse_comment_score_fallback(comment: str, default_score: float) -> tuple[str, float]:
    """兜底：从纯文本中按「最后一行分数」解析。"""
    score = default_score
    if not comment:
        return "", default_score
    lines = comment.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line and line[-1].isdigit():
            try:
                raw = "".join(c for c in line if c.isdigit() or c == ".")[:4]
                if raw:
                    score = max(0.0, min(1.0, float(raw)))
                    comment = "\n".join(lines[:-1]).strip() if len(lines) > 1 else comment
            except ValueError:
                pass
            break
    return comment.strip(), score


def parse_comment_score(comment: str, default_score: float) -> tuple[str, float]:
    """
    从 LLM 回复中解析出评论正文与 0-1 分数（兼容旧逻辑，内部调用 parse_agent_output）。
    """
    return parse_agent_output(comment, default_score)


class BaseAgent(ABC):
    """多角色 Agent 基类，统一接口 review(answer, question)。"""

    @abstractmethod
    def review(self, answer: str, question: str = "") -> dict[str, Any]:
        """对回答进行审视，返回评论与分数。"""
        ...

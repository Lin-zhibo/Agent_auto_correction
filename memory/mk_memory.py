# -*- coding: utf-8 -*-
"""MK Memory：按问题类型存储策略、Agent 优先级、阈值等，支持从 LTM 更新。"""

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from config import DEFAULT_QUESTION_TYPE, MK_MEMORY_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


def load_mk(path: Path | None = None) -> dict[str, Any]:
    """从 JSON 加载 MK 数据。"""
    p = path or MK_MEMORY_PATH
    if not p.exists():
        return _default_mk()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "question_types" not in data:
        return _default_mk()
    return data


def save_mk(mk: dict[str, Any], path: Path | None = None) -> None:
    """将 MK 保存到 JSON。"""
    p = path or MK_MEMORY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(mk, f, ensure_ascii=False, indent=2)


def _default_mk() -> dict[str, Any]:
    """无文件时的默认 MK 结构（单类型 general）。"""
    return {
        "question_types": {
            "general": {
                "description": "通用问题",
                "strategy": {
                    "similarity_threshold": 0.9,
                    "improvement_min": 0.05,
                    "max_loops": 3,
                },
                "agent_priorities": {
                    "questioner": 0.9,
                    "logic_analyzer": 0.8,
                    "authority_checker": 0.7,
                    "explainer": 0.6,
                },
                "thresholds": {
                    "confidence_low": 0.6,
                    "consistency_low": 0.7,
                    "confidence_high": 0.8,
                    "consistency_high": 0.8,
                },
                "agent_effectiveness": {
                    "questioner": 0.87,
                    "logic_analyzer": 0.78,
                    "authority_checker": 0.82,
                    "explainer": 0.85,
                },
            },
        },
        "default_type": "general",
        "last_update": "",
    }


def infer_question_type(question: str, mk: dict[str, Any] | None = None) -> str:
    """
    根据问题文本推断问题类型（与 LTM 顶层分类一致），用于选择 MK 中对应的策略。
    规则：与 LTM 顶层一致，只到「问题的上一层」——数学类、哲学类、技术类；否则 default_type。
    """
    q = question.strip()
    types = (mk or {}).get("question_types", {})
    default = (mk or {}).get("default_type", DEFAULT_QUESTION_TYPE)
    # 与 LTM 顶层分类对齐：按关键词映射到 数学类、哲学类、技术类
    if "数学" in q or "计算" in q or "逻辑" in q or "加法" in q or "乘法" in q or "命题" in q:
        return "数学类" if "数学类" in types else default
    if "哲学" in q or "社会" in q or "结构" in q:
        return "哲学类" if "哲学类" in types else default
    if "技术" in q or "机器" in q or "学习" in q or "深度" in q or "神经网络" in q or "自监督" in q:
        return "技术类" if "技术类" in types else default
    return default


def get_config_for_question_type(
    mk: dict[str, Any],
    question_type: str,
) -> dict[str, Any]:
    """
    获取指定问题类型对应的完整配置（策略、优先级、阈值）。
    若该类型不存在，则使用 default_type 对应配置。
    """
    types = mk.get("question_types", {})
    default_type = mk.get("default_type", DEFAULT_QUESTION_TYPE)
    if question_type not in types:
        question_type = default_type
    if question_type not in types:
        return _default_mk()["question_types"]["general"]
    t = types[question_type]
    return {
        "strategy": t.get("strategy", {}),
        "agent_priorities": t.get("agent_priorities", {}),
        "thresholds": t.get("thresholds", {}),
        "agent_effectiveness": t.get("agent_effectiveness", {}),
    }


def update_mk_from_ltm(mk: dict[str, Any], ltm: dict[str, Any]) -> None:
    """
    根据 LTM 树形结构中每个节点的 question_type 与使用情况，更新 MK。
    当前实现：按 question_type 统计节点数量，并确保 MK 中存在的类型都有对应配置；
    若 LTM 中出现 MK 中不存在的 question_type，则在 MK 中新增该类型（复制 default 配置）。
    """
    tree = ltm.get("tree", {})
    type_counts: dict[str, int] = {}
    
    def traverse_tree(node: dict[str, Any]) -> None:
        """递归遍历树形结构，统计每个 question_type 的数量。"""
        qt = node.get("question_type")
        if qt:
            type_counts[qt] = type_counts.get(qt, 0) + 1
        children = node.get("children", {})
        if isinstance(children, dict):
            for child in children.values():
                traverse_tree(child)
    
    for root_node in tree.values():
        traverse_tree(root_node)
    
    # 若树为空或没有 question_type，使用默认类型
    if not type_counts:
        default_type = mk.get("default_type", DEFAULT_QUESTION_TYPE)
        type_counts[default_type] = 0

    types = mk.setdefault("question_types", {})
    default_type = mk.get("default_type", DEFAULT_QUESTION_TYPE)
    default_config = types.get(default_type) or _default_mk()["question_types"]["general"]

    for qt, count in type_counts.items():
        if qt and qt not in types:
            types[qt] = copy.deepcopy(default_config)
            types[qt]["description"] = f"由 LTM 自动衍生（来自 {count} 条记录）"

    mk["last_update"] = datetime.now().strftime("%Y-%m-%d")
    mk["type_counts"] = type_counts


def evolve_mk_from_random_agent(
    mk: dict[str, Any],
    question_type: str,
    random_agent_name: str,
    *,
    effectiveness_delta: float = 0.02,
    priority_delta: float = 0.02,
) -> None:
    """
    当随机加入的 Agent 参与后结果较好时，用于 MK 进化：提升该 Agent 在该问题类型下的
    优先级与有效性，便于后续更常被选中。
    """
    types = mk.get("question_types", {})
    if question_type not in types:
        return
    t = types[question_type]
    eff = t.get("agent_effectiveness", {})
    prio = t.get("agent_priorities", {})
    if isinstance(eff, dict):
        old = eff.get(random_agent_name, 0.5)
        eff[random_agent_name] = min(1.0, old + effectiveness_delta)
    if isinstance(prio, dict):
        old = prio.get(random_agent_name, 0.5)
        prio[random_agent_name] = min(1.0, old + priority_delta)
    mk["last_update"] = datetime.now().strftime("%Y-%m-%d")
    logger.info("MK evolve_mk_from_random_agent question_type=%s random_agent=%s", question_type, random_agent_name)


def select_better_agents_from_wm(
    wm: dict[str, Any],
    final_answer: str,
    *,
    max_agents: int = 3,
) -> list[str]:
    """
    根据正确答案与 WM 中各 Agent 的输出进行对比，选出对优化贡献更大的 Agent。
    返回 Agent 名称列表，用于更新 MK。
    """
    history = wm.get("agent_feedback_history", [])
    if not history:
        return []
    from collections import defaultdict
    agent_comments: dict[str, list[str]] = defaultdict(list)
    for round_feedbacks in history:
        for f in round_feedbacks:
            name = f.get("agent", "")
            comment = (f.get("comment") or "").strip()
            if name and comment:
                agent_comments[name].append(comment)
    if not agent_comments:
        return []
    from utils.llm import llm_call, parse_json_from_llm
    lines = []
    for agent, comments in agent_comments.items():
        text = "；".join(comments[:3]) if len(comments) > 3 else "；".join(comments)
        text = text[:300] + "..." if len(text) > 300 else text
        lines.append(f"- {agent}: {text}")
    prompt = '''最终优化后的答案：
{final_answer}

各 Agent 在本轮中的反馈摘要：
{agent_summary}

请从上述 Agent 中选出对得到该最终答案贡献最大的 {max_agents} 个。
请输出的格式严格按照下面（只输出该 JSON，不要其他文字）：
{{
    "agents": ["questioner", "logic_analyzer", "explainer"]
}}
agents 为 Agent 名称列表，仅包含上述摘要中出现的 Agent。
'''.format(final_answer=final_answer[:800], agent_summary="\n".join(lines), max_agents=max_agents)
    try:
        raw = llm_call(prompt)
        obj = parse_json_from_llm(raw)
        if obj is not None:
            agents = obj.get("agents")
            if isinstance(agents, list):
                chosen = [str(a).strip().lower() for a in agents if a]
            elif isinstance(agents, str):
                chosen = [s.strip().lower() for s in agents.replace("，", ",").split(",") if s.strip()]
            else:
                chosen = []
            chosen = [a for a in chosen if a in agent_comments][:max_agents]
            logger.info("MK select_better_agents_from_wm 选出: %s", chosen)
            return chosen
        # 兜底：按原始文本解析
        raw_stripped = (raw or "").strip().lower().replace("，", ",")
        chosen = [s.strip() for s in raw_stripped.split(",") if s.strip()]
        chosen = [a for a in chosen if a in agent_comments][:max_agents]
        logger.info("MK select_better_agents_from_wm 选出(兜底): %s", chosen)
        return chosen
    except Exception:
        return []


def evolve_mk_from_better_agents(
    mk: dict[str, Any],
    question_type: str,
    better_agent_names: list[str],
    *,
    effectiveness_delta: float = 0.03,
    priority_delta: float = 0.03,
) -> None:
    """
    当一致性判断通过并输出正确答案后，根据 WM 选出的更好 Agent 更新 MK：
    提升这些 Agent 在该问题类型下的优先级与有效性。
    """
    types = mk.get("question_types", {})
    if question_type not in types:
        return
    t = types[question_type]
    eff = t.get("agent_effectiveness", {})
    prio = t.get("agent_priorities", {})
    for name in better_agent_names:
        if isinstance(eff, dict):
            old = eff.get(name, 0.5)
            eff[name] = min(1.0, old + effectiveness_delta)
        if isinstance(prio, dict):
            old = prio.get(name, 0.5)
            prio[name] = min(1.0, old + priority_delta)
    mk["last_update"] = datetime.now().strftime("%Y-%m-%d")
    logger.info("MK evolve_mk_from_better_agents question_type=%s agents=%s", question_type, better_agent_names)
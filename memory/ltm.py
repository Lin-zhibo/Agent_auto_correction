# -*- coding: utf-8 -*-
"""长期记忆 (LTM)：树形结构，支持逐层加载与更新。"""

import json
from pathlib import Path
from typing import Any

from config import LTM_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_LTM_DEPTH_EXCLUDING_ROOT = 3  # 不包含根节点，最多三层：一级分类/二级分类/条目


def _tree_structure_summary(ltm: dict[str, Any]) -> str:
    """从 LTM 树中提取一、二级结构摘要，供 LLM 选择路径。"""
    tree = ltm.get("tree", {})
    if not tree:
        return "（当前无分类）"
    lines = []
    for l1, node1 in tree.items():
        children = node1.get("children", {}) if isinstance(node1, dict) else {}
        if children:
            lines.append(f"- {l1}: " + "、".join(children.keys()))
        else:
            lines.append(f"- {l1}")
    return "\n".join(lines)


def load_ltm(path: Path | None = None) -> dict[str, Any]:
    """从 JSON 加载 LTM（树形结构）。"""
    p = path or LTM_PATH
    if not p.exists():
        return {"tree": {}}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "tree" not in data:
        return {"tree": {}}
    return data


def save_ltm(ltm: dict[str, Any], path: Path | None = None) -> None:
    """将 LTM（树形结构）保存到 JSON。"""
    p = path or LTM_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ltm, f, ensure_ascii=False, indent=2)


def infer_question_type_for_ltm(question: str) -> str:
    """
    为 LTM 条目推断问题类型（与 MK 的 infer_question_type 规则一致）。
    避免循环依赖，此处仅做简单规则；若需与 MK 一致可在外层传入 question_type。
    """
    q = question.strip()
    if "为什么" in q or "如何" in q or "怎样" in q:
        return "why_how"
    if "什么是" in q or "是什么" in q or "定义" in q or "含义" in q:
        return "what_is"
    return "general"


def _find_or_create_path(
    tree: dict[str, Any],
    path: list[str],
    create_if_not_exists: bool = True,
) -> dict[str, Any] | None:
    """
    在树中按路径查找节点（如 ["数学类", "计算类"]）。
    若 create_if_not_exists=True 且路径不存在，则创建中间节点。
    返回找到的节点（字典），若不存在且不创建则返回 None。
    """
    current = tree
    for key in path:
        if key not in current:
            if not create_if_not_exists:
                return None
            current[key] = {
                "content": "",
                "question_type": None,
                "children": {},
            }
        current = current[key].get("children", {})
        if not isinstance(current, dict):
            current = {}
    return current


def infer_topic_path_for_ltm(question: str, ltm: dict[str, Any]) -> list[str]:
    """
    根据问题与 LTM 树形结构，推断应插入的合适路径（用于找到树形结构中的位置）。
    优先使用 LLM 从现有树结构中选择或建议路径；失败时退回关键词规则。
    """
    summary = _tree_structure_summary(ltm)
    try:
        from utils.llm import llm_call, parse_json_from_llm
        prompt = '''当前长期记忆的树形结构（一级分类及其子分类）：
{summary}

问题：{question}

请根据问题内容，选择最合适的一级和二级分类路径。若现有分类都不合适，可填「未分类」或建议新的一级名（如「经济学类」）。
请输出的格式严格按照下面（只输出该 JSON，不要其他文字）：
{{
    "topic_path": "一级,二级"
}}
路径用英文逗号分隔，例如：技术类,机器学习 或 数学类,计算类。最多两级。
'''.format(summary=summary, question=question[:500])
        raw = llm_call(prompt)
        obj = parse_json_from_llm(raw)
        if obj is not None:
            path_str = (obj.get("topic_path") or obj.get("path") or "").strip().replace("，", ",")
            path = [s.strip() for s in path_str.split(",") if s.strip()]
            if path and path[0].lower() not in ("无", "无分类", "无合适"):
                return path[:3]
        # 兜底：按原始文本解析
        raw_stripped = (raw or "").strip().replace("，", ",")
        path = [s.strip() for s in raw_stripped.split(",") if s.strip()]
        if path and path[0].lower() not in ("无", "无分类", "无合适"):
            return path[:3]
    except Exception:
        pass
    # 退回关键词规则
    if "数学" in question or "计算" in question or "逻辑" in question:
        return ["数学类", "计算类"]
    if "哲学" in question or "社会" in question:
        return ["哲学类", "社会学"]
    if "机器" in question or "学习" in question or "深度" in question:
        return ["技术类", "机器学习"]
    if "经济" in question:
        return ["经济学类", "通用"]
    return ["未分类"]


def _sanitize_topic_name(name: str) -> str:
    """清理 topic 名称，避免出现换行/逗号等导致路径解析异常的字符。"""
    n = (name or "").strip()
    n = n.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    n = n.replace("，", ",")
    # 不允许逗号出现在单个节点名里（逗号用于路径分隔）
    n = n.replace(",", " ").strip()
    # 合并多空格
    while "  " in n:
        n = n.replace("  ", " ")
    return n


def _llm_choose_or_create_node(
    *,
    level: int,
    question: str,
    parent_path: list[str],
    candidates: list[str],
) -> str:
    """
    让 LLM 在当前层的候选节点里选择最合适的一个；若都不合适则创建一个新节点名。
    返回节点名（保证非空，失败时退回 '未分类'）。
    """
    try:
        from utils.llm import llm_call, parse_json_from_llm

        cand_text = "\n".join(f"- {c}" for c in candidates) if candidates else "（无候选节点）"
        parent_text = "/".join(parent_path) if parent_path else "root"
        prompt = f'''你正在为长期记忆（LTM）的分类树选择/创建第 {level} 层节点。

【父路径】
{parent_text}

【候选节点】
{cand_text}

【问题】
{question}

请在候选节点中选择最贴切的一个；若都不合适，请创建一个新的节点名（中文，尽量短，2-8 字）。
请输出严格符合下面结构的 JSON，仅输出该 JSON，无其他文字：
{{
  "action": "choose_or_create",
  "name": "节点名"
}}
其中 name 必须是一个节点名（不要包含逗号或换行）。
'''
        raw = llm_call(prompt)
        obj = parse_json_from_llm(raw)
        if obj is not None:
            name = _sanitize_topic_name(str(obj.get("name") or ""))
            if name:
                return name
    except Exception:
        pass
    # 兜底
    return "未分类"


def _llm_generate_entry_title(
    *,
    question: str,
    parent_path: list[str],
) -> str:
    """
    生成第 3 层条目标题（用于存储具体问答）。
    返回标题（非空，失败时用问题前若干字兜底）。
    """
    try:
        from utils.llm import llm_call, parse_json_from_llm

        parent_text = "/".join(parent_path) if parent_path else "root"
        prompt = f'''你正在为长期记忆（LTM）生成第 3 层“条目标题”（用于存放具体问答内容）。

【父路径】
{parent_text}

【问题】
{question}

请给出一个简短、可读、能概括问题主题的标题（中文优先，4-12 字；不要包含逗号或换行）。
请输出严格符合下面结构的 JSON，仅输出该 JSON，无其他文字：
{{
  "entry_title": "标题"
}}
'''
        raw = llm_call(prompt)
        obj = parse_json_from_llm(raw)
        if obj is not None:
            title = _sanitize_topic_name(str(obj.get("entry_title") or ""))
            if title:
                return title
    except Exception:
        pass
    # 兜底：取问题前 12 个字符
    q = _sanitize_topic_name(question)
    return q[:12] if q else "条目"


def infer_topic_path_layered_3level(question: str, ltm: dict[str, Any]) -> list[str]:
    """
    结合 LLM 做逐层检索/创建，得到固定三层（不含根）路径：
    [一级分类, 二级分类, 条目标题]。

    逐层策略：
    - 第 1 层：从现有一级分类中选择，否则创建一级分类；
    - 第 2 层：在已选一级下从现有二级中选择，否则创建二级；
    - 第 3 层：生成条目标题，用于存放具体问答内容。
    """
    tree = ltm.get("tree", {}) if isinstance(ltm, dict) else {}
    l1_candidates = list(tree.keys()) if isinstance(tree, dict) else []
    l1 = _llm_choose_or_create_node(level=1, question=question, parent_path=[], candidates=l1_candidates)
    if not l1:
        l1 = "未分类"

    # 确保一级节点存在
    if "tree" not in ltm or not isinstance(ltm.get("tree"), dict):
        ltm["tree"] = {}
    if l1 not in ltm["tree"]:
        ltm["tree"][l1] = {"content": "", "question_type": None, "children": {}}

    l1_node = ltm["tree"][l1]
    children1 = l1_node.get("children", {}) if isinstance(l1_node, dict) else {}
    l2_candidates = list(children1.keys()) if isinstance(children1, dict) else []
    l2 = _llm_choose_or_create_node(level=2, question=question, parent_path=[l1], candidates=l2_candidates)
    if not l2:
        l2 = "通用"

    # 确保二级节点存在
    if not isinstance(l1_node.get("children", {}), dict):
        l1_node["children"] = {}
    if l2 not in l1_node["children"]:
        l1_node["children"][l2] = {"content": "", "question_type": None, "children": {}}

    l3 = _llm_generate_entry_title(question=question, parent_path=[l1, l2])
    if not l3:
        l3 = "条目"

    return [l1, l2, l3]


def update_ltm(
    ltm: dict[str, Any],
    question: str,
    answer: str,
    question_type: str | None = None,
    topic_path: list[str] | None = None,
    max_depth: int = 5,
) -> str:
    """
    根据本轮问答更新 LTM（树形结构）。
    
    :param ltm: LTM 数据（包含 "tree" 键）
    :param question: 问题文本
    :param answer: 答案文本
    :param question_type: 问题类型（若未传入则推断）
    :param topic_path: 话题路径，如 ["数学类", "计算类"]，表示要添加到该路径下。
                      若为 None，则调用 infer_topic_path_for_ltm 推断合适树形位置。
    :param max_depth: 树的最大深度限制
    :return: 本条目使用的问题类型
    """
    tree = ltm.setdefault("tree", {})
    qt = question_type or infer_question_type_for_ltm(question)
    
    if topic_path is None:
        # 固定三层：一级/二级/条目。逐层选择/创建。
        topic_path = infer_topic_path_layered_3level(question, ltm)
        logger.info("LTM 逐层推断路径(3层): %s", topic_path)
    else:
        # 外部传入路径时也强制裁剪到三层（不含根）
        topic_path = topic_path[:MAX_LTM_DEPTH_EXCLUDING_ROOT]

    # 限制深度（兼容旧参数，但实际固定为三层）
    if len(topic_path) > min(max_depth, MAX_LTM_DEPTH_EXCLUDING_ROOT):
        topic_path = topic_path[: min(max_depth, MAX_LTM_DEPTH_EXCLUDING_ROOT)]
    
    # 找到或创建目标路径的父节点
    parent = tree
    for i, key in enumerate(topic_path[:-1]):
        if key not in parent:
            parent[key] = {
                "content": "",
                "question_type": None,
                "children": {},
            }
        parent = parent[key].get("children", {})
        if not isinstance(parent, dict):
            parent = {}
    
    # 创建或更新叶子节点
    leaf_key = topic_path[-1]
    # 若标题冲突，避免覆盖已有条目：自动追加后缀
    if leaf_key in parent:
        i = 2
        new_key = f"{leaf_key}_{i}"
        while new_key in parent and i < 50:
            i += 1
            new_key = f"{leaf_key}_{i}"
        if new_key not in parent:
            logger.info("LTM 条目标题冲突，改用新标题: %s -> %s", leaf_key, new_key)
            leaf_key = new_key
            topic_path = topic_path[:-1] + [leaf_key]

    if leaf_key not in parent:
        parent[leaf_key] = {
            "content": f"问：{question}\n答：{answer}",
            "question_type": qt,
            "children": {},
        }
        logger.info("LTM 新增节点 路径: %s", topic_path)
    else:
        # 若节点已存在，更新内容（追加或替换）
        parent[leaf_key]["content"] = f"问：{question}\n答：{answer}"
        parent[leaf_key]["question_type"] = qt
        logger.info("LTM 更新节点 路径: %s", topic_path)

    # 同步更新知识库 ltmk.json（供 Chroma RAG 使用）
    _sync_to_ltmk(question, answer)

    return qt


def _sync_to_ltmk(question: str, answer: str) -> None:
    """将新问答同步到 ltmk.json 知识库（供 Chroma RAG 使用）。"""
    from config import LTMK_PATH
    
    content = f"Question: {question}\nAnswer: {answer}"
    
    # 加载现有知识库
    knowledge = []
    if LTMK_PATH.exists():
        try:
            with open(LTMK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            knowledge = data.get("knowledge", [])
        except Exception as e:
            logger.warning("加载 ltmk.json 失败: %s", e)
    
    # 避免重复添加
    if content not in knowledge:
        knowledge.append(content)
        LTMK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LTMK_PATH, "w", encoding="utf-8") as f:
            json.dump({"knowledge": knowledge}, f, ensure_ascii=False, indent=2)
        logger.info("同步新知识到 ltmk.json，当前共 %d 条", len(knowledge))


def get_all_concepts(ltm: dict[str, Any]) -> list[dict[str, Any]]:
    """
    从树形 LTM 中提取所有叶子节点（有实际内容的条目）为扁平列表。
    用于兼容性：某些地方可能需要扁平列表格式。
    """
    concepts = []
    
    def traverse(node: dict[str, Any], path: list[str] = []) -> None:
        content = node.get("content", "").strip()
        if content and node.get("question_type"):
            concepts.append({
                "id": "-".join(path) if path else "root",
                "topic": "/".join(path) if path else "root",
                "content": content,
                "question_type": node.get("question_type"),
            })
        children = node.get("children", {})
        for key, child in children.items():
            traverse(child, path + [key])
    
    tree = ltm.get("tree", {})
    for key, node in tree.items():
        traverse(node, [key])
    
    return concepts

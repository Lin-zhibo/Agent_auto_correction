# -*- coding: utf-8 -*-
"""固定 16 类长期记忆（LTM）与检索工具。"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from config import LTM_EMBEDDINGS_PATH, LTM_PATH
from utils.llm import get_embedding, llm_call, parse_json_from_llm
from utils.logger import get_logger

logger = get_logger(__name__)
_LTM_EMBEDDINGS_CACHE_KEY: tuple[str, int, int] | None = None
_LTM_EMBEDDINGS_CACHE_VALUE: dict[str, Any] | None = None

QUESTION_TYPES_16 = [
    "TEXT_WRITING",
    "SUMMARIZATION",
    "CODE_DEVELOPMENT",
    "KNOWLEDGE_QA",
    "EDUCATIONAL_TUTORING",
    "TRANSLATION_LOCALIZATION",
    "CREATIVE_IDEATION",
    "DATA_PROCESSING",
    "ROLE_PLAYING",
    "CAREER_BUSINESS",
    "LIFE_EMOTIONAL",
    "MARKETING_COPYWRITING",
    "LOGICAL_REASONING",
    "MATH_COMPUTATION",
    "MULTIMODAL",
    "OTHER_GENERAL_Q",
]

def empty_ltm() -> dict[str, Any]:
    """返回固定 16 类的空知识库结构。"""
    return {
        "version": "2.0",
        "categories": {k: [] for k in QUESTION_TYPES_16},
    }


def _normalize_ltm(data: dict[str, Any]) -> dict[str, Any]:
    out = empty_ltm()
    categories = data.get("categories", {})
    if not isinstance(categories, dict):
        return out
    for category in QUESTION_TYPES_16:
        rows = categories.get(category, [])
        if not isinstance(rows, list):
            continue
        valid_rows: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if q and a:
                valid_rows.append({"question": q, "answer": a})
        out["categories"][category] = valid_rows
    return out


def load_ltm(path: Path | None = None) -> dict[str, Any]:
    """加载固定结构 LTM，不合法则回退为空库。"""
    p = path or LTM_PATH
    if not p.exists():
        return empty_ltm()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return empty_ltm()
    return _normalize_ltm(data)


def save_ltm(ltm: dict[str, Any], path: Path | None = None) -> None:
    """保存固定结构 LTM。"""
    p = path or LTM_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_normalize_ltm(ltm), f, ensure_ascii=False, indent=2)


def empty_ltm_embeddings() -> dict[str, Any]:
    """返回空的 LTM embedding 索引结构。"""
    return {
        "version": "1.0",
        "source_ltm": str(LTM_PATH),
        "entry_count": 0,
        "entries": [],
    }


def _make_embeddings_cache_key(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path)
    return (resolved, stat.st_mtime_ns, stat.st_size)


def _safe_resolve_str(path_str: str) -> str:
    try:
        return str(Path(path_str).resolve())
    except OSError:
        return str(path_str)


def _normalize_question_type_label(label: str) -> str:
    raw = (label or "").strip().upper()
    if not raw:
        return ""
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in QUESTION_TYPES_16:
        return normalized
    aliases = {
        "KNOWLEDGE_QA_EXPERT": "KNOWLEDGE_QA",
        "MATH": "MATH_COMPUTATION",
        "CODING": "CODE_DEVELOPMENT",
        "GENERAL": "OTHER_GENERAL_Q",
    }
    return aliases.get(normalized, "")


def _extract_question_type_candidates_from_obj(obj: dict[str, Any] | None) -> list[str]:
    if not isinstance(obj, dict):
        return []

    candidates: list[str] = []

    def _append_candidate(val: Any) -> None:
        if val is None:
            return
        if isinstance(val, list):
            for item in val:
                _append_candidate(item)
            return
        qt = _normalize_question_type_label(str(val))
        if qt and qt not in candidates:
            candidates.append(qt)

    for key in (
        "primary_category",
        "primary_question_type",
        "question_type",
        "type",
        "label",
        "category",
        "top_categories",
        "categories",
        "candidate_categories",
        "candidates",
    ):
        _append_candidate(obj.get(key))
    return candidates


def _extract_question_type_from_obj(obj: dict[str, Any] | None) -> str:
    candidates = _extract_question_type_candidates_from_obj(obj)
    return candidates[0] if candidates else ""


def _build_question_type_prompt(question: str, top_n: int = 2) -> str:
    category_lines = "\n".join(f"- {name}" for name in QUESTION_TYPES_16)
    return f"""
你是一个问题分类器，需要将问题严格划分到下列 16 个类别中最可能的前 {max(1, top_n)} 个。

候选类别：
{category_lines}

要求：
1. 按置信度从高到低返回最多 {max(1, top_n)} 个类别。
2. 返回 JSON，且只能返回 JSON。
3. JSON 格式为：{{"primary_category": "类别名", "categories": ["类别1", "类别2"], "reason": "一句简短原因"}}
4. `categories` 中必须包含 `primary_category`，且所有类别都必须来自候选列表。

待分类问题：
{question}
""".strip()


def _classify_question_types_via_llm(question: str, top_n: int = 2) -> tuple[list[str], str]:
    prompt = _build_question_type_prompt(question, top_n=top_n)
    system_prompt = "你是一个严格的问题分类器，只返回合法 JSON。"
    raw = llm_call(prompt, system_prompt=system_prompt)
    candidates = _extract_question_type_candidates_from_obj(parse_json_from_llm(raw))
    return candidates[: max(1, top_n)], raw


def infer_question_types_for_ltm(question: str, top_n: int = 2) -> list[str]:
    """仅从固定 16 类中判定问题类型，返回按置信度排序的候选类别。"""
    q = (question or "").strip()
    if not q:
        logger.info("[TRACE] classify source=fallback_empty question_types=%s", ["OTHER_GENERAL_Q"])
        return ["OTHER_GENERAL_Q"]

    try:
        candidates, raw = _classify_question_types_via_llm(q, top_n=top_n)
        if candidates:
            logger.info("[TRACE] classify source=llm question_types=%s", candidates)
            return candidates
        logger.warning("infer_question_type_for_ltm 分类结果解析失败，原始输出: %s", raw)
    except Exception as e:
        logger.warning("infer_question_type_for_ltm 失败，回退 OTHER_GENERAL_Q: %s", e)
    logger.info("[TRACE] classify source=fallback_invalid question_types=%s", ["OTHER_GENERAL_Q"])
    return ["OTHER_GENERAL_Q"]


def infer_question_type_for_ltm(question: str) -> str:
    """仅从固定 16 类中判定问题类型，返回最高置信类别。"""
    candidates = infer_question_types_for_ltm(question, top_n=1)
    return candidates[0] if candidates else "OTHER_GENERAL_Q"


def _normalize_text(text: str) -> str:
    return "".join((text or "").strip().lower().split())


def get_all_qa_entries(ltm: dict[str, Any]) -> list[dict[str, str]]:
    """获取全库 QA 条目，附带 category。"""
    out: list[dict[str, str]] = []
    categories = (ltm or {}).get("categories", {})
    if not isinstance(categories, dict):
        return out
    for category in QUESTION_TYPES_16:
        rows = categories.get(category, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if q and a:
                out.append({"category": category, "question": q, "answer": a})
    return out


def build_embedding_text(question: str, answer: str) -> str:
    """构造用于 QA 检索的 embedding 文本。"""
    return f"Question: {str(question or '').strip()}\nAnswer: {str(answer or '').strip()}"


def _normalize_ltm_embeddings(data: dict[str, Any]) -> dict[str, Any]:
    out = empty_ltm_embeddings()
    if not isinstance(data, dict):
        return out

    out["version"] = str(data.get("version") or "1.0")
    out["source_ltm"] = str(data.get("source_ltm") or str(LTM_PATH))
    raw_entries = data.get("entries", [])
    if not isinstance(raw_entries, list):
        return out

    valid_entries: list[dict[str, Any]] = []
    for row in raw_entries:
        if not isinstance(row, dict):
            continue
        category = str(row.get("category", "")).strip().upper()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        embedding = row.get("embedding")
        if category not in QUESTION_TYPES_16 or not question or not answer:
            continue
        if not isinstance(embedding, list) or not embedding:
            continue
        try:
            embedding_values = [float(x) for x in embedding]
        except (TypeError, ValueError):
            continue
        valid_entries.append(
            {
                "category": category,
                "question": question,
                "answer": answer,
                "embedding": embedding_values,
            }
        )

    out["entries"] = valid_entries
    out["entry_count"] = len(valid_entries)
    return out


def load_ltm_embeddings(path: Path | None = None) -> dict[str, Any]:
    """加载预计算的 LTM embedding 索引。"""
    global _LTM_EMBEDDINGS_CACHE_KEY, _LTM_EMBEDDINGS_CACHE_VALUE
    p = path or LTM_EMBEDDINGS_PATH
    if not p.exists():
        logger.warning("LTM embedding 文件不存在: %s", p)
        return empty_ltm_embeddings()
    cache_key = _make_embeddings_cache_key(p)
    if (
        cache_key is not None
        and _LTM_EMBEDDINGS_CACHE_KEY == cache_key
        and _LTM_EMBEDDINGS_CACHE_VALUE is not None
    ):
        logger.info(
            "LTM embedding 索引命中内存缓存 path=%s entry_count=%s",
            p,
            _LTM_EMBEDDINGS_CACHE_VALUE.get("entry_count", 0),
        )
        return _LTM_EMBEDDINGS_CACHE_VALUE
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    normalized = _normalize_ltm_embeddings(data)
    source_ltm = str(normalized.get("source_ltm") or "").strip()
    if source_ltm:
        resolved_source = _safe_resolve_str(source_ltm)
        resolved_current = _safe_resolve_str(str(LTM_PATH))
        if resolved_source != resolved_current:
            logger.warning(
                "LTM embedding 索引来源与当前 LTM 路径不一致 source_ltm=%s current_ltm=%s",
                source_ltm,
                LTM_PATH,
            )
    logger.info(
        "LTM embedding 索引加载完成 path=%s entry_count=%s source_ltm=%s",
        p,
        normalized.get("entry_count", 0),
        source_ltm or "N/A",
    )
    if cache_key is not None:
        _LTM_EMBEDDINGS_CACHE_KEY = cache_key
        _LTM_EMBEDDINGS_CACHE_VALUE = normalized
    return normalized


def save_ltm_embeddings(index_data: dict[str, Any], path: Path | None = None) -> None:
    """保存预计算的 LTM embedding 索引。"""
    p = path or LTM_EMBEDDINGS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_ltm_embeddings(index_data)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


def _build_single_entry_embedding(idx: int, row: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """单条 QA 的 embedding 计算，供并行调用。返回 (原始下标, 带 embedding 的条目)。"""
    text = build_embedding_text(row["question"], row["answer"])
    embedding = get_embedding(text)
    return (
        idx,
        {
            "category": row["category"],
            "question": row["question"],
            "answer": row["answer"],
            "embedding": embedding,
        },
    )


def build_ltm_embeddings(
    ltm: dict[str, Any] | None = None,
    *,
    source_path: Path | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """根据当前 LTM 构建 embedding 索引数据。max_workers>1 时并行计算。"""
    ltm_data = _normalize_ltm(ltm) if ltm is not None else load_ltm(source_path)
    entries = get_all_qa_entries(ltm_data)
    total = len(entries)
    logger.info("开始构建 LTM embedding 索引 entry_count=%s max_workers=%s", total, max_workers)

    if max_workers is not None and max_workers <= 1:
        max_workers = None

    if max_workers is None:
        built_entries = []
        for idx, row in enumerate(entries, start=1):
            _, built = _build_single_entry_embedding(0, row)
            built_entries.append(built)
            if idx == total or idx % 100 == 0:
                logger.info("LTM embedding 构建进度 %s/%s", idx, total)
    else:
        index_to_built: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_build_single_entry_embedding, idx, row): idx
                for idx, row in enumerate(entries)
            }
            done = 0
            for future in as_completed(futures):
                idx, built = future.result()
                index_to_built[idx] = built
                done += 1
                if done == total or done % 100 == 0:
                    logger.info("LTM embedding 构建进度 %s/%s", done, total)
        built_entries = [index_to_built[i] for i in range(len(entries))]

    return {
        "version": "1.0",
        "source_ltm": str(source_path or LTM_PATH),
        "entry_count": len(built_entries),
        "entries": built_entries,
    }


def exact_search_in_category(
    ltm: dict[str, Any],
    question: str,
    question_type: str,
) -> list[dict[str, str]]:
    """在指定类别做精确匹配（规范化后字符串相等）。"""
    qt = (question_type or "").strip().upper()
    if qt not in QUESTION_TYPES_16:
        return []
    target = _normalize_text(question)
    if not target:
        return []
    rows = ((ltm or {}).get("categories", {}) or {}).get(qt, [])
    if not isinstance(rows, list):
        return []
    hits: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if q and a and _normalize_text(q) == target:
            hits.append({"category": qt, "question": q, "answer": a})
    return hits


def exact_search_in_categories(
    ltm: dict[str, Any],
    question: str,
    question_types: list[str] | None,
) -> list[dict[str, str]]:
    """在多个指定类别内做精确匹配（规范化后字符串相等）。"""
    ordered_types: list[str] = []
    for item in question_types or []:
        qt = (item or "").strip().upper()
        if qt in QUESTION_TYPES_16 and qt not in ordered_types:
            ordered_types.append(qt)

    hits: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for qt in ordered_types:
        for row in exact_search_in_category(ltm, question, qt):
            key = (row["category"], row["question"], row["answer"])
            if key in seen:
                continue
            seen.add(key)
            hits.append(row)
    return hits


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vector_search_qa(
    ltm: dict[str, Any],
    question: str,
    *,
    top_k: int = 5,
    question_type: str | None = None,
) -> list[dict[str, str]]:
    """在全库或指定类别做向量检索。"""
    rows = vector_search_qa_scored(
        ltm,
        question,
        top_k=top_k,
        question_type=question_type,
        question_types=None,
    )
    return [
        {
            "category": str(row.get("category", "")),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
        }
        for row in rows
    ]


def vector_search_qa_scored(
    ltm: dict[str, Any],
    question: str,
    *,
    top_k: int = 5,
    question_type: str | None = None,
    question_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """优先基于预计算 embedding 索引做向量检索，并返回相似度分数。"""
    q = (question or "").strip()
    if not q:
        return []

    scoped_types: list[str] = []
    if question_types:
        for item in question_types:
            qt = (item or "").strip().upper()
            if qt in QUESTION_TYPES_16 and qt not in scoped_types:
                scoped_types.append(qt)
    elif question_type:
        qt = question_type.strip().upper()
        if qt in QUESTION_TYPES_16:
            scoped_types = [qt]
        else:
            scoped_types = []
    index_data = load_ltm_embeddings()
    indexed_rows = index_data.get("entries", [])
    if not isinstance(indexed_rows, list):
        indexed_rows = []
    if scoped_types:
        indexed_rows = [
            row for row in indexed_rows
            if isinstance(row, dict) and row.get("category") in scoped_types
        ]
    elif question_types is not None or question_type:
        indexed_rows = []

    q_emb = get_embedding(q)
    scored: list[tuple[float, dict[str, Any]]] = []

    if indexed_rows:
        logger.info(
            "向量检索使用预计算 embedding 索引 scoped_types=%s candidates=%s",
            scoped_types,
            len(indexed_rows),
        )
        for row in indexed_rows:
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue
            scored.append((_cosine(q_emb, embedding), row))
    else:
        all_rows = get_all_qa_entries(ltm)
        if scoped_types:
            all_rows = [x for x in all_rows if x["category"] in scoped_types]
        elif question_types is not None or question_type:
            all_rows = []
        if not all_rows:
            logger.warning("向量检索无可用候选：embedding 索引为空且 LTM 范围内无条目 scoped_types=%s", scoped_types)
            return []
        logger.warning(
            "向量检索回退到在线 embedding 计算 scoped_types=%s candidates=%s",
            scoped_types,
            len(all_rows),
        )
        for row in all_rows:
            text = build_embedding_text(row["question"], row["answer"])
            emb = get_embedding(text)
            scored.append((_cosine(q_emb, emb), row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_rows = [
        {
            "category": str(row["category"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "similarity": score,
        }
        for score, row in scored[: max(1, top_k)]
    ]
    logger.info(
        "向量检索 top_hits=%s",
        [
            {
                "category": row["category"],
                "similarity": round(float(row["similarity"]), 4),
                "question_preview": row["question"].replace("\n", " ")[:100],
            }
            for row in top_rows[: min(3, len(top_rows))]
        ],
    )
    return top_rows


def format_qa_context(rows: list[dict[str, str]]) -> str:
    """将检索条目格式化为 LLM 上下文。"""
    if not rows:
        return "（知识库暂无匹配）"
    chunks = []
    for row in rows:
        chunks.append(
            f"[{row['category']}]\nQuestion: {row['question']}\nAnswer: {row['answer']}"
        )
    return "\n\n".join(chunks)

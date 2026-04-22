# -*- coding: utf-8 -*-
"""OpenAI API 调用与语义相似度计算。"""

import threading
from typing import Any

import requests
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_API_EMBEDDING_KEY, 
    OPENAI_BASE_EMBEDDING_URL,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from utils.parse_llm_json import parse_llm_output_to_dict

# 单次 run 内累计的 token 用量，由 reset_token_usage / get_token_usage 配合 main 使用
_token_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
# 单次会话内 embedding 缓存，避免重复文本重复请求 API
_embedding_cache: dict[str, list[float]] = {}
_embedding_cache_lock = threading.Lock()


def reset_token_usage() -> None:
    """重置 token 累计值，在每次 run_system 前调用。"""
    _token_usage["prompt_tokens"] = 0
    _token_usage["completion_tokens"] = 0
    _token_usage["total_tokens"] = 0


def get_token_usage() -> dict[str, int]:
    """返回当前累计的 token 用量（复制），供保存结果使用。"""
    return dict(_token_usage)


def reset_embedding_cache() -> None:
    """重置 embedding 缓存，在每次会话开始前调用。"""
    _embedding_cache.clear()


def _accumulate_usage(usage: Any) -> None:
    """将单次 API 返回的 usage 累加到 _token_usage。"""
    if not usage:
        return
    _token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
    _token_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    _token_usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0


def _get_client() -> OpenAI:
    """获取 OpenAI 客户端。"""
    if not OPENAI_API_KEY:
        raise ValueError(
            "请设置 OPENAI_API_KEY：在 config.py 中配置或设置环境变量 OPENAI_API_KEY"
        )
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def _get_embedding_client() -> OpenAI:
    """获取 OpenAI 客户端。"""
    if not OPENAI_API_EMBEDDING_KEY:
        raise ValueError(
            "请设置 OPENAI_API_KEY：在 config.py 中配置或设置环境变量 OPENAI_API_KEY"
        )
    return OpenAI(api_key=OPENAI_API_EMBEDDING_KEY, base_url=OPENAI_BASE_EMBEDDING_URL)

def parse_json_from_llm(raw: str) -> dict[str, Any] | None:
    """
    从 LLM 回复中解析出 JSON 对象，供调用方用 .get() 获取字段。
    内部调用 utils.parse_llm_json.parse_llm_output_to_dict，便于其他 LLM 回答复用同一解析逻辑。
    :return: 解析得到的字典，失败返回 None
    """
    return parse_llm_output_to_dict(raw)


def llm_call(
    text: str,
    system_prompt: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> str:
    """
    调用大模型生成回复。

    根据配置中的 ``LLM_PROVIDER`` 或调用时指定的 ``provider`` 决定使用哪
    个后端。当前支持 ``openai`` 和 ``huggingface``。

    :param text: 用户/问题文本
    :param system_prompt: 可选系统提示
    :param model: 可选模型名，默认使用配置中与提供者对应的模型
    :param provider: ``openai`` 或 ``huggingface``，优先于配置
    :return: 模型回复文本
    """
    provider = provider or "openai"

    # fallback to OpenAI-compatible interface (openai or any other unknown value)
    client = _get_client()
    model = model or OPENAI_MODEL
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=OPENAI_TEMPERATURE,
    )
    _accumulate_usage(getattr(resp, "usage", None))
    return (resp.choices[0].message.content or "").strip()

def get_embedding(text: str) -> list[float]:
    """获取单段文本的 embedding（多线程安全）。"""
    key = text.strip()
    with _embedding_cache_lock:
        cached = _embedding_cache.get(key)
        if cached is not None:
            return cached[:]

    client = _get_embedding_client()
    resp = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=key,
    )
    _accumulate_usage(getattr(resp, "usage", None))
    embedding = resp.data[0].embedding

    with _embedding_cache_lock:
        if key not in _embedding_cache:
            _embedding_cache[key] = embedding
        return _embedding_cache[key][:]


def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    计算两段文本的语义相似度（余弦相似度）。
    若任一段为空，返回 0.0。
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0
    a_emb = get_embedding(text_a)
    b_emb = get_embedding(text_b)
    dot = sum(x * y for x, y in zip(a_emb, b_emb))
    norm_a = sum(x * x for x in a_emb) ** 0.5
    norm_b = sum(x * x for x in b_emb) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_improvement(prev_answer: str, new_answer: str) -> float:
    """
    计算从 prev_answer 到 new_answer 的改进程度。
    使用 1 - semantic_similarity 作为改进量：改动越大，改进分数越高（上限 1）。
    """
    if not prev_answer.strip() or not new_answer.strip():
        return 0.0
    sim = semantic_similarity(prev_answer, new_answer)
    return max(0.0, 1.0 - sim)

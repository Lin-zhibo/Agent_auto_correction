# -*- coding: utf-8 -*-
"""
基于 Judge LLM 的评估脚本：
1. 读取数据集（question + answer）
2. 调用 main.run_system 生成模型答案
3. 用 Judge LLM 判断模型答案与标准答案是否一致
4. 输出准确率、总耗时、总 token
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from main import run_system
from utils.llm import get_token_usage, parse_json_from_llm, reset_token_usage

# Judge LLM 独立配置（不依赖 config.py 全局配置）
KEY = os.getenv("JUDGE_API_KEY", "sk-XcGaOx4wJHKQ8Rq694E001E4209347E6A27174272d687bBa")
BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.gpt.ge/v1/")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5")


def _get_judge_client() -> OpenAI:
    if not KEY:
        raise ValueError("请设置 JUDGE_API_KEY（环境变量）用于 Judge LLM。")
    return OpenAI(api_key=KEY, base_url=BASE_URL)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"数据集格式错误，期望 JSON 数组：{path}")
    return data


def _judge_consistency(
    question: str,
    reference_answer: str,
    model_answer: str,
    judge_model: str,
    judge_usage: dict[str, int],
) -> bool:
    """
    由 Judge LLM 判断模型答案是否与标准答案一致（语义一致、事实不冲突）。
    """
    system_prompt = (
        "You are a strict answer consistency judge. "
        "Return JSON only."
    )
    prompt = f"""
Question:
{question}

Reference Answer:
{reference_answer}

Model Answer:
{model_answer}

Task:
Determine whether the Model Answer is consistent with the Reference Answer.
Consistent means semantically equivalent or factually compatible with no contradiction.
If Model Answer is wrong, contradictory, or misses core required facts, mark as false.

Output JSON only:
{{
  "consistent": true or false
}}
""".strip()

    client = _get_judge_client()
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    usage = getattr(resp, "usage", None)
    if usage:
        judge_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        judge_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        judge_usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
    raw = (resp.choices[0].message.content or "").strip()
    obj = parse_json_from_llm(raw)
    if isinstance(obj, dict) and isinstance(obj.get("consistent"), bool):
        return obj["consistent"]

    # 兜底：无法解析 JSON 时做简单文本判断
    lower = raw.strip().lower()
    if "true" in lower:
        return True
    if "false" in lower:
        return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Judge LLM 一致性评估")
    parser.add_argument(
        "--dataset",
        type=str,
        default="testdata/test.json",
        help="数据集路径（JSON 数组，元素需包含 question 与 answer）",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=JUDGE_MODEL,
        help="Judge LLM 模型名（默认 JUDGE_MODEL）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评估前 N 条（0 表示全部）",
    )
    args = parser.parse_args()

    # 仅保留脚本自身输出，屏蔽 main/httpx/rag 等日志（含 WARNING）
    logging.disable(logging.CRITICAL)

    dataset_path = Path(args.dataset).expanduser()
    data = _load_dataset(dataset_path)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    total = len(data)
    if total == 0:
        print("数据集为空，无法评估。")
        return

    reset_token_usage()
    judge_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    correct = 0
    skipped = 0
    processed = 0
    t_all_start = time.perf_counter()

    def _progress_bar(done: int, total_count: int, width: int = 28) -> str:
        if total_count <= 0:
            return "[" + "-" * width + "]"
        ratio = max(0.0, min(1.0, done / total_count))
        filled = int(ratio * width)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    for i, item in enumerate(data, start=1):
        question = str(item.get("question", "")).strip()
        reference = str(item.get("answer", "")).strip()
        model_answer = ""
        if not question or not reference:
            skipped += 1
            processed += 1
            usage = get_token_usage()
            bar = _progress_bar(processed, total)
            print(
                f"\r{bar} {processed}/{total} | "
                f"当前正确率: {(correct / max(1, processed - skipped)) * 100:.2f}% | "
                f"当前耗时: 0.00s | "
                f"当前消耗token: {usage.get('total_tokens', 0)}",
                end="",
                flush=True,
            )
            continue

        t0 = time.perf_counter()
        try:
            result = run_system(
                question,
                selected_agents=None,
                do_update_ltm=False,
                do_update_mk=False,
            )
            model_answer = str(result.get("final_answer", "")).strip()
            is_correct = _judge_consistency(
                question=question,
                reference_answer=reference,
                model_answer=model_answer,
                judge_model=args.judge_model,
                judge_usage=judge_usage,
            )
        except Exception as e:
            is_correct = False
            _ = e

        if is_correct:
            correct += 1

        processed += 1
        elapsed = time.perf_counter() - t0
        usage = get_token_usage()
        effective_processed = max(1, processed - skipped)
        bar = _progress_bar(processed, total)
        print(
            f"\r{bar} {processed}/{total} | "
            f"当前正确率: {(correct / effective_processed) * 100:.2f}% | "
            f"当前耗时: {elapsed:.2f}s | "
            f"当前消耗token: {usage.get('total_tokens', 0)}",
            end="",
            flush=True,
        )
        print()
        print(f"模型答案: {model_answer}")
        print(f"标准答案: {reference}")

    total_elapsed = time.perf_counter() - t_all_start
    usage = get_token_usage()
    effective_total = total - skipped
    accuracy = (correct / effective_total * 100) if effective_total > 0 else 0.0

    print()
    print(f"最终正确率: {accuracy:.2f}%")
    print(f"总统计耗时: {total_elapsed:.2f}s")
    print(f"总消耗token: {usage.get('total_tokens', 0)}")


if __name__ == "__main__":
    main()

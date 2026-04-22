# -*- coding: utf-8 -*-
"""
Simple evaluation script:

1. Call `utils.llm.llm_call` (which defaults to `OPENAI_MODEL`) to get answers.
2. Use a judge model to compare generated answers against reference answers.
"""

import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import OPENAI_MODEL
from utils.llm import get_token_usage, llm_call, parse_json_from_llm, reset_token_usage
from utils.logger import setup_logging

logger = logging.getLogger("evaluate")

KEY = os.getenv("JUDGE_API_KEY", "<API KEY>")
BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1/")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5-chat-latest")

CSV_FIELDNAMES = [
    "\u9898\u53f7",
    "\u672c\u9898\u6240\u7528\u65f6\u957f",
    "\u672c\u9898\u6240\u7528token\u6570",
    "\u7d2f\u8ba1\u65f6\u957f",
    "\u7d2f\u8ba1\u4f7f\u7528token\u6570",
]


def _get_judge_client() -> OpenAI:
    if not KEY:
        logger.error("Missing JUDGE_API_KEY")
        raise ValueError("Please configure Judge API KEY")
    return OpenAI(api_key=KEY, base_url=BASE_URL)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    logger.info("Loading dataset: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.error("Dataset format is invalid")
        raise ValueError("Dataset must be a JSON array")
    logger.info("Dataset loaded successfully, total=%s", len(data))
    return data


def _sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "unknown"


def _get_model_name_for_filename(model_name: str) -> str:
    candidate = model_name.strip() or "unknown"
    if "/" in candidate or "\\" in candidate:
        candidate = Path(candidate).name
    return _sanitize_filename_part(candidate)


def _build_csv_path(dataset_path: Path, model_name: str, csv_output: str | None = None) -> Path:
    dataset_name = _sanitize_filename_part(dataset_path.stem or "dataset")
    safe_model_name = _get_model_name_for_filename(model_name)
    default_name = f"{dataset_name}_{safe_model_name}.csv"
    if not csv_output:
        return Path("output") / "without" / default_name

    output_path = Path(csv_output).expanduser()
    if output_path.suffix.lower() == ".csv":
        return output_path
    return output_path / default_name


def _build_log_filename(dataset_path: Path, model_name: str) -> str:
    dataset_name = _sanitize_filename_part(dataset_path.stem or "dataset")
    safe_model_name = _get_model_name_for_filename(model_name)
    return f"{safe_model_name}_{dataset_name}_without.log"


def _build_metrics_row(
    index: int,
    question_elapsed: float,
    question_tokens: int,
    cumulative_elapsed: float,
    cumulative_tokens: int,
) -> dict[str, Any]:
    return {
        CSV_FIELDNAMES[0]: index,
        CSV_FIELDNAMES[1]: round(question_elapsed, 4),
        CSV_FIELDNAMES[2]: question_tokens,
        CSV_FIELDNAMES[3]: round(cumulative_elapsed, 4),
        CSV_FIELDNAMES[4]: cumulative_tokens,
    }


def _write_metrics_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _judge_consistency(
    question: str,
    reference_answer: str,
    model_answer: str,
    judge_model: str,
    judge_usage: dict[str, int],
) -> bool:
    system_prompt = "You are a strict answer consistency judge. Return JSON only."
    prompt = f"""
Question:
{question}

Reference Answer:
{reference_answer}

Model Answer:
{model_answer}

Task:
Determine whether Model Answer is consistent with Reference Answer.
Return:
{{"consistent": true/false}}
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
        judge_usage["prompt_tokens"] += usage.prompt_tokens
        judge_usage["completion_tokens"] += usage.completion_tokens
        judge_usage["total_tokens"] += usage.total_tokens

    raw = (resp.choices[0].message.content or "").strip()
    obj = parse_json_from_llm(raw)
    if isinstance(obj, dict) and isinstance(obj.get("consistent"), bool):
        return obj["consistent"]

    lower = raw.lower()
    if "true" in lower:
        return True
    if "false" in lower:
        return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="在数据集上评估当前配置模型")
    parser.add_argument("--dataset", type=str, default="./dataset/HaluEval_qa.json", help="数据集路径")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL, help="裁判模型名称")
    parser.add_argument("--limit", type=int, default=0, help="限制评估样本数")
    parser.add_argument(
        "--csv-output",
        type=str,
        default="",
        help="CSV保存路径；传目录时使用默认文件名，传.csv路径时按指定文件保存",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    setup_logging(log_file=_build_log_filename(dataset_path, OPENAI_MODEL))

    data = _load_dataset(dataset_path)
    if args.limit > 0:
        data = data[: args.limit]
        logger.info("Evaluation limit applied: %s", args.limit)

    total = len(data)
    if total == 0:
        print("数据集为空。")
        logger.warning("Dataset is empty.")
        return

    reset_token_usage()
    judge_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    correct = 0
    skipped = 0
    processed = 0
    prev_total_tokens = 0
    csv_rows: list[dict[str, Any]] = []
    t_all_start = time.perf_counter()

    def _progress(done: int, total_count: int, width: int = 28) -> str:
        ratio = max(0.0, min(1.0, done / total_count))
        filled = int(width * ratio)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    model_answer = ""
    reference = ""

    for i, item in enumerate(data, start=1):
        question = str(item.get("question", "")).strip()
        reference = str(item.get("answer", "")).strip()

        if not question or not reference:
            skipped += 1
            processed += 1
            usage = get_token_usage()
            current_total_tokens = usage.get("total_tokens", 0)
            csv_rows.append(
                _build_metrics_row(
                    index=i,
                    question_elapsed=0.0,
                    question_tokens=0,
                    cumulative_elapsed=time.perf_counter() - t_all_start,
                    cumulative_tokens=current_total_tokens,
                )
            )
            continue

        t0 = time.perf_counter()
        try:
            model_answer = llm_call(question)
            is_correct = _judge_consistency(
                question,
                reference,
                model_answer,
                args.judge_model,
                judge_usage,
            )
        except Exception as e:
            logger.error("Evaluation failed at item %s: %s", i, e)
            is_correct = False

        if is_correct:
            correct += 1
            logger.info("Item %s/%s: correct", i, total)
        else:
            logger.info("Item %s/%s: incorrect", i, total)

        processed += 1

        question_elapsed = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_all_start
        usage = get_token_usage()
        current_total_tokens = usage.get("total_tokens", 0)
        csv_rows.append(
            _build_metrics_row(
                index=i,
                question_elapsed=question_elapsed,
                question_tokens=current_total_tokens - prev_total_tokens,
                cumulative_elapsed=elapsed,
                cumulative_tokens=current_total_tokens,
            )
        )
        prev_total_tokens = current_total_tokens
        bar = _progress(processed, total)

        print(
            f"\r{bar} {processed}/{total} | "
            f"当前正确率: {(correct / max(1, processed - skipped)) * 100:.2f}% | "
            f"当前耗时: {elapsed:.2f}s | "
            f"当前消耗token: {current_total_tokens}\n",
            end="",
            flush=True,
        )

    print()
    print(f"模型答案: {model_answer}")
    print(f"标准答案: {reference}")

    total_elapsed = time.perf_counter() - t_all_start
    usage = get_token_usage()
    effective_total = total - skipped
    accuracy = (correct / max(1, effective_total)) * 100

    logger.info("Final accuracy: %.2f%%", accuracy)
    logger.info("Total elapsed: %.2fs", total_elapsed)
    logger.info("Total tokens: %s", usage.get("total_tokens", 0))

    print()
    print(f"最终正确率: {accuracy:.2f}%")
    print(f"总计耗时: {total_elapsed:.2f}s")
    print(f"总消耗token: {usage.get('total_tokens', 0)}")

    csv_path = _build_csv_path(dataset_path, OPENAI_MODEL, args.csv_output)
    _write_metrics_csv(csv_path, csv_rows)
    logger.info("CSV saved to %s", csv_path)
    print(f"CSV已保存到: {csv_path}")


if __name__ == "__main__":
    main()

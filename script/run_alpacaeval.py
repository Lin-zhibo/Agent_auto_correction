# -*- coding: utf-8 -*-
"""
AlpacaEval 评估脚本：
1. 从 parquet 文件读取 AlpacaEval 数据集
2. 将每条样本构造成 question，喂给 stuLLM（整个多 Agent 系统）
3. 使用 judgeLLM 对 stuLLM 的回答与参考答案进行对比判分（正确 / 错误）
4. 计算准确率，并将详细结果保存为 JSON
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# 将项目根目录添加到 sys.path，以便导入项目模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from main import run_system
from utils.llm import reset_token_usage, get_token_usage
from utils.logger import get_logger, setup_logging

# ==================== 配置 ====================
# TODO: 如有需要，请修改为你自己的 KEY / BASE_URL / 模型名称
KEY = "sk-XcGaOx4wJHKQ8Rq694E001E4209347E6A27174272d687bBa"
BASE_URL = "https://api.gpt.ge/v1/"
JUDGE_MODEL = "gpt-5"

# 路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
# AlpacaEval parquet 文件路径（请根据实际文件名修改）
ALPACA_PATH = SCRIPT_DIR / "alpacaeval.parquet"
# 评估结果输出路径
OUTPUT_PATH = SCRIPT_DIR / "alpacaeval_results.json"

# 初始化日志
setup_logging()
logger = get_logger(__name__)


# ==================== stuLLM（多 Agent 系统） ====================

def call_stu_llm(question: str) -> str:
    """
    调用 stuLLM（整个多 Agent 系统）获取答案

    :param question: 问题（由 AlpacaEval 的 instruction/input 组合而成）
    :return: 最终答案
    """
    logger.info("stuLLM 开始处理问题: %s", question[:80] + "..." if len(question) > 80 else question)
    reset_token_usage()

    try:
        result = run_system(
            question,
            selected_agents=None,  # 由 MK 自动选择 Agent
            do_update_ltm=False,   # 评估时不更新 LTM
            do_update_mk=False,    # 评估时不更新 MK
        )
        answer = result.get("final_answer", "")
        logger.info("stuLLM 返回答案长度: %d", len(answer))
        return answer
    except Exception as e:
        logger.exception("stuLLM 处理失败: %s", e)
        return ""


# ==================== judgeLLM：对比参考答案判定对错 ====================

def get_judge_client() -> OpenAI:
    """获取 Judge LLM 的 OpenAI 客户端"""
    return OpenAI(api_key=KEY, base_url=BASE_URL)


def call_judge_llm_quality(
    instruction: str,
    input_text: str,
    reference_output: str,
    stu_answer: str,
) -> bool:
    """
    调用 judgeLLM 对 stuLLM 的回答是否与参考答案语义一致进行判定

    :param instruction: AlpacaEval 的 instruction
    :param input_text: AlpacaEval 的 input（可能为空）
    :param reference_output: 参考答案（ground truth）
    :param stu_answer: stuLLM 给出的答案
    :return: True 表示 stu_answer 被判定为“正确”（与参考答案语义一致），False 表示“错误”
    """
    client = get_judge_client()

    # 构造任务描述
    if input_text:
        task_desc = f"Instruction:\n{instruction}\n\nInput:\n{input_text}"
    else:
        task_desc = f"Instruction:\n{instruction}"

    prompt = f"""You are an expert judge evaluating whether a model's answer is semantically correct compared to a reference answer for an instruction-following task.

{task_desc}

Reference answer:
{reference_output}

Model answer to evaluate:
{stu_answer}

Task:
Determine whether the model answer is semantically equivalent to the reference answer, i.e., it correctly follows the instruction and provides all the essential information, even if the wording is different.

Instructions:
1. Focus on semantic correctness, not surface-form similarity.
2. If the model answer is mostly correct but has minor wording differences or extra harmless details, treat it as correct.
3. If the model answer misses key points, adds wrong information, contradicts the reference, or does not follow the instruction, treat it as incorrect.

Output:
Respond with ONLY a JSON object in the following format (no other text):
{{"is_correct": true}}  if the model answer is semantically correct
{{"is_correct": false}} if the model answer is not semantically correct
"""

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an evaluation expert. Respond only with JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        raw_response = (response.choices[0].message.content or "").strip()
        logger.info("judgeLLM 原始响应: %s", raw_response)

        # 解析 JSON 响应
        import re

        json_match = re.search(r"\{.*?\}", raw_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            is_correct = result.get("is_correct", False)
            return bool(is_correct)

        # 如果无法解析 JSON，尝试简单判断
        lower_response = raw_response.lower()
        if "true" in lower_response:
            return True
        elif "false" in lower_response:
            return False
        else:
            logger.warning("judgeLLM 响应无法解析，默认为错误: %s", raw_response)
            return False

    except Exception as e:
        logger.exception("judgeLLM 调用失败: %s", e)
        # 出错时保守地认为“不正确”，以免高估系统性能
        return False


# ==================== 数据加载与结果保存 ====================

def load_alpaca_dataset() -> List[Dict[str, Any]]:
    """
    加载 AlpacaEval 数据集（parquet）

    当前假设 parquet 中仅包含两列：
    - prompt: 问题 / 指令
    - selected: 参考答案（ground truth）
    """
    logger.info("加载 AlpacaEval 数据集: %s", ALPACA_PATH)
    df = pd.read_parquet(ALPACA_PATH)

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        # 你的 parquet 中的字段：prompt = 问题，selected = 答案
        instruction = str(row.get("prompt", "") or "")
        input_text = ""  # 没有单独的 input 列，这里始终为空
        reference_output = str(row.get("selected", "") or "")

        # 组合成传给 stuLLM 的 question
        if input_text:
            question = f"{instruction}\n\nInput:\n{input_text}"
        else:
            question = instruction

        records.append(
            {
                "instruction": instruction,
                "input": input_text,
                "reference_output": reference_output,
                "question": question,
            }
        )

    logger.info("AlpacaEval 数据集加载完成，共 %d 条记录", len(records))
    return records


def save_results(results: List[Dict[str, Any]]) -> None:
    """保存评估结果到 OUTPUT_PATH"""
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("结果已保存至: %s", OUTPUT_PATH)


# ==================== 主评估流程 ====================

def run_alpaca_evaluation() -> List[Dict[str, Any]]:
    """运行完整的 AlpacaEval 评估流程"""
    logger.info("=" * 60)
    logger.info("开始 AlpacaEval 评估（stuLLM + judgeLLM 计算准确率）")
    logger.info("=" * 60)

    dataset = load_alpaca_dataset()

    # 只随机抽取 100 条进行评估（如果数据少于 100 条则全用）
    import random
    if len(dataset) > 100:
        logger.info("原始数据集大小为 %d 条，随机抽取 100 条进行评估", len(dataset))
        dataset = random.sample(dataset, 100)
    else:
        logger.info("数据集大小为 %d 条，小于等于 100 条，全部用于评估", len(dataset))

    results: List[Dict[str, Any]] = []
    total = len(dataset)
    correct_count = 0

    for idx, item in enumerate(dataset):
        instruction = item["instruction"]
        input_text = item["input"]
        reference_output = item["reference_output"]
        question = item["question"]

        logger.info("")
        logger.info("=" * 40)
        logger.info("处理第 %d/%d 条", idx + 1, total)
        logger.info(
            "instruction: %s",
            instruction[:100] + "..." if len(instruction) > 100 else instruction,
        )

        # Step 1: 调用 stuLLM 获取答案
        t0 = time.perf_counter()
        stu_answer = call_stu_llm(question)
        stu_time = time.perf_counter() - t0
        token_usage = get_token_usage()

        logger.info(
            "stuLLM 答案: %s",
            stu_answer[:200] + "..." if len(stu_answer) > 200 else stu_answer,
        )
        logger.info(
            "stuLLM 耗时: %.2f 秒, Token: %d",
            stu_time,
            token_usage.get("total_tokens", 0),
        )

        # Step 2: 调用 judgeLLM 判定 stu_answer 是否与参考答案一致
        t1 = time.perf_counter()
        is_correct = call_judge_llm_quality(
            instruction=instruction,
            input_text=input_text,
            reference_output=reference_output,
            stu_answer=stu_answer,
        )
        judge_time = time.perf_counter() - t1

        if is_correct:
            correct_count += 1

        logger.info(
            "judgeLLM 判定: %s",
            "正确（与参考答案语义一致）" if is_correct else "错误（与参考答案不一致）",
        )
        logger.info("judgeLLM 耗时: %.2f 秒", judge_time)

        # 记录结果
        result_item: Dict[str, Any] = {
            "index": idx,
            "instruction": instruction,
            "input": input_text,
            "reference_output": reference_output,
            "stu_answer": stu_answer,
            "is_correct": is_correct,
            "stu_time_seconds": round(stu_time, 2),
            "judge_time_seconds": round(judge_time, 2),
            "stu_token_usage": token_usage,
        }
        results.append(result_item)

        # 实时保存（防止中断丢失）
        save_results(results)

        logger.info(
            "当前准确率: %.2f%% (%d/%d)",
            100 * correct_count / (idx + 1),
            correct_count,
            idx + 1,
        )

    # 输出最终统计
    logger.info("")
    logger.info("=" * 60)
    logger.info("AlpacaEval 评估完成！")
    logger.info("总样本数: %d", total)
    logger.info("判定为正确: %d", correct_count)
    logger.info(
        "最终准确率: %.2f%%",
        100 * correct_count / total if total > 0 else 0,
    )
    logger.info("结果文件: %s", OUTPUT_PATH)
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_alpaca_evaluation()


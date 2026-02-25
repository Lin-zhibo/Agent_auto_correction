# -*- coding: utf-8 -*-
"""
HaluEval 评估脚本：
1. 读取 halueval_qa-100.json 数据集
2. 将每个问题的 question 字段喂给 stuLLM（整个多 Agent 系统）
3. 获取 answer 后，将 knowledge、question、answer 交给 judgeLLM 判断是否幻觉
4. 结果保存到 testdata_halueval_result.json
"""

import json
import sys
import time
from pathlib import Path

# 将项目根目录添加到 sys.path，以便导入项目模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from main import run_system
from utils.llm import reset_token_usage, get_token_usage
from utils.logger import get_logger, setup_logging

# 配置
KEY = "sk-XcGaOx4wJHKQ8Rq694E001E4209347E6A27174272d687bBa"
BASE_URL = "https://api.gpt.ge/v1/"
JUDGE_MODEL = "gpt-5"

# 路径
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "halueval_qa-100.json"
OUTPUT_PATH = SCRIPT_DIR / "testdata_halueval_result.json"

# 初始化日志
setup_logging()
logger = get_logger(__name__)


def get_judge_client() -> OpenAI:
    """获取 Judge LLM 的 OpenAI 客户端"""
    return OpenAI(api_key=KEY, base_url=BASE_URL)


def call_stu_llm(question: str) -> str:
    """
    调用 stuLLM（整个多 Agent 系统）获取答案
    :param question: 问题
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


# Web Search 工具配置
# 注意：web_search_preview 类型不被某些 API 支持（如 api.gpt.ge），暂时禁用
# WEB_SEARCH_TOOL = {"type": "web_search_preview"}
WEB_SEARCH_TOOL = None
ENABLE_WEB_SEARCH = False  # 是否启用 web search（已禁用：当前 API 不支持 web_search_preview）


def call_judge_llm(knowledge: str, question: str, answer: str) -> dict:
    """
    调用 judgeLLM 判断答案是否存在幻觉
    :param knowledge: 背景知识
    :param question: 问题
    :param answer: stuLLM 给出的答案
    :return: dict 包含 hallucinated, correct_answer, reasoning
    """
    client = get_judge_client()
    
    prompt = f"""You are an expert judge evaluating whether an answer contains hallucinations based on the given knowledge.

**Knowledge:**
{knowledge}

**Question:**
{question}

**Answer to evaluate:**
{answer}

**Task:**
Determine if the answer contains hallucinations (i.e., information that is factually incorrect, unsupported by or contradicts the given knowledge).

**Instructions:**
1. Carefully compare the answer against the provided knowledge.
2. If the answer contains ANY factual errors, made-up information, or contradictions with the knowledge, it is hallucinated.
3. If the answer is fully supported by and consistent with the knowledge, it is NOT hallucinated.
4. You MUST also provide the correct answer based on the knowledge and your reasoning.

**Output:**
Respond with ONLY a JSON object in the following format (no other text):
{{
    "hallucinated": true or false,
    "correct_answer": "The correct answer based on the knowledge (if hallucinated, provide what the answer should be; if not hallucinated, you can repeat the evaluated answer or provide a refined version)",
    "reasoning": "Your detailed reasoning for why the answer is or is not hallucinated, including specific factual errors or confirmations"
}}
"""

    try:
        # 构建请求参数
        request_params = {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": "You are a hallucination detection expert with web search capability. Respond only with JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
        }
        
        # 如果启用 web search，添加工具参数
        # 注意：web_search_preview 类型不被某些 API 支持，已禁用
        if ENABLE_WEB_SEARCH and WEB_SEARCH_TOOL is not None:
            request_params["tools"] = [WEB_SEARCH_TOOL]
        
        response = client.chat.completions.create(**request_params)
        
        raw_response = (response.choices[0].message.content or "").strip()
        logger.info("judgeLLM 原始响应: %s", raw_response[:500] + "..." if len(raw_response) > 500 else raw_response)
        
        # 解析 JSON 响应
        import re
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "hallucinated": bool(result.get("hallucinated", True)),
                "correct_answer": result.get("correct_answer", ""),
                "reasoning": result.get("reasoning", "")
            }
        
        # 如果无法解析 JSON，尝试简单判断
        lower_response = raw_response.lower()
        if "true" in lower_response:
            return {"hallucinated": True, "correct_answer": "", "reasoning": "JSON解析失败，根据响应中包含true判定"}
        elif "false" in lower_response:
            return {"hallucinated": False, "correct_answer": "", "reasoning": "JSON解析失败，根据响应中包含false判定"}
        else:
            logger.warning("judgeLLM 响应无法解析，默认为幻觉: %s", raw_response)
            return {"hallucinated": True, "correct_answer": "", "reasoning": "响应无法解析，默认判定为幻觉"}
            
    except Exception as e:
        logger.exception("judgeLLM 调用失败: %s", e)
        return {"hallucinated": True, "correct_answer": "", "reasoning": f"调用失败: {e}"}  # 出错时保守地认为是幻觉


def load_dataset() -> list[dict]:
    """加载 HaluEval 数据集"""
    logger.info("加载数据集: %s", INPUT_PATH)
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("数据集加载完成，共 %d 条记录", len(data))
    return data


def save_results(results: list[dict]) -> None:
    """保存评估结果"""
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("结果已保存至: %s", OUTPUT_PATH)


def run_evaluation():
    """运行完整的评估流程"""
    logger.info("=" * 60)
    logger.info("开始 HaluEval 评估")
    logger.info("=" * 60)
    
    # 加载数据集
    dataset = load_dataset()
    
    results = []
    total = len(dataset)
    correct_count = 0
    
    for idx, item in enumerate(dataset):
        knowledge = item.get("knowledge", "")
        question = item.get("question", "")
        ground_truth_answer = item.get("answer", "")
        ground_truth_judge = item.get("judge", 0)  # 0=无幻觉, 1=有幻觉
        
        logger.info("")
        logger.info("=" * 40)
        logger.info("处理第 %d/%d 条", idx + 1, total)
        logger.info("问题: %s", question[:100] + "..." if len(question) > 100 else question)
        
        # Step 1: 调用 stuLLM 获取答案
        t0 = time.perf_counter()
        stu_answer = call_stu_llm(question)
        stu_time = time.perf_counter() - t0
        token_usage = get_token_usage()
        
        logger.info("stuLLM 答案: %s", stu_answer[:200] + "..." if len(stu_answer) > 200 else stu_answer)
        logger.info("stuLLM 耗时: %.2f 秒, Token: %d", stu_time, token_usage.get("total_tokens", 0))
        
        # Step 2: 调用 judgeLLM 判断是否幻觉
        t1 = time.perf_counter()
        judge_result = call_judge_llm(knowledge, question, stu_answer)
        judge_time = time.perf_counter() - t1
        
        is_hallucinated = judge_result["hallucinated"]
        judge_correct_answer = judge_result["correct_answer"]
        judge_reasoning = judge_result["reasoning"]
        
        # judge 结果: 1=有幻觉, 0=无幻觉
        predicted_judge = 1 if is_hallucinated else 0
        
        # stuLLM 回答正确 = 无幻觉 (predicted_judge == 0)
        stu_is_correct = (predicted_judge == 0)
        
        logger.info("judgeLLM 判断: %s", "有幻觉" if is_hallucinated else "无幻觉")
        logger.info("judgeLLM 认为的正确答案: %s", judge_correct_answer[:300] + "..." if len(judge_correct_answer) > 300 else judge_correct_answer)
        logger.info("judgeLLM 判断理由: %s", judge_reasoning[:500] + "..." if len(judge_reasoning) > 500 else judge_reasoning)
        logger.info("stuLLM 回答: %s", "正确" if stu_is_correct else "错误（存在幻觉）")
        logger.info("judgeLLM 耗时: %.2f 秒", judge_time)
        
        # stuLLM 回答正确（无幻觉）计入正确数
        if stu_is_correct:
            correct_count += 1
        
        # 记录结果
        result_item = {
            "index": idx,
            "knowledge": knowledge,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "stu_answer": stu_answer,
            "ground_truth_judge": ground_truth_judge,
            "predicted_judge": predicted_judge,
            "stu_is_correct": stu_is_correct,  # stuLLM回答是否正确（无幻觉=正确）
            "judge_correct_answer": judge_correct_answer,  # judgeLLM认为的正确答案
            "judge_reasoning": judge_reasoning,  # judgeLLM判断理由
            "stu_time_seconds": round(stu_time, 2),
            "judge_time_seconds": round(judge_time, 2),
            "stu_token_usage": token_usage,
        }
        results.append(result_item)
        
        # 实时保存（防止中断丢失）
        save_results(results)
        
        logger.info("当前 stuLLM 回答准确率: %.2f%% (%d/%d)", 
                   100 * correct_count / (idx + 1), correct_count, idx + 1)
    
    # 输出最终统计
    logger.info("")
    logger.info("=" * 60)
    logger.info("评估完成！")
    logger.info("总样本数: %d", total)
    logger.info("stuLLM 回答正确数（无幻觉）: %d", correct_count)
    logger.info("stuLLM 回答错误数（有幻觉）: %d", total - correct_count)
    logger.info("stuLLM 回答准确率: %.2f%%", 100 * correct_count / total if total > 0 else 0)
    logger.info("结果文件: %s", OUTPUT_PATH)
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    run_evaluation()
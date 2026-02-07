# -*- coding: utf-8 -*-
"""
多 Agent 反思系统主流程：初答 → MK 选 Agent（或用户指定）→ 多 Agent 反馈 → Insight 整合 → Student 修正 → 循环/输出 → 可选更新 LTM/MK。
"""

from typing import Any

from utils.logger import get_logger

from memory import (
    build_knowledge_tree,
    create_wm,
    evolve_mk_from_better_agents,
    evolve_mk_from_random_agent,
    load_ltm,
    load_mk,
    save_ltm,
    save_mk,
    select_better_agents_from_wm,
    update_ltm,
    update_mk_from_ltm,
    update_wm,
    RAGSearch,
)
from agents import AgentFactory, InsightAgent, MetaKnowledge, StudentAgent
from utils.llm import (
    compute_improvement,
    get_token_usage,
    reset_token_usage,
    semantic_similarity,
)

logger = get_logger(__name__)


def get_suggest(question: str) -> dict[str, Any]:
    """
    根据问题返回 MK 建议的 Agent 列表与初答，供前端展示为系统建议勾选。
    返回: initial_answer, question_type, suggested_agents, all_agents
    """
    logger.info("get_suggest 开始 question=%s", question[:80] + "..." if len(question) > 80 else question)
    ltm = load_ltm()
    mk_data = load_mk()
    ltm_root = build_knowledge_tree(ltm)
    rag_search = RAGSearch(ltm_root)
    mk = MetaKnowledge(mk_data)
    student = StudentAgent(rag_search)
    answer, question_type = student.answer(question, ltm, mk_data)
    confidence, consistency = student.evaluate_answer(answer, ltm)
    suggested_agents = mk.select_agents(
        question, confidence, consistency, question_type=question_type
    )
    from agents.meta_knowledge import ALL_AGENT_NAMES
    logger.info("get_suggest 完成 question_type=%s suggested_agents=%s", question_type, suggested_agents)
    return {
        "initial_answer": answer,
        "question_type": question_type,
        "suggested_agents": suggested_agents,
        "all_agents": list(ALL_AGENT_NAMES),
    }


def run_system(
    question: str,
    *,
    selected_agents: list[str] | None = None,
    do_update_ltm: bool = True,
    do_update_mk: bool = True,
) -> dict[str, Any]:
    """
    主流程。
    :param question: 用户问题
    :param selected_agents: 用户选择的 Agent 列表；若为 None 则每轮由 MK 选择
    :param do_update_ltm: 是否在结束后更新并保存 LTM
    :param do_update_mk: 是否在结束后更新并保存 MK（含 better_agents 与 random_agent 进化、update_mk_from_ltm）
    :return: {"initial_answer": 初答, "final_answer": 最终答案}
    """
    logger.info(
        "run_system 开始 question=%s selected_agents=%s do_update_ltm=%s do_update_mk=%s",
        question[:80] + "..." if len(question) > 80 else question,
        selected_agents,
        do_update_ltm,
        do_update_mk,
    )
    ltm = load_ltm()
    mk_data = load_mk()
    ltm_root = build_knowledge_tree(ltm)
    rag_search = RAGSearch(ltm_root)
    mk = MetaKnowledge(mk_data)
    student = StudentAgent(rag_search)
    insight = InsightAgent()
    wm = create_wm()

    answer, question_type = student.answer(question, ltm, mk_data)
    answer2, question_type2 = student.answer(question, ltm, mk_data)
    logger.info("初始轮次再回答：(全文): %s", answer2)
    confidence, consistency = student.evaluate_answer(answer, ltm)
    logger.info("初答完成 question_type=%s confidence=%.3f consistency=%.3f", question_type, confidence, consistency)
    logger.info("初答内容(全文): %s", answer)
    initial_answer = answer
    loop_count = 0
    # should_continue 应当与双向蕴含结果相反
    should_continue = not mk.is_bientail(answer1=answer, answer2=answer2)  # 双向蕴含判断初始是否继续


    while should_continue:
        if selected_agents is not None and len(selected_agents) > 0:
            agents_this_round = selected_agents[:4]
        else:
            agents_this_round = mk.select_agents(
                question, confidence, consistency, question_type=question_type
            )
        round_num = loop_count + 1
        logger.info("========== 第 %s 轮 ========== 本轮 Agent: %s", round_num, agents_this_round)
        feedbacks = [AgentFactory.create(a).review(answer) for a in agents_this_round]
        for fb in feedbacks:
            comment = (fb.get("comment") or "").strip()
            if comment:
                logger.info("  [%s] 反馈: %s", fb.get("agent", "?"), comment)

        organized_feedback = insight.integrate_feedback(question, answer, feedbacks)
        logger.info("Insight 综合指导(全文): %s", organized_feedback)
        improved_answer = student.revise_answer(question, answer, organized_feedback)
        improvement_score = compute_improvement(answer, improved_answer)
        logger.info("本轮改进后答案(全文): %s | improvement_score=%.3f", improved_answer, improvement_score)
        # if round_num == 1:
        #     # 第一轮
        #     answer2 , _ = student.answer(question, ltm, mk_data)
        #     logger.info("初始轮次再回答：(全文): %s", answer2)
        #     improved_answer = answer2

        should_continue = mk.should_continue(
            answer,
            # answer2,
            improved_answer,
            improvement_score,
            loop_count,
            initial_answer=initial_answer,
        )

        update_wm(
            wm,
            question=question,
            student_answer=answer,
            agent_feedback=feedbacks,
            improved_answer=improved_answer,
            iteration=loop_count,
        )

        prev_answer_this_round = answer
        answer = improved_answer
        loop_count += 1
        logger.info("第 %s 轮结束，是否继续下一轮: %s", loop_count, should_continue)
        confidence, consistency = student.evaluate_answer(
            answer, ltm, initial_answer=initial_answer, previous_answer=prev_answer_this_round
        )

    if do_update_mk:
        logger.info("更新并保存 MK")
        better_agents = select_better_agents_from_wm(wm, answer)
        if better_agents:
            evolve_mk_from_better_agents(mk_data, question_type, better_agents)
        random_agent = mk.get_last_random_agent()
        if random_agent and loop_count >= 2:
            sim_final_initial = semantic_similarity(answer, initial_answer)
            if sim_final_initial >= 0.5:
                evolve_mk_from_random_agent(mk_data, question_type, random_agent)
        update_mk_from_ltm(mk_data, ltm)
        save_mk(mk_data)

    if do_update_ltm:
        logger.info("更新并保存 LTM")
        update_ltm(ltm, question, answer, question_type=question_type, topic_path=None)
        save_ltm(ltm)

    logger.info("run_system 完成 共 %s 轮，最终答案(全文): %s", loop_count, answer)
    return {"initial_answer": initial_answer, "final_answer": answer}


if __name__ == "__main__":
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    test_path = Path(__file__).parent / "testdata" / "test.json"
    with open(test_path, encoding="utf-8") as f:
        items = json.load(f)

    results: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        q = item.get("question", "")
        if not q:
            continue
        print(f"\n========== 第 {i + 1}/{len(items)} 题 ==========")
        print("问题:", q)
        reset_token_usage()
        t0 = time.perf_counter()
        run_result = run_system(q)
        elapsed = time.perf_counter() - t0
        usage = get_token_usage()
        record = {
            "问题": q,
            "第一次回答的答案": run_result["initial_answer"],
            "最终回答的答案": run_result["final_answer"],
            "回答时间": datetime.now(timezone.utc).isoformat(),
            "耗时_秒": round(elapsed, 2),
            "消耗的token": usage["total_tokens"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
        }
        results.append(record)
        print("=== 最终答案 ===")
        print(run_result["final_answer"])

    out_path = Path(__file__).parent / "testdata" / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至 {out_path}")

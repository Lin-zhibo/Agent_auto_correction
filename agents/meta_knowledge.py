# -*- coding: utf-8 -*-
"""Meta-Knowledge (MK)：策略控制中心，从 MK 数据按问题类型选择 Agent、控制循环、决策是否继续。"""

import random
from typing import Any

from config import MK_JUDGE_MODEL
from memory.mk_memory import get_config_for_question_type, infer_question_type, load_mk
from utils.llm import llm_call, semantic_similarity
from utils.logger import get_logger

logger = get_logger(__name__)

# 全部可选 Agent 名称（与 AgentFactory 一致）
ALL_AGENT_NAMES = [
    "questioner",
    "logic_analyzer",
    "authority_checker",
    "explainer",
    "economics_expert",
    "math_expert",
    "philosophy_expert",
    "science_expert",
    "critic",
    "supporter",
    "clarity_checker",
    "completeness_checker",
    "evidence_checker",
    "brevity_advisor",
    "audience_advisor",
]


class MetaKnowledge:
    """策略中枢：从 MK 数据中按问题类型读取策略、优先级、阈值，选择参与反馈的 Agent、决策是否继续优化。"""

    def __init__(self, mk: dict[str, Any] | None = None):
        self.mk = mk if mk is not None else load_mk()
        self._current_config: dict[str, Any] = {}
        self._current_question_type: str = ""
        self._last_random_agent: str | None = None

    def _ensure_config(self, question: str | None = None, question_type: str | None = None) -> None:
        """
        加载当前问题类型对应的 MK 配置，供 select_agents / should_continue 使用。
        若传入 question_type 则直接按类型取配置；否则根据 question 推断类型。
        """
        if question_type is not None:
            self._current_question_type = question_type
        elif question is not None:
            self._current_question_type = infer_question_type(question, self.mk)
        else:
            return
        self._current_config = get_config_for_question_type(self.mk, self._current_question_type)

    def select_agents(
        self,
        question: str,
        confidence: float,
        consistency: float,
        question_type: str | None = None,
    ) -> list[str]:
        """
        按 MK 中当前问题类型的 agent_priorities 选出优先级最高的 3 个 Agent，
        再在未入选的 Agent 中随机加入 1 个，共返回 4 个。
        随机加入的 Agent 会记录在 _last_random_agent，若最终结果好可用于 MK 进化。
        """
        self._ensure_config(question=question, question_type=question_type)
        agent_priorities = self._current_config.get("agent_priorities", {})

        # 按 MK 优先级排序，取前 3 个（保证始终有 3 个）
        sorted_by_priority = sorted(
            ALL_AGENT_NAMES,
            key=lambda a: agent_priorities.get(a, 0),
            reverse=True,
        )
        mk_three = sorted_by_priority[:3]

        # 在未入选的 Agent 中随机选一个（保证共 4 个）
        remaining = [a for a in ALL_AGENT_NAMES if a not in mk_three]
        self._last_random_agent = None
        if remaining:
            self._last_random_agent = random.choice(remaining)
            mk_three = mk_three + [self._last_random_agent]
        logger.info("MK 选 Agent: %s (随机加入: %s)", mk_three, self._last_random_agent)
        return mk_three

    def get_last_random_agent(self) -> str | None:
        """返回本轮 select_agents 中随机加入的那个 Agent 名称，用于结果好时参与 MK 进化。"""
        return self._last_random_agent

    def should_continue(
        self,
        prev_answer: str,
        new_answer: str,
        improvement_score: float,
        loop_count: int,
        *,
        initial_answer: str | None = None,
    ) -> bool:
        """
        判定是否继续下一轮反思循环。
        - 至少进行两次循环（loop_count < 2 时一定继续）。
        - 之后：与「首答」「上一轮回答」做一致性判断；若与上一轮几乎无变化或相对首答无继续变好趋势，则结束。
        - when round 1, new_answer is another answer.
        """
        
        if not self._current_config:
            return False
        strategy = self._current_config.get("strategy", {})
        max_loops = int(strategy.get("max_loops", 3))
        similarity_threshold = float(strategy.get("similarity_threshold", 0.9))

        # 至少进行两次循环
        if loop_count < 2:
            if self.is_bientail(prev_answer, new_answer):
                logger.info("should_continue 结束原因: 双向蕴含 (bidirectional entailment) 检测PASS，停止循环")
                return False
            if loop_count >= max_loops:
                logger.info("should_continue 结束原因: 达到设置的最高轮数 max_loops=%s (当前 loop_count=%s)", max_loops, loop_count)
                return False
            logger.info("should_continue 继续: 未满最小轮数 2 (当前 loop_count=%s)", loop_count)
            return True

        if loop_count >= max_loops:
            logger.info("should_continue 结束原因: 达到设置的最高轮数 max_loops=%s (当前 loop_count=%s)", max_loops, loop_count)
            return False

        # 与上一轮回答的一致性：若几乎不变则认为优化完毕
        sim_new_prev = semantic_similarity(new_answer, prev_answer)
        if sim_new_prev > similarity_threshold:
            logger.info(
                "should_continue 结束原因: 改动过小 (与上一轮相似度 %.3f > 阈值 %.3f，设置的最高轮数 max_loops=%s)",
                sim_new_prev, similarity_threshold, max_loops,
            )
            return False

        # 与首答的一致性趋势：若相对上一轮没有继续变好（相对首答更一致），则不再继续
        if initial_answer and initial_answer.strip():
            sim_new_initial = semantic_similarity(new_answer, initial_answer)
            sim_prev_initial = semantic_similarity(prev_answer, initial_answer)
            if sim_new_initial <= sim_prev_initial:
                logger.info(
                    "should_continue 结束原因: 无变好趋势 (sim_new_initial=%.3f <= sim_prev_initial=%.3f)，设置的最高轮数 max_loops=%s",
                    sim_new_initial, sim_prev_initial, max_loops,
                )
                return False
        logger.info("should_continue 继续: 未达结束条件 (sim_new_prev=%.3f，阈值=%.3f，max_loops=%s)", sim_new_prev, similarity_threshold, max_loops)
        return True
    
    def is_bientail(
        self,
        answer1: str,
        answer2: str,
    ) -> bool:
        """
        判定两个回答是否为双向蕴含 (bidirectional entailment)。
        使用 LLM 进行双向蕴含判断：
        1. 判断 answer1 是否语义蕴含 answer2
        2. 判断 answer2 是否语义蕴含 answer1
        若两个方向均为 Yes，则认为两个回答语义等价，返回 True。

        Prompt 参考自 STaR (Self-Taught Reasoner) 论文中关于语义等价性检测的方法。
        """
        # 双向蕴含判断 Prompt 模板
        logger.info("双向蕴含判断开始: answer1='%s', answer2='%s'", answer1, answer2)
        entail_prompt_template = """You are a semantic equivalence judge. Given two answers, determine if the first answer semantically entails the second answer.

Answer 1: {answer1}
Answer 2: {answer2}

Does Answer 1 semantically entail Answer 2? That is, if Answer 1 is true, must Answer 2 also be true?
Reply with "Yes" or "No" only."""

        # 方向1: answer1 -> answer2
        prompt_1_to_2 = entail_prompt_template.format(answer1=answer1, answer2=answer2)
        try:
            response_1_to_2 = llm_call(prompt_1_to_2, model=MK_JUDGE_MODEL).strip().lower()
        except Exception as e:
            logger.warning("is_bientail 调用 LLM 失败 (1->2): %s", e)
            return False

        # 方向2: answer2 -> answer1
        prompt_2_to_1 = entail_prompt_template.format(answer1=answer2, answer2=answer1)
        try:
            response_2_to_1 = llm_call(prompt_2_to_1, model=MK_JUDGE_MODEL).strip().lower()
        except Exception as e:
            logger.warning("is_bientail 调用 LLM 失败 (2->1): %s", e)
            return False

        # 双向判断：两个方向都为 Yes 才认为是双向蕴含
        entail_1_to_2 = response_1_to_2.startswith("yes")
        entail_2_to_1 = response_2_to_1.startswith("yes")

        result = entail_1_to_2 and entail_2_to_1
        logger.info(
            "双向蕴含 判断结果: %s (1->2: %s, 2->1: %s)",
            result, response_1_to_2, response_2_to_1
        )
        return result
        

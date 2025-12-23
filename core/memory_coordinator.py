"""
记忆协调器 (Memory Coordinator)

职责：
- 协调三种记忆（WM, LTM, MK）的交互
- 提供统一的记忆访问接口
- 管理记忆之间的数据流转
"""

from typing import Any, Dict, List, Optional

from memory.working_memory import WorkingMemoryManager
from memory.long_term_memory import LongTermMemoryManager
from memory.meta_knowledge import MetaKnowledgeManager
from core.schemas import (
    WorkingMemory,
    LongTermMemoryEntry,
    MetaKnowledge,
    AgentOutput,
    AgentType,
)


class MemoryCoordinator:
    """
    记忆协调器
    
    统一管理三种记忆的交互：
    - Working Memory (WM): 当前任务的临时状态
    - Long-Term Memory (LTM): 知识经验存储
    - Meta-Knowledge (MK): Agent表现和决策参数
    """
    
    def __init__(
        self,
        wm_manager: Optional[WorkingMemoryManager] = None,
        ltm_manager: Optional[LongTermMemoryManager] = None,
        mk_manager: Optional[MetaKnowledgeManager] = None
    ):
        """
        初始化记忆协调器
        
        Args:
            wm_manager: 工作记忆管理器
            ltm_manager: 长期记忆管理器
            mk_manager: 元知识管理器
        """
        self.wm = wm_manager or WorkingMemoryManager()
        self.ltm = ltm_manager or LongTermMemoryManager()
        self.mk = mk_manager or MetaKnowledgeManager()
    
    # ==================== 任务初始化 ====================
    
    def init_task(
        self,
        task_id: str,
        question: str,
        question_type: str = "general"
    ) -> WorkingMemory:
        """
        初始化任务记忆
        
        Args:
            task_id: 任务ID
            question: 问题文本
            question_type: 问题类型
            
        Returns:
            初始化的WorkingMemory
        """
        # 创建工作记忆
        wm = self.wm.create(task_id, question)
        
        # 确保元知识存在
        self.mk.get_or_create(question_type)
        
        return wm
    
    # ==================== LTM查询 ====================
    
    def get_relevant_knowledge(
        self,
        question_type: str,
        error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取相关的长期记忆知识
        
        Args:
            question_type: 问题类型
            error_type: 错误类型（可选）
            
        Returns:
            包含纠正策略、思考角度、错误案例的字典
        """
        # 获取纠正策略
        correction_strategies = self.ltm.get_correction_strategies(
            topic_category=question_type,
            error_type=error_type
        )
        
        # 获取思考角度
        thinking_angles = self.ltm.get_thinking_angles(
            topic_category=question_type,
            error_type=error_type
        )
        
        # 获取相关条目的错误案例
        entries = self.ltm.search(
            topic_category=question_type,
            error_type=error_type
        )
        
        error_examples = []
        for entry in entries[:3]:  # 最多3个条目
            for ex in entry.error_examples[:2]:  # 每个条目最多2个案例
                error_examples.append({
                    "question": ex.question,
                    "wrong_answer": ex.wrong_answer,
                    "correct_answer": ex.correct_answer,
                    "error_analysis": ex.error_analysis
                })
        
        return {
            "correction_strategies": correction_strategies,
            "thinking_angles": thinking_angles,
            "error_examples": error_examples
        }
    
    # ==================== WM更新 ====================
    
    def update_student_response(
        self,
        task_id: str,
        response: str,
        confidence: float
    ) -> None:
        """
        更新Student Agent的回答
        
        Args:
            task_id: 任务ID
            response: 回答内容
            confidence: 置信度
        """
        self.wm.update_student_response(task_id, response, confidence)
    
    def add_agent_output(
        self,
        task_id: str,
        agent_id: str,
        agent_type: AgentType,
        response: str,
        confidence: float,
        reasoning: str = ""
    ) -> None:
        """
        添加Agent输出
        
        Args:
            task_id: 任务ID
            agent_id: Agent ID
            agent_type: Agent类型
            response: 回答内容
            confidence: 置信度
            reasoning: 推理过程
        """
        self.wm.add_agent_output(
            task_id=task_id,
            agent_id=agent_id,
            agent_type=agent_type,
            response=response,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def set_analysis_results(
        self,
        task_id: str,
        conflicts: List[str],
        complementary_points: List[str],
        summary_prompt: str,
        insight_evaluation: Optional[Dict[str, Any]] = None,
        quality_score: float = 0.0,
        correctness: str = "uncertain",
        recommended_action: str = "refine"
    ) -> None:
        """
        设置Insight Agent的分析结果
        
        Args:
            task_id: 任务ID
            conflicts: 冲突点列表
            complementary_points: 互补观点列表
            summary_prompt: 汇总提示词
            insight_evaluation: 完整评估结果（新增）
            quality_score: 答案质量分数（新增）
            correctness: 正确性判断（新增）
            recommended_action: 建议行动（新增）
        """
        self.wm.set_conflicts(task_id, conflicts)
        self.wm.set_complementary_points(task_id, complementary_points)
        self.wm.set_summary_prompt(task_id, summary_prompt)
        
        # 设置Insight评估结果（新增）
        memory = self.wm.get(task_id)
        if memory:
            memory.insight_evaluation = insight_evaluation or {}
            memory.quality_score = quality_score
            memory.correctness = correctness
            memory.recommended_action = recommended_action
            self.wm.update(memory)
    
    # ==================== MK查询和更新 ====================
    
    def get_recommended_agents(
        self,
        question_type: str,
        top_k: int = 3
    ) -> List[str]:
        """
        获取推荐的Agent列表
        
        Args:
            question_type: 问题类型
            top_k: 返回数量
            
        Returns:
            推荐的Agent ID列表
        """
        return self.mk.get_recommended_agents(question_type, top_k)
    
    def update_agent_performance(
        self,
        question_type: str,
        agent_id: str,
        agent_type: AgentType,
        success: bool,
        confidence: float
    ) -> None:
        """
        更新Agent表现
        
        Args:
            question_type: 问题类型
            agent_id: Agent ID
            agent_type: Agent类型
            success: 是否成功
            confidence: 置信度
        """
        self.mk.update_agent_performance(
            question_type=question_type,
            agent_id=agent_id,
            agent_type=agent_type,
            success=success,
            confidence=confidence
        )
    
    def should_stop_loop(
        self,
        question_type: str,
        current_confidence: float,
        current_round: int
    ) -> bool:
        """
        判断是否停止循环
        
        Args:
            question_type: 问题类型
            current_confidence: 当前置信度
            current_round: 当前轮次
            
        Returns:
            是否停止
        """
        return self.mk.should_stop(question_type, current_confidence, current_round)
    
    # ==================== 轮次管理 ====================
    
    def advance_round(self, task_id: str) -> int:
        """
        进入下一轮
        
        Args:
            task_id: 任务ID
            
        Returns:
            新的轮次号
        """
        return self.wm.advance_round(task_id)
    
    def get_current_round(self, task_id: str) -> int:
        """
        获取当前轮次
        
        Args:
            task_id: 任务ID
            
        Returns:
            当前轮次号
        """
        wm = self.wm.get(task_id)
        return wm.round_index if wm else 0
    
    # ==================== 反馈上下文构建 ====================
    
    def build_feedback_context(self, task_id: str) -> Dict[str, Any]:
        """
        构建反馈上下文（用于Student Agent的下一轮修正）
        
        Args:
            task_id: 任务ID
            
        Returns:
            反馈上下文字典
        """
        wm = self.wm.get(task_id)
        if wm is None:
            return {}
        
        return {
            "previous_answer": wm.student_response,
            "conflicts": wm.conflicts_detected,
            "complementary_points": wm.complementary_points,
            "insight_feedback": wm.summary_prompt,
            "round_index": wm.round_index
        }
    
    # ==================== 学习和持久化 ====================
    
    def learn_from_task(
        self,
        task_id: str,
        question_type: str,
        final_answer: str,
        correct_answer: Optional[str] = None,
        success: bool = True
    ) -> None:
        """
        从任务中学习（更新长期记忆）
        
        Args:
            task_id: 任务ID
            question_type: 问题类型
            final_answer: 最终答案
            correct_answer: 正确答案（用于评估）
            success: 是否成功
        """
        wm = self.wm.get(task_id)
        if wm is None:
            return
        
        # 如果失败，记录错误案例
        if not success and correct_answer:
            # 查找或创建LTM条目
            entries = self.ltm.find_by_topic(question_type)
            if entries:
                entry = entries[0]
            else:
                entry = LongTermMemoryEntry(
                    topic_category=question_type,
                    content_pattern="",
                    error_type="general"
                )
                self.ltm.add_entry(entry)
            
            # 添加错误案例
            self.ltm.add_error_example(
                entry_id=entry.entry_id,
                question=wm.question_text,
                wrong_answer=final_answer,
                correct_answer=correct_answer,
                error_analysis="待分析"
            )
        
        # 更新成功率
        entries = self.ltm.find_by_topic(question_type)
        for entry in entries:
            self.ltm.update_success_rate(entry.entry_id, success)
    
    def save_task_state(self, task_id: str) -> None:
        """
        保存任务状态
        
        Args:
            task_id: 任务ID
        """
        self.wm.save(task_id)
    
    def cleanup_task(self, task_id: str) -> None:
        """
        清理任务状态
        
        Args:
            task_id: 任务ID
        """
        self.wm.delete(task_id)


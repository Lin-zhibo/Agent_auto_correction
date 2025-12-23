"""
流程编排器 (Orchestrator)

职责：
- 协调整个多Agent系统的工作流程
- 管理Agent的调度和执行
- 实现图1中的逻辑拓扑
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from agent.base_agent import BaseAgent, AgentResponse
from agent.student_agent import StudentAgent
from agent.insight_agent import InsightAgent
from agent.multi_agents.authority_agent import AuthorityAgent
from agent.multi_agents.questioner_agent import QuestionerAgent
from agent.multi_agents.logic_analyst_agent import LogicAnalystAgent
from agent.multi_agents.listener_agent import ListenerAgent
from agent.multi_agents.companion_agent import CompanionAgent
from agent.multi_agents.heuristic_solver_agent import HeuristicSolverAgent
from agent.multi_agents.explanation_generator_agent import ExplanationGeneratorAgent
from agent.multi_agents.concept_analogist_agent import ConceptAnalogistAgent
from core.loop_controller import LoopController
from core.memory_coordinator import MemoryCoordinator
from memory.meta_knowledge import MetaKnowledgeManager
from core.schemas import (
    Task,
    TaskResult,
    TaskStatus,
    AgentType,
    AgentOutput,
)


class Orchestrator:
    """
    流程编排器
    
    实现图1中的多Agent协作逻辑拓扑：
    
    1. Student Agent首先作答（结合LTM）
    2. Meta-Knowledge决定选择哪些Agent参与
    3. 多Agent阶段：各专家Agent提供反馈
    4. Insight Agent汇总生成反馈
    5. 循环反思，直到满足停止条件
    
    核心流程：
    init -> student -> multi_agent -> insight -> [循环] -> complete
    """
    
    def __init__(
        self,
        memory_coordinator: Optional[MemoryCoordinator] = None,
        llm_client: Any = None
    ):
        """
        初始化编排器
        
        Args:
            memory_coordinator: 记忆协调器
            llm_client: LLM客户端
        """
        self.memory = memory_coordinator or MemoryCoordinator()
        self.llm_client = llm_client
        
        # 循环控制器
        self.loop_controller = LoopController(self.memory.mk)
        
        # Agent注册表
        self._agents: Dict[str, BaseAgent] = {}
        
        # 初始化核心Agent
        self._init_core_agents()
    
    def _init_core_agents(self) -> None:
        """初始化核心Agent"""
        # ==================== 核心 Agent ====================
        
        # Student Agent - 主答题者
        student = StudentAgent(
            agent_id="student_agent_001",
            llm_client=self.llm_client
        )
        self._agents["student"] = student
        self._agents["student_agent_001"] = student
        
        # Insight Agent - 综合评估者
        insight = InsightAgent(
            agent_id="insight_agent_001",
            llm_client=self.llm_client
        )
        self._agents["insight"] = insight
        self._agents["insight_agent_001"] = insight
        
        # ==================== Multi-Agent 专家组 ====================
        
        # 1. Authority Agent - 权威专家
        authority = AuthorityAgent(
            agent_id="authority_agent_001",
            llm_client=self.llm_client
        )
        self._agents["authority"] = authority
        self._agents["authority_agent_001"] = authority
        
        # 2. Questioner Agent - 质疑者
        questioner = QuestionerAgent(
            agent_id="questioner_agent_001",
            llm_client=self.llm_client
        )
        self._agents["questioner"] = questioner
        self._agents["questioner_agent_001"] = questioner
        
        # 3. Logic Analyst Agent - 逻辑分析员
        logic_analyst = LogicAnalystAgent(
            agent_id="logic_analyst_agent_001",
            llm_client=self.llm_client
        )
        self._agents["logic_analyst"] = logic_analyst
        self._agents["logic_analyst_agent_001"] = logic_analyst
        
        # 4. Listener Agent - 倾听者/综合者
        listener = ListenerAgent(
            agent_id="listener_agent_001",
            llm_client=self.llm_client
        )
        self._agents["listener"] = listener
        self._agents["listener_agent_001"] = listener
        
        # 5. Companion Agent - 协作伙伴
        companion = CompanionAgent(
            agent_id="companion_agent_001",
            llm_client=self.llm_client
        )
        self._agents["companion"] = companion
        self._agents["companion_agent_001"] = companion
        
        # 6. Heuristic Solver Agent - 启发式解决者
        heuristic_solver = HeuristicSolverAgent(
            agent_id="heuristic_solver_agent_001",
            llm_client=self.llm_client
        )
        self._agents["heuristic_solver"] = heuristic_solver
        self._agents["heuristic_solver_agent_001"] = heuristic_solver
        
        # 7. Explanation Generator Agent - 解释生成者
        explanation_generator = ExplanationGeneratorAgent(
            agent_id="explanation_generator_agent_001",
            llm_client=self.llm_client
        )
        self._agents["explanation_generator"] = explanation_generator
        self._agents["explanation_generator_agent_001"] = explanation_generator
        
        # 8. Concept Analogist Agent - 概念类比者
        concept_analogist = ConceptAnalogistAgent(
            agent_id="concept_analogist_agent_001",
            llm_client=self.llm_client
        )
        self._agents["concept_analogist"] = concept_analogist
        self._agents["concept_analogist_agent_001"] = concept_analogist
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """
        注册新的Agent
        
        同时用短名称和 agent_id 注册，确保能被正确查找。
        
        Args:
            name: Agent名称
            agent: Agent实例
        """
        self._agents[name] = agent
        # 同时用 agent_id 注册，支持 get_recommended_agents 返回的 agent_id
        if agent.agent_id and agent.agent_id != name:
            self._agents[agent.agent_id] = agent
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        获取Agent
        
        Args:
            name: Agent名称
            
        Returns:
            Agent实例或None
        """
        return self._agents.get(name)
    
    def _calibrate_confidence(
        self,
        raw_confidence: float,
        current_round: int
    ) -> float:
        """
        校准置信度
        
        防止LLM过度自信，确保循环反思机制能有效运作。
        
        校准规则：
        - 第1轮：置信度上限 0.70（强制降低，确保有改进空间）
        - 第2轮：置信度上限 0.85
        - 第3轮及之后：无上限
        
        Args:
            raw_confidence: LLM返回的原始置信度
            current_round: 当前轮次
            
        Returns:
            校准后的置信度
        """
        # 定义各轮次的置信度上限
        confidence_caps = {
            1: 0.70,  # 首轮强制保守
            2: 0.85,  # 第二轮略微放宽
        }
        
        cap = confidence_caps.get(current_round, 1.0)
        calibrated = min(raw_confidence, cap)
        
        # 日志记录（如果有显著调整）
        if raw_confidence > cap:
            print(f"  [置信度校准] 轮次{current_round}: {raw_confidence:.2f} → {calibrated:.2f} (上限{cap})")
        
        return calibrated
    
    def _update_error_rate(self, question_type: str, task_success: bool) -> None:
        """
        更新历史错误率
        
        使用指数移动平均（EMA）平滑更新：
        new_error_rate = α × current_error + (1-α) × old_error_rate
        
        这样可以：
        1. 让近期结果对错误率影响更大
        2. 避免单次结果造成剧烈波动
        3. 随着任务积累，error_rate 逐渐反映真实错误率
        
        Args:
            question_type: 问题类型
            task_success: 本次任务是否成功
        """
        # EMA 平滑因子（0.2 表示新结果占20%权重）
        ema_alpha = 0.2
        
        # 获取当前错误率
        mk = self.memory.mk.get_or_create(question_type)
        old_error_rate = mk.error_rate
        
        # 本次的错误值（成功=0，失败=1）
        current_error = 0.0 if task_success else 1.0
        
        # EMA 更新
        new_error_rate = ema_alpha * current_error + (1 - ema_alpha) * old_error_rate
        
        # 保存更新
        self.memory.mk.update_error_rate(question_type, new_error_rate)
        
        print(f"  [错误率更新] {question_type}: {old_error_rate:.2f} → {new_error_rate:.2f} (本次{'成功' if task_success else '失败'})")
    
    def execute(
        self,
        question: str,
        question_type: str = "general",
        task_id: Optional[str] = None,
        correct_answer: Optional[str] = None,
        verbose: bool = False
    ) -> TaskResult:
        """
        执行完整的多Agent协作流程
        
        Args:
            question: 问题文本
            question_type: 问题类型
            task_id: 任务ID（可选，自动生成）
            correct_answer: 正确答案（可选，用于学习和错误案例记录）
            
        Returns:
            TaskResult: 任务结果
        """
        start_time = datetime.now()
        
        # 生成任务ID
        task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        
        # 初始化任务
        wm = self.memory.init_task(task_id, question, question_type)
        
        # 获取LTM知识
        ltm_knowledge = self.memory.get_relevant_knowledge(question_type)
        
        # 开始循环
        loop_state = self.loop_controller.start_loop(
            task_id=task_id,
            question_type=question_type,
            initial_confidence=0.0,
            verbose=verbose
        )
        
        try:
            # ==================== 主循环：循环反思机制 ====================
            # 
            # 核心逻辑：
            # 1. 首轮必须完整执行（Student作答 → 多Agent反馈 → Insight汇总）
            # 2. 只有在第2轮及之后，才考虑基于评估结果提前停止
            # 3. 这确保Student至少有一次机会根据反馈改进答案
            #
            while True:
                current_round = self.loop_controller.get_state(task_id).current_round
                
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"📍 第 {current_round} 轮开始")
                    print(f"{'='*60}")
                
                # 执行一轮循环
                self._execute_round(
                    task_id=task_id,
                    question=question,
                    question_type=question_type,
                    ltm_knowledge=ltm_knowledge,
                    verbose=verbose
                )
                
                # 更新工作记忆引用
                wm = self.memory.wm.get(task_id)
                
                # ========== 停止条件判断 ==========
                
                if verbose:
                    print(f"\n📊 第 {current_round} 轮评估结果:")
                    print(f"   • Student 置信度: {wm.student_confidence:.2f}")
                    print(f"   • 质量分数: {wm.quality_score:.2f}")
                    print(f"   • 正确性: {wm.correctness}")
                    print(f"   • 建议行动: {wm.recommended_action}")
                
                # 条件1: 达到最大轮次，强制停止
                loop_state = self.loop_controller.get_state(task_id)
                if current_round >= loop_state.max_rounds:
                    stop_reason = f"达到最大循环次数 ({current_round}/{loop_state.max_rounds})"
                    self.loop_controller.force_stop(task_id, stop_reason)
                    if verbose:
                        print(f"\n🛑 停止: {stop_reason}")
                    break
                
                # 条件2: 至少执行2轮后，才考虑基于质量的提前停止
                # 这确保Student有机会根据第一轮的反馈进行改进
                if current_round >= 2:
                    # Insight Agent 明确建议接受
                    if wm.recommended_action == "accept":
                        stop_reason = f"Insight Agent 建议接受答案 (轮次:{current_round}, 质量:{wm.quality_score:.2f})"
                        self.loop_controller.force_stop(task_id, stop_reason)
                        if verbose:
                            print(f"\n✅ 停止: {stop_reason}")
                        break
                    
                    # 答案质量高且正确
                    if wm.quality_score >= 0.85 and wm.correctness == "correct":
                        stop_reason = f"答案质量达标 (轮次:{current_round}, 分数:{wm.quality_score:.2f})"
                        self.loop_controller.force_stop(task_id, stop_reason)
                        if verbose:
                            print(f"\n✅ 停止: {stop_reason}")
                        break
                    
                    # 置信度达到高阈值（提高到0.9，避免过早停止）
                    if wm.student_confidence >= 0.9:
                        stop_reason = f"置信度达到阈值 (轮次:{current_round}, 置信度:{wm.student_confidence:.2f})"
                        self.loop_controller.force_stop(task_id, stop_reason)
                        if verbose:
                            print(f"\n✅ 停止: {stop_reason}")
                        break
                
                # 条件3: 如果Insight建议reject（答案有严重问题），继续循环
                # 条件4: 如果recommended_action是refine，继续循环改进
                
                if verbose:
                    print(f"\n➡️  进入下一轮 (当前: {wm.recommended_action})")
                
                # 进入下一轮
                self.loop_controller.advance_round(task_id)
                self.memory.advance_round(task_id)
            
            # 构建结果
            loop_state = self.loop_controller.get_state(task_id)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = TaskResult(
                task_id=task_id,
                final_answer=wm.student_response,
                confidence=wm.student_confidence,
                total_rounds=wm.round_index,
                success=True,
                working_memory=wm,
                execution_time=execution_time,
                metadata={
                    "stop_reason": loop_state.stop_reason if loop_state else "unknown",
                    "question_type": question_type,
                    # Insight Agent 评估结果（新增）
                    "quality_score": wm.quality_score,
                    "correctness": wm.correctness,
                    "recommended_action": wm.recommended_action,
                    "insight_evaluation": wm.insight_evaluation
                }
            )
            
            # ==================== 学习阶段：更新 LTM 和 MK ====================
            
            # 基于 Insight Agent 的评估判断任务是否成功
            # recommended_action == "accept" 表示答案质量达标
            task_success = (
                wm.recommended_action == "accept" or 
                wm.quality_score >= 0.8 or
                wm.correctness == "correct"
            )
            
            # 如果提供了外部正确答案，优先使用外部答案进行校验
            if correct_answer:
                # 结合 Insight Agent 的评估和外部答案
                external_check = correct_answer.lower() in wm.student_response.lower()
                task_success = task_success and external_check
            
            # 更新各Agent的表现到 Meta-Knowledge
            for output in wm.agent_outputs:
                self.memory.update_agent_performance(
                    question_type=question_type,
                    agent_id=output.agent_id,
                    agent_type=output.agent_type,
                    success=task_success,
                    confidence=output.confidence
                )
            
            # 从任务中学习，更新 Long-Term Memory
            self.memory.learn_from_task(
                task_id=task_id,
                question_type=question_type,
                final_answer=wm.student_response,
                correct_answer=correct_answer,
                success=task_success
            )
            
            # ========== 更新 error_rate（历史错误率） ==========
            # 根据任务成功/失败，动态更新该问题类型的错误率
            # 使用指数移动平均（EMA）平滑更新，避免单次结果影响过大
            self._update_error_rate(question_type, task_success)
            
            # 保存任务状态
            self.memory.save_task_state(task_id)
            
            return result
            
        except Exception as e:
            # 错误处理
            
            # 更新失败情况下的Agent表现到 MK
            if wm and wm.agent_outputs:
                for output in wm.agent_outputs:
                    self.memory.update_agent_performance(
                        question_type=question_type,
                        agent_id=output.agent_id,
                        agent_type=output.agent_type,
                        success=False,  # 任务失败
                        confidence=output.confidence
                    )
            
            return TaskResult(
                task_id=task_id,
                final_answer="",
                confidence=0.0,
                total_rounds=wm.round_index if wm else 0,
                success=False,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": str(e)}
            )
        finally:
            # 清理循环状态
            self.loop_controller.cleanup(task_id)
    
    def _execute_round(
        self,
        task_id: str,
        question: str,
        question_type: str,
        ltm_knowledge: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """
        执行一轮循环
        
        Args:
            task_id: 任务ID
            question: 问题
            question_type: 问题类型
            ltm_knowledge: LTM知识
            verbose: 是否输出详细信息
        """
        wm = self.memory.wm.get(task_id)
        current_round = wm.round_index
        
        # ==================== 阶段1: Student Agent作答 ====================
        self.memory.wm.set_phase(task_id, "student")
        
        if verbose:
            print(f"\n🎓 [Student Agent] 作答中...")
        
        # 构建Student Agent上下文
        student_context = {
            "ltm_knowledge": ltm_knowledge,
            "round_index": current_round
        }
        
        # 如果是第二轮及之后，添加反馈上下文
        if current_round > 1:
            feedback_context = self.memory.build_feedback_context(task_id)
            student_context["previous_feedback"] = feedback_context
        
        # 执行Student Agent
        student_agent = self._agents.get("student")
        if student_agent:
            student_response = student_agent.execute(question, student_context)
            
            # ========== 置信度校准机制 ==========
            # 防止 LLM 过度自信，确保循环反思机制能生效
            raw_confidence = student_response.confidence
            calibrated_confidence = self._calibrate_confidence(
                raw_confidence=raw_confidence,
                current_round=current_round
            )
            
            self.memory.update_student_response(
                task_id=task_id,
                response=student_response.content,
                confidence=calibrated_confidence
            )
            
            if verbose:
                answer_preview = student_response.content[:100] + "..." if len(student_response.content) > 100 else student_response.content
                print(f"   答案: {answer_preview}")
                print(f"   置信度: {calibrated_confidence:.2f} (原始: {raw_confidence:.2f})")
        
        # ==================== 阶段2: 多Agent反馈 ====================
        self.memory.wm.set_phase(task_id, "multi_agent")
        
        # 获取推荐的Agent列表
        recommended_agents = self.memory.get_recommended_agents(question_type, top_k=3)
        
        # 如果没有推荐，使用默认Agent
        if not recommended_agents:
            # 默认使用 3 个核心专家 Agent
            recommended_agents = ["authority", "questioner", "logic_analyst"]
        
        if verbose:
            # 获取实际可用的 Agent 名称
            available_agents = [name for name in recommended_agents if self._agents.get(name)]
            print(f"\n👥 [Multi-Agent] 专家评估 (参与: {len(available_agents)} 个 Agent)")
        
        # 获取当前Student回答
        wm = self.memory.wm.get(task_id)
        student_response_text = wm.student_response
        student_confidence = wm.student_confidence
        
        # 执行各专家Agent
        agent_outputs: List[AgentOutput] = []
        for agent_name in recommended_agents:
            agent = self._agents.get(agent_name)
            if agent is None:
                continue
            
            agent_context = {
                "student_response": student_response_text,
                "student_confidence": student_confidence,
                "round_index": current_round
            }
            
            try:
                response = agent.execute(question, agent_context)
                output = agent.to_output(response)
                agent_outputs.append(output)
                
                # 添加到工作记忆
                self.memory.add_agent_output(
                    task_id=task_id,
                    agent_id=agent.agent_id,
                    agent_type=agent.agent_type,
                    response=response.content,
                    confidence=response.confidence,
                    reasoning=response.reasoning
                )
                
                if verbose:
                    # 提取关键评估信息
                    stance = response.metadata.get("stance", "N/A")
                    accuracy = response.metadata.get("professional_assessment", {}).get("accuracy_score", "N/A")
                    validity = response.metadata.get("logical_validity", {}).get("validity_score", "N/A")
                    content_preview = response.content[:80] + "..." if len(response.content) > 80 else response.content
                    
                    print(f"\n   📋 [{agent_name.upper()}]")
                    print(f"      观点: {content_preview}")
                    print(f"      置信度: {response.confidence:.2f}", end="")
                    if stance != "N/A":
                        print(f" | 立场: {stance}", end="")
                    if accuracy != "N/A":
                        print(f" | 准确性: {accuracy}", end="")
                    if validity != "N/A":
                        print(f" | 逻辑性: {validity}", end="")
                    print()
                    
            except Exception as e:
                print(f"Agent {agent_name} 执行失败: {e}")
        
        # ==================== 阶段3: Insight Agent汇总和评估 ====================
        self.memory.wm.set_phase(task_id, "insight")
        
        if verbose:
            if not agent_outputs:
                print(f"\n⚠️  警告: 没有专家Agent输出！可能是Agent注册问题。")
            print(f"\n🔍 [Insight Agent] 综合评估中...")
        
        insight_agent = self._agents.get("insight")
        if insight_agent and agent_outputs:
            insight_context = {
                "student_response": student_response_text,
                "student_confidence": student_confidence,
                "agent_outputs": agent_outputs,
                "round_index": current_round  # 传入当前轮次，影响Insight的评估决策
            }
            
            insight_response = insight_agent.execute(question, insight_context)
            
            # 提取分析结果
            conflicts = insight_response.metadata.get("conflicts_detected", [])
            complementary = insight_response.metadata.get("complementary_insights", [])
            
            # 提取答案评估结果（新增）
            answer_evaluation = insight_response.metadata.get("answer_evaluation", {})
            quality_score = insight_response.metadata.get("quality_score", 0.5)
            correctness = insight_response.metadata.get("correctness", "uncertain")
            recommended_action = insight_response.metadata.get("recommended_action", "refine")
            
            # 更新工作记忆（包含评估结果）
            self.memory.set_analysis_results(
                task_id=task_id,
                conflicts=conflicts,
                complementary_points=complementary,
                summary_prompt=insight_response.content,
                insight_evaluation=answer_evaluation,
                quality_score=quality_score,
                correctness=correctness,
                recommended_action=recommended_action
            )
            
            if verbose:
                print(f"   质量分数: {quality_score:.2f}")
                print(f"   正确性: {correctness}")
                print(f"   建议行动: {recommended_action}")
                if conflicts:
                    print(f"   冲突点: {', '.join(conflicts[:3])}" + ("..." if len(conflicts) > 3 else ""))
                if complementary:
                    print(f"   互补观点: {', '.join(complementary[:3])}" + ("..." if len(complementary) > 3 else ""))
    
    def execute_single_round(
        self,
        question: str,
        question_type: str = "general"
    ) -> Dict[str, Any]:
        """
        执行单轮（不循环），用于测试或简单场景
        
        Args:
            question: 问题
            question_type: 问题类型
            
        Returns:
            单轮结果
        """
        task_id = f"single_{uuid.uuid4().hex[:8]}"
        
        # 初始化
        self.memory.init_task(task_id, question, question_type)
        ltm_knowledge = self.memory.get_relevant_knowledge(question_type)
        
        # 执行一轮
        self._execute_round(task_id, question, question_type, ltm_knowledge)
        
        # 获取结果
        wm = self.memory.wm.get(task_id)
        
        result = {
            "task_id": task_id,
            "question": question,
            "answer": wm.student_response,
            "confidence": wm.student_confidence,
            "conflicts": wm.conflicts_detected,
            "complementary_points": wm.complementary_points,
            "summary": wm.summary_prompt,
            "agent_outputs": [
                {
                    "agent_id": o.agent_id,
                    "type": o.agent_type.value,
                    "response": o.response,
                    "confidence": o.confidence
                }
                for o in wm.agent_outputs
            ]
        }
        
        # 清理
        self.memory.cleanup_task(task_id)
        
        return result


"""
数据结构体定义模块

本模块定义了多Agent自动纠错系统的核心数据结构，包括：
- Working Memory (工作记忆): 存储当前任务的临时状态
- Long-Term Memory (长期记忆): 存储知识经验和错误模式
- Meta-Knowledge (元知识): 存储Agent表现和决策参数

结构体优化说明：
1. 命名规范: 遵循PEP8，使用snake_case命名
2. 字段合理性: 剔除冗余字段，补充必要的元数据字段
3. 层级划分: 将复杂结构拆分为独立的子结构
4. 扩展性: 预留多Agent扩展接口和配置字段
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ==================== 枚举类型定义 ====================

class AgentType(Enum):
    """
    Agent类型枚举
    
    优化说明：
    - 新增枚举类型，便于类型检查和IDE提示
    - 涵盖图1中所有Agent角色
    """
    STUDENT = "student"           # 学生Agent - 主答题者
    AUTHORITY = "authority"       # 权威者 - 提供权威观点
    QUESTIONER = "questioner"     # 质疑者 - 挑战现有答案
    LOGIC_ANALYST = "logic_analyst"  # 逻辑分析员 - 逻辑推理
    LISTENER = "listener"         # 倾听者 - 综合意见
    COMPANION = "companion"       # 同伴 - 协作支持
    HEURISTIC_SOLVER = "heuristic_solver"  # 启发式解决者
    EXPLANATION_GENERATOR = "explanation_generator"  # 解释生成者
    CONCEPT_ANALOGIST = "concept_analogist"  # 概念类比者
    INSIGHT = "insight"           # 洞察Agent - 汇总反馈


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# ==================== 基础子结构体 ====================

@dataclass
class AgentOutput:
    """
    单个Agent的输出记录
    
    优化说明：
    - 原始设计中agent_outputs为简单列表，优化为独立结构体
    - 新增agent_type便于分类统计
    - 新增reasoning字段记录推理过程
    - 新增metadata字段支持扩展信息
    """
    agent_id: str                          # Agent唯一标识
    agent_type: AgentType                  # Agent类型
    response: str                          # Agent的回答内容
    confidence: float                      # 置信度 (0-1)
    reasoning: str = ""                    # 推理过程说明
    timestamp: datetime = field(default_factory=datetime.now)  # 输出时间
    metadata: Dict[str, Any] = field(default_factory=dict)     # 扩展元数据
    
    def __post_init__(self):
        """验证置信度范围"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence必须在0-1之间，当前值: {self.confidence}")


@dataclass
class ErrorExample:
    """
    错误案例记录
    
    优化说明：
    - 从Long-Term Memory中的error_examples字段拆分为独立结构
    - 新增example_id便于索引和检索
    - 新增error_analysis记录错误分析
    """
    example_id: str                        # 案例唯一标识
    question: str                          # 原始问题
    wrong_answer: str                      # 错误答案
    correct_answer: str                    # 正确答案
    error_analysis: str                    # 错误原因分析
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentPerformanceRecord:
    """
    Agent表现记录
    
    优化说明：
    - 从Meta-Knowledge中的agent_performance_log拆分为独立结构
    - 新增统计字段：total_tasks, successful_tasks
    - 新增confidence_variance记录置信度波动
    """
    agent_id: str                          # Agent标识
    agent_type: AgentType                  # Agent类型
    accuracy: float = 0.0                  # 正确率
    confidence_variance: float = 0.0       # 置信度方差（波动）
    total_tasks: int = 0                   # 总任务数
    successful_tasks: int = 0              # 成功任务数
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_performance(self, success: bool, confidence: float):
        """更新Agent表现统计"""
        self.total_tasks += 1
        if success:
            self.successful_tasks += 1
        self.accuracy = self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0
        self.last_updated = datetime.now()


# ==================== 核心记忆结构体 ====================

@dataclass
class WorkingMemory:
    """
    工作记忆 (Working Memory)
    
    用途：存储当前任务的临时状态和各Agent产生的回答
    
    优化说明：
    - 保留原始核心字段：task_id, question_text, round_index等
    - 新增created_at/updated_at时间戳便于追踪
    - 新增current_phase字段标记当前处理阶段
    - agent_outputs改用List[AgentOutput]结构化存储
    - 新增history字段记录历史轮次摘要
    """
    # 基础信息
    task_id: str                           # 当前问题的唯一编号
    question_text: str                     # 当前处理的问题
    
    # 循环状态
    round_index: int = 1                   # 当前反思循环次数（从1起）
    current_phase: str = "init"            # 当前阶段: init/student/multi_agent/insight/complete
    
    # Agent输出
    agent_outputs: List[AgentOutput] = field(default_factory=list)  # 各Agent输出记录
    
    # 分析结果
    conflicts_detected: List[str] = field(default_factory=list)     # 检测到的冲突点
    complementary_points: List[str] = field(default_factory=list)   # 互补观点
    summary_prompt: str = ""               # Meta-Synthesizer生成的引导prompt
    
    # Insight Agent评估结果（新增）
    insight_evaluation: Dict[str, Any] = field(default_factory=dict)  # 完整评估结果
    quality_score: float = 0.0             # 答案质量分数 (0-1)
    correctness: str = "uncertain"         # 正确性: correct/partially_correct/incorrect/uncertain
    recommended_action: str = "refine"     # 建议行动: refine/accept/escalate
    
    # Student Agent状态
    student_response: str = ""             # Student Agent本轮回答
    student_confidence: float = 0.0        # Student Agent信心值 (0-1)
    
    # 历史记录
    round_history: List[Dict[str, Any]] = field(default_factory=list)  # 历史轮次摘要
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_agent_output(self, output: AgentOutput):
        """添加Agent输出并更新时间戳"""
        self.agent_outputs.append(output)
        self.updated_at = datetime.now()
    
    def advance_round(self):
        """进入下一轮循环，保存当前轮次摘要"""
        round_summary = {
            "round": self.round_index,
            "student_response": self.student_response,
            "student_confidence": self.student_confidence,
            "agent_count": len(self.agent_outputs),
            "conflicts": self.conflicts_detected.copy(),
            "timestamp": datetime.now().isoformat()
        }
        self.round_history.append(round_summary)
        self.round_index += 1
        self.agent_outputs.clear()
        self.conflicts_detected.clear()
        self.complementary_points.clear()
        self.updated_at = datetime.now()
    
    def get_current_round_outputs(self) -> List[AgentOutput]:
        """获取当前轮次的所有Agent输出"""
        return self.agent_outputs.copy()


@dataclass 
class LongTermMemoryEntry:
    """
    长期记忆条目 (Long-Term Memory Entry)
    
    用途：存储知识经验、错误类型、正确处理方式
    
    优化说明：
    - 保留原始核心字段：topic_category, error_type, correction_strategies等
    - error_examples改用List[ErrorExample]结构化存储
    - 新增entry_id便于索引
    - 新增usage_count记录使用频率
    - thinking_angles保持列表形式，便于扩展
    """
    # 标识信息
    entry_id: str = ""                     # 条目唯一标识
    topic_category: str = ""               # 问题所属主题
    content_pattern: str = ""              # 问题内容类型模板
    
    # 错误相关
    error_type: str = ""                   # 错误类别标记
    error_examples: List[ErrorExample] = field(default_factory=list)  # 历史错误案例
    
    # 纠正策略
    correction_strategies: List[str] = field(default_factory=list)    # 正确处理方法
    thinking_angles: List[str] = field(default_factory=list)          # 有效思考角度
    
    # 统计信息
    success_rate: float = 0.0              # 此主题的总体正确率
    usage_count: int = 0                   # 使用次数
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    update_timestamp: datetime = field(default_factory=datetime.now)
    
    def add_error_example(self, example: ErrorExample):
        """添加错误案例"""
        self.error_examples.append(example)
        self.update_timestamp = datetime.now()
    
    def update_success_rate(self, success: bool):
        """更新成功率"""
        self.usage_count += 1
        # 使用滑动平均更新成功率
        alpha = 0.1  # 平滑因子
        success_value = 1.0 if success else 0.0
        self.success_rate = alpha * success_value + (1 - alpha) * self.success_rate
        self.update_timestamp = datetime.now()


@dataclass
class MetaKnowledge:
    """
    元知识 (Meta-Knowledge)
    
    用途：存储Agent表现统计和循环决策参数
    
    优化说明：
    - 保留原始核心字段：question_type, error_rate, avg_confidence等
    - agent_performance_log改用Dict[str, AgentPerformanceRecord]结构化存储
    - 新增循环控制参数：max_rounds, confidence_threshold, alpha, beta
    - 公式实现：N = min(N_max, [αE(1-C)^β × N_max])
    - 停止条件：C ≥ τ_c (confidence_threshold)
    """
    # 问题分类
    question_type: str = ""                # 问题类别标签
    
    # 历史统计
    error_rate: float = 0.0                # 历史错误率 (E)
    avg_confidence: float = 0.0            # 过去信心均值
    
    # Agent表现记录
    agent_performance_log: Dict[str, AgentPerformanceRecord] = field(default_factory=dict)
    
    # Agent推荐
    agent_recommendation_list: List[str] = field(default_factory=list)  # 推荐Agent列表
    exploration_prob: float = 0.1          # 随机探索比例 (ε-greedy)
    
    # 循环控制参数
    max_rounds: int = 10                   # 最大循环次数 (N_max)
    confidence_threshold: float = 0.8      # 信心阈值 (τ_c)
    alpha: float = 1.0                     # 循环次数计算参数 α
    beta: float = 1.0                      # 循环次数计算参数 β
    
    # 时间戳
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_max_rounds(self, current_confidence: float, verbose: bool = False) -> int:
        """
        计算推荐的最大循环次数
        
        公式: N = min(N_max, ceil(α × E × (1-C)^β × N_max))
        
        特殊情况：
        - 当 error_rate = 0（无历史数据）时，使用默认的 max_rounds
        - 确保最少执行 2 轮（循环反思机制的核心要求）
        
        Args:
            current_confidence: 当前置信度 (C)
            verbose: 是否输出调试信息
            
        Returns:
            推荐的循环次数
        """
        import math
        
        E = self.error_rate
        C = current_confidence
        
        if verbose:
            print(f"  [calculate_max_rounds] E={E:.2f}, C={C:.2f}, max_rounds={self.max_rounds}")
        
        # 特殊情况：当没有历史错误率数据时，使用默认最大轮次
        # 这确保新系统或新问题类型不会因为 E=0 而只执行2轮
        if E <= 0.01:  # 近似为0
            if verbose:
                print(f"  [calculate_max_rounds] E≈0，使用默认 max_rounds={self.max_rounds}")
            return self.max_rounds
        
        # 计算动态循环次数
        dynamic_rounds = self.alpha * E * ((1 - C) ** self.beta) * self.max_rounds
        recommended_rounds = min(self.max_rounds, math.ceil(dynamic_rounds))
        result = max(2, recommended_rounds)
        
        if verbose:
            print(f"  [calculate_max_rounds] dynamic={dynamic_rounds:.2f}, recommended={recommended_rounds}, result={result}")
        
        # 确保至少执行2轮（循环反思机制的核心要求）
        # 第1轮：Student作答 + 专家反馈
        # 第2轮：Student根据反馈改进
        return result
    
    def should_stop(self, current_confidence: float, current_round: int) -> bool:
        """
        判断是否应该停止循环
        
        停止条件：
        1. 当前置信度 >= 阈值 (C >= τ_c)
        2. 已达到最大循环次数
        
        Args:
            current_confidence: 当前置信度
            current_round: 当前轮次
            
        Returns:
            是否应该停止
        """
        if current_confidence >= self.confidence_threshold:
            return True
        if current_round >= self.max_rounds:
            return True
        return False
    
    def update_agent_performance(self, agent_id: str, agent_type: AgentType, 
                                  success: bool, confidence: float):
        """更新Agent表现记录"""
        if agent_id not in self.agent_performance_log:
            self.agent_performance_log[agent_id] = AgentPerformanceRecord(
                agent_id=agent_id,
                agent_type=agent_type
            )
        self.agent_performance_log[agent_id].update_performance(success, confidence)
        self.last_updated = datetime.now()
    
    def get_recommended_agents(self, top_k: int = 3) -> List[str]:
        """
        获取推荐的Agent列表（基于历史表现）
        
        Args:
            top_k: 返回前k个Agent
            
        Returns:
            推荐的Agent ID列表
        """
        import random
        
        # ε-greedy策略：以exploration_prob概率随机探索
        if random.random() < self.exploration_prob:
            all_agents = list(self.agent_performance_log.keys())
            if len(all_agents) <= top_k:
                return all_agents
            return random.sample(all_agents, top_k)
        
        # 按正确率排序
        sorted_agents = sorted(
            self.agent_performance_log.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        )
        
        return [agent_id for agent_id, _ in sorted_agents[:top_k]]


# ==================== 任务和配置结构体 ====================

@dataclass
class AgentConfig:
    """
    Agent配置
    
    用途：定义单个Agent的配置参数
    """
    agent_id: str                          # Agent唯一标识
    agent_type: AgentType                  # Agent类型
    name: str                              # Agent显示名称
    description: str = ""                  # Agent描述
    prompt_template: str = ""              # 提示词模板
    model_name: str = "gpt-4"              # 使用的模型
    temperature: float = 0.7               # 生成温度
    max_tokens: int = 2000                 # 最大token数
    enabled: bool = True                   # 是否启用
    priority: int = 0                      # 优先级（数值越大优先级越高）
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """
    任务定义
    
    用途：封装待处理的任务信息
    """
    task_id: str                           # 任务唯一标识
    question: str                          # 问题内容
    context: str = ""                      # 上下文信息
    expected_answer: Optional[str] = None  # 期望答案（用于评估）
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TaskResult:
    """
    任务结果
    
    用途：封装任务执行结果
    """
    task_id: str                           # 任务ID
    final_answer: str                      # 最终答案
    confidence: float                      # 最终置信度
    total_rounds: int                      # 总循环次数
    success: bool = False                  # 是否成功
    working_memory: Optional[WorkingMemory] = None  # 工作记忆快照
    execution_time: float = 0.0            # 执行时间（秒）
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)


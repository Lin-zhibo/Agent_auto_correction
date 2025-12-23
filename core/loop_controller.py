"""
循环控制器 (Loop Controller)

职责：
- 控制反思循环的执行
- 判断停止条件
- 动态调整循环次数
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from memory.meta_knowledge import MetaKnowledgeManager


@dataclass
class LoopState:
    """循环状态"""
    current_round: int = 1
    max_rounds: int = 10
    should_continue: bool = True
    stop_reason: str = ""
    started_at: datetime = None
    
    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now()


class LoopController:
    """
    循环控制器
    
    基于Meta-Knowledge实现动态循环控制：
    - 根据置信度和错误率计算推荐循环次数
    - 判断是否满足停止条件
    - 支持早停和强制停止
    
    核心公式：
    N = min(N_max, ceil(α × E × (1-C)^β × N_max))
    
    停止条件：
    1. C >= τ_c (置信度达到阈值)
    2. 达到最大循环次数
    """
    
    def __init__(self, meta_knowledge_manager: MetaKnowledgeManager):
        """
        初始化循环控制器
        
        Args:
            meta_knowledge_manager: 元知识管理器
        """
        self.mk_manager = meta_knowledge_manager
        self._states: Dict[str, LoopState] = {}
    
    def start_loop(
        self,
        task_id: str,
        question_type: str,
        initial_confidence: float = 0.0,
        verbose: bool = False
    ) -> LoopState:
        """
        开始新的循环
        
        Args:
            task_id: 任务ID
            question_type: 问题类型
            initial_confidence: 初始置信度
            verbose: 是否输出调试信息
            
        Returns:
            LoopState: 循环状态
        """
        # 计算推荐的最大循环次数
        max_rounds = self.mk_manager.calculate_max_rounds(
            question_type=question_type,
            current_confidence=initial_confidence,
            verbose=verbose
        )
        
        state = LoopState(
            current_round=1,
            max_rounds=max_rounds,
            should_continue=True,
            started_at=datetime.now()
        )
        
        self._states[task_id] = state
        return state
    
    def check_continue(
        self,
        task_id: str,
        question_type: str,
        current_confidence: float
    ) -> bool:
        """
        检查是否应该继续循环
        
        Args:
            task_id: 任务ID
            question_type: 问题类型
            current_confidence: 当前置信度
            
        Returns:
            是否继续
        """
        state = self._states.get(task_id)
        if state is None:
            return False
        
        # 检查停止条件
        should_stop = self.mk_manager.should_stop(
            question_type=question_type,
            current_confidence=current_confidence,
            current_round=state.current_round
        )
        
        if should_stop:
            # 确定停止原因
            mk = self.mk_manager.get_or_create(question_type)
            if current_confidence >= mk.confidence_threshold:
                state.stop_reason = f"置信度达到阈值 ({current_confidence:.2f} >= {mk.confidence_threshold})"
            else:
                state.stop_reason = f"达到最大循环次数 ({state.current_round} >= {state.max_rounds})"
            state.should_continue = False
            return False
        
        return True
    
    def advance_round(self, task_id: str) -> int:
        """
        进入下一轮
        
        Args:
            task_id: 任务ID
            
        Returns:
            新的轮次号
        """
        state = self._states.get(task_id)
        if state is None:
            raise ValueError(f"循环状态不存在: {task_id}")
        
        state.current_round += 1
        return state.current_round
    
    def force_stop(self, task_id: str, reason: str = "手动停止") -> None:
        """
        强制停止循环
        
        Args:
            task_id: 任务ID
            reason: 停止原因
        """
        state = self._states.get(task_id)
        if state:
            state.should_continue = False
            state.stop_reason = reason
    
    def get_state(self, task_id: str) -> Optional[LoopState]:
        """
        获取循环状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            LoopState或None
        """
        return self._states.get(task_id)
    
    def get_progress(self, task_id: str) -> Dict[str, Any]:
        """
        获取循环进度
        
        Args:
            task_id: 任务ID
            
        Returns:
            进度信息字典
        """
        state = self._states.get(task_id)
        if state is None:
            return {"error": "循环状态不存在"}
        
        elapsed = (datetime.now() - state.started_at).total_seconds()
        
        return {
            "current_round": state.current_round,
            "max_rounds": state.max_rounds,
            "progress_percentage": (state.current_round / state.max_rounds) * 100,
            "should_continue": state.should_continue,
            "stop_reason": state.stop_reason,
            "elapsed_seconds": elapsed
        }
    
    def cleanup(self, task_id: str) -> None:
        """
        清理循环状态
        
        Args:
            task_id: 任务ID
        """
        if task_id in self._states:
            del self._states[task_id]


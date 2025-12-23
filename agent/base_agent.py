"""
Agent基类 (Base Agent)

定义所有Agent的通用接口和基础功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.schemas import AgentConfig, AgentOutput, AgentType


@dataclass
class AgentResponse:
    """Agent响应结构"""
    content: str                           # 响应内容
    confidence: float                      # 置信度 (0-1)
    reasoning: str = ""                    # 推理过程
    metadata: Dict[str, Any] = None        # 扩展元数据
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence必须在0-1之间: {self.confidence}")


class BaseAgent(ABC):
    """
    Agent基类
    
    所有具体Agent实现需要继承此类并实现以下方法:
    - execute(): 执行Agent逻辑
    - _build_prompt(): 构建提示词
    
    Attributes:
        agent_id: Agent唯一标识
        agent_type: Agent类型
        config: Agent配置
        llm_client: LLM客户端（由子类注入）
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        config: Optional[AgentConfig] = None,
        llm_client: Any = None
    ):
        """
        初始化Agent
        
        Args:
            agent_id: Agent唯一标识
            agent_type: Agent类型
            config: Agent配置
            llm_client: LLM客户端
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or self._default_config()
        self.llm_client = llm_client
        
        # 执行历史
        self._execution_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def execute(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        执行Agent逻辑
        
        Args:
            question: 输入问题
            context: 上下文信息（可包含其他Agent的输出、历史记录等）
            
        Returns:
            AgentResponse: Agent响应
        """
        pass
    
    @abstractmethod
    def _build_prompt(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        构建提示词
        
        Args:
            question: 输入问题
            context: 上下文信息
            
        Returns:
            构建好的提示词
        """
        pass
    
    def _default_config(self) -> AgentConfig:
        """返回默认配置"""
        return AgentConfig(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            name=f"{self.agent_type.value}_agent",
            description="",
            prompt_template="",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=2000,
            enabled=True,
            priority=0
        )
    
    def to_output(self, response: AgentResponse) -> AgentOutput:
        """
        将AgentResponse转换为AgentOutput
        
        Args:
            response: Agent响应
            
        Returns:
            AgentOutput实例
        """
        return AgentOutput(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            response=response.content,
            confidence=response.confidence,
            reasoning=response.reasoning,
            timestamp=datetime.now(),
            metadata=response.metadata
        )
    
    def _call_llm(self, prompt: str) -> str:
        """
        调用LLM
        
        Args:
            prompt: 提示词
            
        Returns:
            LLM响应文本
        """
        if self.llm_client is None:
            raise RuntimeError("LLM客户端未初始化")
        
        return self.llm_client.chat(
            prompt=prompt,
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
    
    def _log_execution(
        self,
        question: str,
        response: AgentResponse,
        prompt: str
    ) -> None:
        """记录执行历史"""
        self._execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "prompt": prompt,
            "response": response.content,
            "confidence": response.confidence
        })
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self._execution_history.copy()
    
    def clear_history(self) -> None:
        """清空执行历史"""
        self._execution_history.clear()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type.value})"


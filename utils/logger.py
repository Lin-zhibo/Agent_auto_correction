"""
日志记录器 (Logger)

提供统一的日志记录功能
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """
    日志记录器
    
    支持同时输出到控制台和文件
    """
    
    _instances = {}
    
    def __new__(cls, name: str = "agent_system", log_dir: str = "log"):
        """单例模式"""
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]
    
    def __init__(self, name: str = "agent_system", log_dir: str = "log"):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志目录
        """
        if hasattr(self, '_initialized'):
            return
        
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            self._setup_handlers()
        
        self._initialized = True
    
    def _setup_handlers(self) -> None:
        """设置日志处理器"""
        # 日志格式
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """记录调试信息"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        """记录一般信息"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """记录警告信息"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        """记录错误信息"""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs) -> None:
        """记录严重错误"""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """格式化日志消息"""
        if kwargs:
            extra_info = ' '.join(f'{k}={v}' for k, v in kwargs.items())
            return f"{message} | {extra_info}"
        return message
    
    def log_agent_execution(
        self,
        agent_id: str,
        action: str,
        details: Optional[dict] = None
    ) -> None:
        """
        记录Agent执行日志
        
        Args:
            agent_id: Agent ID
            action: 执行动作
            details: 详细信息
        """
        message = f"[Agent: {agent_id}] {action}"
        if details:
            message += f" | Details: {details}"
        self.info(message)
    
    def log_task_progress(
        self,
        task_id: str,
        phase: str,
        round_index: int,
        confidence: float
    ) -> None:
        """
        记录任务进度
        
        Args:
            task_id: 任务ID
            phase: 当前阶段
            round_index: 轮次
            confidence: 置信度
        """
        self.info(
            f"[Task: {task_id}] Phase: {phase}, Round: {round_index}, "
            f"Confidence: {confidence:.2f}"
        )


def get_logger(name: str = "agent_system") -> Logger:
    """
    获取日志记录器实例
    
    Args:
        name: 日志记录器名称
        
    Returns:
        Logger实例
    """
    return Logger(name)


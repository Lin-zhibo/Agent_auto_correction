"""
配置管理 (Settings)

集中管理系统配置
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LLMSettings:
    """LLM相关配置"""
    api_key: str = ""
    base_url: str = "https://api.gpt.ge/v1/"
    model_name: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60
    default_headers: Dict[str, str] = field(default_factory=lambda: {"x-foo": "true"})


@dataclass
class LoopSettings:
    """循环控制配置"""
    max_rounds: int = 10
    confidence_threshold: float = 0.8
    alpha: float = 1.0
    beta: float = 1.0
    exploration_prob: float = 0.1


@dataclass
class StorageSettings:
    """存储配置"""
    working_memory_dir: str = "memory/storage/working"
    long_term_memory_dir: str = "memory/storage/long_term"
    meta_knowledge_dir: str = "memory/storage/meta"
    log_dir: str = "log"


@dataclass
class AgentSettings:
    """Agent配置"""
    enabled_agents: List[str] = field(default_factory=lambda: [
        "student",
        "authority",
        "insight"
    ])
    default_top_k_agents: int = 3


@dataclass
class Settings:
    """
    系统配置
    
    支持从配置文件、环境变量加载配置
    """
    llm: LLMSettings = field(default_factory=LLMSettings)
    loop: LoopSettings = field(default_factory=LoopSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    
    # 调试模式
    debug: bool = False
    
    @classmethod
    def load(cls, config_dir: str = "config") -> "Settings":
        """
        加载配置
        
        Args:
            config_dir: 配置目录
            
        Returns:
            Settings实例
        """
        settings = cls()
        config_path = Path(config_dir)
        
        # 加载key.json
        key_file = config_path / "key.json"
        if key_file.exists():
            with open(key_file, 'r', encoding='utf-8') as f:
                key_config = json.load(f)
                settings.llm.api_key = key_config.get("key", "")
                settings.llm.base_url = key_config.get("base_url", "")
        
        # 加载model.json
        model_file = config_path / "model.json"
        if model_file.exists():
            with open(model_file, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
                settings.llm.model_name = model_config.get("model_name", settings.llm.model_name)
                settings.llm.temperature = model_config.get("temperature", settings.llm.temperature)
                settings.llm.max_tokens = model_config.get("max_tokens", settings.llm.max_tokens)
                settings.llm.base_url = model_config.get("base_url", settings.llm.base_url)
                settings.llm.default_headers = model_config.get("default_headers", settings.llm.default_headers)
        
        # 加载settings.json（如果存在）
        settings_file = config_path / "settings.json"
        if settings_file.exists():
            with open(settings_file, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
                settings._apply_config(full_config)
        
        # 从环境变量覆盖
        settings._apply_env_vars()
        
        return settings
    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """应用配置字典"""
        # LLM配置
        if "llm" in config:
            llm_config = config["llm"]
            self.llm.model_name = llm_config.get("model_name", self.llm.model_name)
            self.llm.temperature = llm_config.get("temperature", self.llm.temperature)
            self.llm.max_tokens = llm_config.get("max_tokens", self.llm.max_tokens)
            self.llm.timeout = llm_config.get("timeout", self.llm.timeout)
        
        # 循环配置
        if "loop" in config:
            loop_config = config["loop"]
            self.loop.max_rounds = loop_config.get("max_rounds", self.loop.max_rounds)
            self.loop.confidence_threshold = loop_config.get(
                "confidence_threshold", self.loop.confidence_threshold
            )
            self.loop.alpha = loop_config.get("alpha", self.loop.alpha)
            self.loop.beta = loop_config.get("beta", self.loop.beta)
            self.loop.exploration_prob = loop_config.get(
                "exploration_prob", self.loop.exploration_prob
            )
        
        # 存储配置
        if "storage" in config:
            storage_config = config["storage"]
            self.storage.working_memory_dir = storage_config.get(
                "working_memory_dir", self.storage.working_memory_dir
            )
            self.storage.long_term_memory_dir = storage_config.get(
                "long_term_memory_dir", self.storage.long_term_memory_dir
            )
            self.storage.meta_knowledge_dir = storage_config.get(
                "meta_knowledge_dir", self.storage.meta_knowledge_dir
            )
            self.storage.log_dir = storage_config.get("log_dir", self.storage.log_dir)
        
        # Agent配置
        if "agent" in config:
            agent_config = config["agent"]
            self.agent.enabled_agents = agent_config.get(
                "enabled_agents", self.agent.enabled_agents
            )
            self.agent.default_top_k_agents = agent_config.get(
                "default_top_k_agents", self.agent.default_top_k_agents
            )
        
        # 其他配置
        self.debug = config.get("debug", self.debug)
    
    def _apply_env_vars(self) -> None:
        """从环境变量应用配置"""
        # API密钥
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.llm.api_key = api_key
        
        # Base URL
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            self.llm.base_url = base_url
        
        # 调试模式
        debug = os.environ.get("DEBUG")
        if debug:
            self.debug = debug.lower() in ("true", "1", "yes")
    
    def save(self, config_dir: str = "config") -> None:
        """
        保存配置到文件
        
        Args:
            config_dir: 配置目录
        """
        config_path = Path(config_dir)
        config_path.mkdir(parents=True, exist_ok=True)
        
        # 保存settings.json
        settings_file = config_path / "settings.json"
        config = {
            "llm": {
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout
            },
            "loop": {
                "max_rounds": self.loop.max_rounds,
                "confidence_threshold": self.loop.confidence_threshold,
                "alpha": self.loop.alpha,
                "beta": self.loop.beta,
                "exploration_prob": self.loop.exploration_prob
            },
            "storage": {
                "working_memory_dir": self.storage.working_memory_dir,
                "long_term_memory_dir": self.storage.long_term_memory_dir,
                "meta_knowledge_dir": self.storage.meta_knowledge_dir,
                "log_dir": self.storage.log_dir
            },
            "agent": {
                "enabled_agents": self.agent.enabled_agents,
                "default_top_k_agents": self.agent.default_top_k_agents
            },
            "debug": self.debug
        }
        
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


# 全局配置实例
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    获取全局配置实例
    
    Args:
        reload: 是否重新加载配置
        
    Returns:
        Settings实例
    """
    global _settings
    if _settings is None or reload:
        _settings = Settings.load()
    return _settings

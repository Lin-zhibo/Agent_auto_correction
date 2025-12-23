"""
LLM客户端 (LLM Client)

封装大语言模型的调用接口
支持多种模型提供商：OpenAI, Azure, 本地模型等
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# 尝试导入openai，如果不存在则跳过
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class LLMConfig:
    """LLM配置"""
    api_key: str = ""
    base_url: str = "https://api.gpt.ge/v1/"
    model_name: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60
    default_headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.default_headers is None:
            self.default_headers = {"x-foo": "true"}


class LLMClient:
    """
    LLM客户端
    
    封装OpenAI API调用
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4"
    ):
        """
        初始化LLM客户端
        
        Args:
            config_path: 配置文件路径
            api_key: API密钥（优先级高于配置文件）
            base_url: API基础URL
            model_name: 模型名称
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai库未安装，请运行: pip install openai")
        
        # 加载配置
        self.config = self._load_config(config_path, api_key, base_url, model_name)
        
        # 配置OpenAI客户端
        client_kwargs = {
            "api_key": self.config.api_key,
        }
        
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        
        if self.config.default_headers:
            client_kwargs["default_headers"] = self.config.default_headers
        
        self.client = openai.OpenAI(**client_kwargs)
    
    def _load_config(
        self,
        config_path: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        model_name: str
    ) -> LLMConfig:
        """加载配置"""
        config = LLMConfig(model_name=model_name)
        
        # 从配置文件加载
        if config_path:
            config_file = Path(config_path)
        else:
            # 默认配置文件路径
            config_file = Path("config/key.json")
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.api_key = file_config.get("key", "")
                config.base_url = file_config.get("base_url", config.base_url)
                config.default_headers = file_config.get("default_headers", config.default_headers)
        
        # 从环境变量加载
        config.api_key = os.environ.get("OPENAI_API_KEY", config.api_key)
        config.base_url = os.environ.get("OPENAI_BASE_URL", config.base_url)
        
        # 参数覆盖
        if api_key:
            config.api_key = api_key
        if base_url:
            config.base_url = base_url
        
        # 验证配置
        if not config.api_key:
            raise ValueError(
                "未找到API密钥，请通过以下方式之一配置：\n"
                "1. 创建 config/key.json 文件\n"
                "2. 设置环境变量 OPENAI_API_KEY\n"
                "3. 初始化时传入 api_key 参数"
            )
        
        return config
    
    def chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        发送聊天请求
        
        Args:
            prompt: 提示词
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            模型响应文本
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_with_history(messages, model, temperature, max_tokens, **kwargs)
    
    def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        带历史记录的聊天
        
        Args:
            messages: 消息历史
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            模型响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.config.model_name,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM调用失败: {str(e)}")

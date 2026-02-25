# -*- coding: utf-8 -*-
"""
系统配置文件：API Key、模型参数、路径等。
请将 OPENAI_API_KEY 设置为你的密钥，或通过环境变量 OPENAI_API_KEY 传入。
"""

import json
import os
from pathlib import Path


def _load_conf(path: Path) -> dict:
	if not path.is_file():
		return {}
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		return {}

CONF_PATH = Path(__file__).resolve().parent / "cfg" / "conf.json"
_CONF = _load_conf(CONF_PATH)


def _conf_get(key: str, default: str) -> str:
	return _CONF.get(key, default)


# ---------- OpenAI ----------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", _conf_get("OPENAI_API_KEY", ""))
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", _conf_get("OPENAI_BASE_URL", ""))
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", _conf_get("OPENAI_MODEL", ""))
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", _conf_get("OPENAI_EMBEDDING_MODEL", ""))

# ---------- MK Judge 模型（用于双向蕴含判断） ----------
MK_JUDGE_MODEL: str = os.getenv("MK_JUDGE_MODEL", _conf_get("MK_JUDGE_MODEL", ""))

# ---------- 项目路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"
LTM_PATH = DATA_DIR / "ltm.json"
LTMK_PATH = DATA_DIR / "ltmk.json"  # 知识库（用于 Chroma RAG）
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"  # Chroma 持久化目录
MK_MEMORY_PATH = DATA_DIR / "mk_memory.json"

# ---------- MK 默认问题类型（当无法推断类型时使用） ----------
DEFAULT_QUESTION_TYPE: str = _conf_get("DEFAULT_QUESTION_TYPE", "general")

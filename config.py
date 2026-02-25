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


def _conf_get(key: str, default: str = "") -> str:
    return _CONF.get(key, default)


def _get_model_config(key: str) -> dict[str, str]:
    """读取模型配置，兼容旧格式（字符串）与新格式（对象）。"""
    raw = _CONF.get(key, "")
    if isinstance(raw, dict):
        return {
            "NAME": str(raw.get("NAME", "") or "").strip(),
            "URL": str(raw.get("URL", "") or "").strip(),
            "KEY": str(raw.get("KEY", "") or "").strip(),
        }
    if isinstance(raw, str):
        return {"NAME": raw.strip(), "URL": "", "KEY": ""}
    return {"NAME": "", "URL": "", "KEY": ""}


# ---------- OpenAI ----------
STU_MODEL = _get_model_config("STU_MODEL")
EMBEDDING_MODEL = _get_model_config("EMBEDDING_MODEL")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", _conf_get("OPENAI_API_KEY", "") or STU_MODEL["KEY"])
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", _conf_get("OPENAI_BASE_URL", "") or STU_MODEL["URL"])
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", _conf_get("OPENAI_MODEL", "") or STU_MODEL["NAME"])
OPENAI_EMBEDDING_MODEL: str = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    _conf_get("OPENAI_EMBEDDING_MODEL", "") or EMBEDDING_MODEL["NAME"] or "text-embedding-3-small",
)
OPENAI_EMBEDDING_BASE_URL: str = os.getenv(
    "OPENAI_EMBEDDING_BASE_URL",
    _conf_get("OPENAI_EMBEDDING_BASE_URL", "") or EMBEDDING_MODEL["URL"] or OPENAI_BASE_URL,
)
OPENAI_EMBEDDING_API_KEY: str = os.getenv(
    "OPENAI_EMBEDDING_API_KEY",
    _conf_get("OPENAI_EMBEDDING_API_KEY", "") or EMBEDDING_MODEL["KEY"] or OPENAI_API_KEY,
)

# ---------- 业务模型配置（兼容旧字段） ----------
MK_MODEL = _get_model_config("MK_MODEL")
MULTIAGENTS_MODEL = _get_model_config("MULTIAGENTS_MODEL")
INSIGHT_MODEL = _get_model_config("INSIGHT_MODEL")

MK_MODEL_NAME: str = os.getenv("MK_MODEL_NAME", os.getenv("MK_MODEL", "") or MK_MODEL["NAME"])
MK_MODEL_URL: str = os.getenv("MK_MODEL_URL", MK_MODEL["URL"])
MK_MODEL_KEY: str = os.getenv("MK_MODEL_KEY", MK_MODEL["KEY"])

MULTIAGENTS_MODEL_NAME: str = os.getenv(
    "MULTIAGENTS_MODEL_NAME", os.getenv("MULTIAGENTS_MODEL", "") or MULTIAGENTS_MODEL["NAME"]
)
MULTIAGENTS_MODEL_URL: str = os.getenv("MULTIAGENTS_MODEL_URL", MULTIAGENTS_MODEL["URL"])
MULTIAGENTS_MODEL_KEY: str = os.getenv("MULTIAGENTS_MODEL_KEY", MULTIAGENTS_MODEL["KEY"])

INSIGHT_MODEL_NAME: str = os.getenv("INSIGHT_MODEL_NAME", os.getenv("INSIGHT_MODEL", "") or INSIGHT_MODEL["NAME"])
INSIGHT_MODEL_URL: str = os.getenv("INSIGHT_MODEL_URL", INSIGHT_MODEL["URL"])
INSIGHT_MODEL_KEY: str = os.getenv("INSIGHT_MODEL_KEY", INSIGHT_MODEL["KEY"])

# ---------- MK Judge 模型（用于双向蕴含判断） ----------
MK_JUDGE_MODEL: str = os.getenv("MK_JUDGE_MODEL", MK_MODEL_NAME)

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

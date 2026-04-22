# -*- coding: utf-8 -*-
"""FastAPI 后端：提供 /api/suggest 与 /api/run，并托管前端静态页。"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from main import get_suggest, run_system
from utils.logger import get_logger, setup_logging

# 启动时初始化日志（创建 log 目录并配置输出）
setup_logging()

app = FastAPI(title="多 Agent 反思系统 API", version="1.0")
logger = get_logger(__name__)

# 前端静态目录
STATIC_DIR = Path(__file__).resolve().parent / "static"


class SuggestRequest(BaseModel):
    question: str


class RunRequest(BaseModel):
    question: str
    selected_agents: list[str]
    update_ltm: bool = True
    update_mk: bool = True


# Agent 显示名（供前端展示）
AGENT_LABELS = {
    "logic_checker": "逻辑检查",
    "clarity_editor": "清晰度编辑",
    "completeness_checker": "完整性检查",
    "evidence_checker": "证据与引用检查",
    "brevity_advisor": "简洁性建议",
    "consistency_checker": "一致性检查",
    "relevancy_checker": "相关性检查",
    "harmlessness_checker": "无害性检查",
    "compliance_checker": "合规性检查",
    "fluency_editor": "流畅自然优化",
    "text_writing_expert": "文案写作专家",
    "summarization_expert": "摘要总结专家",
    "code_development_expert": "代码开发专家",
    "knowledge_qa_expert": "知识问答专家",
    "educational_tutoring_expert": "教育辅导专家",
    "translation_localization_expert": "翻译本地化专家",
    "creative_ideation_expert": "创意构思专家",
    "data_processing_expert": "数据处理专家",
    "role_playing_expert": "角色扮演专家",
    "career_business_expert": "职业商业专家",
    "life_emotional_expert": "生活情感专家",
    "marketing_copywriting_expert": "营销文案专家",
    "logical_reasoning_expert": "逻辑推理专家",
    "math_computation_expert": "数学计算专家",
    "multimodal_expert": "多模态专家",
    "other_general_q_expert": "通用问答专家",
}


@app.post("/api/suggest")
def api_suggest(req: SuggestRequest):
    """根据问题返回 MK 建议的 Agent 与初答，供前端作为系统建议勾选。"""
    if not (req.question or req.question.strip()):
        logger.warning("api_suggest 请求问题为空")
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        logger.info("api_suggest 请求 question=%s", req.question[:80] + "..." if len(req.question) > 80 else req.question)
        out = get_suggest(req.question.strip())
        out["all_agents_with_labels"] = [
            {"id": a, "label": AGENT_LABELS.get(a, a)} for a in out["all_agents"]
        ]
        logger.info("api_suggest 成功 question_type=%s", out.get("question_type"))
        return out
    except Exception as e:
        logger.exception("api_suggest 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run")
def api_run(req: RunRequest):
    """使用用户选择的 Agent 进行多轮问询，并按用户选择是否更新 LTM、MK。"""
    if not (req.question or req.question.strip()):
        logger.warning("api_run 请求问题为空")
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        logger.info(
            "api_run 请求 question=%s selected_agents=%s update_ltm=%s update_mk=%s",
            req.question[:80] + "..." if len(req.question) > 80 else req.question,
            req.selected_agents,
            req.update_ltm,
            req.update_mk,
        )
        agents = req.selected_agents if req.selected_agents else None
        final_answer = run_system(
            req.question.strip(),
            selected_agents=agents,
            do_update_ltm=req.update_ltm,
            do_update_mk=req.update_mk,
        )
        logger.info("api_run 成功")
        return {
            "final_answer": final_answer,
            "success": True,
            "update_ltm": req.update_ltm,
            "update_mk": req.update_mk,
        }
    except Exception as e:
        logger.exception("api_run 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def index():
    """返回前端页面。"""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="前端页面未找到")
    return FileResponse(index_file)


@app.get("/health")
def health():
    return {"status": "ok"}

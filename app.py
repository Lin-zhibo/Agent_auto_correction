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
    "questioner": "质疑者",
    "logic_analyzer": "逻辑分析者",
    "authority_checker": "事实核查者",
    "explainer": "解释者",
    "economics_expert": "经济学专家",
    "math_expert": "数学专家",
    "philosophy_expert": "哲学专家",
    "science_expert": "科学专家",
    "critic": "批判者",
    "supporter": "追随者/支持者",
    "clarity_checker": "清晰度检查",
    "completeness_checker": "完整性检查",
    "evidence_checker": "证据与引用检查",
    "brevity_advisor": "简洁性建议",
    "audience_advisor": "受众适配",
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

# 多 Agent 反思系统（减少幻觉、优化回答）

具备自我反思、多角度审视、逐步优化回答的智能 Agent 系统。  
流程：**Student 初答 → MK 选 Agent → 多 Agent 反馈 → Insight 整合 → Student 修正 → 循环/输出 → 更新 LTM**。

## 环境与依赖

- Python 3.10+
- 依赖：`pip install -r requirements.txt`

## 配置

1. 复制或编辑项目根目录下的 **`config.py`**，设置 OpenAI 相关配置：
   - **`OPENAI_API_KEY`**：必填，也可通过环境变量 `OPENAI_API_KEY` 设置。
   - **`OPENAI_BASE_URL`**：可选，默认 `https://api.openai.com/v1`（兼容代理或其它兼容接口）。
   - **`OPENAI_MODEL`**：对话模型，默认 `gpt-4o-mini`。
   - **`OPENAI_EMBEDDING_MODEL`**：语义相似度用，默认 `text-embedding-3-small`。
2. 其它如 `MAX_LOOPS`、`SIMILARITY_THRESHOLD`、`IMPROVEMENT_MIN`、Agent 优先级等均在 `config.py` 中可调。

## 运行

### 命令行

在项目根目录执行：

```bash
# 默认问题：「什么是机器学习？」
python main.py

# 自定义问题
python main.py "为什么深度学习需要大量数据？"
```

输出为经过多轮反思后的最终答案，并会自动更新 `data/ltm.json`（长期记忆）。

### 前端页面

提供 Web 界面：MK 建议的 Agent 作为系统勾选，支持用户自选 Agent 进行多轮问询，并可选择是否更新 LTM、MK。

```bash
# 安装依赖后
pip install -r requirements.txt
# 启动后端（含前端静态页）
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

浏览器访问 **http://localhost:8000**：

1. 输入问题，点击「获取建议」→ 显示初答与 MK 建议的 Agent（已勾选）。
2. 可勾选/取消勾选任意 Agent，并勾选「更新 LTM」「更新 MK」。
3. 点击「开始多轮问询」→ 使用所选 Agent 多轮优化，并按选项决定是否更新 LTM/MK；最终答案展示在下方。

## 项目结构概览

- **`config.py`**：API Key、模型、路径、MK 策略参数等配置。
- **`main.py`**：主流程 `run_system(question)`。
- **`memory/`**：LTM 加载/保存/更新、WM、RAG 树形检索（`rag_search.py`、`ltm.py`、`wm.py`）。
- **`agents/`**：Student、Meta-Knowledge、多角色 Agent（Questioner / LogicAnalyzer / AuthorityChecker / Explainer）、Insight、AgentFactory。
- **`utils/`**：OpenAI 调用（`llm_call`）、语义相似度（`semantic_similarity`）、改进量（`compute_improvement`）。
- **`data/`**：`ltm.json`（长期知识）、`mk_memory.json`（MK 策略/元数据，可扩展）。

## 流程简述

1. **Student Agent** 结合 LTM（RAG 检索）生成初答，并计算置信度与一致性。
2. **Meta-Knowledge (MK)** 根据问题与置信度/一致性选择参与反思的 Agent（最多 3 个）。
3. **多 Agent 阶段** 各角色对当前答案进行审视并给出反馈。
4. **Insight Agent** 整合反馈，生成改写后的答案。
5. **MK** 根据相似度、改进量、循环次数决定是否继续；若继续则用新答案进入下一轮。
6. 满足停止条件后输出最终答案，并调用 **`update_ltm`** 更新长期记忆。

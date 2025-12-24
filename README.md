# Agent 自动纠错系统

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于多Agent协作的自动纠错系统，通过循环反思机制实现答案的持续优化。

## 目录

- [项目概述](#项目概述)
- [架构设计](#架构设计)
- [结构体设计](#结构体设计)
- [Agent模块](#agent模块)
- [环境依赖](#环境依赖)
- [快速启动](#快速启动)
- [后续规划](#后续规划)
- [注意事项](#注意事项)

---

## 项目概述

### 背景

在大语言模型（LLM）应用中，单次生成的答案往往存在不确定性和错误。本项目通过多Agent协作机制，实现答案的自动检验和迭代优化，提高输出质量和可靠性。

### 核心目标

1. **多Agent协作**：通过多种角色的Agent（权威者、质疑者、逻辑分析员等）对答案进行多角度审视
2. **循环反思**：基于置信度和错误率动态调整循环次数，直到达到满意的答案
3. **知识积累**：通过长期记忆存储历史经验，持续改进系统表现

### 适用场景

- 需要高可靠性的问答系统
- 复杂推理任务的自动验证
- 教育场景中的答案评估和纠正
- 任何需要迭代优化输出的LLM应用

---

## 架构设计

### 逻辑拓扑（基于方案架构图）

```
                    ┌─────────────────────────────────────────────────┐
                    │                  方案架构                         │
                    └─────────────────────────────────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
          ▼                               ▼                               ▼
    ┌──────────┐                   ┌──────────────┐                ┌──────────┐
    │   WM     │◄─────────────────►│ Student Agent│◄──────────────►│   LTM    │
    │ 工作记忆  │                   │   学生Agent   │                │ 长期记忆  │
    └──────────┘                   └──────┬───────┘                └──────────┘
          │                               │
          │                               │ 决定
          │                               ▼
          │                        ┌──────────────┐
          │                        │     MK       │
          │                        │   元知识     │
          │                        └──────┬───────┘
          │                               │ 决定
          │                               ▼
          │         ┌─────────────────────────────────────────────┐
          │         │              多Agent阶段                     │
          │         │  ┌─────────┐ ┌─────────┐ ┌─────────────┐    │
          │         │  │ 权威者  │ │ 质疑者  │ │ 逻辑分析员  │    │
          │         │  └─────────┘ └─────────┘ └─────────────┘    │
          │         │  ┌─────────┐ ┌─────────┐ ┌─────────────┐    │
          │         │  │ 倾听者  │ │  同伴   │ │启发式解决者 │    │
          │         │  └─────────┘ └─────────┘ └─────────────┘    │
          │         │  ┌─────────────┐ ┌─────────────┐            │
          │         │  │ 解释生成者 │ │ 概念类比者  │  ...       │
          │         │  └─────────────┘ └─────────────┘            │
          │         └─────────────────────┬───────────────────────┘
          │                               │
          │                               ▼
          │                        ┌──────────────┐
          │                        │ Insight Agent│
          │                        │  洞察Agent   │
          │                        └──────┬───────┘
          │                               │
          │                               │ 反馈
          │                               ▼
          │                        ┌──────────────┐
          └────────────────────────┤  循环反思    │
                                   └──────────────┘
```

### 核心流程

1. **Student Agent 作答**：结合长期记忆（LTM）中的知识，对问题进行初步回答
2. **Meta-Knowledge 决策**：根据问题类型和历史表现，选择参与的专家Agent
3. **多Agent 反馈**：各专家Agent从不同角度对答案进行评估
4. **Insight Agent 汇总**：提取冲突、互补观点，生成反馈提示词
5. **循环反思**：基于反馈更新答案，直到置信度达标或达到最大轮次

### 核心模块划分

| 模块 | 路径 | 职责 |
|------|------|------|
| Agent模块 | `agent/` | 实现各类Agent |
| 配置管理 | `config/` | 系统配置和API密钥 |
| 核心编排 | `core/` | 流程控制、协调、数据结构定义 (`schemas.py`) |
| 数据集 | `data/` | 预留给数据集 |
| 数据库 | `db/` | 存储检索的索引 |
| 文档 | `doc/` | 预留给文档 |
| 日志信息 | `log/` | 存放日志信息 |
| 记忆系统 | `memory/` | 管理WM、LTM、MK三种记忆 |
| 本地模型 | `models/` | 预留给本地模型文件存放 |
| 环境配置脚本 | `script/` | 预留给环境配置脚本 |
| 调试代码 | `test/` | 用于调试代码正确性 |
| 工具模块 | `utils/` | LLM客户端、日志等 |

---

## 结构体设计

> 📁 数据结构定义位于 `core/schemas.py`

### 设计原则

1. **命名规范**：遵循Python PEP8，使用snake_case
2. **字段合理性**：剔除冗余字段，补充必要的元数据
3. **层级划分**：复杂结构拆分为独立子结构
4. **扩展性**：预留多Agent扩展接口

### 使用方式

```python
from core.schemas import WorkingMemory, MetaKnowledge, AgentType
# 或通过 core 包导入
from core import WorkingMemory, MetaKnowledge
```

### 核心结构体

#### Working Memory (工作记忆)

存储当前任务的临时状态。

```python
@dataclass
class WorkingMemory:
    task_id: str                    # 任务唯一标识
    question_text: str              # 问题文本
    round_index: int                # 当前轮次 (从1起)
    current_phase: str              # 当前阶段
    agent_outputs: List[AgentOutput]  # Agent输出列表
    conflicts_detected: List[str]   # 检测到的冲突
    complementary_points: List[str] # 互补观点
    summary_prompt: str             # 汇总提示词
    student_response: str           # Student回答
    student_confidence: float       # 置信度 (0-1)
    round_history: List[Dict]       # 历史轮次
    created_at: datetime
    updated_at: datetime
```

**优化说明**：
- 新增 `current_phase` 标记处理阶段
- 新增 `round_history` 记录历史
- `agent_outputs` 使用结构化的 `AgentOutput` 类型

#### Long-Term Memory (长期记忆)

存储知识经验和错误模式。

```python
@dataclass
class LongTermMemoryEntry:
    entry_id: str                   # 条目唯一标识
    topic_category: str             # 主题类别
    content_pattern: str            # 内容模式
    error_type: str                 # 错误类型
    error_examples: List[ErrorExample]  # 错误案例
    correction_strategies: List[str]    # 纠正策略
    thinking_angles: List[str]      # 思考角度
    success_rate: float             # 成功率
    usage_count: int                # 使用次数
    update_timestamp: datetime
```

**优化说明**：
- 新增 `entry_id` 便于索引
- 新增 `usage_count` 记录使用频率
- `error_examples` 使用独立的 `ErrorExample` 结构

#### Meta-Knowledge (元知识)

存储Agent表现和决策参数。

```python
@dataclass
class MetaKnowledge:
    question_type: str              # 问题类型
    error_rate: float               # 历史错误率 (E)
    avg_confidence: float           # 平均置信度
    agent_performance_log: Dict[str, AgentPerformanceRecord]
    agent_recommendation_list: List[str]  # 推荐Agent
    exploration_prob: float         # 探索概率
    # 循环控制参数
    max_rounds: int                 # 最大轮次 (N_max)
    confidence_threshold: float     # 置信度阈值 (τ_c)
    alpha: float                    # 参数 α
    beta: float                     # 参数 β
```

**循环控制公式**：
$$N = \min\left(N_{max}, \lceil \alpha \cdot E \cdot (1-C)^{\beta} \cdot N_{max} \rceil\right)$$

**停止条件**：$C \geq \tau_c$



---

## Agent模块

### 已实现的Agent

#### 1. Student Agent (学生Agent)

**功能**：作为主答题者，结合LTM知识对问题进行回答。

**核心提示词逻辑**：
- 角色：严谨的问题解答者
- 输入：问题、LTM知识、历史反馈
- 输出：JSON格式（analysis, reasoning, answer, confidence）
- 异常处理：解析失败时返回原始响应

**使用示例**：
```python
from agent.student_agent import StudentAgent
from utils.llm_client import LLMClient

llm = LLMClient(api_key="your-key")
student = StudentAgent(llm_client=llm)

response = student.execute(
    question="什么是机器学习?",
    context={"ltm_knowledge": {...}}
)
print(response.content)       # 答案
print(response.confidence)    # 置信度
```

#### 2. Authority Agent (权威者Agent)

**功能**：从专业角度审视答案，提供权威性评估。

**核心提示词逻辑**：
- 角色：领域权威专家
- 输入：问题、Student回答、置信度
- 输出：JSON格式（professional_assessment, expert_opinion, error_analysis, recommendations）
- 评估维度：准确性、概念错误、逻辑漏洞

#### 3. Insight Agent (洞察Agent)

**功能**：汇总各Agent输出，生成综合反馈。

**核心提示词逻辑**：
- 角色：Meta-Synthesizer（元综合者）
- 输入：问题、Student回答、各Agent输出
- 输出：JSON格式（conflicts, complementary_insights, feedback_prompt, recommended_action）

#### 4. Questioner Agent (质疑者Agent)

**功能**：挑战现有答案，从批判性角度提出质疑。

**核心提示词逻辑**：
- 角色：批判性思维专家
- 输入：问题、Student回答、置信度
- 输出：JSON格式（challenges, severity_assessment, suggested_improvements）
- 评估维度：隐含假设、边界条件、反例场景

#### 5. Logic Analyst Agent (逻辑分析员Agent)

**功能**：分析推理链条，检测逻辑漏洞和谬误。

**核心提示词逻辑**：
- 角色：逻辑推理专家
- 输出：JSON格式（logic_chain_analysis, validity_score, fallacies_detected）
- 检测：因果倒置、循环论证、滑坡谬误等

#### 6. Listener Agent (倾听者Agent)

**功能**：综合各方意见，寻求共识。

**核心提示词逻辑**：
- 角色：协调综合者
- 输出：JSON格式（consensus_points, divergence_points, synthesis）
- 职责：提取共识、识别分歧、提出整合方案

#### 7. Companion Agent (同伴Agent)

**功能**：提供协作支持和情感关怀。

**核心提示词逻辑**：
- 角色：学习伙伴
- 输出：JSON格式（encouragement, collaborative_insights, areas_for_improvement）
- 风格：鼓励性、建设性

#### 8. Heuristic Solver Agent (启发式解决者Agent)

**功能**：提供创造性解决方案和替代思路。

**核心提示词逻辑**：
- 角色：创意问题解决专家
- 输出：JSON格式（alternative_approaches, creative_solutions, heuristic_strategies）
- 方法：类比迁移、逆向思维、分治策略

#### 9. Explanation Generator Agent (解释生成者Agent)

**功能**：生成多层次、易理解的解释。

**核心提示词逻辑**：
- 角色：教育解释专家
- 输出：JSON格式（eli5_explanation, intermediate_explanation, detailed_explanation）
- 层次：简单(ELI5)、中级、详细

#### 10. Concept Analogist Agent (概念类比者Agent)

**功能**：通过类比帮助理解抽象概念。

**核心提示词逻辑**：
- 角色：类比思维专家
- 输出：JSON格式（analogies, mappings, limitations）
- 方法：结构映射、跨领域类比

---

## 环境依赖

### 系统要求

- Python 3.8+
- 可访问的OpenAI API (或兼容的API服务)

### 依赖包

```bash
# 核心依赖
openai>=1.0.0

# 可选依赖
tiktoken>=0.5.0      # Token计数
numpy>=1.24.0        # 数值计算
```

---

## 快速启动

### 1. 克隆项目

```bash
git clone https://github.com/Lin-zhibo/Agent_auto_correction.git
cd agent_auto_correction
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置API密钥

创建 `config/key.json` 文件：

```json
{
    "key": "your-openai-api-key",
    "base_url": ""
}
```

### 4. 运行系统

**基础模式** (`main.py`)：
```bash
python main.py
```

**调试模式** (`shell.py`)：
```bash
python shell.py
```

**单次执行**：
```bash
python main.py --question "什么是人工智能?" --type knowledge
```

### 5. Shell 命令说明

`shell.py` 提供了交互式调试环境：

| 命令 | 说明 |
|------|------|
| `ask <问题>` | 完整执行，显示详细的每轮调试信息 |
| `askq <问题>` | 快速执行，只显示最终结果 |
| `test [verbose]` | 运行测试用例，可选详细输出 |
| `status` | 查看系统状态 |
| `clear` | 清空记忆数据 |
| `help` | 显示帮助信息 |
| `quit` | 退出系统 |

**示例**：
```bash
(agent) >>> ask 什么是机器学习？
# 显示每轮的 Student 回答、专家评估、Insight 汇总等详细信息

(agent) >>> test verbose
# 运行测试用例并显示详细过程
```

### 6. 验证运行

成功运行后，您将看到：

```
============================================================
欢迎使用 Agent 自动纠错系统
输入问题开始处理，输入 'quit' 或 'exit' 退出
============================================================

请输入问题: 
```

---

## 后续规划

### 多Agent扩展路径

1. **Phase 1** ✅ 已完成: 实现核心框架和基础Agent (Student, Authority, Insight)
2. **Phase 2** ✅ 已完成: 实现全部8个专家Agent (Questioner, LogicAnalyst, Listener, Companion, HeuristicSolver, ExplanationGenerator, ConceptAnalogist)
3. **Phase 3** (进行中): 实现Agent动态选择和协作策略优化（MAB-RAG）
4. **Phase 4**: 加入语义熵迭代控制机制增强agent的自我判断能力

### 结构体迭代方向

1. LTM、Meta-Knowledge升级为“海马体图谱” (基于 HippoRAG, LORIL)
2. LTM、Meta-Knowledge增加关键词搜索技术（精确匹配->BM25缩减匹配->向量检索兜底）

### 功能完善计划

**已完成**：
- [x] 实现核心框架和基础Agent (Student, Authority, Insight)
- [x] 全部8个专家Agent的Prompt实现
- [x] Insight Agent 综合评估机制 (quality_score, correctness, recommended_action)
- [x] 循环反思机制（最少2轮保障）
- [x] 调试模式 (shell.py verbose输出)
- [x] WM → LTM/MK 学习更新

**待完成**：
- [ ] 实现Agent动态选择和协作策略优化（MAB-RAG）
- [ ] LTM、Meta-Knowledge升级为“海马体图谱” (基于 HippoRAG, LORIL)
- [ ] LTM、Meta-Knowledge增加关键词搜索技术（精确匹配->BM25缩减匹配->向量检索兜底）
- [ ] 加入语义熵迭代控制机制增强agent的自我判断能力
- [ ] 数据集下载
- [ ] 环境安装脚本编写
- [ ] 本地模型下载（可能）

## 注意事项

### 潜在风险

1. **API成本**：多Agent协作会增加API调用次数，注意监控使用量
2. **循环死锁**：极端情况下可能出现置信度无法提升的情况，已设置最大轮次限制
3. **响应延迟**：多轮循环会增加响应时间，建议设置合理的超时

### 调试技巧

1. 使用 `shell.py` 的 `ask` 命令获取详细的每轮执行信息
2. 使用 `test verbose` 命令查看测试用例的详细执行过程
3. 查看 `log/` 目录获取历史执行记录
4. 检查 `memory/storage/` 目录了解记忆状态
5. 使用 `status` 命令查看当前系统配置和记忆统计

### 编码规范

- 遵循 PEP8 编码规范
- 所有公开接口需要完整的文档字符串
- 新增Agent需继承 `BaseAgent` 并实现 `execute()` 和 `_build_prompt()`
- 提交前运行类型检查：`mypy .`

### 假设前提

以下设计基于合理默认假设：

1. **Agent选择策略**：默认使用ε-greedy策略（exploration_prob=0.1）
2. **循环参数**：默认 α=1.0, β=1.0, N_max=10, τ_c=0.8
3. **问题类型**：默认支持 general, math, logic, knowledge 四种类型
4. **LLM模型**：默认使用 gpt-4，可通过配置修改

---

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
